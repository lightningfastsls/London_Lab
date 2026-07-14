# CNN Detection Pipeline (Production)

> **What this is:** The production USV detection CNN (`models/hard_neg_retrain/best_model.pt`)
> plus its full post-processing chain — temperature calibration, hysteresis event detection,
> false-positive (FP) filter, soft-notch, and recording-level triage.
> **Status:** CURRENT (production model dated 2026-03-31, FP-filter/hysteresis re-fit 2026-04-01).
> **Production artifact:** `models/hard_neg_retrain/best_model.pt`
> **DO NOT USE:** `models/matched_windows/best_model.pt`, `models/production/best_model.pt` (older baselines — see [§2.7](#27-deprecated-models)).
> **Companion doc:** [batch_detection.md](batch_detection.md) (batch runner mechanics).

---

## CRITICAL DISTINCTION — read this first

There are **two different detection paths** in this repo and they produce **different results**.
Conflating them has caused real bugs (an FP-filter applied in the app context dropped 8 real
detections to 0 on file `0004954`).

| | **PyQt6 App "Detect" button** | **Batch pipeline (`run_batch_detection.py`)** |
|---|---|---|
| CNN sliding window | yes | yes |
| MAD normalization | yes (global, whole spectrogram) | yes (global, whole spectrogram) |
| Energy gate | **0.35** (`main_window.py:72`) | **0.1** (`SlidingInference` default, `sliding_inference.py:48`) |
| Temperature calibration | **NO** (raw sigmoid probs) | yes (if `--temperature` passed) |
| Hysteresis | yes — `HysteresisDetector` (`detection_logic.py`, app variant) | yes — `hysteresis_detect` (`postprocessing/hysteresis.py`, batch variant) |
| FP-filter | **NO** | yes (if `--fp-filter` passed) |
| Soft-notch | **NO** | optional (`--soft-notch`, lab-only) |
| Triage tiers | **NO** | yes |

**Rule of thumb:** "What the app shows" = CNN + hysteresis on **raw** probabilities, energy gate
0.35. To reproduce the app, do **not** apply the temperature/FP-filter/soft-notch artifacts.
The FULL pipeline (calibrated probs + FP-filter + triage) is **batch-only**.

The two hysteresis implementations also differ — see [§2.4](#24-two-hysteresis-implementations).

---

## 1. Operate

### 1.1 Required environment

- Interpreter: `.venv/bin/python` (Linux/WSL).
- Signal-processing constants are fixed in `src/usv_spectrogram/corpus.py`
  (`SAMPLE_RATE_HZ = 300_000`, `STFT_N_FFT = 512`, `STFT_HOP = 128`,
  `USV_FREQ_MIN_HZ = 20_000`, `USV_FREQ_MAX_HZ = 120_000` — `corpus.py:30-36`).
  Never override these; the CNN was trained against this exact grid.

### 1.2 Running the FULL production pipeline (batch)

This is the canonical command. **All five flags are required for correct results.** Omitting
`--fp-filter` or `--hysteresis-config` produces an incomplete pipeline with unreliable triage.

```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir <WAV_FOLDER>/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_<NAME>/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

#### CLI flags (`run_batch_detection.py`, `main()` at line 612)

| Flag | Required | Default | What it does |
|------|----------|---------|--------------|
| `--wav-dir` | yes | — | Directory of WAVs, searched **recursively** (`**/*.wav`, line 361). |
| `--model` | yes | — | Path to CNN `.pt`. Use the production model above. |
| `--output-dir` | yes | — | Writes `summary.parquet` + `detections/*.json` here. |
| `--temperature` | no | `None` | Fitted `TemperatureScaler` JSON. Calibrates logits → probs. **Required for production correctness.** |
| `--fp-filter` | no | `None` | Fitted `FalsePositiveFilter` **pickle** (`.pkl`, not `.json`). Second-stage USV-vs-FP classifier. **Required for production correctness.** |
| `--hysteresis-config` | no | `None` → `HysteresisConfig()` defaults | JSON with `best_params`. **Required** — the dataclass defaults (`onset 0.75 / sustain 0.40 / gap 3 / min_dur 5`, `hysteresis.py:36-40`) are NOT the production values. |
| `--workers` | no | `1` | Parallel worker processes. Use `4`. |
| `--no-resume` | no | off | Reprocess files even if their JSON already exists (default behavior skips them, line 373-379). |
| `--subtract-baseline` | no | off | **Lab-only.** Per-bin temporal-baseline subtraction before the CNN. Wild runs (5970/3452/9252) MUST omit for byte-identical results (line 651-660). |
| `--subtraction-method` | no | `percentile` | `percentile` (Boll 1979 floor, p10) or `median_envelope`. Only consulted with `--subtract-baseline` (line 661-670). |
| `--soft-notch` | no | off | **Lab-only.** Pass a `TonalLibrary` JSON path or the literal `auto`. Removes rig-specific equipment tonals before STFT. Wild runs MUST omit (line 671-682). |

### 1.3 Inputs

- WAV files at 300 kHz. There is **no single canonical WAV directory** — recordings span
  `5970 USV/`, `USV_3452_sample_reviewed/`, `USV_9252/`, etc. (see `docs/DATA_LOCATIONS.md`).
- Model + companion artifacts: all in `models/hard_neg_retrain/` (see [§1.6](#16-companion-artifacts-reference)).

### 1.4 Outputs

Written under `--output-dir`:

| File | Producer | Contents |
|------|----------|----------|
| `detections/<stem>.json` | `_process_and_save_one` (line 247-253) | One file per WAV, written immediately (crash-safe). List of events in ADR-010 dict format. |
| `summary.parquet` | `_write_summary_parquet` (line 506) | One row per recording with tier + QC metrics. |
| `soft_notch_applied.parquet` | only with `--soft-notch` (line 522) | One row per applied/audit notch event. |
| `soft_notch_summary.json` | only with `--soft-notch` | Library metadata + stale-library warning. |

**`summary.parquet` columns** (from `RecordingResult`, populated at `run_batch_detection.py:319-330`):

| Column | Meaning |
|--------|---------|
| `filepath` | Source WAV path. |
| `tier` | `auto_accept` / `manual_review` / `auto_reject` (see [§1.5](#15-triage-tiers)). |
| `n_events` | Number of surviving events (after hysteresis + FP-filter). |
| `max_confidence` | Max `peak_probability` across events (0.0 if none). |
| `mean_event_confidence` | Mean per-event `peak_probability` (0.0 if none). |
| `total_usv_duration_ms` | Sum of event `duration_ms` (center-to-center). |
| `noise_floor_p90` | 90th-percentile window probability across the recording. |
| `confidence_score` | Equals `mean_event_confidence` (`triage.py:232`). |
| `qc_flags` | List of flags: `high_noise_floor`, `high_event_count`, `long_event_duration`, `high_total_usv_duration`, `event_spans_most_of_recording`, `outlier_event_count`. |

**Per-event JSON fields** (ADR-010, via `_event_to_adr010_dict`): see
`postprocessing/batch_output.py`. Each event carries start/end time, duration, and
peak/mean probability derived from the `USVEvent` dataclass (`hysteresis.py:56-73`).

### 1.5 Triage tiers

Assigned per recording by `triage_recording` (`postprocessing/triage.py:140`). Order matters
(`triage.py:216-229`):

1. `n_events == 0` → **`auto_reject`** (no USVs detected).
2. else if `max(probabilities) <= auto_reject_max_window` (0.10) → **`auto_reject`** (clearly empty).
3. else if any `long_event_duration` or `event_spans_most_of_recording` flag → **`manual_review`**.
4. else if **every** event has `peak_probability >= auto_accept_min_peak` (0.90) → **`auto_accept`**.
5. else → **`manual_review`**.

`TriageConfig` defaults (`triage.py:50-57`) — these are the live thresholds (the runner never
overrides `TriageConfig`, line 393-394):

| Field | Default | Role |
|-------|---------|------|
| `auto_accept_min_peak` | **0.90** | Every event must clear this to auto-accept. |
| `auto_reject_max_window` | **0.10** | Whole-recording prob ceiling for auto-reject. |
| `noise_floor_p90_threshold` | **0.4** | p90 prob above this raises `high_noise_floor`. |
| `outlier_count_zscore` | **2.0** | Batch-level event-count z above this flags `outlier_event_count`. |
| `max_event_duration_ms` | **600.0** | Any event longer → `long_event_duration` → review. |
| `total_duration_review_ms` | **600.0** | Summed duration over this → `high_total_usv_duration`. |
| `high_event_count_threshold` | **10** | More events than this → `high_event_count`. |
| `max_event_fraction_of_recording` | **0.8** | Event spanning ≥80% of timeline → `event_spans_most_of_recording` → review. |

### 1.6 Companion artifacts reference

All in `models/hard_neg_retrain/`. Read each before relying on it.

#### `temperature.json` (calibration)
```json
{ "temperature": 0.9019383780691683, "fitted": true,
  "nll_before": 0.16908450424671173, "nll_after": 0.16809050738811493 }
```
Applied in `TemperatureScaler.calibrate` (`calibration.py:72`): `probs = sigmoid(logits / T)`.
T = 0.902 (< 1) **sharpens** predictions slightly. Loaded only when `--temperature` is passed
(`run_batch_detection.py:135-136`); requires `return_logits=True` from inference (line 129).
The app does **not** request logits, so the app never applies temperature.

#### `fp_filter.pkl` / `fp_filter.json`
- `.pkl` = the pickled `FalsePositiveFilter` object (`StandardScaler → LogisticRegression`,
  `class_weight="balanced"`, `C=1.0`, `max_iter=1000` — `fp_filter.py:58-65`). **This is the
  artifact the pipeline loads** (`_load_fp_filter`, line 83-86).
- `.json` = the training/CV record (NOT loaded at inference). Headline from `fp_filter.json`:
  mean F2 **0.8233** ±0.0352 over 5 folds, `n_events=1319`, `n_usv=1234`, `n_fp=85`.
  Top feature importances: `peak_probability` 1.65, `mean_probability` 0.56, `prob_kurtosis` 0.36,
  `prob_std` 0.31, `duration_windows` 0.25.
- The FP-filter consumes 11 `EventFeatures` (`event_features.py:31-43`): `peak_probability`,
  `mean_probability`, `prob_std`, `prob_kurtosis`, `prob_roughness`, `duration_windows`,
  `tonality`, `mean_peak_freq_bin`, `freq_range_bins`, `freq_modulation_rate`, `snr_db`.

#### `fp_filter_no_duration.pkl` / `.json` (variant — added 2026-04-12)
Same model but **excludes the `duration_windows` feature** (`excluded_features: ["duration_windows"]`).
Trains a 10-feature model so duration cannot drive the USV/FP decision (useful when event
duration is an artifact of hysteresis settings rather than biology). CV mean F2 **0.8333** ±0.0244.
**Not** wired into the canonical command above — use `fp_filter.pkl` unless you have a specific
reason to exclude duration.

#### `hysteresis_optimization_v2.json` (event thresholds — the production values)
`best_params` (`hysteresis_optimization_v2.json`):

| Param | Value | Role (`hysteresis.py:23-40`) |
|-------|-------|------------------------------|
| `onset_threshold` | **0.6** | Seed gate — start an event when prob ≥ this. |
| `sustain_threshold` | **0.4** | Extend gate — grow event while prob ≥ this. |
| `gap_fill_windows` | **0** | Merge events separated by ≤ this many windows (0 = no merging). |
| `min_duration_windows` | **3** | Drop events shorter than 3 windows. |

Selected at CV mean F2 **0.8669** ±0.0506. The file also records a more conservative
`one_se_params` (`onset 0.8 / sustain 0.5 / min_dur 9`, F2 0.8295) — **not** the production
choice. The runner reads `best_params` (`_load_hysteresis_config`, line 89-100); `max_duration_ms`
falls back to **600.0** since the JSON omits it.
(`hysteresis_optimization.json` without the `_v2` suffix is the older v1 result — use `_v2`.)

#### `training_history.json` + `evaluation/`
Held-out test metrics from `evaluation/test_metrics.json`:

| Metric | Value |
|--------|-------|
| accuracy | 0.9388 |
| **precision** | **0.9055** |
| **recall** | **0.8854** |
| f1 | 0.8953 |
| specificity | 0.9612 |
| TP / FP / TN / FN | 479 / 50 / 1238 / 62 |

(These are **window/patch-level** classifier metrics, not whole-file detection rates.)

### 1.7 Worked example (wild cohort, full pipeline)

```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir "5970 USV/" \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_5970/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

Produces `results/batch_5970/detections/<stem>.json` (per file) and
`results/batch_5970/summary.parquet`. Inspect the tier split in the log line
`Triage distribution: {...}` (`run_batch_detection.py:501`).

### 1.8 Troubleshooting / Gotchas

- **"FP-filter killed my detections" / 8 → 0.** You applied the batch FP-filter in an
  app-style context. The app's Detect is CNN + hysteresis only — do not run the FP-filter to
  reproduce what the app shows. (Recorded for file `0004954`.)
- **Wrong / too few events vs the app.** The app uses energy gate **0.35** (`main_window.py:72`)
  while batch uses the `SlidingInference` default **0.1**. They also use different hysteresis
  implementations ([§2.4](#24-two-hysteresis-implementations)). Expect non-identical output.
- **Hysteresis "defaults" are wrong.** Omitting `--hysteresis-config` silently uses the
  dataclass defaults (`onset 0.75`, `min_dur 5`), **not** production (`onset 0.6`, `min_dur 3`).
  Always pass `hysteresis_optimization_v2.json`.
- **`.pkl` vs `.json` for the FP-filter.** Pass the **`.pkl`** to `--fp-filter`. The `.json` is a
  training record and is not loadable by `_load_fp_filter`.
- **Temperature has no effect in the app.** The app calls `inference.infer(...)` without
  `return_logits` (`main_window.py:77-80`), so temperature scaling is structurally impossible there.
- **Lab vs wild flags.** `--subtract-baseline` and `--soft-notch` are **lab-only**. Adding either
  to a wild run breaks byte-identical reproducibility (5970/3452/9252).
- **Resume skips files.** By default, files with an existing `detections/<stem>.json` are skipped
  (line 373-379). Use `--no-resume` to force a clean re-run.
- **Probabilities must be in [0,1].** `hysteresis_detect` raises if passed raw logits
  (`hysteresis.py:111-114`) — calibration/sigmoid must happen first.

---

## 2. Internals

### 2.1 Data flow

```
WAV → AudioLoader → spectrogram_db (freqs × times)
    → SlidingInference.infer()                              [global MAD norm → windows → CNN]
    → probabilities (+ logits if return_logits)
    → [TemperatureScaler.calibrate(logits)]                 batch-only, if --temperature
    → hysteresis_detect(probs, times, HysteresisConfig)     → list[USVEvent]
    → [extract_event_features + FalsePositiveFilter.predict] batch-only, if --fp-filter
    → triage_recording(...)                                 → RecordingResult (tier + QC)
    → JSON + parquet
```
Orchestrated by `process_one_recording` (`run_batch_detection.py:107`).

### 2.2 Sliding-window CNN inference

`SlidingInference` — `src/usv_spectrogram/app/core/sliding_inference.py` (424 lines).

- `__init__` (line 41): `window_width_px=100` (~43 ms at 300 kHz), `hop_px=10`, `batch_size=32`,
  `energy_threshold=0.1`, `enable_per_window_norm=False`.
- `infer` (line 119): generates window centers, runs the CNN in batches, returns
  `InferenceResult(probabilities, column_indices, times, logits)` (line 23-30).
- `_load_model` (line 79): `torch.load(..., weights_only=False)`, instantiates `USVClassifierCNN`.

#### INVARIANT — global MAD normalization (`_apply_mad_normalization`, line 388)

> The whole spectrogram is MAD-normalized **once** *before* windows are cropped (line 147-152).

```python
median = np.median(spec_db)
mad    = np.median(np.abs(spec_db - median))
vmin   = median - 2.0 * mad          # mad_vmin_scale = 2.0  (line 409)
vmax   = median + 4.0 * mad          # mad_vmax_scale = 4.0  (line 410)
spec_clipped = np.clip(spec_db, vmin, vmax)   # CLIP before normalize (line 416)
spec_norm    = (spec_clipped - vmin) / (vmax - vmin + 1e-12)
```

This preserves the **global** energy distribution so quiet regions stay quiet and loud USV
regions stay loud. Per-window MAD (the wrong thing) silently kills high-confidence USVs — see
the memory note `feedback_cnn_inference_global_mad`. The clip-before-normalize order must match
the training pipeline's `_render_training`.

#### `_prepare_batch` (line 274) — must match training EXACTLY
magma colormap → vertical flip (`np.flipud`) → uint8 → resize to **256 px height** (LANCZOS) →
grayscale (`0.299R + 0.587G + 0.114B`) → per-image min/max norm. Skipping the 256-px resize or
the flip silently shifts the input distribution.

#### Energy pre-filter (`_should_skip_window_by_energy`, line 376)
Windows whose max < `energy_threshold` skip the CNN and get probability 0.0 (logit −20.0,
line 178-180). App = 0.35; batch = 0.1.

### 2.3 Temperature calibration

`TemperatureScaler` — `postprocessing/calibration.py:20`. `calibrate` (line 72) divides logits by
T then sigmoid (numerically stable, line 90-95). Production T = **0.902**. `load`/`save` are JSON
(line 98-118). The transform is monotonic → does not change ROC AUC, only calibration.

### 2.4 Two hysteresis implementations

There are **two**, and they are not interchangeable:

| | Batch | App |
|---|---|---|
| File | `postprocessing/hysteresis.py` | `app/core/detection_logic.py` |
| Entry | `hysteresis_detect()` (line 76) | `HysteresisDetector.detect()` (line 139) |
| Config | `HysteresisConfig` (line 23) | constructor kwargs (line 91) |
| Seed/extend | seeds at `onset`, grows bidirectionally while ≥ `sustain` (line 116-136) | scans, starts at `high_threshold`, ends below `low_threshold` (line 188+) |
| Returns | `USVEvent` (window indices) | `DetectedUSV` (column indices) |
| Extra filters | duration (min windows + max ms) | duration ms, prob-stability (`min_sustained_prob`), temporal-position exclusion |

Batch production config comes from `hysteresis_optimization_v2.json` (`onset 0.6 / sustain 0.4 /
gap 0 / min_dur 3`). App defaults (`main_window.py:147-149`, set 2026-05-11 to match batch):
`high_threshold=0.60`, `low_threshold=0.40`, `min_sustained_prob=0.0`, energy gate 0.35,
`merge_gap_columns=3`, `min_duration_ms=10`, `max_duration_ms=500` (`main_window.py:798-807`).
Note the app's `min_duration_ms=10` and `max_duration_ms=500` are **time-based** and differ from
the batch's window-count / 600 ms gates — output will not be byte-identical even with matched
thresholds.

### 2.5 FP-filter application

In `process_one_recording` (`run_batch_detection.py:146-154`): for each event, call
`extract_event_features(event, spectrogram_db)` then `fp_filter.predict(features)`; keep only
events where `predict` returns `True`. `FalsePositiveFilter.predict` (`fp_filter.py:121`) applies
the column mask (excluded features), runs the sklearn pipeline, returns Python bools. Degenerate
single-class fit is handled via `_constant_label` (line 131-132).

True-effect caveat (memory `project_fp_filter_true_effect`): the FP-filter's real cost is ~7.67%
interval-level USV loss (keeps ~87.3% of GT USVs), **not** the ~70% wrongly reported by an earlier
session. The "80%" figures in older notes were event-level, not interval-level.

### 2.6 Triage

`triage_recording` (`postprocessing/triage.py:140`) computes QC metrics and assigns the tier per
[§1.5](#15-triage-tiers). `TriageConfig` is frozen and validated in `__post_init__`
(`triage.py:59`) — `auto_reject_max_window` must be strictly below `auto_accept_min_peak`.
Batch-level outlier flagging (`outlier_event_count`) is applied in the runner after all files
complete (`run_batch_detection.py:474-485`) using z-score > 2.0.

### 2.7 Deprecated models

Both directories exist and are kept **only as baselines** — do not point production at them.

| Path | Why deprecated |
|------|----------------|
| `models/matched_windows/best_model.pt` | Prior iteration (matched-window retrain), superseded by hard_neg_retrain. |
| `models/production/best_model.pt` | Older "March" CNN (~1.24 MB vs ~2.5 MB). Calibration collapse: sliding-window probs compress to ≤0.486 so the 0.6 onset never fires (memory `project_march_model_calibration_collapse`). Ranks USVs fine but is unusable at production thresholds. |

The `_baseline` copies inside `models/production/` are explicit baseline snapshots.
The PyQt app defaults to the production model (`app/main.py:25`).

### 2.8 Where to change things

| Want to change… | Edit |
|-----------------|------|
| Onset/sustain/min-duration (batch) | re-fit and point `--hysteresis-config` at a new JSON; do not hand-edit `_v2`. |
| Energy gate | `SlidingInference(energy_threshold=...)` — batch default in `sliding_inference.py:48`; app in `main_window.py:72`. |
| MAD scale factors | `sliding_inference.py:409-410` — **HIGH RISK**, must match training render. |
| Triage thresholds | `TriageConfig` defaults (`triage.py:50-57`) or pass a custom config. |
| FP-filter behavior | re-train via `scripts/train_fp_filter.py`; swap the `.pkl`. |
| Temperature | re-fit `TemperatureScaler`, save JSON, pass `--temperature`. |

**Locked / review-gated** (per CLAUDE.md Red Flags): `scripts/run_batch_detection.py`,
`app/core/sliding_inference.py`, `postprocessing/`, and any `ExtractionConfig` value. Changes
require DSP/CNN review and `/kcheck`.
