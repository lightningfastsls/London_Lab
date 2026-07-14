# Batch USV Detection — `scripts/run_batch_detection.py`

> **What this is:** the production entry point for running the CNN USV-detection
> pipeline across a whole folder of WAV files. One command in, a `summary.parquet`
> triage table + one `detections/<stem>.json` per recording out.
> **Status:** CURRENT / production. This is the canonical way to detect USVs at scale.
> **Production model artifact:** `models/hard_neg_retrain/best_model.pt` (with its
> sibling `temperature.json`, `fp_filter.pkl`, `hysteresis_optimization_v2.json`).
> **Do NOT use** `models/matched_windows/best_model.pt` or
> `models/production/best_model.pt` — older baselines, deprecated.
> **Interpreter:** `.venv/bin/python` (Linux/WSL).

For the interactive desktop app's "Detect" button (a *different*, lighter pipeline —
CNN + hysteresis only, **no** FP-filter/temperature/soft-notch), see the run-app
skill and `docs/modules/cnn-classifier.md`. The two pipelines do **not** produce
identical detections; this document covers the batch pipeline only.

---

## 1. Operate

### 1.1 The canonical command

This is the exact invocation from `CLAUDE.md`. Every flag below has been verified
against the argparse block in `scripts/run_batch_detection.py:615-682`.

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

**Why all five model-related flags are required for a correct production run.**
The script will *run* with only `--wav-dir`, `--model`, and `--output-dir` (those three
are the only `required=True` args). But omitting the others silently degrades the
pipeline:

| Omitted flag | What you lose | Symptom |
|---|---|---|
| `--temperature` | Probability calibration. Raw sigmoid outputs are over-confident; thresholds tuned on calibrated probs no longer mean what they should. | Triage tiers drift; `noise_floor_p90` inflated. |
| `--fp-filter` | The learned false-positive rejector. **Detection list is bloated with in-recording FPs and noise events.** | More events per file; precision drops. |
| `--hysteresis-config` | The fitted onset/sustain/gap/min-duration grid. Falls back to **library defaults** (`onset=0.75, sustain=0.40, gap_fill=3, min_duration=5`, `hysteresis.py:36-40`), which are **not** the production-tuned values (`onset=0.6, sustain=0.4, gap_fill=0, min_duration=3`). | Event boundaries and counts differ from the validated pipeline. |

Treat all five flags as mandatory. A run missing `--fp-filter` or `--hysteresis-config`
is an *incomplete pipeline with unreliable triage* — do not report its numbers as
production results.

### 1.2 Every command-line flag

Defined in `main()` at `scripts/run_batch_detection.py:612-682`.

| Flag | Type | Default | Required | What it controls |
|---|---|---|---|---|
| `--wav-dir` | Path | — | **yes** | Directory of WAVs. Searched **recursively** (`**/*.wav`, line 361). |
| `--model` | Path | — | **yes** | Trained CNN checkpoint (`.pt`). Use `models/hard_neg_retrain/best_model.pt`. |
| `--output-dir` | Path | — | **yes** | Where `summary.parquet` + `detections/*.json` are written. Created if absent. |
| `--temperature` | Path | `None` | no | Fitted temperature-scaling JSON. Calibrates logits → probabilities (`run_batch_detection.py:631-633`). |
| `--fp-filter` | Path | `None` | no | Fitted false-positive-filter pickle. Drops events the classifier flags as FPs (`635-637`). |
| `--hysteresis-config` | Path | `None` | no | Hysteresis-optimization JSON. Sets onset/sustain/gap/min-duration (`639-641`). |
| `--workers` | int | `1` | no | Parallel worker processes. `>1` → `multiprocessing.Pool` (`397-415`). Use `4`. |
| `--no-resume` | flag | off | no | Reprocess everything. By default the run **skips** files that already have a JSON in `<output-dir>/detections/` (`646-649`, resume logic `372-379`). |
| `--subtract-baseline` | flag | off | no | **Lab-only.** Per-frequency-bin temporal-baseline subtraction *before* the CNN. Removes stationary equipment harmonics the wild-trained CNN never saw. **Wild runs (5970/3452/9252) MUST omit this** for byte-identical results (`650-660`). |
| `--subtraction-method` | str | `percentile` | no | Only consulted with `--subtract-baseline`. Choices: `percentile` (Boll 1979 floor, p10) or `median_envelope` (sliding-median per bin, ~0.5 s kernel) (`661-670`). |
| `--soft-notch` | str (`PATH` or `auto`) | `None` | no | **Lab-only.** Adaptive soft-notch in the audio domain *before* STFT. Pass a `TonalLibrary` JSON path (library mode, preferred) or the literal `auto` (pure per-chunk detect, no library). **Wild runs MUST omit this** for byte-identical results (`671-682`, dispatch `700-712`). If a library path is given that does not exist, the script logs an error and exits with code **2** (`705-707`). |

There is **no** flag to change `--workers`' batch size, the CNN window width, the
energy threshold, or the triage thresholds from the CLI — those are code-level
constants (see §2). The triage config is always `TriageConfig()` defaults
(`run_batch_detection.py:393-394`).

### 1.3 Required environment

- Interpreter: `.venv/bin/python` (the repo venv; has torch, pandas, pyarrow, scipy, PIL).
- No environment variables are required by this script. `src/` is added to `sys.path`
  automatically (`run_batch_detection.py:43-46`).
- GPU is used automatically if available (`SlidingInference` picks `cuda` when
  `torch.cuda.is_available()`, else `cpu` — `sliding_inference.py:69-72`). CPU works; it
  is just slower.

### 1.4 WAV input locations

**There is no single canonical WAV directory.** Recordings span multiple folders
(e.g. `5970 USV/`, the 3452 sample/reviewed dirs, `USV_9252/`, lab `131204` sets).
Point `--wav-dir` at whichever cohort you are processing; the glob recurses, so a
parent folder containing sub-cohorts works too. Never assume a fixed path. For the
authoritative map of which WAVs live where and how labels resolve to them, see
[labels & data](labels_and_data.md).

### 1.5 Outputs written to `--output-dir`

| File | Written by | Contents |
|---|---|---|
| `detections/<stem>.json` | per-recording, **immediately** after each file (crash-safe) | List of detected events for that WAV, in ADR-010 dict format. Empty list `[]` if no events. (`run_batch_detection.py:247-253`) |
| `summary.parquet` | once, at end of batch | One row per recording: tier + QC metrics. (`_write_summary_parquet`, `506-519`) |
| `soft_notch_applied.parquet` | only with `--soft-notch` | One row per applied/audit notch event. (`_write_soft_notch_sidecars`, `522-551`) |
| `soft_notch_summary.json` | only with `--soft-notch` | Library metadata + stale-library warning. (`553-601`) |

#### `detections/<stem>.json` — per-event schema

Each list element comes from `_event_to_adr010_dict` (`batch_output.py:32-47`):

| Field | Meaning |
|---|---|
| `start_time_s` | Event start, seconds (window-center time of first window). |
| `end_time_s` | Event end, seconds (window-center time of last window). |
| `duration_s` | `end_time_s − start_time_s`. **Center-to-center**; a single-window event is `0.0` (see `hysteresis.py:62-69`). To get physical span, add one window step. |
| `start_col` | `start_window × hop_px` (hop_px=10). Spectrogram column index of the start window center. |
| `end_col` | `end_window × hop_px`. Column index of the end window center. |
| `max_probability` | Peak per-window probability across the event (calibrated if `--temperature` used). |
| `mean_probability` | Mean per-window probability across the event. |

#### `summary.parquet` — per-recording schema

Columns are `_PARQUET_COLUMNS` (`batch_output.py:19-29`); values come from
`RecordingResult` (`triage.py:96-133`):

| Column | Meaning |
|---|---|
| `filepath` | Path to the source WAV (verbatim). |
| `tier` | `auto_accept`, `auto_reject`, or `manual_review` (see triage logic below). |
| `n_events` | Number of detected events after hysteresis + FP filter. |
| `max_confidence` | Max `peak_probability` across all events (`0.0` if none). |
| `mean_event_confidence` | Mean of per-event `peak_probability` (`0.0` if none). |
| `total_usv_duration_ms` | Sum of event `duration_ms` (center-to-center). |
| `noise_floor_p90` | 90th-percentile of the **raw window-probability array** — a recording-level noise proxy. |
| `confidence_score` | Currently equals `mean_event_confidence` (`triage.py:232`). |
| `qc_flags` | List of QC strings (see below). |

**Triage tier rules** (`triage.py:216-229`, order matters):
1. `n_events == 0` → `auto_reject`.
2. else if `max(prob) ≤ auto_reject_max_window` (0.10) → `auto_reject`.
3. else if any of `{long_event_duration, event_spans_most_of_recording}` flagged → `manual_review`.
4. else if **every** event has `peak_probability ≥ auto_accept_min_peak` (0.90) → `auto_accept`.
5. else → `manual_review`.

**QC flags** (`triage.py:186-214`), with their `TriageConfig` defaults (`triage.py:50-57`):

| Flag | Raised when |
|---|---|
| `high_noise_floor` | `noise_floor_p90 > 0.4`. |
| `high_event_count` | `n_events > 10`. |
| `long_event_duration` | any event `duration_ms > 600.0`. |
| `high_total_usv_duration` | `total_usv_duration_ms > 600.0`. |
| `event_spans_most_of_recording` | an event covers ≥ `0.8` of the probability timeline. |
| `outlier_event_count` | `n_events` z-score `> 2.0` vs the batch mean (added post-hoc in `run_batch_detection.py:475-485`). |

#### Soft-notch sidecars (only with `--soft-notch`)

`soft_notch_applied.parquet` columns (`run_batch_detection.py:543-547`):
`recording_path, chunk_idx (always 0), center_hz, width_hz, peak_db, local_median_db,
cut_depth_db, source ("library"|"audit"), is_drift, intensity_drift_sigma`.

`soft_notch_summary.json` (`589-596`): library metadata (`library_path`,
`library_rig_id`, `library_calibrated_at`, `library_n_entries`), `batch_n_chunks`,
`n_chunks_with_unmatched`, `unmatched_rate`, and `stale_library_warning_fired`
(true when `unmatched_rate > 0.10`, `_UNMATCHED_RATE_WARNING_THRESHOLD`, line 68).

### 1.6 Worked example

Detect USVs on a folder of wild-mouse recordings, 4 workers:

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

Reading the result:

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('results/batch_5970/summary.parquet')
print(df['tier'].value_counts())
print(df.loc[df.tier=='manual_review', ['filepath','n_events','max_confidence','qc_flags']].head())
"
```

A single recording's events:

```bash
.venv/bin/python -c "
import json; print(json.load(open('results/batch_5970/detections/<stem>.json')))
"
```

Logs (INFO level, `run_batch_detection.py:686-689`) report file count, per-50-file
progress + ETA, batch mean/std event count, and the final triage distribution
(`auto_accept` / `auto_reject` / `manual_review`).

### 1.7 Troubleshooting / Gotchas

- **A re-run "does nothing" / processes 0 files.** Resume is **on by default**: any WAV
  whose `<stem>.json` already exists in `<output-dir>/detections/` is skipped
  (`run_batch_detection.py:372-379`). To force a full reprocess, add `--no-resume` or
  point at a fresh `--output-dir`. Resume keys on **stem**, so two WAVs with the same
  filename in different subfolders collide — one will be skipped.
- **A single file crashing does not abort the batch.** Per-file exceptions are logged and
  the file is dropped from results (`331-333` parallel, `471-472` serial). Check the log
  for `Failed to process ...` lines; those recordings produce no JSON.
- **`summary.parquet` only contains files processed *this* run.** Resumed (already-done)
  files are **not** re-added to the parquet (`_write_summary_parquet` only sees
  `raw_results`). If you resumed a partial batch, the parquet is incomplete — rebuild from
  the `detections/*.json` files or run once with `--no-resume`.
- **Don't pass raw logits as probabilities.** Hysteresis rejects any non-finite or
  out-of-`[0,1]` value (`hysteresis.py:111-114`). The pipeline always sigmoid-transforms;
  this only bites if you call the internals directly.
- **Sample-rate mismatch raises.** The loader requires 300 kHz WAVs and raises `ValueError`
  if the file header says otherwise (`audio_loader.py:154-158`). Resampled/decimated WAVs
  will not run.
- **Very short recordings yield no events.** A spectrogram narrower than the CNN window
  (100 px) raises inside inference (`sliding_inference.py:141-145`); the file then counts
  as a per-file failure.
- **`--soft-notch` library file missing → exit code 2** (`705-707`). The literal `auto`
  is always accepted (no file check).
- **Lab vs wild flags.** `--subtract-baseline` and `--soft-notch` change the spectrogram
  the CNN sees. For wild cohorts (5970/3452/9252) omit both, or you break byte-identity
  with the validated wild results.

---

## 2. Internals

### 2.1 Pipeline data flow

Per recording (`process_one_recording`, `run_batch_detection.py:107-156`):

```
WAV
 └─ AudioLoader.load()                      audio_loader.py:133
     ├─ load_wav_mono()                     (300 kHz mono float32)
     ├─ [auto_soft_notch] in audio domain   audio_loader.py:166-169   (lab opt-in)
     ├─ STFT → dB, band-limited 20–120 kHz  audio_loader.py:196-239   (n_fft=512, hop=128, Hann, magnitude-normalized so max dB = 0)
     └─ [subtract_temporal_baseline]        audio_loader.py:247-…     (lab opt-in, dB↔linear round-trip)
 └─ SlidingInference.infer()               sliding_inference.py:119
     ├─ MAD-normalize WHOLE spectrogram ONCE sliding_inference.py:147-152  (global, NOT per-window)
     ├─ slide 100 px window, hop 10 px      sliding_inference.py:157-165
     ├─ energy pre-filter (skip quiet → p=0) sliding_inference.py:201-205
     ├─ magma colormap + vertical flip → CNN sliding_inference.py:274-…   (matches training RGB pipeline)
     └─ per-window probabilities (+ logits if return_logits)
 └─ [TemperatureScaler.calibrate(logits)]  run_batch_detection.py:135-136   (if --temperature)
 └─ [normalize_fn]                          run_batch_detection.py:139-140   (unused by CLI; always None here)
 └─ hysteresis_detect(probs, times, cfg)    hysteresis.py:76          → List[USVEvent]
 └─ [event_features → FPFilter.predict]     run_batch_detection.py:146-154   (if --fp-filter; drops events)
 └─ triage_recording(...)                   triage.py:140             → RecordingResult
 └─ write detections/<stem>.json            run_batch_detection.py:247-253
```

Then, batch-level: outlier z-score flagging (`474-485`) → `summary.parquet`
(`488`) → optional soft-notch sidecars (`491-495`) → triage distribution log.

### 2.2 Key signatures (file:line)

- `process_one_recording(wav_path, loader, inference, hysteresis_config, temperature_scaler=None, normalize_fn=None, fp_filter=None, spectrogram_for_features=False) -> (List[USVEvent], np.ndarray, ReconciliationResult|None)` — `run_batch_detection.py:107`.
- `_process_and_save_one(...)` — runs one file + writes its JSON, returns `(RecordingResult|None, soft_notch_rows)` — `run_batch_detection.py:222`.
- `_worker_init(...)` / `_worker_process_file(wav_path)` — multiprocessing worker setup + task; worker state held in module dict `_worker_state` — `run_batch_detection.py:267, 303`.
- `run_batch(wav_dir, model_path, output_dir, temperature_path=None, fp_filter_path=None, hysteresis_config_path=None, triage_config=None, n_workers=1, resume=True, subtract_baseline=False, subtraction_method="percentile", soft_notch_enabled=False, soft_notch_library_path=None) -> List[dict]` — `run_batch_detection.py:340`.
- `_load_hysteresis_config(path) -> HysteresisConfig` — reads `data["best_params"]` (falls back to the top-level dict), keys `onset_threshold, sustain_threshold, gap_fill_windows, min_duration_windows`, and optional `max_duration_ms` (default `600.0`) — `run_batch_detection.py:89-100`.
- `SlidingInference.__init__(model_path, window_width_px=100, hop_px=10, batch_size=32, device=None, energy_threshold=0.1, enable_per_window_norm=False)` — `sliding_inference.py:41-49`.
- `hysteresis_detect(probabilities, times, config=None) -> List[USVEvent]` — `hysteresis.py:76`.
- `triage_recording(filepath, events, probabilities, config=None, batch_stats=None) -> RecordingResult` — `triage.py:140`.
- `FalsePositiveFilter.load(path)` / `.predict(features) -> List[bool]` — `fp_filter.py:177, 121`.
- `TemperatureScaler.load(path)` / `.calibrate(logits) -> np.ndarray` — `calibration.py:111, 72`.
- `extract_event_features(event, spectrogram_db)` — `event_features.py:46`.

### 2.3 Invariants (do not break)

- **Corpus constants are locked to the CNN.** `sample_rate=300000`, band `20–120 kHz`,
  `n_fft=512`, `hop=128` — single source of truth `src/usv_spectrogram/corpus.py:30-36`.
  `ExtractionConfig` re-asserts these at import (`detection/extraction_config.py:148+`).
  Changing the band silently corrupts inference (same pixel grid, different Hz/pixel).
- **Global MAD normalization, once, before windowing.** Per-window MAD silently kills
  high-confidence USVs (`sliding_inference.py:147-152`; see also memory note
  `feedback_cnn_inference_global_mad`).
- **Hysteresis defaults ≠ production values.** Library defaults are
  `onset=0.75 / sustain=0.40 / gap_fill=3 / min_duration=5` (`hysteresis.py:36-40`).
  Production uses the JSON: `onset=0.6 / sustain=0.4 / gap_fill=0 / min_duration=3`
  (`models/hard_neg_retrain/hysteresis_optimization_v2.json`, `best_params`). Always pass
  `--hysteresis-config`.
- **`duration_ms` / `duration_s` are center-to-center.** Single-window events report `0.0`
  (`hysteresis.py:62-69`). Don't treat these as physical onset-to-offset spans without
  adding a window step.
- **`start_col`/`end_col` in JSON are `window_index × hop_px` (hop_px=10)**
  (`batch_output.py:43-44`) — these are coarse window-center columns, not the full event
  span. (A separate `convert_to_detection_format` with `boundary_padding_cols` exists in
  `hysteresis.py:176` but the batch JSON writer does **not** use it.)
- **Constants the CLI cannot change** (edit code to change): triage thresholds are always
  `TriageConfig()` defaults (`run_batch_detection.py:393-394`); CNN window `100 px`, hop
  `10 px`, batch `32`, energy threshold `0.1` are `SlidingInference` defaults
  (`sliding_inference.py:44-49`).

### 2.4 Where to change things

- **Add a CLI knob** (e.g. expose a triage threshold): add to the argparse block
  (`run_batch_detection.py:615-682`), thread it through `run_batch` (`340`) into
  `TriageConfig`/`HysteresisConfig`.
- **Change triage policy:** `triage.py` — `TriageConfig` defaults (`50-57`) and tier
  logic (`216-229`).
- **Change detection geometry** (window, hop, energy gate): `SlidingInference.__init__`
  (`sliding_inference.py:41-66`). These are not CLI-exposed by design (locked to the CNN).
- **Change the spectrogram the CNN sees:** `AudioLoader._compute_spectrogram`
  (`audio_loader.py:196-239`) — but this is a HIGH-risk path; corpus constants are locked.
- **Output schema:** `batch_output.py` (`_PARQUET_COLUMNS` at `19`, `_event_to_adr010_dict`
  at `32`).

### 2.5 Production artifacts (verified present)

| Artifact | Path | Note |
|---|---|---|
| CNN checkpoint | `models/hard_neg_retrain/best_model.pt` | Production model (2026-04-01). |
| Temperature | `models/hard_neg_retrain/temperature.json` | `temperature = 0.9019…`, `fitted: true`. |
| FP filter | `models/hard_neg_retrain/fp_filter.pkl` | Learned false-positive rejector. |
| Hysteresis config | `models/hard_neg_retrain/hysteresis_optimization_v2.json` | `best_params`: onset 0.6 / sustain 0.4 / gap_fill 0 / min_duration 3 (no `max_duration_ms` key → falls back to 600.0 ms). |

Deprecated (do not use): `models/matched_windows/best_model.pt`,
`models/production/best_model.pt`.

### 2.6 Sibling docs

- [labels & data](labels_and_data.md) — where WAVs live, label resolution.
- [CNN detection / classifier](cnn_detection_pipeline.md) — model training & the
  classifier internals (also `docs/modules/cnn-classifier.md`).
- Full v2 pipeline results: `docs/handoffs/v2-full-pipeline-results.md`.
- Soft-notch design: `docs/handoffs/2026-05-11_adaptive-soft-notch.md`.
- Pre-CNN spectral subtraction: `docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md`.
