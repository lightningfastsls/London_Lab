# Spectrogram Cleaning Pipeline

> **What this is** — The reference for the USV lab's spectrogram/audio *cleaning*
> code. There is no single "cleaning pipeline"; there are **four distinct stacks**
> that historically got conflated. This doc names each, then documents the two that
> are LIVE on `main` in operational detail.
> **Status** — Current as of 2026-06-21. Built from source at HEAD (read line-by-line,
> not from prior docs).
> **Canonical "our cleaning pipeline"** — **Stack 4**, the DeepSqueak focus-STFT contour
> port (`scripts/deepsqueak_focus_stft.py` + `scripts/contour_mask_utils.py`), designated
> canonical 2026-05-28 by user directive. It feeds the contour-masked VAE — see
> [production VAE](production_vae.md).
> **Production-detection cleaning** — **Stacks 2a + 2b** (`app/core/notch.py` +
> `app/core/denoise.py`). These are what the live PyQt6 app and `run_batch_detection.py`
> actually run. They are NOT "our cleaning pipeline" in conversation, but they ARE the
> cleaning in the production detection path.

---

## Read this first: the canonical-vs-detection distinction

Two phrases mean two different things, and mixing them up is a documented past error
([cleaning-subsystems.md](../modules/cleaning-subsystems.md)):

| Phrase | Means | Stack |
|--------|-------|-------|
| "our cleaning pipeline" / "the canonical cleaning" | the contour-mask port that produces VAE patches | **Stack 4** |
| "production-detection cleaning" / "the cleaning the app does" | the pre-CNN audio/spectrogram cleaning in the detection path | **Stacks 2a + 2b** |

If someone says "run our cleaning pipeline on these calls" they mean Stack 4. If someone
asks "what cleaning does the detector apply" they mean Stacks 2a/2b. These are completely
separate code paths operating on different data representations.

### At-a-glance: all four stacks

| Stack | Status | File | Operates on | Sample-rate convention | Canonical FOR |
|-------|--------|------|-------------|------------------------|---------------|
| **1. Classifier 4-layer** | **ARCHIVED 2026-05-28** | `archive/cleaning_legacy/stack1/src/classifier/cleaning_pipeline.py` | dB spectrogram | 250 kHz (VocalMat-aligned), default in code | (was) pre-CNN cleaning for the Module 18.x lab VocalMat classifier train/eval |
| **2a. Production — soft-notch** | **LIVE (detection)** | `src/usv_spectrogram/app/core/notch.py` | **time-domain audio** (before STFT) | 300 kHz (caller passes `fs_hz`) | live PyQt6 + `run_batch_detection.py` equipment-tonal removal |
| **2b. Production — baseline subtraction** | **LIVE (detection)** | `src/usv_spectrogram/app/core/denoise.py` | linear-magnitude spectrogram (after STFT) | 300 kHz (kernel derived from `corpus.STFT_HOP`/`SAMPLE_RATE_HZ`) | live PyQt6 + batch detection temporal-baseline subtraction |
| **3. SIS prefilter** | **ARCHIVED 2026-05-28** | `archive/cleaning_legacy/stack3/src/features/spectrogram_filter.py` | linear-magnitude spectrogram | 300 kHz | (was) the SIS benchmark (ridge / Oren / AMVOC) + 6 shape-VAE experiments |
| **4. DeepSqueak contour port** | **LIVE (canonical "our cleaning")** | `scripts/deepsqueak_focus_stft.py` + `scripts/contour_mask_utils.py` | per-call focus STFT → power-spectrogram mask | 300 kHz (`corpus.SAMPLE_RATE_HZ`) | contour-masked VAE patch generation; the canonical "our cleaning pipeline" |

Stacks 1 and 3 were archived (their analysis families dead-ended: 18.x DANN and all six
shape-VAE attempts). **They were NOT deleted** — see the
[archive-still-imported gotcha](#gotcha-archived-stacks-are-still-imported-at-runtime)
in Internals. Full inventory and rationale: [cleaning-subsystems.md](../modules/cleaning-subsystems.md).

---

# 1. Operate

This section is split by the two things you might actually want to run:

- **[1A — Stack 4 (the canonical cleaning):](#1a--stack-4-the-canonical-cleaning)** generate contour-masked VAE patches.
- **[1B — Stacks 2a/2b (production-detection cleaning):](#1b--stacks-2a2b-production-detection-cleaning)** how the detector cleans, and the flags that turn it on.

---

## 1A — Stack 4 (the canonical cleaning)

Stack 4 turns a folder of WAVs + a per-call contour parquet into **masked power-spectrogram
patches** for the contour-masked VAE. Two scripts:

| Script | Purpose | When |
|--------|---------|------|
| `scripts/sweep_contour_mask.py` | Render a 12-cell visual sweep (4 bandwidths × 3 tonality thresholds) + 1 raw-reference PNG, for **human selection** of the mask parameters. | Run FIRST, review the PNGs, pick `(bandwidth, threshold)`. |
| `scripts/mass_apply_contour_mask.py` | Apply the chosen mask to **all** accepted windows; emit `patches.npz` + `patches_manifest.parquet`. | Run SECOND with the chosen params. |

### Required environment

```bash
.venv/bin/python <script>        # repo interpreter (Linux/WSL)
```
Both scripts add `scripts/` and `src/` to `sys.path` themselves, so cwd does not matter.
Dependencies used at runtime: `numpy`, `scipy`, `statsmodels`, `librosa`, `pandas`,
`matplotlib`. The focus-STFT port (`deepsqueak_focus_stft.py`) additionally needs
`statsmodels` for the robust LOWESS smoother.

### Input files

| Input | What it is | Schema (columns used) |
|-------|-----------|-----------------------|
| `--contours-parquet` | Per-call contour bins from the DeepSqueak port (produced upstream by the contour extractor; one row per ridge bin). | `wav_stem`, `call_id`, `time_bin_index`, `frequency_kHz`, `tonality`, `accepted` (bool) |
| `--window-index-parquet` | The fixed-width windowing of accepted calls (one row per patch window). | `wav_stem`, `call_id`, `window_idx`, `start_bin_index`, `end_bin_index`, `abs_time_start_s`, `abs_time_end_s`, `num_contour_bins_in_window` |
| `--wav-search-dirs` | One or more directories searched recursively for `<wav_stem>.wav`. First match wins (search-dir order). | — |

Default parquet paths (5970 cohort) are baked into `mass_apply_contour_mask.py`:
`results/contour_extraction/5970/contours.parquet` and
`results/masked_patches/5970/window_index.parquet`
(`scripts/mass_apply_contour_mask.py:60`, `:65`).

### Step 1 — visual sweep (`sweep_contour_mask.py`)

```bash
.venv/bin/python scripts/sweep_contour_mask.py \
    --contours-parquet results/contour_extraction/5970/contours.parquet \
    --window-index-parquet results/masked_patches/5970/window_index.parquet \
    --wav-search-dirs "5970 USV/" \
    --output-dir results/masked_patches/5970/sweep/ \
    --n-example-calls 20 \
    --seed 42 \
    --db-floor -60.0
```

| Flag | Default | Meaning / when to change |
|------|---------|--------------------------|
| `--contours-parquet` | (required) | Contour parquet (see schema above). |
| `--window-index-parquet` | (required) | Window-index parquet. |
| `--wav-search-dirs` | (required) | One or more WAV roots (nargs `+`). |
| `--output-dir` | (required) | Where the 13 PNGs are written. |
| `--n-example-calls` | `20` | How many accepted first-windows (`window_idx == 0`) to sample for the sweep grid. The grid is rendered 5×4, so 20 fills it exactly. |
| `--seed` | `42` | Random seed for the deterministic sample of example calls. |
| `--db-floor` | `-60.0` | vmin (dB power) shared across all panels for an honest visual comparison. Lower it to see more of the noise floor. |

The sweep matrix is **hard-coded in the script** (not flags):
- Bandwidths (`scripts/sweep_contour_mask.py:87-92`): `bw_2kHz` (±2 kHz hard), `bw_5kHz`
  (±5 kHz hard), `bw_10kHz` (±10 kHz hard), `bw_gauss3kHz` (Gaussian σ=3 kHz soft).
- Tonality thresholds (`scripts/sweep_contour_mask.py:94-98`): `thr_p25` = **94.4**,
  `thr_p50` = **314.1**, `thr_p75` = **1254.7**. These are percentile anchors on the
  *peak-to-median tonality* distribution (NOT DeepSqueak's internal 0.3, which lived on a
  different scale). Higher value = keep fewer / more-tonal contour bins.

**Outputs** (into `--output-dir`): `cell_raw_unmasked.png` (reference, no mask) plus
`cell_<bw>_<thr>.png` for each of the 12 combinations. The script also prints, per cell,
how many of the example patches were fully zeroed (no contour bin passed the threshold) —
a high zeroed count means the threshold is too aggressive for that cell.

### Step 2 — mass apply (`mass_apply_contour_mask.py`)

```bash
.venv/bin/python scripts/mass_apply_contour_mask.py \
    --contours-parquet results/contour_extraction/5970/contours.parquet \
    --window-index-parquet results/masked_patches/5970/window_index.parquet \
    --wav-search-dirs "5970 USV/" \
    --output-patches-npz results/masked_patches/5970/patches.npz \
    --output-manifest-parquet results/masked_patches/5970/patches_manifest.parquet \
    --bandwidth-kHz 5.0 \
    --tonality-threshold 0.0
```

| Flag | Default | Meaning / when to change |
|------|---------|--------------------------|
| `--contours-parquet` | `results/contour_extraction/5970/contours.parquet` | Contour parquet. |
| `--window-index-parquet` | `results/masked_patches/5970/window_index.parquet` | Window-index parquet. |
| `--wav-search-dirs` | (required) | WAV roots (nargs `+`). |
| `--output-patches-npz` | `results/masked_patches/5970/patches.npz` | Output patch tensor + freq axis. |
| `--output-manifest-parquet` | `results/masked_patches/5970/patches_manifest.parquet` | Output manifest (one row per patch). |
| `--bandwidth-kHz` | `5.0` | Half-width (kHz) of the **hard** bandwidth mask around the ridge. This is the chosen production value (`±5 kHz`). |
| `--tonality-threshold` | `0.0` | Minimum tonality for a contour bin to qualify a column. `0.0` keeps every contour bin (the chosen production value). |

> The mass-apply step uses **only** the hard-bandwidth mask
> (`apply_hard_bandwidth_mask`); the Gaussian variant is sweep-only.

**Production-chosen mask: hard ±5 kHz, tonality threshold 0.0** — declared in the script
docstring (`scripts/mass_apply_contour_mask.py:3-4`) and as the CLI defaults
(`:78-79`).

### Stack 4 outputs and their fields

`patches.npz` (`np.savez`, `scripts/mass_apply_contour_mask.py:211`):
| Key | Shape / dtype | Meaning |
|-----|---------------|---------|
| `patches` | `(N, F, T)` float32 | Masked **raw power** patches. `F = STFT_N_FFT//2 + 1 = 257`, `T` is the fixed window width (`234` bins = 100 ms; see Internals). Raw power, NOT normalized — per-patch normalization happens at VAE *training* time (Refinement D, `scripts/mass_apply_contour_mask.py:10-11`). |
| `freqs_kHz` | `(F,)` float32 | The frequency axis in kHz (shared across all patches). |

`patches_manifest.parquet` (one row per patch, column order enforced at
`scripts/mass_apply_contour_mask.py:259-276`):
| Column | dtype | Meaning |
|--------|-------|---------|
| `patch_idx` | int32 | Row index into `patches` (0-based). |
| `wav_stem` | string | Source WAV stem. |
| `call_id` | int64 | Source DeepSqueak call id. |
| `window_idx` | int32 | Which window of that call (0 = first). |
| `start_bin_index`, `end_bin_index` | int32 | STFT-column span of the patch in the full recording. |
| `abs_time_start_s`, `abs_time_end_s` | float32 | Absolute time span of the patch within the WAV. |
| `num_contour_bins_in_window` | int32 | Contour bins falling in this window (provenance). |
| `mask_kind` | string | `"hard"` (always, for this script). |
| `bandwidth_kHz` | float32 | The `--bandwidth-kHz` used. |
| `tonality_threshold` | float32 | The `--tonality-threshold` used. |
| `n_nonzero_freqs` | int32 | Diagnostic: how many freq rows survived the mask. `0` = fully-zeroed patch. |
| `patch_max_power` | float32 | Diagnostic: max power in the masked patch. |

The script prints a summary: windows processed, WAV stems used, patch tensor shape,
count of all-zero patches, and average mask coverage (`n_nonzero_freqs / F`).

### Worked example (Stack 4)

```bash
# 1. Visual sweep — produces 13 PNGs for human review.
.venv/bin/python scripts/sweep_contour_mask.py \
    --contours-parquet results/contour_extraction/5970/contours.parquet \
    --window-index-parquet results/masked_patches/5970/window_index.parquet \
    --wav-search-dirs "5970 USV/" \
    --output-dir results/masked_patches/5970/sweep/

# (human reviews results/masked_patches/5970/sweep/*.png, picks ±5 kHz / thr 0.0)

# 2. Mass-apply the chosen mask to every accepted window.
.venv/bin/python scripts/mass_apply_contour_mask.py \
    --wav-search-dirs "5970 USV/" \
    --bandwidth-kHz 5.0 --tonality-threshold 0.0
# -> writes results/masked_patches/5970/patches.npz (+ manifest)
```
The resulting `patches.npz` is the input to the contour-masked VAE training script
`scripts/train_contour_vae_v2.py` — see [production VAE](production_vae.md).

### Stack 4 — Troubleshooting / Gotchas

- **WAV not found** → both scripts raise `FileNotFoundError` listing the missing stems.
  The lookup is `<wav_stem>.wav` recursively under `--wav-search-dirs`; first match wins.
  WAVs span multiple roots (5970/, 3452*/, 9252/) — pass all relevant dirs.
- **"Not enough accepted first-windows"** (sweep) → fewer than `--n-example-calls`
  accepted windows with `window_idx == 0`. Lower `--n-example-calls`, or confirm the
  contour extractor set `accepted == True` on enough calls.
- **"No accepted windows after merging"** (mass-apply) → the `accepted` column in the
  contour parquet is all-False or the merge keys don't line up. Confirm the upstream
  contour extractor populated `accepted`.
- **All-zero / heavily-zeroed patches** → mask is too aggressive (high tonality threshold
  or tiny bandwidth). The sweep's per-cell zeroed count is the diagnostic; the mass-apply
  manifest's `n_nonzero_freqs == 0` count is the post-hoc check.
- **Non-uniform window widths** → mass-apply asserts a single fixed `T` across all windows
  (`scripts/mass_apply_contour_mask.py:134-140`). If the window-index parquet has mixed
  widths it raises `AssertionError`. The windowing script produces a fixed 234-bin width.
- **Memory** — `patches.npz` for the lab cohort is large (an ~11 GiB box has OOM'd
  full-scanning a lab `patches.npz` before; see memory note `project_c06_empty_cluster`).
  Never load the whole array if you only need a slice; mass-apply itself holds one WAV's
  STFT at a time, not all patches in RAM.
- **FFT join key gotcha (downstream, not these scripts)** — the contour/FPCA join is
  `(wav_stem, call_id-1) == (wav_stem, det_index)`, and `(wav_stem, call_id)` is
  NON-unique; dedupe first. (From memory note on WS-B/WS-C.) Relevant if you cross-join
  these patches with detection or FPCA tables.

---

## 1B — Stacks 2a/2b (production-detection cleaning)

This is the cleaning the **detector** applies. It lives behind opt-in flags and is
**default-OFF** so wild-mouse runs are byte-identical without them. Both are invoked
through `app/core/audio_loader.py` (live PyQt6 path) and `scripts/run_batch_detection.py`
(batch path).

> **Hard invariant:** wild-mouse batches (5970, 3452, 9252) MUST omit both
> `--soft-notch` and `--subtract-baseline` to reproduce the published detections
> byte-for-byte. These flags are **lab-only opt-ins** that target rig equipment tonals the
> wild-trained CNN never saw. (`notch.py:30-34`, `run_batch_detection.py:656-657`.)

### Where they sit in the detection path

```
run_batch_detection.py
  → audio_loader.SpectrogramAudioLoader
      → (optional) notch.auto_soft_notch        # Stack 2a — on TIME-DOMAIN audio, before STFT
      → STFT
      → (optional) denoise.subtract_temporal_baseline   # Stack 2b — on LINEAR-MAG spectrogram, after STFT
      → sliding_inference._apply_mad_normalization      # (MAD norm — NOT part of "cleaning")
```

### Stack 2a — soft-notch (`notch.py`)

Removes stationary equipment **tonal lines** from the time-domain audio before the STFT,
via complementary-bandpass subtraction: `audio - alpha * bandpass(audio)`, with
`alpha = 1 - 10**(-cut_depth_db/20)` (`notch.py:18-27`, `:313`). Two modes:
- **Library mode** (preferred): a per-rig `TonalLibrary` JSON drives which bands to cut;
  cut depth is measured per-chunk from the local PSD.
- **Auto mode** (no library): `discover_tonals` finds tonals per-chunk via Welch PSD and
  filters them all.

CLI: `--soft-notch PATH|auto` on `run_batch_detection.py`
(`scripts/run_batch_detection.py:671-682`) — pass a `TonalLibrary` JSON path (library
mode) or the literal string `auto` (pure auto-detect). Default `None` (off).

Key defaults (read from `notch.py`; these are function defaults, change only with reason):

| Parameter | Default | Source | Meaning |
|-----------|---------|--------|---------|
| `discovery_threshold_db` | `10.0` | `notch.py:345` | A PSD bin must exceed its rolling-median neighborhood by this many dB to count as a tonal. |
| `median_window_hz` | `4_000.0` | `notch.py:346` | Width of the local-median neighborhood for tonal discovery / per-chunk cut-depth. |
| `nperseg` | `8192` | `notch.py:347` | Welch segment length (~37 Hz/bin at 300 kHz). |
| `usv_band_min_hz` / `usv_band_max_hz` | `USV_FREQ_MIN_HZ` / `USV_FREQ_MAX_HZ` (`20_000` / `120_000`) | `notch.py:343-344`, `corpus.py:32-33` | Band searched for tonals. |
| `freq_tolerance_hz` | `200.0` | `notch.py:428` | Center-frequency match tolerance, library reconciliation. |
| `intensity_drift_sigma` | `2.0` | `notch.py:429` | Library-drift alarm: matched tonal deviates this many σ from the library mean. |
| `min_width_hz` | `200.0` | `notch.py:485` | Floor on filter bandwidth for auto-detected tonals. |
| `width_safety_factor` | `2.0` | `notch.py:486` | Auto-detected width is multiplied by this. |
| `safety_margin_db` | `0.0` | `notch.py:487` | Extra dB added to every cut depth. |
| `order` | `4` | `notch.py:488` | Butterworth order (effective `2*order` via `sosfiltfilt`). |

`auto_soft_notch` returns `(cleaned_audio, ReconciliationResult)`. When no tonals are
discovered and no library entries exist, `cleaned_audio` is the **same ndarray** as the
input (byte-identical) — this is what guarantees the default-off invariant
(`notch.py:599-600`, `:611-613`).

### Stack 2b — baseline subtraction (`denoise.py`)

Subtracts a per-frequency-bin **temporal noise floor** from the linear-magnitude
spectrogram after the STFT (`subtract_temporal_baseline`). Two methods:

CLI on `run_batch_detection.py`:
- `--subtract-baseline` (store_true; default off) — enables it
  (`scripts/run_batch_detection.py:650-660`).
- `--subtraction-method {percentile,median_envelope}` (default `percentile`) — only
  consulted with `--subtract-baseline` (`scripts/run_batch_detection.py:661-670`).

Defaults (read from `denoise.py`):

| Parameter | Default | Source | Meaning |
|-----------|---------|--------|---------|
| `method` | `"percentile"` | `denoise.py:62` | `percentile` = Boll-1979 floor subtraction; `median_envelope` = sliding per-bin median (tracks slow band amplitude modulation). |
| `percentile` | `10.0` (`DEFAULT_BASELINE_PERCENTILE`) | `denoise.py:45`, `:63` | The temporal percentile per bin used as the floor (percentile method only). |
| `envelope_kernel_frames` | `DEFAULT_ENVELOPE_KERNEL_FRAMES` ≈ `1172` | `denoise.py:54-57`, `:64` | Median-filter width in frames, derived as `round(0.5 s * SAMPLE_RATE_HZ / STFT_HOP)` = `round(0.5*300000/128)`. Wider than the longest USV (~300 ms) so bursts can't dominate the median. |
| `epsilon` | `1e-10` (`DEFAULT_EPSILON`) | `denoise.py:46`, `:65` | Post-subtraction floor so downstream `log` is safe. |

> **Why linear, not dB:** subtraction must happen in linear magnitude — subtracting in dB
> is mathematically wrong (`denoise.py:22-24`). The caller owns the dB↔linear conversion.
> Baseline is computed **per 2-second chunk**, not per recording (`denoise.py:25-26`),
> keeping the function stateless.

### Worked example (production cleaning)

```bash
# Wild cohort — NO cleaning flags (byte-identical to published results):
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir "5970 USV/" \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_5970/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4

# Lab cohort — opt in to both cleaning layers:
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir lab_131204/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_lab_131204/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --subtract-baseline --subtraction-method percentile \
    --soft-notch data/lab_tonal_lines/<rig_id>.json \
    --workers 4
```
The five model flags are the canonical detection pipeline — see
[CNN detection pipeline](cnn_detection_pipeline.md). The cleaning flags are additive
lab-only options.

### Stacks 2a/2b — Troubleshooting / Gotchas

- **PyQt6 "Detect" button does NOT run these.** The app's live Detect path is CNN +
  hysteresis only (raw probs, no FP-filter / no temperature / no soft-notch / no baseline
  by default). Soft-notch and baseline are wired through `audio_loader` flags
  (`subtract_baseline`, `auto_soft_notch`) but the batch CLI is where they are turned on
  for production. (Memory note `feedback_app_detect_vs_batch_pipeline`.)
- **MAD normalization is not "cleaning."** `sliding_inference._apply_mad_normalization`
  runs after these but is part of the CNN input normalization, not this pipeline. It is
  **global** (whole-spectrogram), never per-window — per-window MAD silently kills
  high-confidence USVs (memory note `feedback_cnn_inference_global_mad`).
- **Soft-notch library is the source of truth; auto-detections are an alarm.** In library
  mode, unmatched discovered tonals are *logged, not filtered* — they signal a stale
  library, they don't change the output (`notch.py:13-15`).
- **Never hardcode the kernel / sample rate.** Stack 2b derives its kernel from
  `corpus.STFT_HOP` / `SAMPLE_RATE_HZ`; Stack 2a takes `fs_hz` from the caller. Both are
  300 kHz in production via `corpus`. Do not redeclare these (CLAUDE.md / ADR-001).
- **mean_power_db / tonality are cage artifacts.** Cleaning changes the noise floor; don't
  read post-cleaning `mean_power_db`/`tonality` as biology without cross-cage calibration
  (memory note `feedback_rig_artifact_mean_power_db`).

---

# 2. Internals

## Stack 4 architecture and data flow

Stack 4 is a line-by-line Python port of DeepSqueak's `CreateFocusSpectrogram.m` +
`CalculateStats.m`. Unlike the older `ridge_tracker.py`, it computes a **per-call adaptive
focus STFT** (window size adapts to the call's own duration and frequency range), which is
why its time grid is coarser than the canonical hop-128 grid and why it fixed a prior
2.39× density discrepancy (`scripts/deepsqueak_focus_stft.py:20-26`).

Data flow:
```
WAV + per-call contour parquet
  → (upstream extractor uses deepsqueak_focus_stft) → contours.parquet
  → window_calls_to_patches.py                      → window_index.parquet (fixed 234-bin windows)
  → sweep_contour_mask.py (human picks params)
  → mass_apply_contour_mask.py
      load_full_power_spec(): librosa.stft, hann, n_fft=512, hop=128 @300 kHz → |S|^2  (sweep_contour_mask.py:169-189)
      cut_patch(): slice fixed window
      contour_mask_utils.apply_hard_bandwidth_mask(): keep ±bandwidth_kHz around ridge
  → patches.npz (raw power float32) + patches_manifest.parquet
  → train_contour_vae_v2.py  (see production_vae.md)
```

### Focus-STFT render convention (canonical)

Two different STFTs live in Stack 4 — do not conflate them:

1. **Contour extraction** (`deepsqueak_focus_stft.py`): a *per-call adaptive-window* STFT.
   Window seconds come from `compute_optimal_window` (`deepsqueak_focus_stft.py:144-166`):
   `optimalWindow = sqrt(duration / (2000 * freq_range_kHz))`, then `× 1.5`
   (noverlap factor 0.5). Window/overlap/nfft samples = `round(sr * win_s)` etc.
   (`:202-204`). Defensive bounds: `freq_range >= 1 kHz`
   (`MIN_FREQ_RANGE_KHZ = 1.0`, `:79`) and windowsize capped at `4096` samples
   (`MAX_WINDOWSIZE_SAMPLES = 4096`, `:80`) so a degenerate CSV row can't trigger a
   multi-GB STFT. Uses a **Hamming** window (`scipy_signal.windows.hamming(..., sym=False)`,
   `:233`).

2. **Patch render** (`sweep_contour_mask.load_full_power_spec`,
   `sweep_contour_mask.py:169-189`): the *global canonical* STFT used to build the patches
   that get masked — **Hann window, `n_fft = corpus.STFT_N_FFT = 512`,
   `hop = corpus.STFT_HOP = 128`, `sr = corpus.SAMPLE_RATE_HZ = 300000`, `center=True`,
   raw power `|S|^2`**. This is the "global Hann 512/128 @300 kHz, fixed 234-bin windows,
   raw-power" convention from memory note `feedback_cleaning_pipeline_impl_on_rig`.

   `F = STFT_N_FFT // 2 + 1 = 257` (`mass_apply_contour_mask.py:133`). The window width
   `T = 234` bins = 100 ms is `round(0.100 * SAMPLE_RATE_HZ / STFT_HOP)`
   (`window_calls_to_patches.py:72`; verified `round(0.100*300000/128) = 234`). The long-call
   sliding step is `STEP_BINS = round(0.050*...) = 117` (`window_calls_to_patches.py:75`).

> **The actual rig artifacts live elsewhere.** The masking *scripts* are on the box, but
> the canonical `patches.npz` + the trained VAE + latents live ONLY on the GPU rig at
> `/data/shachar/contour_vae/` (memory note `feedback_cleaning_pipeline_impl_on_rig`,
> `docs/DATA_LOCATIONS.md`). Check the rig before re-rendering — a chat reinvented this
> render 2026-05-29 and diverged six ways.

### Key function signatures (Stack 4)

| Function | file:line | Returns |
|----------|-----------|---------|
| `compute_optimal_window(duration_s, freq_range_kHz, noverlap_fraction=0.5)` | `deepsqueak_focus_stft.py:144` | `(windowsize_s, overlap_s, nfft_s)` |
| `create_focus_spectrogram(audio_full, sr, call_box, frequency_padding_kHz=0.0)` | `deepsqueak_focus_stft.py:169` | `FocusSpectrogramResult` |
| `calculate_stats(I, windowsize, noverlap, nfft, sr, call_box, entropy_threshold=0.215, amplitude_threshold=0.825)` | `deepsqueak_focus_stft.py:351` | `CalculateStatsResult` (ridge bins + scales) |
| `extract_contour_for_call(audio_full, sr, call_box, ...)` | `deepsqueak_focus_stft.py:474` | `ContourBins(time_s, freq_kHz, tonality)` |
| `apply_hard_bandwidth_mask(S_pow, contour_t_bins, contour_freqs_kHz, contour_tonalities, freqs_kHz_axis, bandwidth_kHz, tonality_threshold)` | `contour_mask_utils.py:112` | masked `(F,T)` ndarray (fresh array) |
| `apply_soft_gaussian_mask(..., sigma_kHz, tonality_threshold)` | `contour_mask_utils.py:164` | masked `(F,T)` ndarray |

DeepSqueak algorithm constants (`deepsqueak_focus_stft.py:54-62`): `DS_ENTROPY_THRESHOLD =
0.215`, `DS_AMPLITUDE_THRESHOLD = 0.825`, `DS_NOVERLAP_FRACTION = 0.5`,
`DS_ENTROPY_SMOOTH_SPAN = 0.1`, `DS_RIDGE_SMOOTH_SPAN = 0.025`, `DS_RLOWESS_ITERATIONS = 5`,
`DS_LOWERING_FACTOR = 1.1`, `DS_MIN_RIDGE_POINTS = 5`, `DS_MAX_LOWERING_ITERS = 10`.

### Mask semantics (`contour_mask_utils.py`)

For each time column, `_qualifying_contour_per_column`
(`contour_mask_utils.py:73`) picks the contour bin with the **highest** tonality that is
`>= tonality_threshold` and in-range. Columns with no qualifying bin are zeroed entirely.
The hard mask keeps `|freq_axis - f_ridge| <= bandwidth_kHz` per active column
(`:156`); the Gaussian mask multiplies by `exp(-0.5*((f - f_ridge)/sigma)^2)` on active
columns (`:210-211`) and raises `ValueError` if `sigma_kHz <= 0` (`:189`). Both are
vectorized over time (no per-column Python loop over the F×T grid).

### Invariants (Stack 4)

- Patch tensor: `F = STFT_N_FFT//2+1 = 257`; single fixed `T` (234) enforced by assertion
  (`mass_apply_contour_mask.py:134-140`).
- Frequency axis must be identical across all WAVs (asserted, `:157-159`) — guaranteed by
  the canonical STFT params.
- Patches stored as **raw power float32, un-normalized** — normalization is the VAE
  trainer's job, not this stack's (`mass_apply_contour_mask.py:10-11`).

## Stacks 2a/2b internals (key signatures)

| Function | file:line | Returns |
|----------|-----------|---------|
| `auto_soft_notch(audio, fs_hz, library=None, ...)` | `notch.py:568` | `(cleaned_audio, ReconciliationResult)` |
| `discover_tonals(audio, fs_hz, *, discovery_threshold_db=10.0, median_window_hz=4000.0, nperseg=8192, ...)` | `notch.py:339` | `list[DetectedTonal]` |
| `apply_soft_notches(audio, fs_hz, tonals, *, min_width_hz=200.0, width_safety_factor=2.0, safety_margin_db=0.0, order=4, cut_depths_db=None)` | `notch.py:480` | cleaned audio (same shape/dtype) |
| `reconcile(library, detections, *, freq_tolerance_hz=200.0, intensity_drift_sigma=2.0)` | `notch.py:424` | `ReconciliationResult` |
| `subtract_temporal_baseline(spec_linear, method="percentile", percentile=10.0, envelope_kernel_frames=..., epsilon=1e-10)` | `denoise.py:60` | cleaned linear-mag spectrogram |

### Invariants (Stacks 2a/2b)

- **Default-off / byte-identical:** with no library and no discovered tonals,
  `auto_soft_notch` returns the input array unchanged (`notch.py:611-613`,
  `:599-600`). This is the published-result reproducibility guarantee.
- **Linear-domain subtraction only** (`denoise.py:22-24`).
- **Kernel derived from corpus** — `DEFAULT_ENVELOPE_KERNEL_FRAMES` recomputes from
  `SAMPLE_RATE_HZ` / `STFT_HOP` (`denoise.py:55-57`); never hardcode the frame count.

## Gotcha: archived stacks are still imported at runtime

Stacks 1 and 3 were moved to `archive/cleaning_legacy/` (2026-05-28) but **NOT deleted**,
because several LIVE render/experiment scripts import them at runtime by adding the archive
dir to `sys.path`. Verified call sites:

- `scripts/experiments/render_v1_faithful_patches.py:59-83` adds
  `archive/cleaning_legacy/stack1/{src,scripts}` to `sys.path`, then *re-attaches* the
  archived `CleaningConfig` / `clean_spectrogram` onto the live `usv_spectrogram.classifier`
  namespace so the archived `cnn_prepare_training_data` imports unchanged (preserving
  byte-identical 844-pixel patches).
- `scripts/experiments/render_vocalmat_style_patches.py:55`,
  `scripts/experiments/label_patches_v1.py:18` reference the Stack-1 archive.
- `scripts/experiments/render_vocalmat_gt_sample.py:29-30` inserts the Stack-3 archive
  onto `sys.path` at runtime (`ARCHIVE_SRC = archive/cleaning_legacy/stack3/src`).
  `scripts/experiments/train_shape_vae_alpha3.py:15` references the Stack-3 archive in a
  docstring only (points at `stack3/scripts/experiments/rig_M10_image_vae.py`) — it does
  not itself add the archive to `sys.path`.
- `src/usv_spectrogram/classifier/__init__.py:12` documents that `CleaningConfig` /
  `clean_spectrogram` were moved to the Stack-1 archive.

**Do not delete `archive/cleaning_legacy/`** — these live scripts break at import if you
do. (Verified in the 2026-06-08 dead-code audit, memory note
`project_deadcode_audit_2026-06-08`.)

## Where to change things

| You want to... | Change |
|----------------|--------|
| Adjust the canonical mask bandwidth/threshold | `--bandwidth-kHz` / `--tonality-threshold` on `mass_apply_contour_mask.py` (re-run sweep first). |
| Add a sweep bandwidth/threshold variant | `BANDWIDTH_VARIANTS` / `THRESHOLD_VARIANTS` lists in `sweep_contour_mask.py:87-98`. |
| Change the focus-STFT contour algorithm | `deepsqueak_focus_stft.py` (keep MATLAB line refs — it's a faithful port). |
| Change the patch render STFT | `load_full_power_spec` in `sweep_contour_mask.py:169` — but it MUST stay on `corpus` constants (ADR-002). |
| Tune production detection cleaning | `notch.py` (2a) / `denoise.py` (2b) defaults, or the `run_batch_detection.py` flags. **High-risk: any change here alters what the CNN sees.** |

## Cross-references

- [cleaning-subsystems.md](../modules/cleaning-subsystems.md) — the full four-stack
  inventory, call-site map, and VocalMat-equivalence adjudication (H1b).
- [production VAE](production_vae.md) — consumer of Stack 4's `patches.npz`.
- [CNN detection pipeline](cnn_detection_pipeline.md) — the detection path Stacks 2a/2b
  sit inside.
- `docs/DATA_LOCATIONS.md` — where `patches.npz` / VAE artifacts actually live (rig).
- ADR-001 (sample rate 300 kHz) / ADR-002 (STFT 512/128) — `DECISIONS.md`; constants
  enforced in `src/usv_spectrogram/corpus.py`.
