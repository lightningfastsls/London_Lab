# Data Regeneration Recipes

> **Produced during the 2026-06-08 handoff cleanup.** This document is the safety net
> that makes deletion of large *regenerable* data directories reversible. Every directory
> below is mechanically derived from upstream source data (raw WAVs, label CSVs, batch
> parquets) and can be rebuilt from scratch with the command in its **Regenerate** block.
>
> **All directories documented here are CANDIDATES FOR DELETION** — but only once the
> downstream analysis that consumes them is frozen. Raw cohort recordings
> (`USV_lab_131204/`, `5970/`, `USV_9252/`, `USV_3452_sample/`, label CSVs, Raven tables)
> are **irreplaceable** and are NOT in this list.
>
> Conventions:
> - All commands assume the repo root as CWD and use the WSL interpreter `.venv/bin/python`.
> - Sizes are as measured on 2026-06-08.
> - Where a script exposes a parameter not pinned by a recorded run, the doc says
>   "see `<script> --help`" rather than inventing a value.
> - **Do NOT delete anything based on this doc alone.** It is documentation only.

---

## 1. `USV_lab_131204_chunked_2s_full/`

**Purpose.** The wild detection pipeline expects short trigger-recorded snippets
(~2 s). The lab cohort (`USV_lab_131204/`) is continuous 10-minute recordings at
250 kHz; feeding a 10-min file through the loader OOMs (a single-pass 600 s
spectrogram at 300 kHz needs ~1.4 GB). This directory slices each long lab WAV into
overlapping 2 s chunks resampled to the canonical 300 kHz, so the pipeline sees many
small files identical in shape to wild data. Chunking is undone downstream via
`chunk_manifest.csv` (`original_begin_time_s = chunk_begin_time_s + start_s_in_original`).

**Source data.** `USV_lab_131204/` — 83 raw recordings, 250 kHz, ~10 min each (24 GB).
**Irreplaceable; do not delete.**

**Created by.** `scripts/chunk_and_resample_lab_to_300k.py`. The actual run parameters
are recorded inside the output `chunk_manifest.csv` (verified header row):
`chunk_duration_s = 2.0`, `overlap_s = 0.1`, `src_sample_rate_hz = 250000`,
`dst_sample_rate_hz = 300000` (resample_poly up=6 / down=5), `min_chunk_s = 1.0`
(script default). Naming scheme: `{original_stem}_chunk_{index:03d}.wav`
(e.g. `131204_1400_m1fm1_chunk_000.wav`). The target rate is pinned to
`corpus.SAMPLE_RATE_HZ`, not a CLI flag.

**Regenerate.**
```bash
.venv/bin/python scripts/chunk_and_resample_lab_to_300k.py \
    --src-dir USV_lab_131204/ \
    --dst-dir USV_lab_131204_chunked_2s_full/ \
    --manifest USV_lab_131204_chunked_2s_full/chunk_manifest.csv \
    --chunk-duration-s 2.0 \
    --overlap-s 0.1 \
    --min-chunk-s 1.0 \
    --workers 4
```

**Size.** 58 GB, 25,770 WAVs. (Largest single regenerable artifact in the repo.)

---

## 2. `usv_language/prepared_data/`

**Purpose.** Preprocessed bout dataset for the USV-language transformer training. Holds
`train/`, `val/`, `test/` `.npy` spectrogram tensors plus `normalization_stats.npz` and
`pipeline_config.json`. Built once per corpus by extracting bouts from detection results,
computing per-bout spectrograms, splitting by recording, and computing train-set
normalization stats.

**Source data.** Detection JSONs + WAVs of the 5970 cohort:
`--detection-dir USV_Detections/5970` and `--wav-dir "5970 USV"`. (Both irreplaceable.)

**Created by.** `usv_language/data/prepare_data.py` (run as a module). Actual parameters
recorded in `usv_language/prepared_data/pipeline_config.json` (verified):
`bout_gap_threshold_ms = 500.0`, `context_padding_ms = 200.0`, `min_bout_duration_ms = 50.0`,
`max_bout_duration_ms = 10000.0`; spectrogram `n_fft = 512`, `hop_length = 128`,
`freq_min/max_hz = 20000/120000`, `sr = 300000`, `window = hann`; dataset
`max_seq_len = 512`, `overlap_ratio = 0.5`, split 0.8/0.1/0.1, `batch_size = 32`,
`seed = 42`. Produced 1,658 bouts / 1,658 spectrograms over 1,414 recordings.

**Regenerate.**
```bash
.venv/bin/python -m usv_language.data.prepare_data \
    --detection-dir USV_Detections/5970 \
    --wav-dir "5970 USV" \
    --output-dir usv_language/prepared_data \
    --bout-gap-ms 500 \
    --padding-ms 200 \
    --max-seq-len 512 \
    --batch-size 32 \
    --seed 42
```
(`min-bout`/`max-bout`/STFT params are script defaults baked into the preparer; see
`usv_language/data/prepare_data.py --help` for the full flag list.)

**Size.** ~3.0 GB.

---

## 3. `data/lab_finetune_v1/`

**Purpose.** Working/labeling scratch directory for the lab CNN fine-tune (active-learning
cycle). It is NOT the training PNG output (that goes to `data/training/lab_finetune_v1/`).
It holds the human-labeling workspace: `labels_audit_72.csv` (the 72 re-elicited audit
labels), `labeled/` (labeled review PNGs, `bor*/typ*` stable-ID prefixes), `labeling_queue/`,
and `mining_candidates_500/` (stratified-random mining renders + `candidates_seed42.csv`).

**Source data.** The production lab batch parquet
`results/batch_lab_131204_full/merged_events_with_filter.parquet`, the chunk WAVs
`USV_lab_131204_chunked_2s_full/`, and the prior audit labels
`data/lab_finetune_v1/labels_audit_72.csv`. The 72 audit labels are
**human-produced — irreplaceable**; back them up before deleting the directory.

**Created by.** `scripts/mine_lab_finetune_candidates.py` produces the
`mining_candidates_*` subdir (stratified-random sample of typical / long_event / borderline
auto-accept events, excluding already-labeled events, rendered with the same `render_event`
style as the audit). The `mining_500` run used `--seed 42` (see `candidates_seed42.csv`).
The `labeled/`, `labeling_queue/` subtrees are human/interactive labeling output, not a
single script's deterministic product.

**Regenerate** (the mining candidates — the deterministic part):
```bash
.venv/bin/python scripts/mine_lab_finetune_candidates.py \
    --n-typical 100 --n-long-event 30 --n-borderline 20 \
    --out-dir data/lab_finetune_v1/mining_candidates_500 \
    --exclude-labels-csv data/lab_finetune_v1/labels_audit_72.csv \
    --seed 42
```
(Adjust `--n-*` to match the desired sample size; defaults are 100/30/20. The 500-candidate
run scaled these up — see `scripts/mine_lab_finetune_candidates.py --help`. The
`labeled/`/`labeling_queue/` contents are human review products and are NOT regenerable
by script — preserve any labels.)

**Size.** ~2.1 GB.

---

## 4. alpha3 patch-render caches (`data/alpha3_*`)

These four caches are spectrogram-patch PNG/tensor renders for the α₃-C shape-VAE
labeling-oracle work. All derive from the production lab classification CSV
`classified_detections_lab_131204_clean.csv` (40,787 rows) and the chunk WAVs
`USV_lab_131204_chunked_2s_full/`. `call_id` convention: `{wav_stem}__det{det_index}`.

> **KEEP-WORTHY:** `data/alpha3_human_patches/` is the **human shape-labeling substrate**
> (normal-spectrogram, tight-crop renders). The lab's 204 shape labels were made on these.
> The masked `alpha3_patches` are illegible vertical bands and are NOT the labeling substrate
> (per project memory `feedback_shape_labeling_substrate`). Treat `alpha3_human_patches`
> as analysis input, not deletion fodder, until shape labeling is fully frozen.

### 4a. `data/alpha3_patches/` — VocalMat-style masked oracle substrate
**Purpose.** 227×227 3-channel grayscale PNG patches rendered from the Stack-4
contour-masked focus spectrogram, the substrate the in-house VocalMat oracle
(`results/lab_classifier_v1/best.pt`) labels in A4 (`label_patches_v1.py`).
**Created by** `scripts/experiments/render_vocalmat_style_patches.py`.
```bash
.venv/bin/python scripts/experiments/render_vocalmat_style_patches.py \
    --csv classified_detections_lab_131204_clean.csv \
    --wav-root USV_lab_131204_chunked_2s_full \
    --out-dir data/alpha3_patches \
    --workers 4
```
(Source-rate default 250000; pad/bandwidth/tonality thresholds are script defaults —
see `render_vocalmat_style_patches.py --help`.) **Size.** 168 MB.

### 4b. `data/alpha3_oracle_patches/` — v1-faithful UNMASKED patches
**Purpose.** "v1-faithful" UNMASKED 227×227 RGB patches (0.22 s window centred on the
detection midpoint) that reproduce the v1 training render byte-for-byte, so the
`lab_classifier_v1` oracle sees in-distribution inputs (masked patches made it collapse to
~75% Noise). Imports the archived Stack-1 building blocks; rewrites no DSP math.
**Created by** `scripts/experiments/render_v1_faithful_patches.py`.
```bash
.venv/bin/python scripts/experiments/render_v1_faithful_patches.py \
    --wav-root USV_lab_131204_chunked_2s_full \
    --out-dir data/alpha3_oracle_patches/ \
    --workers 4
```
(Input CSV / manifest selection: see `render_v1_faithful_patches.py --help`;
`--limit 0` renders all rows.) **Size.** 3.0 GB.

### 4c. `data/alpha3_human_patches/` — HUMAN-labeling substrate (KEEP-WORTHY)
**Purpose.** Tight per-call crop renders (normal spectrogram, neighbouring USVs pushed out
of frame) for the human to *see* the target call's shape. This is the substrate the lab's
shape labels were made on. Output: `<call_id>.png` + `manifest.csv`.
**Created by** `scripts/experiments/render_human_view_patches.py`.
```bash
.venv/bin/python scripts/experiments/render_human_view_patches.py \
    --csv classified_detections_lab_131204_clean.csv \
    --labels data/labels_vocalmat_v1_on_131204.csv \
    --extra-manifest data/alpha3_gamma_manifest.csv \
    --wav-root USV_lab_131204_chunked_2s_full \
    --out-dir data/alpha3_human_patches \
    --pad-frac 0.2 --min-window-s 0.05 --workers 8
```
(`--isolate-dim 0.22`, `--isolate-pad-frac 0.06` are defaults; see
`render_human_view_patches.py --help`.) **Size.** 3.1 GB.

### 4d. `data/alpha3_a6/` — A6 eval-baseline artifacts (not PNGs)
**Purpose.** Small parquet/JSON artifacts for the α₃-C A6 eval-validity baselines (the
mandatory random-init-encoder and column-mean-identity baselines, plus latent-bridge and
binding manifests). NOT patch images. Largely produced on the GPU rig (imports the
production VAE's own `ImageVAE`/`MaskedPatchDataset` for byte-identical preprocessing).
**Created by** `scripts/experiments/rig_extract_a6_baselines.py` (rig) and consumed/extended
by `scripts/experiments/eval_a6_existing_latents.py`; provenance in
`docs/handoffs/2026-05-30_alpha3-C-A8-A6-binding.md`.
```bash
# Runs on the GPU rig against results/masked_patches/combined_all_cohorts/patches.npz
.venv/bin/python scripts/experiments/rig_extract_a6_baselines.py \
    --keymap <bridge_keymap> --out data/alpha3_a6/a6_baselines.parquet \
    --cohort lab_131204 --seed 42 --agg mean --device cuda:0
```
(See `rig_extract_a6_baselines.py --help` for required `--keymap`/`--out` paths; the
contour-VAE patch tensors live only on the rig at `/data/shachar/contour_vae/`.)
**Size.** 54 MB.

---

## 5. Scratch chunker dev / review-cascade directories

**Purpose.** Iterative chunker-development experiments — small inputs run through
`scripts/chunk_and_resample_lab_to_300k.py` while tuning chunk length / sample rate /
overlap before the full 58 GB run. Plus a manual "hot" review cascade where each `_reviewed`
level is a human-curated subset of the prior level. All are regenerable from
`USV_lab_131204/` via the same chunker with a `--limit` or smaller chunk params.

**Source data.** `USV_lab_131204/` (raw recordings, irreplaceable).

**Directories & sizes (2026-06-08):**

| Directory | Files | Size | Note |
|---|---|---|---|
| `USV_lab_131204_2chunk_test/` | 2 | 138 MB | smoke test, `--limit`-style small input |
| `USV_lab_131204_10chunk_test/` | 10 | 23 MB | smoke test |
| `USV_lab_131204_100chunk_test/` | 100 | 229 MB | smoke test |
| `USV_lab_131204_chunked_300k_test/` | 12 | 695 MB | early 300 kHz resample test |
| `USV_lab_131204_chunked_2s_test/` | 318 | 726 MB | pre-full 2 s chunk test |
| `USV_lab_131204_chunked_2s_hot/` | 101 | 227 MB | soft-notch review subset (handoff 2026-05-11) |
| `USV_lab_131204_chunked_2s_hot_reviewed/` | 1 | 2.3 MB | human-curated subset of `_hot` |
| `USV_lab_131204_chunked_2s_hot_reviewed_reviewed/` | 1 | 2.3 MB | further human-curated subset |
| `USV_lab_131204_chunked_2s_full_reviewed/` | 539 | 1.3 GB | human-curated subset of `_full` |

**Regenerate** (representative — chunk dev test):
```bash
# e.g. reproduce a 100-file chunk smoke test
.venv/bin/python scripts/chunk_and_resample_lab_to_300k.py \
    --src-dir USV_lab_131204/ \
    --dst-dir USV_lab_131204_100chunk_test/ \
    --manifest USV_lab_131204_100chunk_test/chunk_manifest.csv \
    --chunk-duration-s 2.0 --overlap-s 0.1 --limit <N_source_files> \
    --workers 4
```
The `*_test` / `*_300k_test` dirs differ only in `--limit`, `--chunk-duration-s`, and
sample-rate target relative to the full run (recover exact per-dir params from each dir's
`chunk_manifest.csv` if present). The `_hot*` and `_*_reviewed` cascades are **human
review subsets**, not deterministic script output — the chunk WAVs are regenerable, but the
*selection* of which files were reviewed is a manual artifact; preserve any associated
review CSV/verdict before deleting.

---

## 6. Dated result pilots (`results/`)

Point-in-time analysis outputs from the lab detection / cleaning experiments. Each is a
`scripts/run_batch_detection.py` (or preview-script) output with a specific flag
combination; none are inputs to a frozen downstream pipeline. Delete once the analysis they
back is captured in a handoff/report.

### 6a. `results/batch_lab_131204_subtracted_pilot/` (329 MB)
**Purpose.** Pilot of per-frequency-bin spectral background subtraction (Boll 1979,
10th-percentile-over-time baseline) on 382 lab chunks — comparison reference for the
fine-tune. **Source.** Subset of `USV_lab_131204_chunked_2s_full/`. **Preview reproducer.**
`scripts/preview_spectral_subtraction.py` (side-by-side renders). **Batch reproducer.**
```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir USV_lab_131204_chunked_2s_full/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_lab_131204_subtracted_pilot/ \
    --subtract-baseline --subtraction-method percentile \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

### 6b. `results/batch_lab_full_softnotch_20260513_1538/` (448 MB)
**Purpose.** Full lab batch run with the adaptive soft-notch cleaning step enabled
(handoff `docs/handoffs/2026-05-11_adaptive-soft-notch.md`). **Source.**
`USV_lab_131204_chunked_2s_full/`. **Reproducer** — same canonical 5-flag pipeline plus the
`--soft-notch` flag (defaults off; wild runs must stay byte-identical without it):
```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir USV_lab_131204_chunked_2s_full/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_lab_full_softnotch_<TIMESTAMP>/ \
    --soft-notch auto \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```
(Exact soft-notch config: pass `auto` or a config PATH — see
`scripts/run_batch_detection.py --help` and `scripts/notch_filter_wav.py`.)

### 6c. `results/codex_detection_compare_{100,band_wav,one_wav}/` (230 MB + 2.4 MB + 2.4 MB)
**Purpose.** Detection-comparison spot checks (100-file, single-band-WAV, single-WAV) used to
cross-check the detector during the Codex bridge work. **Source.** Lab/wild WAV subsets.
**Reproducer.** Outputs of `scripts/run_batch_detection.py` on the named WAV subsets; the
band/one-wav variants are single-file runs. Recover the exact `--wav-dir`/model from each
dir's run JSONs. See `docs/REPO_CLEANUP_PLAN_2026-06-08.html` for the cleanup-audit context.

### 6d. `results/pipeline_comparison/` (251 MB)
**Purpose.** Three-stage detection ablation for the presentation: `1_raw_cnn/`,
`2_hysteresis_only/`, `3_full_pipeline/` — the same WAV set scored at three pipeline depths
(raw CNN scores → CNN+hysteresis → full 5-flag pipeline). **Source.** Presentation WAV set.
**Reproducer.** Three `scripts/run_batch_detection.py` runs differing only in which
post-processing flags are supplied (none → `--hysteresis-config` only → all five). Provenance:
`docs/handoffs/2026-05-03_presentation-png-provenance-audit.md` and the hard-neg-retrain
results handoff. Re-run the canonical 5-flag command for `3_full_pipeline/`; drop
`--fp-filter`/`--temperature` for the leaner stages.

> Other dated pilots exist (e.g. `results/batch_lab_131204_envelope_pilot/`); the same rule
> applies — they are `run_batch_detection.py` outputs with a documented flag set and are
> regenerable. The authoritative space/regenerability audit is
> `docs/REPO_CLEANUP_PLAN_2026-06-08.html`.
