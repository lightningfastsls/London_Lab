# Handoff: Fix the V2 Post-Processing Pipeline

**Date:** 2026-03-31
**Context:** A new CNN model was retrained with hard negatives (`models/hard_neg_retrain/best_model.pt`). A batch detection was run on all 6,400 WAV files, but the post-processing pipeline was not fully set up. Several stages are missing or misconfigured compared to the working v1 pipeline.

## What's Wrong

### 1. No FP Filter for the New Model
The old pipeline used a logistic regression FP filter (`models/matched_windows/fp_filter.pkl`) trained on 11 event-level features. The new model has **no FP filter at all** — it was never trained.

**Fix:** Train a new FP filter for the new model. The script is `scripts/train_fp_filter.py`. It needs labeled event-level features from the new model's detections. Check the old FP filter training for reference: `models/matched_windows/fp_filter.json` has the config.

### 2. Summary Parquet Only Covers 3,011 of 6,400 Files
The batch was split across 2 machines. Detection JSONs were merged into `results/batch_5970_v2/detections/` (all 6,400 present), but `results/batch_5970_v2/summary.parquet` only covers this machine's 3,011 files.

**Fix:** Regenerate the summary parquet from all 6,400 detection JSONs. The script `scripts/run_batch_detection.py` has a `_write_summary_parquet()` function, or you can write a quick script to:
1. Read all JSONs from `results/batch_5970_v2/detections/`
2. Compute per-file stats (n_events, max_confidence, etc.)
3. Run triage on each to assign tiers
4. Write `results/batch_5970_v2/summary_full.parquet`

### 3. Hysteresis Parameters May Be Unreliable
The hysteresis optimization for the new model only had **5 recordings** with available WAVs (many paths in `unified_labels.json` were missing). Compare:

| Param | Old Model (229 recordings) | New Model (5 recordings) |
|-------|---------------------------|-------------------------|
| onset | 0.60 | 0.60 |
| sustain | **0.45** | **0.20** |
| gap_fill | 0 | 2 |
| min_duration | 3 | 3 |
| F2 | 0.8848 | 0.9471 |

The sustain threshold dropped from 0.45 to 0.20 — this means the new model keeps detections alive at much lower probabilities. Combined with gap_fill=2 (bridging gaps), this makes the pipeline much more permissive.

**Fix:** Either:
- Re-run hysteresis optimization with more recordings (fix the WAV paths in `unified_labels.json` first)
- Or use the old model's hysteresis params (onset=0.60, sustain=0.45) as a starting point and test

### 4. Triage Distribution is Off
On the 3,011-file subset:
- Old model: auto_reject=4,986, auto_accept=1,344, manual_review=70 (out of 6,400)
- New model: auto_reject=2,398, auto_accept=234, manual_review=**379** (out of 3,011)

The manual_review tier jumped from ~1% to ~13% of files. This suggests the triage thresholds need recalibration for the new model's probability distribution, or the missing FP filter is letting through events that would have been filtered.

### 5. The Batch Run Was Done Without Temperature Scaling Check
Temperature was calibrated (T=0.9019 at `models/hard_neg_retrain/temperature.json`), but verify it was actually passed to the batch run command.

## What the V1 Pipeline Had (Working Reference)

The v1 pipeline that produced good results used all of these in sequence:
1. **CNN model:** `models/matched_windows/best_model.pt`
2. **Temperature scaling:** `models/matched_windows/temperature.json` (T=0.905)
3. **Hysteresis detection:** onset=0.60, sustain=0.45, gap_fill=0, min_duration=3
4. **FP filter:** `models/matched_windows/fp_filter.pkl` (logistic regression, 11 features)
5. **Triage:** auto_accept (>0.90), auto_reject (no events), manual_review (middle)

The batch command looked like:
```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir 5970/ \
    --model models/matched_windows/best_model.pt \
    --output-dir results/batch_5970/ \
    --temperature models/matched_windows/temperature.json \
    --fp-filter models/matched_windows/fp_filter.pkl \
    --hysteresis-config results/hysteresis_optimization.json \
    --workers 4
```

## Recommended Fix Order

1. **First:** Evaluate old model on test set (no baseline exists):
   ```bash
   .venv/bin/python scripts/evaluate_model.py \
       --model models/matched_windows/best_model.pt \
       --test-csv data/training/matched_windows/test.csv \
       --output-dir models/matched_windows/evaluation/
   ```
   Compare with new model at `models/hard_neg_retrain/evaluation/test_metrics.json`

2. **Fix WAV paths** in `unified_labels.json` so hysteresis optimization has enough data

3. **Re-run hysteresis optimization** with the new model and more recordings

4. **Train FP filter** for the new model

5. **Re-run batch detection** on all 6,400 files with the full pipeline (temperature + hysteresis + FP filter)

6. **Regenerate summary parquet** for all 6,400 files

7. **Compare tiers** between v1 and v2

## Key File Locations

| File | Purpose |
|------|---------|
| `models/hard_neg_retrain/best_model.pt` | New CNN model |
| `models/matched_windows/best_model.pt` | Old CNN model (working reference) |
| `models/hard_neg_retrain/temperature.json` | New temperature (T=0.9019) |
| `models/hard_neg_retrain/hysteresis_optimization.json` | New hysteresis params (unreliable — only 5 recordings) |
| `models/matched_windows/fp_filter.pkl` | Old FP filter (reference for training new one) |
| `results/batch_5970/` | Old batch results (v1, complete, working) |
| `results/batch_5970_v2/` | New batch results (v2, incomplete pipeline) |
| `results/hysteresis_optimization.json` | Old hysteresis optimization output |
| `src/usv_spectrogram/postprocessing/` | All pipeline modules (hysteresis, calibration, fp_filter, triage, etc.) |
| `scripts/run_batch_detection.py` | Batch detection script |
| `scripts/optimize_hysteresis.py` | Hysteresis parameter optimization |
| `scripts/train_fp_filter.py` | FP filter training (check if exists) |
| `data/unified_labels.json` | Labeled recordings for optimization |

## WAV File Locations

WAV files are NOT in a single directory. They span:
- `5970/` (6,400 files, nested in `USV1-5/usv_lmt_034/`)
- `5970_reviewed/`
- `5970_manual_review/`
- `5970_manual_review_reviewed/`

Use `rglob` to search across all directories.
