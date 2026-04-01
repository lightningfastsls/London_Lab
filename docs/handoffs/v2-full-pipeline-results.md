# V2 Full Pipeline — Final Results

**Date:** 2026-04-01
**Status:** Complete. New model is now the default in the PyQt6 app.
**Significance:** This is the most impactful improvement to the USV detection pipeline to date. The retrained CNN with a properly calibrated post-processing pipeline reduces false positives dramatically while maintaining high recall.

## What Was Done

### Problem
The original CNN model (`models/matched_windows/best_model.pt`) was producing false positive detections on broadband noise patterns (the "1960 pattern"). 18 recordings were known to contain only noise but were being flagged as USVs.

### Solution: Full Retrain + Pipeline Rebuild
1. **CNN retrained** with 620 hard negatives + 144 hard positives from manual review
2. **WAV paths fixed** in `unified_labels.json` — went from 5 to 220 usable recordings
3. **Hysteresis re-optimized** on 220 recordings (was only 5 — badly overfitting)
4. **FP filter trained** for the new model (was completely missing)
5. **Full batch detection** on all 6,400 files with complete pipeline

## Results

### CNN Test Set Comparison

| Metric | Old Model | New Model | Delta |
|--------|-----------|-----------|-------|
| Accuracy | 93.93% | 93.88% | -0.05% |
| **Precision** | **87.20%** | **90.55%** | **+3.35%** |
| Recall | 93.16% | 88.54% | -4.62% |
| F1 | 90.08% | 89.53% | -0.55% |
| Specificity | 94.25% | 96.12% | +1.87% |
| FP | 74 | **50** | **-32%** |

Trade-off is exactly what was intended: fewer false positives at cost of some recall.

### Hysteresis Parameters

| Param | Old Model (229 recs) | New Model v1 (5 recs) | New Model FINAL (220 recs) |
|-------|---------------------|-----------------------|---------------------------|
| onset | 0.60 | 0.60 | 0.60 |
| sustain | 0.45 | 0.20 | **0.40** |
| gap_fill | 0 | 2 | **0** |
| min_duration | 3 | 3 | 3 |
| F2 | — | 0.9471 (inflated) | **0.8669** |

The 5-recording optimization was wildly misleading (sustain=0.20, gap_fill=2). With proper data, params converged close to the old model.

### FP Filter

| Metric | Old Filter | New Filter |
|--------|-----------|------------|
| Events | 1,537 (1,348 USV, 189 FP) | 1,319 (1,234 USV, 85 FP) |
| Mean F2 | 0.8504 | 0.8233 |
| Top feature | peak_probability (1.85) | peak_probability (1.65) |
| SNR importance | **1.45** | **0.19** |
| Tonality importance | **1.12** | **0.01** |

The CNN now handles what the old FP filter used to catch (low-SNR, non-tonal noise). SNR and tonality dropped to near-zero importance.

### Batch Detection (6,400 files)

| Tier | Old Model | New Model | Delta |
|------|-----------|-----------|-------|
| auto_accept | 1,344 | 1,194 | -150 |
| auto_reject | 4,986 | 5,072 | +86 |
| manual_review | 70 | 134 | +64 |

### Agreement Between Models
- **98.6% agreement** on whether a file has any detections
- 95.9% same triage tier (among files in old summary)
- 88 files lost all events (old detected, new doesn't)
- Only 2 files gained new detections

### Known Noise Files (18 total)
- **16 of 18 fixed** (→ 0 events). Up from 9 without FP filter.
- **2 remaining:** 0000716 (9 events, p=1.0) and 0005647 (1 event, p=0.965)

### Manual Review Quality
- 134 files in manual_review, containing 156 sub-threshold events (p < 0.90)
- **154 of 156 are real USVs** (98.7%)
- Only 2 noise events found: 0004134 evt0, 0004242 evt0
- The manual_review tier is triggered by the `all(events >= 0.90)` rule — one borderline event drags the whole file

### Downgraded Files (old auto_accept → new lower tier)
- 20 files downgraded: 13 → auto_reject, 7 → manual_review
- The 13 → auto_reject were confirmed noise (single events with old p=0.90-0.97, now 0)
- The 7 → manual_review contain real USVs, just hit by the strict triage rule

## Pipeline Configuration (Production)

```bash
# Full pipeline command
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir 5970/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_5970_v2_full/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

## App Configuration
- Default model changed: `models/matched_windows/best_model.pt` → `models/hard_neg_retrain/best_model.pt`
- Preset thresholds updated to match new hysteresis optimization (Best F2: onset=0.60, sustain=0.40)
- Files: `src/usv_spectrogram/app/main.py`, `src/usv_spectrogram/app/core/preset_config.py`

## Key Files

| File | Purpose |
|------|---------|
| `models/hard_neg_retrain/best_model.pt` | **Production model** (207K params, mid, epoch 8) |
| `models/hard_neg_retrain/temperature.json` | Temperature scaling (T=0.9019) |
| `models/hard_neg_retrain/hysteresis_optimization_v2.json` | Hysteresis params (220 recordings) |
| `models/hard_neg_retrain/fp_filter.pkl` | FP filter (logistic regression) |
| `models/hard_neg_retrain/fp_filter.json` | FP filter training report |
| `models/hard_neg_retrain/evaluation/` | Test set metrics, curves |
| `models/matched_windows/evaluation/` | Old model test metrics (generated this session) |
| `results/batch_5970_v2_full/` | Full pipeline batch results (6,400 files) |
| `results/batch_5970_v2_full/manual_review_pngs/` | PNGs for sub-threshold events only |
| `results/batch_5970_v2_full/manual_review_wavs/` | Symlinked WAVs for manual review |
| `results/batch_5970_v2_full/downgraded_review_pngs/` | PNGs for old auto_accept → downgraded |
| `results/batch_5970_v2_full/downgraded_review_wavs/` | Symlinked WAVs for downgraded files |
| `data/unified_labels.json` | Fixed WAV paths (220/229 recordings usable) |

## What Could Be Done Next
1. Add the 2 new noise events (4134, 4242) + remaining noise files as hard negatives for another retrain cycle
2. Review the 2 holdout noise files (0000716, 0005647) — may actually contain real USVs given p=1.0
3. The 9 missing noise recordings (nums 8557-11488) need WAVs from the source machine
