# Handoff: Hard Negative Retrain — Results & Model Comparison

**Date:** 2026-03-31
**From:** Session that executed the retrain plan from `retrain-cnn-with-hard-negatives.md`
**Status:** Batch detection complete, awaiting manual review of remaining noise files

## What Was Done

### Training Data (Step 1)
- Merged **620 hard negatives** (noise from manual review, label normalized NOISE → "Not USV")
- Extracted and merged **144 hard positives** (USVs from same manual review recordings — these were missing from the original handoff plan)
- Fixed absolute → relative spectrogram paths for portability
- Dropped extra columns (`original_detection_idx`, `jitter_idx`) to match train schema
- Output: `data/training/matched_windows_v2/train.csv` (11,476 samples: 4,377 USV + 7,099 Not USV)
- Val/test copied unchanged from `data/training/matched_windows/`

### CNN Retraining (Step 2)
```bash
.venv/bin/python scripts/train_cnn.py \
    --train-csv data/training/matched_windows_v2/train.csv \
    --val-csv data/training/matched_windows_v2/val.csv \
    --output-dir models/hard_neg_retrain/ \
    --num-epochs 60 --patience 12 \
    --model-size mid --use-class-weights
```
- Architecture: `mid` — filters [32, 96, 192], dense 64, ~207K params (same as previous model)
- Best epoch: 8 (early stopping at epoch 20)
- Best val loss: 0.2217, val accuracy: 94.25%, val F1: 92.89%

### Test Evaluation (Step 3)
| Metric | New Model |
|--------|-----------|
| Accuracy | 93.88% |
| Precision | 90.55% |
| Recall | 88.54% |
| F1 | 89.53% |
| Specificity | 96.12% |
| TP: 479, FP: 50, FN: 62, TN: 1238 |

Note: No baseline test_metrics.json exists for the old model at `models/matched_windows/evaluation/`. To generate one:
```bash
.venv/bin/python scripts/evaluate_model.py \
    --model models/matched_windows/best_model.pt \
    --test-csv data/training/matched_windows/test.csv \
    --output-dir models/matched_windows/evaluation/
```

### Temperature Calibration (Step 4)
- Temperature: **0.9019** (model already well-calibrated; ECE improved 0.0243 → 0.0205)
- Saved: `models/hard_neg_retrain/temperature.json`

### Hysteresis Optimization (Step 5)
- Best params: onset=0.6, sustain=0.2, gap_fill=2, min_duration=3
- F2 score: 0.9471 (±0.0815), **0 FP across all 5 folds**
- Only 5 recordings had WAVs available for optimization (many WAV paths in unified_labels.json were missing)
- Saved: `models/hard_neg_retrain/hysteresis_optimization.json`

### Batch Detection on 5970 (Step 6)
- Split across 2 machines (this laptop + PC) for speed
- This machine: 3,391 files, PC: 3,009 files
- PC results pushed to `results/5970_batch_v2/detections/`, merged into `results/batch_5970_v2/detections/`

## Model Comparison: Old vs New

### Overall Batch Stats

| Metric | Old Model | New Model | Delta |
|--------|-----------|-----------|-------|
| Total files | 6,400 | 6,400 | 0 |
| Total events detected | 8,036 | 8,248 | **+212** |
| Files with ≥1 detection | 1,414 | 1,586 | **+172** |

**Note:** The new model detects MORE events overall, not fewer. The hysteresis thresholds shifted (onset=0.6/sustain=0.2 vs whatever the old model used). This needs investigation — compare old hysteresis params.

### Known Noise Files (18 total)

**Fixed (→ 0 events): 9 files**
| Stem | Old Events | New Events |
|------|-----------|------------|
| 0001960 | 1 | **0** |
| 0003502 | 5 | **0** |
| 0003503 | 1 | **0** |
| 0003794 | 1 | **0** |
| 0005656 | 9 | **0** |
| 0006086 | 1 | **0** |
| 0000570 | 1 | **0** |
| 0000717 | 1 | **0** |
| 0003579 | 1 | **0** |

**Still detecting: 9 files**
| Stem | Old Events | New Events | Max Prob | Tier |
|------|-----------|------------|----------|------|
| 0000716 | 9 | 9 | 1.000 | auto_accept |
| 0005108 | 1 | 1 | 0.966 | auto_accept |
| 0005647 | 1 | 1 | 0.965 | auto_accept |
| 0004706 | 1 | 1 | 0.947 | auto_accept |
| 0003781 | 1 | 1 | 0.859 | manual_review |
| 0002522 | 1 | 1 | 0.852 | manual_review |
| 0002431 | 1 | 1 | 0.779 | manual_review |
| 0003825 | 1 | **3** | 0.748 | manual_review |
| 0005107 | 1 | 1 | 0.672 | manual_review |

**Key pattern:** The 10 manually-reviewed stems (trained on) had 80% fix rate. The 8 flatness-only stems (NOT trained on) had 12.5% fix rate. The model learned the specific patterns it was shown.

### Known Good USV Files
| Stem | Old Events | New Events |
|------|-----------|------------|
| 0000053 | 5 | **8** (+3 more detected) |
| 0000054 | 3 | 3 |

### Review PNGs
Spectrograms for all events in the 9 remaining noise files:
`results/batch_5970_v2/noise_review_pngs/`

Format: `{stem_short}_evt{N}_p{probability}.png`

## How to Compare the Two Models

### 1. Test Set Evaluation (side-by-side)
Generate old model metrics (missing), then compare:
```bash
# Old model (generate if missing)
.venv/bin/python scripts/evaluate_model.py \
    --model models/matched_windows/best_model.pt \
    --test-csv data/training/matched_windows/test.csv \
    --output-dir models/matched_windows/evaluation/ \
    --save-predictions models/matched_windows/evaluation/predictions.csv

# New model (already done)
# Results at: models/hard_neg_retrain/evaluation/
```
Compare: `models/matched_windows/evaluation/test_metrics.json` vs `models/hard_neg_retrain/evaluation/test_metrics.json`

### 2. Per-File Event Count Comparison
```python
import json, pandas as pd
from pathlib import Path

def load_events(det_dir):
    rows = []
    for j in Path(det_dir).glob('*.json'):
        data = json.loads(j.read_text())
        events = data if isinstance(data, list) else data.get('events', [])
        rows.append({'stem': j.stem, 'n_events': len(events),
                     'max_prob': max((e.get('max_probability',0) for e in events), default=0)})
    return pd.DataFrame(rows)

old = load_events('results/batch_5970/detections')
new = load_events('results/batch_5970_v2/detections')
merged = old.merge(new, on='stem', suffixes=('_old', '_new'))
merged['delta'] = merged['n_events_new'] - merged['n_events_old']

# Files where detection count changed
changed = merged[merged['delta'] != 0].sort_values('delta')
print(f"Files with fewer events: {(merged['delta'] < 0).sum()}")
print(f"Files with more events: {(merged['delta'] > 0).sum()}")
print(f"Files unchanged: {(merged['delta'] == 0).sum()}")
```

### 3. Hysteresis Parameter Comparison
Old model hysteresis params should be at `models/matched_windows/hysteresis_optimization.json` or embedded in the batch config. Compare onset/sustain/gap_fill/min_duration between the two.

### 4. Probability Distribution Comparison
```bash
# Compare CNN probability distributions on same recordings
.venv/bin/python scripts/compare_probability_distributions.py \
    --model-a models/matched_windows/best_model.pt \
    --model-b models/hard_neg_retrain/best_model.pt \
    --wav-dir 5970/ \
    --output results/pipeline_comparison/
```
(Check if this script exists and takes these args)

### 5. Triage Tier Comparison
Need to regenerate summary.parquet for the full 6400-file new batch (current one only covers 3011 files from this machine). Then compare tier distributions.

### 6. Temperature & Calibration Comparison
| | Old Model | New Model |
|--|-----------|-----------|
| Temperature | `models/matched_windows/temperature.json` | 0.9019 |
| Hysteresis onset | ? | 0.6 |
| Hysteresis sustain | ? | 0.2 |

## Open Questions

1. **Why +172 more files with detections?** The new model + hysteresis may be more sensitive overall. Need to check if old batch used different hysteresis params.
2. **Are the 4 auto_accept noise files actually USVs?** Review PNGs at `noise_review_pngs/` — especially 0000716 (9 events, all p>0.94). These were never manually labeled.
3. **Should we run another round?** Labeling the 8 flatness-only stems and retraining again would likely fix most of the remaining 9.
4. **Summary.parquet regeneration** — Current one only covers this machine's 3,011 files. Need to regenerate for all 6,400 with proper triage.

## Key Files

| File | Role |
|------|------|
| `models/hard_neg_retrain/best_model.pt` | New model (207K params, mid) |
| `models/matched_windows/best_model.pt` | Old model (207K params, same architecture) |
| `models/hard_neg_retrain/temperature.json` | New temperature scaling (0.9019) |
| `models/hard_neg_retrain/hysteresis_optimization.json` | New hysteresis params |
| `models/hard_neg_retrain/evaluation/` | Test set metrics, confusion matrix, ROC, PR curves |
| `data/training/matched_windows_v2/` | Merged training data (11,476 samples) |
| `data/training/hard_usvs/` | 144 hard positive PNGs + CSV |
| `data/training/hard_noises/` | 620 hard negative PNGs + CSV |
| `results/batch_5970_v2/detections/` | New batch detection JSONs (6,400 files) |
| `results/batch_5970/detections/` | Old batch detection JSONs (6,400 files, untouched) |
| `results/5970_batch_v2/detections/` | PC's batch results (pushed separately, already merged) |
| `results/batch_5970_v2/noise_review_pngs/` | Review PNGs for 9 remaining noise files |
| `pc_batch_files.txt` | List of 3,009 WAV files processed by PC |
| `scripts/extract_hard_negatives.py` | Extraction script (supports --label-filter usv/noise) |
