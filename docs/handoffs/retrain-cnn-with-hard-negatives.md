# Handoff: Retrain CNN with Hard Negatives + Full Pipeline Recalibration

**Date:** 2026-03-31
**From:** Session that completed 1960-pattern spectral flatness profiling and hard negative extraction
**Priority:** High — this is the recommended path forward for fixing broadband noise FPs

## Background

The current CNN (`models/matched_windows/best_model.pt`) misclassifies broadband noise bursts + low-frequency streaks as USVs (the "1960 pattern"). Spectral flatness profiling (AUC 0.97) confirmed the pattern is detectable but cannot serve as a standalone filter (10-15% per-detection FPR). CNN retraining with hard negatives is the recommended fix.

**Full investigation:** `1960-spectral-flatness-plan.md` and `docs/handoffs/1960-problem-investigation.md`

## What's Already Done

1. **Hard negative training data extracted** — 620 PNGs in exact assembler format
   - `data/training/hard_noises/hard_noises.csv` — 161 original + 459 jittered
   - `data/training/hard_noises/spectrograms/` — 100×256px, magma, MAD-normalized
   - Labels: all "NOISE" (column name: `label`)
   - Source labels: `data/manual_review_labels.csv` (200 detections: 161 noise, 39 USV from 33 files)

2. **Detection window padding** — `boundary_padding_cols` param added to `convert_to_detection_format` in `hysteresis.py` (backward compatible, default=0, all 99 tests pass). Activate with `boundary_padding_cols=25` when regenerating batch detections.

3. **Current training data** in `data/training/matched_windows/`:
   - train.csv: 10,712 samples (4,233 USV, 6,479 Not USV)
   - val.csv: 2,139 samples (872 USV, 1,267 Not USV)
   - test.csv: 1,829 samples (541 USV, 1,288 Not USV)

## Step-by-Step Pipeline

### Step 1: Merge hard negatives into training data

Append hard negative rows to the training CSV. The CSVs have the same format:
```
candidate_id,source_file,label,spectrogram_path
```

**Important:** The hard negatives CSV uses label `"NOISE"` but the existing training data uses `"Not USV"`. Normalize to `"Not USV"` when merging.

```python
import pandas as pd

train = pd.read_csv("data/training/matched_windows/train.csv")
hard_neg = pd.read_csv("data/training/hard_noises/hard_noises.csv")

# Normalize label and drop extra columns
hard_neg["label"] = "Not USV"
hard_neg = hard_neg[["candidate_id", "source_file", "label", "spectrogram_path"]]

# Append to train (not val/test — those should stay clean for fair eval)
merged = pd.concat([train, hard_neg], ignore_index=True)
merged.to_csv("data/training/matched_windows_v2/train.csv", index=False)

# Copy val.csv and test.csv unchanged
```

**Post-merge train stats:** ~11,332 samples (4,233 USV, 7,099 Not USV). The negative class grows by ~10%, which is a moderate shift.

**Recording-level split check:** The hard negatives come from manual_review tier recordings that are NOT in the existing train/val/test splits (they're 5970 batch recordings, not the labeled subset). So there's no train/test leakage risk.

### Step 2: Retrain the CNN

Use the existing training script:

```bash
.venv/bin/python scripts/train_cnn.py \
    --train-csv data/training/matched_windows_v2/train.csv \
    --val-csv data/training/matched_windows/val.csv \
    --output-dir models/hard_neg_retrain/ \
    --epochs 50 \
    --patience 10
```

Check `scripts/train_cnn.py` for exact CLI args — it may differ slightly. Key things:
- Use the same architecture (3 conv blocks + global avg pool)
- Same learning rate schedule as the matched_windows training
- Monitor val loss for early stopping

**Expected outcome:** The model should learn to reject broadband noise patterns while maintaining USV recall. Watch for:
- Val loss should decrease or stay similar (not increase significantly)
- USV recall on val set should stay ≥95%
- False positive rate on noise class should improve

### Step 3: Evaluate on test set

```bash
.venv/bin/python -c "
from src.usv_spectrogram.models.evaluate import evaluate_model
evaluate_model(
    model_path='models/hard_neg_retrain/best_model.pt',
    test_csv='data/training/matched_windows/test.csv',
    output_dir='models/hard_neg_retrain/evaluation/'
)
"
```

Compare metrics against the current model (`models/matched_windows/evaluation/`).

### Step 4: Calibrate temperature scaling

The sigmoid probabilities will have different calibration after retraining. Re-fit temperature:

```bash
.venv/bin/python scripts/calibrate_temperature.py \
    --model models/hard_neg_retrain/best_model.pt \
    --val-csv data/training/matched_windows/val.csv \
    --output models/hard_neg_retrain/temperature.json
```

### Step 5: Optimize hysteresis thresholds

The optimal onset/sustain thresholds will shift with the new model. Re-optimize:

```bash
.venv/bin/python scripts/optimize_hysteresis.py \
    --model models/hard_neg_retrain/best_model.pt \
    --labels data/unified_labels.json \
    --output models/hard_neg_retrain/hysteresis_optimization.json
```

This runs 5-fold cross-validated grid search over onset, sustain, gap_fill, and min_duration parameters.

### Step 6: Re-run batch detection on 5970

With the new model + calibration + hysteresis:

```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir 5970/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_5970_v2/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization.json \
    --no-resume
```

**Optional:** Pass `boundary_padding_cols=25` if wired into the batch script (currently it's in `convert_to_detection_format` but the batch script uses its own `_event_to_adr010_dict` — check if they share the same code path).

### Step 7: Validate against known files

Check the 10 original 1960-problem files + 8 from flatness review:
```
Known noise (should have 0 events or be auto_reject):
  0001960, 0002431, 0002522, 0003502, 0003503, 0003781, 0003794, 0005107, 0005656, 0006086
  0000570, 0000716, 0000717, 0003579, 0003825, 0004706, 0005108, 0005647

Known good USVs (should still be auto_accept):
  0000053, 0000054
```

Also regenerate the manual review PNGs for the new batch and check whether the manual_review tier shrinks (fewer borderline cases).

### Step 8: (Optional) Retrain FP filter

If there was a false-positive logistic regression filter (Stage 3 in the pipeline):

```bash
.venv/bin/python scripts/train_fp_filter.py \
    --model models/hard_neg_retrain/best_model.pt \
    --labels data/unified_labels.json \
    --output models/hard_neg_retrain/fp_filter.pkl
```

This may not be needed if the CNN itself now rejects the noise patterns.

## Key Files

| File | Role |
|------|------|
| `data/training/hard_noises/hard_noises.csv` | 620 hard negative samples (ready to merge) |
| `data/training/matched_windows/train.csv` | Current training data (10,712 samples) |
| `data/manual_review_labels.csv` | Source labels (161 noise, 39 USV from 33 files) |
| `models/matched_windows/best_model.pt` | Current production model (baseline) |
| `scripts/train_cnn.py` | CNN training script |
| `scripts/calibrate_temperature.py` | Temperature scaling calibration |
| `scripts/optimize_hysteresis.py` | Hysteresis parameter optimization |
| `scripts/run_batch_detection.py` | Batch detection pipeline |
| `src/usv_spectrogram/postprocessing/hysteresis.py` | Now has `boundary_padding_cols` param |
| `scripts/extract_hard_negatives.py` | Script that generated the hard negatives (rerun if labels change) |

## Relevant Constraints

- **sr=300000 always** — Never rely on library defaults (ADR-001)
- **STFT params**: n_fft=512, hop=128, window=hann, freq range 20-120 kHz
- **Training images must match inference exactly**: 100-column windows, global MAD normalization, magma colormap, 256px height, RGB PNG (grayscale conversion happens in data loader)
- **Recording-level splits**: Train/val/test split by recording stem to prevent leakage. Hard negatives are from different recordings than the existing splits.
- **Don't modify test expectations to pass** — If the new model scores differently, discuss before changing test thresholds.

## Success Criteria

1. The 18 known noise files are classified as auto_reject (0 events) or at least drop out of auto_accept
2. USV recall on the test set stays ≥95% (no regression on real calls)
3. The manual_review tier shrinks (fewer borderline cases)
4. Total auto_accept count stays within ~10% of current (1,344) — we shouldn't lose many real USVs

## Rollback

If retraining degrades performance, the current model is untouched at `models/matched_windows/best_model.pt` with its calibration files. The batch results stay at `results/batch_5970/`.
