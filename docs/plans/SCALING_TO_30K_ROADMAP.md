# Scaling to 30K Labels - Complete Roadmap

**Date Created:** 2026-02-07
**Current Status:** ~2K labels, all infrastructure complete
**Goal:** Scale to 30K high-quality USV labels with iterative model improvement

---

## Table of Contents

1. [Current State](#current-state)
2. [Overview Strategy](#overview-strategy)
3. [Milestone Checkpoints](#milestone-checkpoints)
4. [Detailed Workflow](#detailed-workflow)
5. [Command Reference](#command-reference)
6. [Troubleshooting](#troubleshooting)

---

## Current State

### What You Have
- **Labels:** ~458 USV, ~374 Not USV, ~8 Uncertain (~840 total)
- **Model:** CNN at `models/full_retrained_cnn/best_model.pt`
- **Performance:** 89.7% precision, 93.8% recall, F1 91.7%
- **Optimal threshold:** 0.05 (app uses 0.04 high / 0.03 low)

### Infrastructure Complete ✅
- Detection app with boundary adjustment
- Progressive labeling workflow (presets, session tracking)
- Constrained jittering for training data generation
- Hard negative mining
- Outlier detection for QC
- Model scaling configurations (small → medium → large)
- Training curve visualization

---

## Overview Strategy

### The Active Learning Loop

```
┌─────────────────────────────────────────────────────┐
│  1. Label new data (detection app + manual review)  │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  2. Generate training data (jittering + negatives)  │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  3. Train CNN (with appropriate model size)         │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  4. Quality control (outlier detection)             │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  5. Mine hard negatives (from new recordings)       │
└──────────────────┬──────────────────────────────────┘
                   ↓
                REPEAT (model improves each cycle)
```

### Recommended Progression

| Milestone | Total Labels | Actions |
|-----------|--------------|---------|
| **Current** | ~840 | Baseline established |
| **Milestone 1** | 2,000 | First major retrain, validate pipeline |
| **Milestone 2** | 5,000 | Evaluate improvement, tune thresholds |
| **Milestone 3** | 10,000 | Consider medium model, hard negative mining |
| **Milestone 4** | 20,000 | Switch to medium model, comprehensive QC |
| **Milestone 5** | 30,000 | Evaluate large model, final optimization |

---

## Milestone Checkpoints

### Milestone 1: Reach 2,000 Labels

**Goal:** Validate pipeline and establish baseline for scaling.

**Labeling Target:** ~1,200 new labels (600 USV, 600 Not USV)

**Steps:**

#### 1.1 Label New Data (Progressive Workflow)

```powershell
# Run detection app on new/unlabeled recordings
.\.venv\Scripts\python.exe -m usv_spectrogram.app.main_window

# Use progressive labeling workflow:
# - Start with "High Confidence" preset (0.10/0.08)
# - Save obvious USVs
# - Switch to "Medium" preset (0.06/0.04)
# - Save clear USVs
# - Switch to "Low" preset (0.04/0.03)
# - Save faint USVs
# - Manually add missed USVs (right-click-drag)
# - Remove false positives (click + Delete key)
```

**Tips:**
- Process 5-10 recordings per session to avoid fatigue
- Use boundary adjustment (drag handles) for precise labels
- Track progress: aim for ~50 new labels per hour

#### 1.2 Generate Jittered Training Data

```powershell
# Generate positionally-diverse positive samples
.\.venv\Scripts\python.exe scripts/generate_jittered_training_data.py `
    --input-dir "5970 USV" `
    --output-dir data/training/jittered_2k `
    --window-ms 40 `
    --context-padding-ms 20 `
    --min-overlap-fraction 0.5 `
    --n-samples 5 `
    --seed 42

# This creates 5 jittered versions of each detection
# Example: 400 USV detections → 2,000 training samples
```

#### 1.3 Generate Negative Samples

```powershell
# Generate balanced negatives (50% random, 30% inter-USV, 20% low-energy)
.\.venv\Scripts\python.exe scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/training/negatives_2k `
    --n-random 500 `
    --n-inter-usv 300 `
    --n-low-energy 200 `
    --seed 42

# Adjust counts to match positive sample count (~2:1 ratio)
```

#### 1.4 Combine Training Data

```powershell
# Merge jittered positives + negatives into single dataset
# (Assuming you have a script for this, or do manually)

# Expected structure:
# data/training/full_2k_dataset/
# ├── spectrograms/
# │   ├── *.png
# ├── train.csv
# ├── val.csv
# └── test.csv
```

#### 1.5 Train CNN (Small Model)

```powershell
# Train with small model (appropriate for 2K samples)
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv data/training/full_2k_dataset/train.csv `
    --val-csv data/training/full_2k_dataset/val.csv `
    --spec-dir data/training/full_2k_dataset/spectrograms `
    --output-dir models/milestone_1_2k `
    --model-size small `
    --weight-decay 1e-4 `
    --epochs 50 `
    --patience 10 `
    --batch-size 32 `
    --learning-rate 0.001

# Training will auto-generate:
# - models/milestone_1_2k/best_model.pt
# - models/milestone_1_2k/training_curves.png
# - models/milestone_1_2k/training_history.json
```

#### 1.6 Evaluate Model

```powershell
# Evaluate on test set
.\.venv\Scripts\python.exe scripts/evaluate_experiment.py `
    --model models/milestone_1_2k/best_model.pt `
    --test-csv data/training/full_2k_dataset/test.csv `
    --spec-dir data/training/full_2k_dataset/spectrograms `
    --output-dir analysis/milestone_1_2k_eval

# Expected outputs:
# - Precision, recall, F1
# - Confusion matrix
# - Probability distributions
```

#### 1.7 Re-optimize Thresholds

```powershell
# Find new optimal threshold for updated model
.\.venv\Scripts\python.exe scripts/optimize_threshold.py `
    --model models/milestone_1_2k/best_model.pt `
    --test-csv data/training/full_2k_dataset/test.csv `
    --spec-dir data/training/full_2k_dataset/spectrograms `
    --output-dir analysis/milestone_1_2k_threshold

# Update detection app thresholds based on results
```

#### 1.8 Quality Control - Outlier Detection

```powershell
# Find potential labeling errors
.\.venv\Scripts\python.exe scripts/find_label_outliers.py `
    --model models/milestone_1_2k/best_model.pt `
    --data-csv data/training/full_2k_dataset/train.csv `
    --spec-dir data/training/full_2k_dataset/spectrograms `
    --output-dir analysis/milestone_1_2k_outliers `
    --threshold 0.7

# Review summary_report.txt
# Use labeling app to review and correct outliers
# If >10% outliers, investigate systematic issues
```

**Success Criteria:**
- ✅ Reached 2,000 total labels
- ✅ Model F1 score improved vs baseline
- ✅ Outlier rate <10%
- ✅ Training curves show convergence (not underfitting/overfitting)

---

### Milestone 2: Reach 5,000 Labels

**Goal:** Expand dataset and mine hard negatives.

**Labeling Target:** ~3,000 new labels

**Steps:**

#### 2.1 Label New Recordings

```powershell
# Continue progressive labeling on new recordings
.\.venv\Scripts\python.exe -m usv_spectrogram.app.main_window

# Load new model:
# File → Load Model → models/milestone_1_2k/best_model.pt
```

#### 2.2 Mine Hard Negatives

```powershell
# Find CNN false positives from unlabeled recordings
.\.venv\Scripts\python.exe scripts/mine_hard_negatives.py `
    --model models/milestone_1_2k/best_model.pt `
    --wav-dir "5970 USV" `
    --labeled-detections-dir data/labeled_detections `
    --output-dir data/hard_negatives_5k `
    --probability-threshold 0.3 `
    --buffer-ms 100 `
    --max-candidates-per-file 50

# Review candidates manually
# Add confirmed false positives to training data
```

#### 2.3 Generate Training Data

```powershell
# Jittered positives
.\.venv\Scripts\python.exe scripts/generate_jittered_training_data.py `
    --input-dir "5970 USV" `
    --output-dir data/training/jittered_5k `
    --n-samples 5 `
    --seed 42

# Negatives (scale proportionally: ~2,500 negatives for ~5,000 positives)
.\.venv\Scripts\python.exe scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/training/negatives_5k `
    --n-random 1250 `
    --n-inter-usv 750 `
    --n-low-energy 500 `
    --seed 42
```

#### 2.4 Train CNN (Small Model, Monitor for Underfitting)

```powershell
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv data/training/full_5k_dataset/train.csv `
    --val-csv data/training/full_5k_dataset/val.csv `
    --spec-dir data/training/full_5k_dataset/spectrograms `
    --output-dir models/milestone_2_5k `
    --model-size small `
    --weight-decay 1e-4 `
    --epochs 50 `
    --patience 10

# Review training_curves.png:
# - If train loss AND val loss both high → model too small (underfitting)
# - If train loss low, val loss high → overfitting (increase weight decay)
# - If both low and converged → good fit
```

#### 2.5 Evaluate and Compare

```powershell
# Evaluate new model
.\.venv\Scripts\python.exe scripts/evaluate_experiment.py `
    --model models/milestone_2_5k/best_model.pt `
    --test-csv data/training/full_5k_dataset/test.csv `
    --spec-dir data/training/full_5k_dataset/spectrograms `
    --output-dir analysis/milestone_2_5k_eval

# Compare with Milestone 1:
# - Precision/recall should improve
# - Probability distributions should be better calibrated
```

#### 2.6 Outlier Detection + Correction

```powershell
.\.venv\Scripts\python.exe scripts/find_label_outliers.py `
    --model models/milestone_2_5k/best_model.pt `
    --data-csv data/training/full_5k_dataset/train.csv `
    --spec-dir data/training/full_5k_dataset/spectrograms `
    --output-dir analysis/milestone_2_5k_outliers `
    --threshold 0.7

# Correct errors and retrain if >5% outlier rate
```

**Success Criteria:**
- ✅ Reached 5,000 total labels
- ✅ Model F1 improved vs Milestone 1
- ✅ Hard negatives successfully incorporated
- ✅ Ready to scale model size if showing underfitting

---

### Milestone 3: Reach 10,000 Labels

**Goal:** Scale to medium model size.

**Labeling Target:** ~5,000 new labels

**Steps:**

#### 3.1 Continue Labeling + Hard Negative Mining

```powershell
# Progressive labeling on new recordings
.\.venv\Scripts\python.exe -m usv_spectrogram.app.main_window

# Mine hard negatives with updated model
.\.venv\Scripts\python.exe scripts/mine_hard_negatives.py `
    --model models/milestone_2_5k/best_model.pt `
    --wav-dir "5970 USV" `
    --labeled-detections-dir data/labeled_detections `
    --output-dir data/hard_negatives_10k `
    --probability-threshold 0.3
```

#### 3.2 Generate Training Data

```powershell
# Jittered positives (~10,000 samples from ~2,000 detections)
.\.venv\Scripts\python.exe scripts/generate_jittered_training_data.py `
    --input-dir "5970 USV" `
    --output-dir data/training/jittered_10k `
    --n-samples 5 `
    --seed 42

# Negatives (~5,000 to maintain 2:1 ratio)
.\.venv\Scripts\python.exe scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/training/negatives_10k `
    --n-random 2500 `
    --n-inter-usv 1500 `
    --n-low-energy 1000 `
    --seed 42
```

#### 3.3 Train CNN (Medium Model)

```powershell
# Scale up to medium model (64→128→256 filters)
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv data/training/full_10k_dataset/train.csv `
    --val-csv data/training/full_10k_dataset/val.csv `
    --spec-dir data/training/full_10k_dataset/spectrograms `
    --output-dir models/milestone_3_10k `
    --model-size medium `
    --weight-decay 1e-4 `
    --epochs 50 `
    --patience 10 `
    --batch-size 32

# Medium model: ~400K parameters (vs 101K for small)
```

#### 3.4 Evaluate + Threshold Optimization

```powershell
# Evaluate
.\.venv\Scripts\python.exe scripts/evaluate_experiment.py `
    --model models/milestone_3_10k/best_model.pt `
    --test-csv data/training/full_10k_dataset/test.csv `
    --spec-dir data/training/full_10k_dataset/spectrograms `
    --output-dir analysis/milestone_3_10k_eval

# Re-optimize threshold
.\.venv\Scripts\python.exe scripts/optimize_threshold.py `
    --model models/milestone_3_10k/best_model.pt `
    --test-csv data/training/full_10k_dataset/test.csv `
    --spec-dir data/training/full_10k_dataset/spectrograms `
    --output-dir analysis/milestone_3_10k_threshold
```

#### 3.5 Quality Control

```powershell
.\.venv\Scripts\python.exe scripts/find_label_outliers.py `
    --model models/milestone_3_10k/best_model.pt `
    --data-csv data/training/full_10k_dataset/train.csv `
    --spec-dir data/training/full_10k_dataset/spectrograms `
    --output-dir analysis/milestone_3_10k_outliers `
    --threshold 0.7
```

**Success Criteria:**
- ✅ 10,000 labels reached
- ✅ Medium model outperforms small model
- ✅ No overfitting (train/val loss converge)
- ✅ F1 score continues to improve

---

### Milestone 4: Reach 20,000 Labels

**Goal:** Large-scale labeling with mature medium model.

**Labeling Target:** ~10,000 new labels

**Steps:**

#### 4.1 Batch Labeling Sessions

```powershell
# Efficient labeling with improved model
.\.venv\Scripts\python.exe -m usv_spectrogram.app.main_window

# Workflow:
# 1. Process recordings in batches (10-20 files)
# 2. Use session tracking to avoid re-labeling
# 3. Save frequently (File → Save Labels)
# 4. Mine hard negatives every 2-3K new labels
```

#### 4.2 Generate Training Data

```powershell
# Jittered positives (~20,000 samples)
.\.venv\Scripts\python.exe scripts/generate_jittered_training_data.py `
    --input-dir "5970 USV" `
    --output-dir data/training/jittered_20k `
    --n-samples 5 `
    --seed 42

# Negatives (~10,000)
.\.venv\Scripts\python.exe scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/training/negatives_20k `
    --n-random 5000 `
    --n-inter-usv 3000 `
    --n-low-energy 2000 `
    --seed 42
```

#### 4.3 Train CNN (Medium Model, Possible Large)

```powershell
# Try medium first
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv data/training/full_20k_dataset/train.csv `
    --val-csv data/training/full_20k_dataset/val.csv `
    --spec-dir data/training/full_20k_dataset/spectrograms `
    --output-dir models/milestone_4_20k_medium `
    --model-size medium `
    --weight-decay 1e-4 `
    --epochs 50 `
    --patience 10

# If showing underfitting, try large model:
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv data/training/full_20k_dataset/train.csv `
    --val-csv data/training/full_20k_dataset/val.csv `
    --spec-dir data/training/full_20k_dataset/spectrograms `
    --output-dir models/milestone_4_20k_large `
    --model-size large `
    --weight-decay 1e-4 `
    --epochs 50 `
    --patience 10

# Large model: ~1.6M parameters
```

#### 4.4 Comprehensive QC Pass

```powershell
# Outlier detection
.\.venv\Scripts\python.exe scripts/find_label_outliers.py `
    --model models/milestone_4_20k_medium/best_model.pt `
    --data-csv data/training/full_20k_dataset/train.csv `
    --spec-dir data/training/full_20k_dataset/spectrograms `
    --output-dir analysis/milestone_4_20k_outliers `
    --threshold 0.7

# Review ALL outliers before proceeding to 30K
# Correct systematic errors now to avoid propagating them
```

**Success Criteria:**
- ✅ 20,000 labels with <3% outlier rate
- ✅ Model performance plateauing (diminishing returns)
- ✅ Probability calibration excellent
- ✅ Ready for final push to 30K

---

### Milestone 5: Reach 30,000 Labels

**Goal:** Complete dataset with production-ready model.

**Labeling Target:** ~10,000 new labels

**Steps:**

#### 5.1 Final Labeling Push

```powershell
# Continue progressive labeling
.\.venv\Scripts\python.exe -m usv_spectrogram.app.main_window

# Focus on:
# - Edge cases (very short/long USVs)
# - Ambiguous samples (get second opinion if needed)
# - Diverse recording conditions
```

#### 5.2 Generate Final Training Data

```powershell
# Jittered positives (~30,000 samples)
.\.venv\Scripts\python.exe scripts/generate_jittered_training_data.py `
    --input-dir "5970 USV" `
    --output-dir data/training/jittered_30k `
    --n-samples 5 `
    --seed 42

# Negatives (~15,000)
.\.venv\Scripts\python.exe scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/training/negatives_30k `
    --n-random 7500 `
    --n-inter-usv 4500 `
    --n-low-energy 3000 `
    --seed 42
```

#### 5.3 Train Production Model

```powershell
# Train large model for production
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv data/training/full_30k_dataset/train.csv `
    --val-csv data/training/full_30k_dataset/val.csv `
    --spec-dir data/training/full_30k_dataset/spectrograms `
    --output-dir models/production_30k `
    --model-size large `
    --weight-decay 1e-3 `
    --epochs 100 `
    --patience 15 `
    --batch-size 32

# Note: Increased weight decay to 1e-3 for large model
```

#### 5.4 Final Evaluation

```powershell
# Comprehensive evaluation
.\.venv\Scripts\python.exe scripts/evaluate_experiment.py `
    --model models/production_30k/best_model.pt `
    --test-csv data/training/full_30k_dataset/test.csv `
    --spec-dir data/training/full_30k_dataset/spectrograms `
    --output-dir analysis/production_30k_eval

# Threshold optimization
.\.venv\Scripts\python.exe scripts/optimize_threshold.py `
    --model models/production_30k/best_model.pt `
    --test-csv data/training/full_30k_dataset/test.csv `
    --spec-dir data/training/full_30k_dataset/spectrograms `
    --output-dir analysis/production_30k_threshold
```

#### 5.5 Final Quality Check

```powershell
# Outlier detection
.\.venv\Scripts\python.exe scripts/find_label_outliers.py `
    --model models/production_30k/best_model.pt `
    --data-csv data/training/full_30k_dataset/train.csv `
    --spec-dir data/training/full_30k_dataset/spectrograms `
    --output-dir analysis/production_30k_outliers `
    --threshold 0.7

# Target: <2% outlier rate
```

#### 5.6 Deploy to Production

```powershell
# Copy best model to production location
Copy-Item models/production_30k/best_model.pt models/production/best_model.pt

# Update detection app to use new model
# File → Load Model → models/production/best_model.pt

# Update thresholds in app based on optimization results
```

**Success Criteria:**
- ✅ 30,000 high-quality labels
- ✅ Outlier rate <2%
- ✅ F1 score >95%
- ✅ Production model deployed
- ✅ Pipeline validated end-to-end

---

## Command Reference

### Quick Reference Table

| Task | Command |
|------|---------|
| **Run detection app** | `.\.venv\Scripts\python.exe -m usv_spectrogram.app.main_window` |
| **Generate jittered data** | `scripts/generate_jittered_training_data.py --input-dir "5970 USV" --output-dir <output> --n-samples 5` |
| **Generate negatives** | `scripts/generate_comprehensive_negatives.py --wav-dir "5970 USV" --labels-csv labels.csv --output-dir <output>` |
| **Train CNN** | `scripts/train_cnn.py --train-csv <train> --val-csv <val> --spec-dir <specs> --output-dir <output> --model-size <size>` |
| **Evaluate model** | `scripts/evaluate_experiment.py --model <model.pt> --test-csv <test> --spec-dir <specs> --output-dir <output>` |
| **Optimize threshold** | `scripts/optimize_threshold.py --model <model.pt> --test-csv <test> --spec-dir <specs> --output-dir <output>` |
| **Find outliers** | `scripts/find_label_outliers.py --model <model.pt> --data-csv <train> --spec-dir <specs> --output-dir <output>` |
| **Mine hard negatives** | `scripts/mine_hard_negatives.py --model <model.pt> --wav-dir "5970 USV" --labeled-detections-dir <labeled> --output-dir <output>` |

### Model Size Selection

| Label Count | Recommended Size | Parameters | Command Flag |
|-------------|------------------|------------|--------------|
| 2K - 10K | Small | ~101K | `--model-size small` |
| 10K - 20K | Medium | ~400K | `--model-size medium` |
| 20K - 30K+ | Large | ~1.6M | `--model-size large` |

### Training Parameters

| Parameter | Small Model | Medium/Large Model | Notes |
|-----------|-------------|-------------------|-------|
| `--weight-decay` | `1e-4` | `1e-4` to `1e-3` | Increase if overfitting |
| `--batch-size` | `32` | `32` | Reduce if GPU memory issues |
| `--learning-rate` | `0.001` | `0.001` | Default works well |
| `--epochs` | `50` | `50-100` | Let early stopping decide |
| `--patience` | `10` | `10-15` | More patience for larger models |

---

## Troubleshooting

### Model Not Improving

**Symptoms:** Val loss not decreasing, F1 score plateaued

**Possible Causes & Solutions:**
1. **Model too small** → Scale up to next size
2. **Data quality issues** → Run outlier detection, correct errors
3. **Insufficient data diversity** → Label more varied recordings
4. **Learning rate too high/low** → Try 1e-4 or 1e-2

### Overfitting

**Symptoms:** Train loss low, val loss high and diverging

**Solutions:**
1. Increase weight decay: `--weight-decay 1e-3` or `1e-2`
2. Add more training data
3. Reduce model size
4. Increase dropout (requires code change)

### Underfitting

**Symptoms:** Both train and val loss high, not converging

**Solutions:**
1. Increase model size
2. Train longer (`--epochs 100`)
3. Check data quality (corrupted images?)
4. Reduce weight decay if too high

### High Outlier Rate (>10%)

**Symptoms:** Outlier detection finds many disagreements

**Causes & Solutions:**
1. **Systematic labeling errors** → Review labeling guidelines
2. **Model undertrained** → Train longer or with more data
3. **Edge cases** → Normal, review and correct
4. **Threshold too strict** → Use `--threshold 0.6` instead of 0.7

### Hard Negative Mining Returns Too Many Candidates

**Symptoms:** Hundreds of candidates per file

**Solutions:**
1. Lower `--probability-threshold` to 0.4 or 0.5 (stricter)
2. Reduce `--max-candidates-per-file` to 20 or 30
3. Model may need retraining if it's producing many false positives

---

## Time Estimates

**Per Milestone (approximate):**

| Milestone | Labeling Time | Training Time | QC Time | Total |
|-----------|---------------|---------------|---------|-------|
| 1 (2K) | ~20 hours | ~30 min | ~2 hours | ~23 hours |
| 2 (5K) | ~50 hours | ~1 hour | ~4 hours | ~55 hours |
| 3 (10K) | ~100 hours | ~2 hours | ~8 hours | ~110 hours |
| 4 (20K) | ~200 hours | ~4 hours | ~16 hours | ~220 hours |
| 5 (30K) | ~200 hours | ~6 hours | ~16 hours | ~222 hours |

**Total to 30K:** ~630 hours (~16 weeks at 40 hours/week)

**Notes:**
- Labeling speed improves with practice
- Better models reduce manual corrections
- Parallel work possible (label while training)

---

## Success Metrics Dashboard

Track these at each milestone:

| Metric | Target |
|--------|--------|
| **F1 Score** | >95% by 30K |
| **Precision** | >93% |
| **Recall** | >95% |
| **Outlier Rate** | <2% by 30K |
| **Hard Negative Rate** | Decreasing over time |
| **Training Convergence** | Early stopping triggered |
| **Label Consistency** | Inter-rater agreement >90% (if using multiple labelers) |

---

## Final Notes

### Best Practices

1. **Save frequently** - Don't lose hours of labeling work
2. **Validate pipeline early** - Complete Milestone 1 thoroughly before scaling
3. **Review training curves** - They tell you if model size is appropriate
4. **Correct errors early** - Cheaper to fix at 5K than 30K
5. **Take breaks** - Labeling fatigue leads to errors
6. **Document decisions** - Note threshold changes, model swaps, etc.

### When to Deviate from Plan

- **Skip a milestone** if model is already excellent
- **Add intermediate checkpoints** if seeing issues
- **Adjust ratios** (positive:negative) based on results
- **Stop early** if hitting diminishing returns

### Questions to Ask at Each Milestone

1. Is the model improving?
2. Are training curves healthy?
3. Is outlier rate acceptable?
4. Should I scale model size?
5. Do I have sufficient data diversity?

---

**Good luck scaling to 30K! This is a marathon, not a sprint. 🚀**
