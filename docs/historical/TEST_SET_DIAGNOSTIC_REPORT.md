# Test Set Performance Diagnostic Report

**Date:** 2026-01-23
**Issue:** Model achieves 92% validation accuracy but only 58% test accuracy with 30% recall

---

## Executive Summary

Initial diagnostics show **NO obvious data leakage or distribution shift** between splits. The test set performance issue is likely due to:
1. Model overfitting to specific recordings despite good validation performance
2. Recording-specific characteristics not captured by basic statistics
3. Low recall (30%) suggests model is being too conservative in USV classification

---

## Diagnostic Results

### 1. Data Leakage Check ✅ PASS

- **No recording leakage** between train/val/test splits
- All 49 unique recordings properly separated:
  - Train: 34 recordings (740 samples)
  - Val: 7 recordings (178 samples)
  - Test: 8 recordings (129 samples)

### 2. Spectrogram Statistics ✅ PASS

**Statistical distributions are nearly identical across splits:**

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Mean pixel intensity | 0.2613 | 0.2621 | 0.2597 |
| Std pixel intensity | 0.0084 | 0.0072 | 0.0078 |
| Shape range | (256, 171-512) | (256, 217-512) | (256, 166-512) |

**Conclusion:** No distribution shift detected. Test spectrograms are statistically similar to train/val.

### 3. Class Distribution ✅ BALANCED

**Class balance is actually good:**
- Train: 58.5% USV / 41.5% Not USV
- Val: 60.7% USV / 39.3% Not USV
- Test: 52.7% USV / 47.3% Not USV

Test set has the most balanced distribution!

### 4. Recording-Level Analysis 🔍 FINDINGS

**Samples per recording:**
- Test: 16.1 avg (range: 8-27)
- Train: 21.8 avg (range: 5-97)
- Val: 25.4 avg (range: 4-54)

**Notable findings:**
1. **Outlier recordings with high pixel intensity (in TRAIN set):**
   - `2024-09-30_11-20-09_0000021.wav` (pixel_mean=0.276)
   - `2024-09-30_11-21-50_0000047.wav` (pixel_mean=0.277)

2. **Recordings with 100% noise (0% USV) in TRAIN set:**
   - 6 recordings with only noise samples (0% USV)
   - This is unusual - real recordings should have some USV activity

3. **Test recordings look normal:**
   - USV ratios: 25-67% (good variety)
   - Pixel intensities: 0.255-0.266 (normal range)

### 5. Test Set Recordings (Detailed)

```
Recording                                 Samples  USV  NotUSV  Ratio
2024-09-30_11-18-27_0000003.wav              27    15     12   55.6%
2024-09-30_11-19-09_0000008.wav              15     8      7   53.3%
2024-09-30_11-19-12_0000009.wav              16     4     12   25.0%
2024-09-30_11-19-15_0000010.wav               8     5      3   62.5%
2024-09-30_11-19-55_0000017.wav              18    12      6   66.7%
2024-09-30_11-19-58_0000018.wav              13     6      7   46.2%
2024-09-30_11-20-07_0000020.wav              15     9      6   60.0%
2024-09-30_11-21-40_0000044.wav              17     9      8   52.9%
```

All test recordings have reasonable USV/noise balance (25-67%).

---

## Key Questions Remaining

Since basic statistics don't explain the poor test performance, we need to investigate:

1. **What patterns is the model learning?**
   - Is it memorizing recording-specific features?
   - Is it learning spurious correlations?

2. **Why is recall so low (30%)?**
   - Model is missing 70% of true USVs
   - Is it being too conservative with threshold?
   - Are test USVs acoustically different from train USVs?

3. **Which test samples are failing?**
   - Need to analyze actual predictions to find patterns

---

## Next Steps

### Step 1: Generate Test Predictions

Run evaluation with prediction export:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py `
  --model models/best_model.pt `
  --test-csv splits/test.csv `
  --save-predictions test_predictions.csv
```

### Step 2: Analyze Prediction Errors

Run the prediction analysis script:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_predictions.py `
  --predictions test_predictions.csv `
  --test-csv splits/test.csv
```

This will show:
- Confidence distributions for correct vs incorrect predictions
- Confusion matrix with precision/recall breakdown
- Which recordings have the most errors
- High-confidence errors (model is very wrong with high confidence)

### Step 3: Visual Inspection

Based on the error analysis:
1. Manually review the top 10-20 misclassified samples
2. Look for patterns (e.g., soft USVs, specific frequencies, recording artifacts)
3. Check if missed USVs have common characteristics

### Step 4: Potential Fixes

Depending on what Step 2-3 reveal:

**If model is too conservative (high false negative rate):**
- Lower classification threshold from 0.5 to 0.3-0.4
- Adjust class weights during training
- Add more soft/weak USV examples to training set

**If model is overfitting to recordings:**
- Add data augmentation (time stretching, frequency shifting)
- Use per-recording normalization
- Consider recording ID as an input feature (with dropout)

**If test USVs are truly different:**
- May need to collect more diverse training data
- Consider domain adaptation techniques
- Check if test recordings are from different populations

---

## Diagnostic Scripts Created

1. **`scripts/diagnose_dataset.py`** - Dataset distribution analysis
   - Checks data leakage
   - Compares spectrogram statistics
   - Analyzes recording-level patterns

2. **`scripts/analyze_predictions.py`** - Prediction error analysis
   - Confidence analysis by correctness
   - Confusion matrix and metrics
   - Per-recording error rates
   - High-confidence error identification

3. **Updated `scripts/evaluate_model.py`** - Added `--save-predictions` flag
   - Export predictions for downstream analysis

---

## Usage Examples

```powershell
# 1. Run full dataset diagnostics
.\.venv\Scripts\python.exe scripts/diagnose_dataset.py

# 2. Generate predictions on test set
.\.venv\Scripts\python.exe scripts/evaluate_model.py `
  --model models/best_model.pt `
  --test-csv splits/test.csv `
  --save-predictions test_predictions.csv

# 3. Analyze prediction errors
.\.venv\Scripts\python.exe scripts/analyze_predictions.py `
  --predictions test_predictions.csv `
  --test-csv splits/test.csv
```

---

## Conclusion

The 92% → 58% accuracy drop between validation and test is **NOT explained by**:
- Data leakage ✅
- Distribution shift ✅
- Class imbalance ✅

The issue is likely **model generalization** - the model works well on recordings similar to those in train/val but struggles with the 8 test recordings despite their normal statistical properties.

**Next action:** Run Steps 1-3 above to identify which specific samples are failing and why.
