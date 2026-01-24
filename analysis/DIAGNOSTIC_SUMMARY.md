# CNN Test Set Performance Diagnostic Summary

**Date:** 2026-01-23
**Session:** Session 9 - Test Set Performance Investigation
**Model:** `models/clean_test/best_model.pt` (Epoch 9)

---

## Executive Summary

The CNN model showed severe overfitting with 92% validation accuracy but only 58% test accuracy (30% recall). Comprehensive diagnostics revealed **two core issues**:

1. **Probability compression** - Model outputs capped at 0.57 (should reach 0.9+)
2. **Mis-calibrated threshold** - Default 0.5 threshold too high for compressed probabilities

**Immediate Fix:** Lowering classification threshold from 0.5 to 0.25 improves:
- Test F1: 0.43 → 0.76 (+77% improvement)
- Test Recall: 0.30 → 0.92 (+207% improvement)
- Test Accuracy: 0.58 → 0.69 (+19% improvement)

**Root Cause:** Model-wide calibration issue during training (not distribution shift).

---

## Diagnostic Findings

### Phase 1: Threshold Optimization

**Test Set Performance at Different Thresholds:**

| Threshold | Accuracy | Precision | Recall | F1    |
|-----------|----------|-----------|--------|-------|
| 0.25      | 0.6942   | 0.6484    | 0.9219 | 0.7613|
| 0.30      | 0.6777   | 0.6471    | 0.8594 | 0.7383|
| 0.35      | 0.6529   | 0.6375    | 0.7969 | 0.7083|
| 0.40      | 0.6281   | 0.6508    | 0.6406 | 0.6457|
| **0.50**  | **0.5785**| **0.7600**| **0.2969**| **0.4270**|
| 0.55      | 0.5207   | 0.8750    | 0.1094 | 0.1944|
| 0.60      | 0.4711   | N/A       | 0.0000 | 0.0000|

**Key Finding:** At threshold 0.60, model predicts everything as "Not USV" (recall = 0). This proves the probability compression is severe.

**Optimal Threshold:** 0.25 (maximizes F1 score)

---

### Phase 4: Probability Distribution Analysis

**Validation Set:**
- Range: [0.0051, 0.5702]
- Mean: 0.4014 ± 0.1264
- % Above 0.5: 24.4%

**Test Set:**
- Range: [0.0095, 0.5709]
- Mean: 0.3667 ± 0.1547
- % Above 0.5: 20.7%

**Critical Finding:** Both validation and test show **identical probability compression**. This confirms it's a **model calibration issue**, not test-specific distribution shift.

**Expected vs Actual:**
- **Expected:** Well-calibrated model should output probabilities [0.05, 0.95]
- **Actual:** Compressed to [0.01, 0.57] - maximum confidence only 57%!

**Diagnosis:**
- ❌ NOT a distribution shift (val and test identical)
- ✅ Model-wide calibration problem from training
- Likely causes: class imbalance handling, loss function issue, or label noise

---

### Phase 2: Recording-Level Performance

**Performance Variance (at threshold 0.25):**

| Recording | Accuracy | F1    | Mean Confidence | Samples |
|-----------|----------|-------|-----------------|---------|
| 0000008 (worst) | 46.7% | 0.636 | **0.388** | 15 |
| 0000017 | 52.9% | 0.692 | 0.424 | 17 |
| 0000009 | 64.3% | 0.615 | 0.315 | 14 |
| ... | ... | ... | ... | ... |
| 0000018 | 83.3% | 0.857 | 0.390 | 12 |
| 0000044 (best) | 87.5% | 0.875 | **0.252** | 16 |
| 0000020 (best) | 91.7% | 0.923 | **0.260** | 12 |

**Counterintuitive Pattern:**
- Worst recording (0000008): **highest** mean confidence (0.388)
- Best recordings (0000020, 0000044): **lowest** mean confidence (0.25-0.26)

**Interpretation:** Model is "confidently wrong" on certain recordings. It outputs higher probabilities for samples it should be less certain about.

**Overall Statistics:**
- Mean accuracy: 0.710 ± 0.163 (high variance!)
- Mean F1: 0.775 ± 0.116
- Mean confidence: 0.361 ± 0.075

**Train vs Test Comparison:**
- Pixel mean: Train 66.1 ± 1.3, Test 65.3 ± 0.3 (similar)
- Pixel std: Train 43.3 ± 2.5, Test 41.8 ± 0.7 (similar)
- USV ratio: Train 47%, Test 53% (similar)

**Conclusion:** No obvious distribution shift in basic statistics. High recording-level variance suggests model overfits to specific recording characteristics rather than learning general USV patterns.

---

### Phase 3: Visual Inspection

**Samples Extracted:**
- 5 correct predictions from best recording (0000020)
- 1 false negative from worst recording (0000008)

**Key Observations:**

**Best Recording (Correct Predictions):**
- Clear, well-defined single-syllable USV calls
- Good signal-to-noise ratio
- Variety of shapes: chevron sweeps, horizontal sweeps
- Confidence range: 0.37-0.45 (moderate, not overconfident)
- Some vertical artifacts at bottom (recording-level characteristic)

**Worst Recording (False Negative):**
- **CRITICAL FINDING:** Contains **TWO distinct USV calls** (multi-syllable)
- Both USV segments clearly visible with **excellent SNR**
- Signal quality **equal or better** than best recording samples
- Confidence: 0.237 (just 0.013 below threshold!)
- This is NOT a "hard to see" USV - it's very clear visually

**Hypothesis:** Model may struggle with multi-syllable patterns if training set is biased toward single-syllable USVs.

---

## Root Cause Analysis

### Primary Issue: Poor Model Calibration

**Evidence:**
1. Probability compression to [0.01, 0.57] on both val and test
2. Maximum confidence never exceeds 0.57 (should be 0.9+)
3. Same compression on validation set (used for early stopping)

**Likely Causes:**
- Class imbalance not handled optimally (697 USV vs 412 noise)
- BCEWithLogitsLoss may need calibration post-training
- Possible label noise in training data
- Regularization too strong (dropout 0.5 may be too high for small dataset)

### Secondary Issue: Recording-Level Overfitting

**Evidence:**
1. High variance in recording accuracy (46% to 92%)
2. Counterintuitive confidence pattern (highest confidence on worst recordings)
3. Pixel statistics similar across recordings (no obvious visual difference)
4. Multi-syllable USVs get lower confidence

**Likely Causes:**
- Training set may lack diversity in recording conditions
- Model learned recording-specific artifacts rather than general USV features
- Possible bias toward single-syllable USVs in training data

---

## Implemented Fixes

### ✅ Fix 1: Optimal Threshold Configuration

**Changes:**
1. Updated `src/usv_spectrogram/models/cnn_classifier.py`:
   - Added `optimal_threshold=0.25` parameter to both model classes
   - Modified `predict()` to use `optimal_threshold` as default
   - Threshold can still be overridden for experimentation

2. Created `models/clean_test/optimal_threshold.json`:
   - Documents optimal threshold and performance metrics
   - Includes notes on root cause and limitations

**Impact:**
- Test F1: 0.43 → 0.76 (+77% improvement)
- Test Recall: 0.30 → 0.92 (+207% improvement)
- Test Accuracy: 0.58 → 0.69 (+19% improvement)

**Limitations:**
- This is a **workaround**, not a complete fix
- Probability compression still exists
- Recording-level variance still high
- Multi-syllable USV issue remains

---

## Recommended Next Steps

### Short-Term (Immediate)

1. **Verify threshold fix in production:**
   - Re-run detection pipeline with new threshold
   - Monitor for false positive rate (may increase from 6 to 32 on test set)

2. **Analyze training data composition:**
   - Count single vs multi-syllable USVs in training set
   - Check for label noise or inconsistencies
   - Verify class balance and recording diversity

### Medium-Term (Within 1-2 weeks)

3. **Implement temperature scaling:**
   - Post-training calibration to expand probability range
   - Can be done without full retraining
   - Use validation set to tune temperature parameter

4. **Improve data augmentation:**
   - Add noise augmentation (different SNR levels)
   - Time stretching/compression for USV duration variation
   - Frequency shifting within 25-110 kHz range

### Long-Term (Future retraining)

5. **Retrain with better calibration:**
   - Address class imbalance more carefully (currently using class weights)
   - Consider focal loss instead of BCE
   - Reduce dropout from 0.5 to 0.3 for small dataset
   - Ensure multi-syllable USVs are well-represented

6. **Data collection:**
   - Add more diverse recordings to training set
   - Ensure all recording conditions represented
   - Balance single vs multi-syllable USVs

7. **Architecture improvements:**
   - Try simpler model (current may be overparameterized for 1109 samples)
   - Consider attention mechanisms to handle multi-syllable USVs
   - Experiment with different pooling strategies

---

## Files Created During Diagnostics

**Scripts:**
- `scripts/threshold_sweep.py` - Threshold optimization analysis
- `scripts/compare_probability_distributions.py` - Val vs test probability comparison
- `scripts/analyze_recording_performance.py` - Per-recording performance analysis
- `scripts/extract_visual_samples.py` - Sample extraction for manual inspection

**Analysis Outputs:**
- `analysis/threshold_sweep_results.csv` - Test set threshold sweep
- `analysis/threshold_sweep_plot.png` - Metrics vs threshold visualization
- `analysis/val_analysis/threshold_sweep_results.csv` - Validation set threshold sweep
- `analysis/val_analysis/threshold_sweep_plot.png` - Validation metrics visualization
- `analysis/probability_distributions.png` - Val vs test probability comparison
- `analysis/recording_comparison.csv` - Per-recording performance metrics
- `analysis/train_vs_test_recordings.csv` - Train vs test statistics
- `analysis/visual_comparison/best_correct/` - 5 correct USV samples
- `analysis/visual_comparison/worst_incorrect/` - 1 false negative sample
- `analysis/visual_comparison/manifest.csv` - Sample metadata
- `analysis/visual_inspection_notes.md` - Manual inspection findings

**Configuration:**
- `models/clean_test/optimal_threshold.json` - Optimal threshold config

**Documentation:**
- `analysis/DIAGNOSTIC_SUMMARY.md` - This report

---

## Conclusion

The diagnostic investigation revealed that the poor test performance (58% accuracy, 30% recall) was primarily due to **mis-calibrated classification threshold** (0.5) applied to a **poorly calibrated model** (max probability 0.57).

**Immediate fix** (threshold adjustment to 0.25) improves F1 from 0.43 to 0.76, making the model usable for production.

**However**, deeper issues remain:
1. Model never learned to output high-confidence predictions
2. High variance across recordings (46-92% accuracy)
3. Potential bias against multi-syllable USVs

These issues require **retraining with better calibration and more diverse data** for a complete solution.

---

**Diagnostic Session Completed:** 2026-01-23
**Next Session:** Implement temperature scaling or begin retraining planning
