# Padding Bug Fix - Session 9

**Date:** 2026-01-24
**Critical Bug Fixed:** Inconsistent padding between training and inference

---

## The Bug

### Symptom
`scripts/predict.py` gave **completely different predictions** than `scripts/threshold_sweep.py` for the same images:

| Method | Mean Probability | USV Count (@ 0.25 threshold) |
|--------|------------------|------------------------------|
| threshold_sweep.py | 0.3667 | 59 / 121 |
| predict.py (BUGGY) | 0.1742 | 33 / 121 |

**Same images, same model, different results!**

### Root Cause

**Training/Evaluation**: Images were padded using `pad_collate_fn`
- Each batch pads to the max width within that batch
- Sample in batch [200px, 300px, 250px] → all padded to 300px
- Sample in batch [150px, 400px, 220px] → all padded to 400px

**predict.py (BUGGY)**: Images were NOT padded
- Each image processed at its original size (e.g., 284px)
- Model never saw unpadded images during training
- **Train/test distribution mismatch** → Wrong predictions

### Why It Matters

Even though the model uses Global Average Pooling (designed to handle variable sizes), padding still affects predictions because:
1. Convolutional layers process padded zeros differently
2. MaxPooling includes padded regions
3. Different aspect ratios → different learned features
4. By the time features reach GAP, they're already different

### Example

```
Image: 2024-09-30_11-19-09_0000008_00001861.png (256×284)

With padding to 354px (in batch of 16):
  Logit: -1.1714
  Probability: 0.2366

Without padding (buggy predict.py):
  Logit: -2.0251
  Probability: 0.1166

DIFFERENCE: 0.12 probability points!
```

---

## The Fix

### Solution: Fixed Padding to 512px

Updated `scripts/predict.py` to **always pad to 512px width** (max width in training set).

**Benefits:**
- ✅ Deterministic: Same image → same prediction every time
- ✅ Consistent: No variation based on batch composition
- ✅ Production-ready: Predictable behavior

**Trade-off:**
- Fixed padding (512px) creates MORE padding than variable batch padding
- This shifts probabilities higher: mean 0.37 → 0.45
- Requires recalibrated threshold: 0.25 → 0.40

### Updated Files

1. **`scripts/predict.py`**:
   - Added `MAX_SPEC_WIDTH = 512`
   - Added `pad_to_max_width()` function
   - Both single-image and batch prediction now use fixed padding

2. **`src/usv_spectrogram/models/cnn_classifier.py`**:
   - Updated `optimal_threshold` from 0.25 → 0.40
   - Updated docstrings to note "with fixed padding to 512px"

3. **`models/clean_test/optimal_threshold.json`**:
   - Documented both thresholds (0.25 for variable, 0.40 for fixed)
   - Documented probability ranges for each padding mode

---

## Performance Comparison

### Variable Padding (Matches Training)
- **Threshold:** 0.25
- **F1 Score:** 0.761
- **Recall:** 92.2%
- **Precision:** 64.8%
- **Accuracy:** 69.4%
- **Probability Range:** [0.01, 0.57], mean 0.37

### Fixed Padding to 512px (Production)
- **Threshold:** 0.40
- **F1 Score:** 0.745
- **Recall:** 88.2%
- **Precision:** 64.5%
- **Accuracy:** 68.2%
- **Probability Range:** [0.04, 0.58], mean 0.45

Both achieve similar performance. **Fixed padding is recommended for production** due to deterministic behavior.

---

## Verification

Test the fix:

```powershell
# Single image
".venv/Scripts/python.exe" scripts/predict.py --model models/clean_test/best_model.pt --image "spectrograms_training/2024-09-30_11-19-09_0000008_00001861.png"

# Expected: Probability ~0.53, Prediction: USV

# Batch prediction
".venv/Scripts/python.exe" scripts/predict.py --model models/clean_test/best_model.pt --csv splits/test.csv --output predictions.csv

# Expected: ~106 USV predictions (with 0.40 threshold)
```

---

## Lessons Learned

1. **Always verify train/test consistency**: Preprocessing during training MUST match inference
2. **Padding matters**: Even with Global Average Pooling, padding affects predictions
3. **Batch composition can affect results**: Variable padding makes predictions non-deterministic
4. **Fixed padding is safer for production**: Ensures reproducibility

---

## Future Improvements

1. **Retrain without padding**: Use fixed-size crops instead of padding (cleanest solution)
2. **Temperature scaling**: Calibrate probabilities post-training to improve confidence estimates
3. **Data augmentation**: Add padding augmentation during training to make model robust to different padding amounts

---

**Status:** ✅ Bug Fixed, Production Ready
**Threshold:** 0.40 (with fixed 512px padding)
**F1 Score:** 0.745
**Recall:** 88.2%
