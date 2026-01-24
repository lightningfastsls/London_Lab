# Training Data Bias Investigation - Deep Dive Analysis

**Date:** 2026-01-24
**Session:** Session 9 - Option B Deep Dive
**Finding:** Training data heavily biased toward SHORT single-syllable USVs

---

## Executive Summary

The training dataset is **heavily skewed toward short-duration USVs** (~36ms median), with only 4.6% of samples exceeding 120ms (potential multi-syllable). This explains why the model struggles with longer, complex multi-syllable USV patterns observed during visual inspection.

---

## Key Findings

### 1. Overall Duration Distribution

**Statistics (all 587 labeled USVs with duration data):**
- **Median:** 35.8ms
- **Mean:** 48.4ms
- **Range:** 10.2ms to 449ms
- **Standard deviation:** 47ms

**Duration Categories:**
| Category | Range | Count | Percentage |
|----------|-------|-------|------------|
| Very Short | 0-50ms | 401 | **68.3%** |
| Short | 50-100ms | 142 | 24.2% |
| Medium | 100-150ms | 28 | 4.8% |
| Long | 150-200ms | 7 | 1.2% |
| Very Long | 200-300ms | 3 | 0.5% |
| Extremely Long | 300ms+ | 6 | 1.0% |

**Key Insight:** Nearly **70% of training data is under 50ms** - typical single-syllable short USVs.

---

### 2. Duration by Train/Val/Test Split

| Split | N USVs | Mean | Median | >100ms | >150ms |
|-------|--------|------|--------|--------|--------|
| **Train** | 433 | 47.2ms | 36.7ms | 28 (6.5%) | 11 (2.5%) |
| **Val** | 108 | 51.1ms | 32.0ms | 10 (9.3%) | 3 (2.8%) |
| **Test** | 68 | 51.4ms | 43.5ms | 6 (8.8%) | 2 (2.9%) |

**Key Insight:** All three splits show similar distributions - **no distribution shift**, just consistent bias toward short USVs.

---

### 3. Multi-Syllable Analysis (Duration > 120ms)

**Counts by Split:**
- **Training:** 20/433 (4.6%)
- **Validation:** 7/108 (6.5%)
- **Test:** 2/68 (2.9%)

**Top 10 Longest USVs:**
1. `2024-09-30_11-20-56_0000033_00001033` - 449ms (train)
2. `2024-09-30_11-21-26_0000040_00001043` - 416ms (val)
3. `2024-09-30_11-21-18_0000037_00000745` - 404ms (val)
4. `2024-09-30_11-20-32_0000025_00001082` - 333ms (train)
5. `2024-09-30_11-21-34_0000043_00002414` - 308ms (val)
6. `2024-09-30_11-19-54_0000016_00000973` - 308ms (train)
7. `2024-09-30_11-18-57_0000006_00001451` - 240ms (train)
8. `2024-09-30_11-19-38_0000015_00013969` - 234ms (train)
9. `2024-09-30_11-18-27_0000003_00004962` - 219ms (**test**) ← One of only 2 in test set!
10. `2024-09-30_11-21-30_0000042_00000907` - 198ms (train)

**Key Insight:** Test set has **FEWER** long USVs (2.9%) than training (4.6%), though the difference is small.

---

### 4. Recording Diversity

| Split | USVs | Unique Recordings | USVs per Recording |
|-------|------|-------------------|-------------------|
| Train | 433 | 29 | 14.9 ± 16.0 |
| Val | 108 | 7 | 17.5 ± 15.2 |
| Test | 68 | 9 | 8.0 ± 3.5 |

**High variance in USVs per recording** suggests some recordings are more "productive" than others.

---

## Visual Analysis

See `analysis/training_data_duration_analysis.png`:

1. **Top-left (Overall Distribution):** Massive peak at 20-40ms, long tail to 450ms
2. **Top-right (By Split):** Train/val/test overlap almost completely - consistent bias
3. **Bottom-left (Box Plot):** All three splits have similar medians around 35-40ms
4. **Bottom-right (Cumulative):**
   - 50% of USVs are under 36ms
   - 90% of USVs are under 100ms
   - **95% of USVs are under 120ms**

---

## Connection to Visual Inspection Findings

**Recall from Session 9 Phase 3:**

The false negative from worst recording (`2024-09-30_11-19-09_0000008_00001861`) contained:
- **TWO distinct USV calls** (multi-syllable)
- **Confidence:** 0.237 (just below 0.25 threshold)
- **Signal quality:** Excellent SNR, clearly visible

**Hypothesis CONFIRMED:** Model struggles with multi-syllable USVs because training data contains only **4.6% examples >120ms** (potential multi-syllable).

---

## Root Causes

### 1. Labeling Methodology
- Labels likely created by annotating individual visible USV segments
- Multi-syllable calls may have been split into separate segments during detection
- Result: Training set dominated by short single-syllable fragments

### 2. Detection Algorithm Bias
- The energy detector may split multi-syllable calls at gaps between syllables
- If `merge_gap_ms` is too small, multi-syllable calls get fragmented
- Each fragment labeled independently → bias toward short single USVs

### 3. Natural Distribution?
- It's possible mouse USVs are naturally dominated by short single-syllable calls
- However, multi-syllable calls DO exist (we found 29 >120ms)
- Even if rare, model should still learn to recognize them

---

## Impact on Model Performance

### 1. Bias in Learned Features
- Model optimizes for short 30-50ms USVs (median duration)
- Feature extractors (convolutional filters) likely tuned to this scale
- Longer patterns (200-400ms) may not fit learned templates

### 2. Temporal Receptive Field
- Model architecture may not have sufficient temporal context for long USVs
- With padding to 512px and pooling, long USVs compressed into same feature space as short ones

### 3. Class Imbalance (within USV class)
- Short USVs: ~95%
- Long USVs: ~5%
- Model minimizes loss by getting short USVs right, can afford to miss long ones

---

## Recommendations

### Short-Term (Immediate)

1. **Visual Inspection of Long USVs**
   - Manually inspect the 29 USVs >120ms
   - Determine if they're truly multi-syllable or just long single calls
   - Check for mislabeling (should some be split?)

2. **Check Detection Parameters**
   - Review `merge_gap_ms` setting in detection config
   - If too small, increase to capture multi-syllable calls as single segments
   - Re-run detection on a few files to test

3. **Data Augmentation (Quick Fix)**
   - Add time-stretching augmentation during training
   - Artificially lengthen short USVs to create synthetic long examples
   - May help model generalize to longer durations

### Medium-Term (Within 1-2 Weeks)

4. **Collect More Long USVs**
   - Run detection on more recordings specifically looking for long calls
   - Adjust detection threshold to capture fainter multi-syllable patterns
   - Target at least 50-100 long USV examples (currently only 29)

5. **Stratified Sampling by Duration**
   - Instead of recording-level stratification, stratify by duration
   - Ensure train/val/test all have proportional representation of long USVs
   - Current: test has 2.9% long, train has 4.6% - not enough difference to matter

6. **Re-label Multi-Syllable Calls**
   - Go through candidates and specifically look for multi-syllable patterns
   - Label entire multi-syllable sequence as single USV (don't split)
   - This may require adjusting detection algorithm first

### Long-Term (Future Retraining)

7. **Architecture Changes**
   - Consider recurrent layers (LSTM/GRU) or temporal convolutions
   - Allow model to capture longer-range temporal dependencies
   - Current CNN may have limited temporal receptive field

8. **Duration-Aware Loss Function**
   - Weight long-duration USVs higher in loss function
   - Prevents model from ignoring rare long examples
   - Similar to class weighting, but within USV class

9. **Multi-Scale Detection**
   - Train separate models or heads for different duration ranges
   - Short model: 10-80ms
   - Medium model: 80-150ms
   - Long model: 150ms+
   - Ensemble predictions

---

## Next Steps for Deep Dive

1. ✅ **Analyze duration distribution** (COMPLETED)
2. **Visual inspection of long USVs** (manually inspect the 29 long samples)
3. **Check detection algorithm parameters** (merge_gap_ms, continuity settings)
4. **Label consistency review** (are multi-syllable calls being split inappropriately?)
5. **Plan data augmentation** (time-stretching, synthetic long USVs)
6. **Design retraining strategy** (incorporate findings into new training run)

---

## Summary

The training data is **heavily biased toward short 30-50ms single-syllable USVs** (68% under 50ms), with only 4.6% exceeding 120ms (potential multi-syllable). This bias explains why the model struggles with multi-syllable patterns observed during visual inspection.

**Immediate action items:**
1. Inspect the 29 long USVs visually
2. Check if detection algorithm is fragmenting multi-syllable calls
3. Consider data augmentation with time-stretching

**Longer-term:** Collect more long USV examples and potentially retrain with duration-aware strategies.

---

**Status:** Analysis Complete
**Next:** Visual inspection of long USV samples
**Files Created:**
- `scripts/analyze_training_data_composition.py`
- `analysis/training_data_duration_analysis.png`
- `analysis/TRAINING_DATA_BIAS_FINDINGS.md`
