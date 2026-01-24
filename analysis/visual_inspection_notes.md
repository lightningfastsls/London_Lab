# Visual Inspection Notes - Test Set Performance Investigation

**Date:** 2026-01-23
**Investigator:** Claude Code (Diagnostic Session 9)
**Threshold:** 0.25 (optimal for F1)

## Overview

Comparing spectrogram samples from:
- **Best recording:** 2024-09-30_11-20-07_0000020.wav (91.7% accuracy, 92.3% F1)
- **Worst recording:** 2024-09-30_11-19-09_0000008.wav (46.7% accuracy, 63.6% F1)

## Key Observation from Statistics

**Counterintuitive confidence pattern:**
- Best recording (0000020): Mean confidence = 0.260 (LOWEST among test recordings)
- Worst recording (0000008): Mean confidence = 0.388 (HIGHEST among test recordings)

This suggests the model outputs higher probabilities for samples it should be less confident about.

## Sample Analysis

### Best Recording - Correct USV Predictions

**Confidence range:** 0.37-0.45 (moderate confidence, correct predictions)

1. `2024-09-30_11-20-07_0000020_00001790_conf0.45.png` - confidence: 0.448
2. `2024-09-30_11-20-07_0000020_00001045_conf0.43.png` - confidence: 0.432
3. `2024-09-30_11-20-07_0000020_00002019_conf0.40.png` - confidence: 0.402
4. `2024-09-30_11-20-07_0000020_00001574_conf0.37.png` - confidence: 0.371
5. `2024-09-30_11-20-07_0000020_00001696_conf0.37.png` - confidence: 0.370

**Visual characteristics observed:**
- Clear, well-defined USV calls with good signal-to-noise ratio
- Variety of shapes: chevron sweeps, horizontal sweeps
- Single-syllable calls (1 USV per spectrogram)
- Moderate background noise (typical pink/purple noise floor)
- Some samples have vertical artifacts at bottom (recording artifacts)
- Confidence range 0.37-0.45 - moderate, not overconfident

### Worst Recording - False Negative (USV misclassified as Not USV)

**Note:** Only 1 false negative found at threshold 0.25 (recall = 87.5%)

1. `2024-09-30_11-19-09_0000008_00001861_conf0.24.png` - confidence: 0.237
   - This is JUST BELOW the 0.25 threshold (borderline case)
   - The model is very uncertain about this sample

**Visual characteristics observed:**
- **CRITICAL FINDING:** Contains TWO distinct USV calls (multi-syllable)
- Both USV segments are CLEARLY visible with good SNR
- Signal quality appears EQUAL OR BETTER than best recording samples
- Confidence was 0.237 (just 0.013 below threshold!)
- **Hypothesis:** Model may struggle with multi-syllable patterns if trained on single-syllable USVs
- This is NOT a "hard to see" USV - it's very clear visually

## Hypothesis

The worst recording (0000008) has higher mean confidence (0.388) because:
1. It may contain noisier/more ambiguous samples that the model incorrectly gives high confidence to
2. The false positives (Not USV classified as USV) likely have high confidence
3. This suggests the model learned incorrect patterns that apply to this recording's noise

The best recording (0000020) has lower mean confidence (0.260) because:
1. The model is appropriately uncertain about many samples
2. It correctly identifies USVs but without overconfidence
3. More calibrated predictions overall

## Recommended Next Steps

1. **Extract false positives** from worst recording:
   - See what "Not USV" samples are being misclassified with high confidence
   - This will reveal what noise patterns confuse the model

2. **Visual comparison:**
   - Compare USV shapes between best/worst recordings
   - Compare noise characteristics
   - Look for recording artifacts or quality differences

3. **Further investigation:**
   - Check if worst recording has unique noise patterns not seen in training
   - Analyze frequency content of false positives

## Preliminary Conclusions

- Model calibration is poor (max probability ~0.57)
- Threshold adjustment (0.5 → 0.25) improves F1 from 0.43 to 0.76
- High recording-level variance (46% to 92% accuracy) suggests dataset heterogeneity
- Counterintuitive confidence pattern indicates model may be "confidently wrong" on certain sample types
- The low overall confidence ceiling suggests training issues rather than test-specific problems
