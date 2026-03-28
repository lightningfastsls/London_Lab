# Per-Recording Score Normalization

## Purpose
Z-normalize CNN probabilities per recording to make fixed thresholds behave adaptively across recordings with different noise floors.

## Location
`src/usv_spectrogram/postprocessing/normalization.py`

## Public Interface

### `normalize_scores_per_recording(probabilities: np.ndarray) -> np.ndarray`
Z-normalize a single recording's CNN scores. Input: 1-D array of per-window probabilities. Output: Z-scores (same shape, can exceed [0,1]).

### `normalize_scores_batch(all_probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]`
Normalize multiple recordings independently. Each recording uses its own noise distribution.

## Pipeline Position
```
CNN inference → probabilities [0,1]
  → normalize_scores_per_recording() → Z-scores
  → hysteresis_detect() → USVEvent list
```

## Algorithm Summary
1. Take bottom 50th percentile of sorted values as noise estimate
2. Compute noise_median (location) from that slice
3. Compute noise_MAD (spread) from full array relative to noise_median
4. Fallback: mean absolute deviation when MAD=0
5. Z-score: `(prob - noise_median) / noise_MAD`

## Key Assumptions
- USVs occupy <5% of total windows in typical recordings, so the 50th percentile cutpoint safely captures the noise distribution without USV contamination.
- Noise distributions are approximately unimodal (one dominant noise floor per recording).

## Algorithm Rationale
- **noise_median from bottom 50%**: Robust to USV contamination. The median of the lower half is roughly Q1 of the full distribution — biased below the true noise center, but this bias is compensated by using the full-array MAD for spread.
- **Full-array MAD (when noise has variation)**: If we used the noise-slice MAD, the centering bias (~1.5× for Gaussian noise) would cause most noise windows to have Z-scores > 1. The full-array MAD is wider, cancelling this bias. Since USVs are rare (<5%), the full-array MAD is still dominated by noise.
- **Mean-AD fallback (when noise is constant)**: Perfectly constant noise makes all median-based spread estimators collapse to 0. Mean absolute deviation is less robust but detects any variation from USV windows mixed into the noise slice or the full array.

## Known Limitations
- When USVs exceed ~50% of windows, the bottom-50% noise slice includes USV windows, degrading the estimate. This is unlikely in practice but can occur in synthetic test data.
- Output dtype is always float64 regardless of input dtype (CNN outputs float32).

## Integration Points
- **Depends on**: CNN inference output (`SlidingInference.infer()` → `InferenceResult.probabilities`)
- **Consumed by**: `hysteresis_detect()` (thresholds operate on Z-scores instead of raw probabilities)
- **Exported via**: `usv_spectrogram.postprocessing`
