"""Per-recording Z-normalization of CNN probabilities.

Noise floors vary across recordings (different cages, equipment, recording
days).  A fixed threshold means different things in quiet vs noisy recordings.
This module estimates the noise distribution from the bottom 50th percentile
of windows — predominantly noise in typical USV recordings where USVs occupy
<5% of total duration — and Z-normalizes each window's probability:

    z = (prob - noise_median) / noise_MAD

Normalized scores can exceed [0, 1]; that is expected.  Hysteresis thresholds
then operate on Z-scores instead of raw probabilities.

Future alternative: PCEN (Per-Channel Energy Normalization) operates at the
spectrogram level and can reduce false alarm rates dramatically, but requires
CNN retraining.  Z-normalization works as a post-hoc fix with the current model.
"""

from __future__ import annotations

import numpy as np

_MAD_EPSILON = 1e-12


def normalize_scores_per_recording(probabilities: np.ndarray) -> np.ndarray:
    """Z-normalize CNN scores using the noise distribution.

    Estimate noise location from the bottom 50th percentile of windows, then
    estimate spread from the full array (less biased when noise slice is a
    truncated distribution).  Express every window as a Z-score relative to
    the noise floor.

    Algorithm:
        1. Sort values, take bottom 50% as the noise slice.
        2. noise_median = median of noise slice (robust location estimate).
        3. Spread estimation (two tiers):
           a. If the noise slice has variation (MAD > 0): use the full-array
              MAD relative to noise_median.  This is wider than the noise-slice
              MAD and avoids the centering bias inherent in half-distribution
              estimation.
           b. If the noise slice is constant (MAD = 0): fall back to the mean
              absolute deviation — first of the noise slice, then of the full
              array.  Mean-AD is less robust but handles constant-noise + USV
              cases where median-based estimators collapse to zero.
        4. If all spread estimates are zero: return all zeros (truly constant
           input has no information to normalize against).

    Args:
        probabilities: 1-D array of per-window CNN output probabilities.
            Values are typically in [0, 1] but non-standard ranges are
            handled gracefully.

    Returns:
        Z-scores with the same shape as *probabilities*.  Values can exceed
        [0, 1] for windows far above the noise floor.  All values are finite.
    """
    if probabilities.size == 0:
        return np.array([], dtype=np.float64)

    # Bottom 50th percentile — these are predominantly noise windows
    n_noise = max(1, len(probabilities) // 2)
    noise_slice = np.sort(probabilities)[:n_noise]

    noise_median = np.median(noise_slice)

    # Two-tier spread estimation
    noise_slice_mad = np.median(np.abs(noise_slice - noise_median))

    if noise_slice_mad > _MAD_EPSILON:
        # Noise slice has variation → use full-array MAD for wider estimate.
        # Full-array MAD >= noise_slice_mad when USVs are present (they add
        # large deviations), so it cannot be zero here.
        full_deviations = np.abs(probabilities - noise_median)
        noise_mad = np.median(full_deviations)
    else:
        # Noise slice is constant → cascading mean-AD fallbacks
        noise_mad = np.mean(np.abs(noise_slice - noise_median))
        if noise_mad < _MAD_EPSILON:
            noise_mad = np.mean(np.abs(probabilities - noise_median))
        if noise_mad < _MAD_EPSILON:
            return np.zeros_like(probabilities, dtype=np.float64)

    return (probabilities.astype(np.float64) - noise_median) / noise_mad


def normalize_scores_batch(
    all_probabilities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Normalize a batch of recordings independently.

    Each recording's probabilities are Z-normalized using its own noise
    distribution — no cross-recording information is shared.

    Args:
        all_probabilities: Mapping of recording stem to per-window
            probability arrays.

    Returns:
        Dict with the same keys, where each value is the Z-normalized
        version of the corresponding input array.
    """
    return {
        stem: normalize_scores_per_recording(probs)
        for stem, probs in all_probabilities.items()
    }
