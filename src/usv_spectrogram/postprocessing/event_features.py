"""Event-level feature extraction for second-stage USV classification.

Extracts discriminative features from each :class:`USVEvent` for downstream
false-positive filtering.  Features fall into two groups:

* **Probability-based** — derived from the CNN output probability curve
  stored on each event (peak, mean, std, kurtosis, roughness, duration).
* **Spectral** — derived from the spectrogram columns corresponding to
  the event (tonality, peak frequency, frequency range, modulation rate, SNR).

Window-to-column mapping extracts one column per event window at hop-spaced
positions: ``cols[i] = (event.start_window + i) * hop_px``.  This samples
across the full event duration while maintaining 1:1 correspondence with
probability values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hysteresis import USVEvent


@dataclass(frozen=True)
class EventFeatures:
    """Discriminative features for second-stage classification."""

    # Probability-based
    peak_probability: float
    mean_probability: float
    prob_std: float
    prob_kurtosis: float
    prob_roughness: float
    duration_windows: int

    # Spectral (require spectrogram access)
    tonality: float
    mean_peak_freq_bin: float
    freq_range_bins: float
    freq_modulation_rate: float
    snr_db: float


def extract_event_features(
    event: USVEvent,
    spectrogram: np.ndarray,
    hop_px: int = 10,
) -> EventFeatures:
    """Extract features from a single event.

    Parameters
    ----------
    event:
        A detected USV event from hysteresis post-processing.
    spectrogram:
        Full recording spectrogram in dB, shape ``(n_freq, n_time)``.
    hop_px:
        SlidingInference stride — maps window indices to spectrogram columns.

    Returns
    -------
    EventFeatures
        Frozen dataclass with 11 discriminative features.

    Raises
    ------
    ValueError
        If the event's column range falls outside the spectrogram bounds.
    """
    # ------------------------------------------------------------------
    # Step 1: Hop-spaced column mapping and bounds check
    # ------------------------------------------------------------------
    col_indices = np.arange(event.window_count) * hop_px + event.start_window * hop_px

    n_freq, n_time = spectrogram.shape
    if col_indices[-1] >= n_time or col_indices[0] >= n_time:
        raise ValueError(
            f"Event columns [{col_indices[0]}, {col_indices[-1]}] exceed "
            f"spectrogram width {n_time}"
        )

    spec_region = spectrogram[:, col_indices]  # (n_freq, window_count)

    # ------------------------------------------------------------------
    # Step 2: Probability features
    # ------------------------------------------------------------------
    probs = event.probabilities
    peak_probability = float(np.max(probs))
    mean_probability = float(np.mean(probs))
    prob_std = float(np.std(probs, ddof=0))
    prob_kurtosis = _excess_kurtosis(probs)
    prob_roughness = _roughness(probs)

    # ------------------------------------------------------------------
    # Step 3: Spectral features
    # ------------------------------------------------------------------
    tonality = _compute_tonality(spec_region)
    peak_bins = np.argmax(spec_region, axis=0)  # per column
    mean_peak_freq_bin = float(np.mean(peak_bins))
    freq_range_bins = float(np.max(peak_bins) - np.min(peak_bins))
    freq_modulation_rate = _freq_modulation_rate(peak_bins)
    snr_db = _compute_snr_db(spec_region)

    return EventFeatures(
        peak_probability=peak_probability,
        mean_probability=mean_probability,
        prob_std=prob_std,
        prob_kurtosis=prob_kurtosis,
        prob_roughness=prob_roughness,
        duration_windows=event.window_count,
        tonality=tonality,
        mean_peak_freq_bin=mean_peak_freq_bin,
        freq_range_bins=freq_range_bins,
        freq_modulation_rate=freq_modulation_rate,
        snr_db=snr_db,
    )


# ======================================================================
# Internal helpers
# ======================================================================


def _excess_kurtosis(values: np.ndarray) -> float:
    """Compute excess kurtosis (0 for Gaussian).

    Returns 0.0 when standard deviation is zero (constant values).
    """
    n = len(values)
    if n < 4:
        return 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=0)
    if std == 0.0:
        return 0.0
    m4 = np.mean((values - mean) ** 4)
    return float(m4 / std**4 - 3.0)


def _roughness(probs: np.ndarray) -> float:
    """Mean absolute second derivative of the probability curve.

    Higher values indicate a jagged (rough) probability curve — typical of
    noise.  Real USVs produce smooth curves with low roughness.
    Returns 0.0 when fewer than 3 values (second derivative undefined).
    """
    if len(probs) < 3:
        return 0.0
    second_diff = np.diff(probs, n=2)
    return float(np.mean(np.abs(second_diff)))


def _compute_tonality(spec_region: np.ndarray) -> float:
    """Compute tonality as ``1 - SFM`` (spectral flatness measure).

    SFM = geometric_mean(power) / arithmetic_mean(power) per column,
    averaged across the event.  Tonal signals → high tonality (near 1),
    broadband noise → low tonality (near 0).

    The spectrogram is in dB; we convert to linear power via
    ``10^(dB / 10)`` before computing the ratio.
    """
    eps = 1e-10
    # Convert dB to linear power
    power = np.power(10.0, spec_region / 10.0)
    power = np.maximum(power, eps)

    n_cols = spec_region.shape[1]
    sfm_sum = 0.0
    for col_idx in range(n_cols):
        col_power = power[:, col_idx]
        # GM via log domain for numerical stability
        log_mean = np.mean(np.log(col_power))
        gm = np.exp(log_mean)
        am = np.mean(col_power)
        if am < eps:
            sfm_sum += 1.0  # flat (silence) → SFM = 1
        else:
            sfm_sum += gm / am

    sfm = sfm_sum / n_cols
    tonality = float(np.clip(1.0 - sfm, 0.0, 1.0))
    return tonality


def _freq_modulation_rate(peak_bins: np.ndarray) -> float:
    """Mean absolute difference in peak frequency bin between adjacent columns.

    Higher values indicate rapid frequency modulation (FM sweeps).
    Tonal signals with stable frequency produce low values.
    Returns 0.0 for single-column events.
    """
    if len(peak_bins) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(peak_bins.astype(np.float64)))))


def _compute_snr_db(spec_region: np.ndarray) -> float:
    """Compute mean SNR in dB across event columns.

    Per column: ``SNR = peak_dB - noise_floor_dB`` where noise floor is
    the 10th percentile.  Averaged across all columns in the event.
    """
    n_cols = spec_region.shape[1]
    snr_sum = 0.0
    for col_idx in range(n_cols):
        col = spec_region[:, col_idx]
        peak_db = float(np.max(col))
        noise_floor_db = float(np.percentile(col, 10))
        snr_sum += peak_db - noise_floor_db

    return float(snr_sum / n_cols)
