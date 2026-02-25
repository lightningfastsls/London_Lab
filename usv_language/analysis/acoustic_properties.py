"""Acoustic property extractors for probing targets.

Extract ground-truth acoustic properties from spectrogram columns
(dB-scaled, shape ``(n_freq, T)``).  These serve as labels for linear
probing classifiers that test whether transformer hidden states encode
acoustic features.

Pure NumPy — no torch / scipy / matplotlib dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcousticPropertyConfig:
    """Parameters for acoustic property extraction.

    Parameters
    ----------
    freq_min_hz:
        Lower frequency bound of the spectrogram slice (Hz).
    freq_max_hz:
        Upper frequency bound of the spectrogram slice (Hz).
    n_freq_bins:
        Number of frequency bins in a spectrogram column.
    sample_rate:
        Audio sample rate (Hz).  ADR-001: always 300 kHz.
    hop_length:
        STFT hop length in samples.  ADR-002: 128.
    voiced_threshold_db:
        A column whose max dB exceeds this is considered voiced.
    direction_threshold:
        Minimum peak-frequency delta (Hz) to count as rising/falling.
    """

    freq_min_hz: float = 20_000.0
    freq_max_hz: float = 120_000.0
    n_freq_bins: int = 170
    sample_rate: int = 300_000
    hop_length: int = 128
    voiced_threshold_db: float = -50.0
    direction_threshold: float = 500.0

    def __post_init__(self) -> None:
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError(
                f"freq_min_hz ({self.freq_min_hz}) must be < "
                f"freq_max_hz ({self.freq_max_hz})"
            )
        if self.n_freq_bins <= 0:
            raise ValueError(
                f"n_freq_bins must be positive, got {self.n_freq_bins}"
            )
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )
        if self.hop_length <= 0:
            raise ValueError(
                f"hop_length must be positive, got {self.hop_length}"
            )
        if self.direction_threshold < 0:
            raise ValueError(
                f"direction_threshold must be >= 0, got "
                f"{self.direction_threshold}"
            )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _freq_array(config: AcousticPropertyConfig) -> np.ndarray:
    """Linearly-spaced frequency axis matching the spectrogram bins.

    Consistent with ``codebook_viz.py`` (line 150) which uses
    ``np.linspace(freq_min, freq_max, n_freq)``.
    """
    return np.linspace(
        config.freq_min_hz, config.freq_max_hz, config.n_freq_bins,
    )


# ---------------------------------------------------------------------------
# Single-column extractors
# ---------------------------------------------------------------------------

def peak_frequency(col: np.ndarray, config: AcousticPropertyConfig) -> float:
    """Frequency of the bin with maximum amplitude.

    Parameters
    ----------
    col:
        1-D array of dB values, shape ``(n_freq_bins,)``.
    config:
        Acoustic property configuration.

    Returns
    -------
    Peak frequency in Hz.
    """
    freqs = _freq_array(config)
    return float(freqs[np.argmax(col)])


def spectral_centroid(
    col: np.ndarray, config: AcousticPropertyConfig,
) -> float:
    """Energy-weighted mean frequency (spectral centroid).

    Converts dB to linear power (``10^(dB/10)``) before weighting.
    An epsilon guard prevents division by zero on silence columns.

    Parameters
    ----------
    col:
        1-D dB column, shape ``(n_freq_bins,)``.
    config:
        Acoustic property configuration.

    Returns
    -------
    Spectral centroid in Hz.
    """
    freqs = _freq_array(config)
    power = np.power(10.0, col / 10.0)
    total = power.sum()
    eps = 1e-30
    if total < eps:
        # Silence — fall back to center frequency
        return float((config.freq_min_hz + config.freq_max_hz) / 2.0)
    return float(np.dot(freqs, power) / total)


def energy(col: np.ndarray) -> float:
    """Total linear power of a spectrogram column.

    Parameters
    ----------
    col:
        1-D dB column, shape ``(n_freq_bins,)``.

    Returns
    -------
    Sum of ``10^(dB/10)`` across bins.
    """
    return float(np.sum(np.power(10.0, col / 10.0)))


def is_voiced(col: np.ndarray, config: AcousticPropertyConfig) -> bool:
    """Whether a column exceeds the voiced-activity threshold.

    Parameters
    ----------
    col:
        1-D dB column.
    config:
        Acoustic property configuration.

    Returns
    -------
    True if the column's max dB > ``config.voiced_threshold_db``.
    """
    return bool(np.max(col) > config.voiced_threshold_db)


def frequency_direction(
    prev: np.ndarray,
    curr: np.ndarray,
    config: AcousticPropertyConfig,
) -> str:
    """Direction of peak-frequency change between two consecutive frames.

    Parameters
    ----------
    prev:
        Previous frame's dB column.
    curr:
        Current frame's dB column.
    config:
        Acoustic property configuration.

    Returns
    -------
    ``'rising'``, ``'falling'``, or ``'flat'``.
    """
    delta = peak_frequency(curr, config) - peak_frequency(prev, config)
    if delta > config.direction_threshold:
        return "rising"
    if delta < -config.direction_threshold:
        return "falling"
    return "flat"


def bout_position(idx: int, bout_length: int) -> float:
    """Normalized position within a bout.

    Parameters
    ----------
    idx:
        0-based frame index within the bout.
    bout_length:
        Total number of frames in the bout.

    Returns
    -------
    ``idx / bout_length`` in range ``[0, (T-1)/T]``, or 0.0 for
    single-frame bouts.  Note: never reaches 1.0 exactly.
    """
    if bout_length <= 1:
        return 0.0
    return float(idx / bout_length)


def time_since_last_usv(
    idx: int,
    onsets: np.ndarray,
    config: AcousticPropertyConfig,
) -> float:
    """Time in milliseconds since the most recent USV onset before *idx*.

    Uses binary search (``np.searchsorted``) for efficiency.

    Parameters
    ----------
    idx:
        Current frame index.
    onsets:
        Sorted array of USV onset frame indices.
    config:
        Acoustic property configuration (for hop→ms conversion).

    Returns
    -------
    Time in milliseconds, or ``-1.0`` if no onset precedes *idx*.
    """
    if onsets.size == 0:
        return -1.0
    # Ensure sorted — searchsorted requires monotonic input
    if onsets.size > 1 and not np.all(np.diff(onsets) >= 0):
        onsets = np.sort(onsets)
    # searchsorted returns the index where idx would be inserted
    pos = int(np.searchsorted(onsets, idx, side="right"))
    if pos == 0:
        return -1.0
    frame_delta = idx - int(onsets[pos - 1])
    ms = frame_delta * config.hop_length / config.sample_rate * 1000.0
    return float(ms)


# ---------------------------------------------------------------------------
# Batch extractor
# ---------------------------------------------------------------------------

def extract_all_properties(
    spec: np.ndarray,
    config: AcousticPropertyConfig,
    onsets: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Extract all acoustic properties for every frame in a spectrogram.

    Parameters
    ----------
    spec:
        dB-scaled spectrogram, shape ``(n_freq_bins, T)``.
    config:
        Acoustic property configuration.
    onsets:
        Sorted array of USV onset frame indices.  Defaults to empty.

    Returns
    -------
    Dictionary mapping property names to arrays of length *T*:
        - ``peak_frequency``: Hz per frame
        - ``spectral_centroid``: Hz per frame
        - ``energy``: linear power per frame
        - ``is_voiced``: bool per frame
        - ``frequency_direction``: ``'rising'``/``'falling'``/``'flat'``
        - ``bout_position``: normalized [0, (T-1)/T] per frame
        - ``time_since_last_usv``: ms per frame (``-1.0`` if none)
    """
    if onsets is None:
        onsets = np.array([], dtype=np.int64)
    # Ensure sorted — the O(T+N) scan requires monotonic onsets
    if onsets.size > 1 and not np.all(np.diff(onsets) >= 0):
        onsets = np.sort(onsets)

    if spec.ndim != 2:
        raise ValueError(
            f"spec must be 2-D (n_freq_bins, T), got shape {spec.shape}"
        )
    T = spec.shape[1]
    if T == 0:
        return {
            "peak_frequency": np.array([], dtype=np.float64),
            "spectral_centroid": np.array([], dtype=np.float64),
            "energy": np.array([], dtype=np.float64),
            "is_voiced": np.array([], dtype=bool),
            "frequency_direction": np.array([], dtype=object),
            "bout_position": np.array([], dtype=np.float64),
            "time_since_last_usv": np.array([], dtype=np.float64),
        }

    freqs = _freq_array(config)

    # --- Vectorized: peak_frequency ---
    peak_indices = np.argmax(spec, axis=0)  # (T,)
    peak_freqs = freqs[peak_indices]

    # --- Vectorized: spectral_centroid ---
    power = np.power(10.0, spec / 10.0)  # (n_freq, T)
    total_power = power.sum(axis=0)  # (T,)
    eps = 1e-30
    safe_total = np.where(total_power < eps, 1.0, total_power)
    centroids = freqs @ power / safe_total  # (T,)
    center_freq = (config.freq_min_hz + config.freq_max_hz) / 2.0
    centroids = np.where(total_power < eps, center_freq, centroids)

    # --- Vectorized: energy ---
    energies = total_power  # already computed

    # --- Vectorized: is_voiced ---
    voiced = np.max(spec, axis=0) > config.voiced_threshold_db  # (T,)

    # --- Vectorized: frequency_direction ---
    deltas = np.diff(peak_freqs, prepend=peak_freqs[0])  # (T,), first=0
    direction = np.where(
        deltas > config.direction_threshold,
        "rising",
        np.where(deltas < -config.direction_threshold, "falling", "flat"),
    )

    # --- Vectorized: bout_position ---
    positions = np.arange(T, dtype=np.float64) / T if T > 1 else np.zeros(T)

    # --- O(T+N) scan: time_since_last_usv ---
    tslu = np.full(T, -1.0, dtype=np.float64)
    hop_ms = config.hop_length / config.sample_rate * 1000.0
    onset_idx = 0
    n_onsets = len(onsets)
    for t in range(T):
        # Advance onset pointer
        while onset_idx < n_onsets and onsets[onset_idx] <= t:
            onset_idx += 1
        if onset_idx > 0:
            tslu[t] = (t - int(onsets[onset_idx - 1])) * hop_ms

    return {
        "peak_frequency": peak_freqs,
        "spectral_centroid": centroids,
        "energy": energies,
        "is_voiced": voiced,
        "frequency_direction": direction,
        "bout_position": positions,
        "time_since_last_usv": tslu,
    }
