"""Stateless pre-filtering for USV magnitude spectrograms.

Shared DSP infrastructure for SIS-benchmark modules 17.3 (ridge tracker),
17.5 (Oren 80-D vectorisation), and 17.6 (AMVOC autoencoder).

Spec: ROADMAP_SIS_BENCHMARK.md §17.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class FilterConfig:
    """Pre-filtering parameters for USV spectrograms.

    ``sample_rate`` is carried in the config as a convenience for downstream
    consumers (17.3, 17.5, 17.6) that need to reconstruct frequency axes — it
    is not used directly by :func:`prefilter_spectrogram`, which receives
    pre-computed frequencies via its ``freqs_hz`` argument.
    """

    sample_rate: int = 300_000
    noise_floor_multiplier: float = 3.0
    noise_floor_window_cols: int = 20
    median_filter_size: int = 3
    freq_min_hz: float = 25_000.0
    freq_max_hz: float = 120_000.0

    def __post_init__(self) -> None:
        if self.noise_floor_multiplier <= 1.0:
            raise ValueError("noise_floor_multiplier must be > 1")
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError("freq_min_hz must be < freq_max_hz")
        if self.median_filter_size % 2 != 1:
            raise ValueError("median_filter_size must be odd")


def prefilter_spectrogram(
    magnitude: np.ndarray,
    freqs_hz: np.ndarray,
    cfg: FilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean a magnitude spectrogram and return ``(cleaned, mask)``.

    Pipeline (see ROADMAP §17.2):
      1. 2-D median filter removes isolated pixel outliers.
      2. Per-column local noise floor = rolling median of per-column medians
         across ``noise_floor_window_cols`` (mode='reflect' handles edges and
         the ``n_time_cols < window`` case).
      3. Amplitude mask ``filtered > noise_floor_multiplier * floor``.
      4. Frequency band mask keeps only bins in ``[freq_min_hz, freq_max_hz]``.
      5. Cleaned = filtered * mask.  Boolean mask returned for downstream
         consumers (ridge tracker, autoencoder).

    Parameters
    ----------
    magnitude:
        Linear-magnitude spectrogram (``np.abs(stft(...))``), shape
        ``(n_freq_bins, n_time_cols)``. Must be non-negative; dB-scaled
        inputs are not supported (the noise-floor threshold assumes linear
        amplitude).
    freqs_hz:
        Frequency of each row, shape ``(n_freq_bins,)``. Must be 1-D with
        length matching ``magnitude.shape[0]``.
    cfg:
        :class:`FilterConfig` with validated parameters.

    Returns
    -------
    cleaned:
        Same shape and dtype as ``magnitude``; masked pixels are zeroed.
    mask:
        Boolean array, same shape as ``magnitude``; True where the pixel
        survived both the amplitude and frequency-band criteria.
    """
    if freqs_hz.ndim != 1 or freqs_hz.shape[0] != magnitude.shape[0]:
        raise ValueError(
            f"freqs_hz shape {freqs_hz.shape} does not match magnitude "
            f"frequency axis ({magnitude.shape[0]},). "
            f"Expected shape ({magnitude.shape[0]},)."
        )

    filtered = median_filter(magnitude, size=cfg.median_filter_size, mode="reflect")

    # Per-column scalar noise floor: median over ALL frequency bins (including
    # out-of-band).  A USV ridge occupies only 3–5 of ~129 bins, so it cannot
    # shift the median off the background level — this makes the noise floor
    # robust to strong in-band signal.
    col_median = np.median(filtered, axis=0)
    noise_floor = median_filter(
        col_median, size=cfg.noise_floor_window_cols, mode="reflect"
    )

    amplitude_mask = filtered > (cfg.noise_floor_multiplier * noise_floor[np.newaxis, :])

    freq_mask = (freqs_hz >= cfg.freq_min_hz) & (freqs_hz <= cfg.freq_max_hz)
    mask = amplitude_mask & freq_mask[:, np.newaxis]

    cleaned = filtered * mask.astype(filtered.dtype)
    return cleaned, mask
