"""In-memory spectrogram computation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal

from .config import SpectrogramConfig
from ._stft_core import compute_stft_frames_db, extract_frames


def compute_spectrogram_db(
    samples: np.ndarray,
    sample_rate_hz: int,
    cfg: SpectrogramConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a dB-scaled spectrogram for an in-memory signal.

    Returns
    -------
    spec_db:
        Spectrogram in dB (no display gain applied), shape (freqs, frames).
    freqs_hz:
        Frequency bins in Hz.
    times_s:
        Time bins in seconds.
    """
    if cfg.enforce_sample_rate and sample_rate_hz != cfg.expected_sample_rate_hz:
        raise ValueError(
            f"Expected {cfg.expected_sample_rate_hz} Hz, got {sample_rate_hz} Hz."
        )

    hop_length = cfg.hop_length(sample_rate_hz)
    if hop_length >= cfg.window_length:
        raise ValueError("Hop length must be smaller than window length.")

    n_fft = cfg.n_fft()
    window = signal.get_window(cfg.window, cfg.window_length, fftbins=True)
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    band_mask = (freqs_hz >= cfg.f_min_hz) & (freqs_hz <= cfg.f_max_hz)

    if samples.size < cfg.window_length:
        return np.empty((band_mask.sum(), 0)), freqs_hz[band_mask], np.empty(0)

    # Extract frames and compute STFT using shared helper
    frames = extract_frames(samples, cfg.window_length, hop_length)
    spec_db = compute_stft_frames_db(frames, window, n_fft, band_mask, cfg.eps)

    n_frames = frames.shape[0]
    times_s = (
        (np.arange(n_frames) * hop_length) + cfg.window_length / 2.0
    ) / sample_rate_hz
    return spec_db, freqs_hz[band_mask], times_s
