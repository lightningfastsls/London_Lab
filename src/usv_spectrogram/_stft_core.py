"""Shared STFT computation utilities."""

from __future__ import annotations

import numpy as np


def compute_stft_frames_db(
    frames: np.ndarray,
    window: np.ndarray,
    n_fft: int,
    band_mask: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Compute dB spectrogram from windowed frames.

    Parameters
    ----------
    frames:
        Input frames, shape (n_frames, window_length).
    window:
        Window function array, shape (window_length,).
    n_fft:
        FFT length (with zero-padding).
    band_mask:
        Boolean mask for frequency bins to keep.
    eps:
        Small constant to avoid log(0).

    Returns
    -------
    spec_db:
        Spectrogram in dB, shape (n_freq_bins, n_frames).
    """
    windowed = frames * window
    stft = np.fft.rfft(windowed, n=n_fft, axis=1)
    magnitude = np.abs(stft)
    spec_db = 20.0 * np.log10(magnitude + eps)
    return spec_db[:, band_mask].T


def extract_frames(
    samples: np.ndarray,
    window_length: int,
    hop_length: int,
) -> np.ndarray:
    """Extract overlapping frames from a 1D signal.

    Parameters
    ----------
    samples:
        Input signal, shape (n_samples,).
    window_length:
        Length of each frame.
    hop_length:
        Hop size between frames.

    Returns
    -------
    frames:
        Extracted frames, shape (n_frames, window_length).
    """
    n_frames = 1 + (len(samples) - window_length) // hop_length
    if n_frames <= 0:
        return np.empty((0, window_length), dtype=samples.dtype)

    # Use stride tricks for efficient frame extraction
    frame_starts = np.arange(n_frames) * hop_length
    frames = np.stack(
        [samples[start:start + window_length] for start in frame_starts],
        axis=0,
    )
    return frames
