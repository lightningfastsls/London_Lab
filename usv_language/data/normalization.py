"""Per-frequency normalization for bout spectrograms.

Computes and applies per-frequency-bin mean/std normalization using
Welford's online algorithm for numerical stability.

Each of the ~170 frequency bins is normalized independently so that
the training set has approximately zero mean and unit variance per bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NormalizationStats:
    """Per-frequency normalization statistics.

    Attributes
    ----------
    mean:
        Mean per frequency bin, shape (n_freq,).
    std:
        Standard deviation per frequency bin, shape (n_freq,).
    count:
        Number of frames (columns) used to compute the statistics.
    """

    mean: np.ndarray
    std: np.ndarray
    count: int

    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be 1D, got shape {self.mean.shape}")
        if self.std.ndim != 1:
            raise ValueError(f"std must be 1D, got shape {self.std.shape}")
        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean shape {self.mean.shape} != std shape {self.std.shape}"
            )
        if self.count < 0:
            raise ValueError(f"count must be non-negative, got {self.count}")


def compute_normalization_stats(
    spectrograms: list[np.ndarray],
    eps: float = 1e-8,
) -> NormalizationStats:
    """Compute per-frequency normalization stats using Welford's algorithm.

    Parameters
    ----------
    spectrograms:
        List of spectrograms, each shape (n_freq, T_i). All must have
        the same n_freq.
    eps:
        Minimum std value to avoid division by zero.

    Returns
    -------
    stats:
        NormalizationStats with mean, std, and count.
    """
    if not spectrograms:
        raise ValueError("Cannot compute stats from empty spectrogram list")

    n_freq = spectrograms[0].shape[0]

    # Welford's online algorithm for numerical stability
    count = 0
    mean = np.zeros(n_freq, dtype=np.float64)
    m2 = np.zeros(n_freq, dtype=np.float64)

    for spec in spectrograms:
        if spec.shape[0] != n_freq:
            raise ValueError(
                f"All spectrograms must have {n_freq} freq bins, "
                f"got {spec.shape[0]}"
            )
        for t in range(spec.shape[1]):
            count += 1
            col = spec[:, t].astype(np.float64)
            delta = col - mean
            mean += delta / count
            delta2 = col - mean
            m2 += delta * delta2

    if count < 2:
        # Can't compute meaningful std with < 2 samples
        std = np.full(n_freq, eps, dtype=np.float32)
    else:
        variance = m2 / (count - 1)
        std = np.sqrt(variance).astype(np.float32)
        std = np.maximum(std, eps)

    return NormalizationStats(
        mean=mean.astype(np.float32),
        std=std,
        count=count,
    )


def normalize(
    spectrogram: np.ndarray,
    stats: NormalizationStats,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply per-frequency normalization to a spectrogram.

    Parameters
    ----------
    spectrogram:
        Input spectrogram, shape (n_freq, T).
    stats:
        Normalization statistics (mean and std per frequency bin).
    eps:
        Guard against zero std (applied on top of stats.std).

    Returns
    -------
    normalized:
        Normalized spectrogram, shape (n_freq, T).
    """
    mean = stats.mean[:, np.newaxis]  # (n_freq, 1)
    std = stats.std[:, np.newaxis]    # (n_freq, 1)
    return ((spectrogram - mean) / (std + eps)).astype(np.float32)


def save_normalization_stats(
    stats: NormalizationStats,
    path: str | Path,
) -> None:
    """Save normalization stats to a .npz file."""
    path = Path(path)
    np.savez(
        str(path),
        mean=stats.mean,
        std=stats.std,
        count=np.array(stats.count),
    )


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    """Load normalization stats from a .npz file."""
    path = Path(path)
    data = np.load(str(path))
    return NormalizationStats(
        mean=data["mean"],
        std=data["std"],
        count=int(data["count"]),
    )
