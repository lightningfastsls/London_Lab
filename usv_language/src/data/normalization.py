"""Spectrogram normalization and denormalization utilities.

Supports min-max normalization (maps to [0, 1]) and standard normalization
(zero mean, unit variance). Normalization parameters can be computed globally
across an HDF5 dataset for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np


@dataclass
class NormalizationParams:
    """Parameters needed to normalize and denormalize spectrograms."""

    min_val: float = 0.0
    max_val: float = 1.0
    mean_val: float = 0.0
    std_val: float = 1.0


def normalize_spectrogram(
    spec: np.ndarray,
    method: str,
    params: Optional[NormalizationParams] = None,
) -> Tuple[np.ndarray, NormalizationParams]:
    """Normalize a spectrogram.

    Parameters
    ----------
    spec:
        Input spectrogram, any shape.
    method:
        "min_max" maps to [0, 1], "standard" maps to zero mean / unit variance.
    params:
        Pre-computed parameters. If None, computed from spec.

    Returns
    -------
    (normalized_spec, params) tuple.
    """
    if params is None:
        params = NormalizationParams(
            min_val=float(np.min(spec)),
            max_val=float(np.max(spec)),
            mean_val=float(np.mean(spec)),
            std_val=float(np.std(spec)),
        )

    if method == "min_max":
        denom = params.max_val - params.min_val
        if denom == 0:
            return np.zeros_like(spec, dtype=np.float32), params
        normalized = (spec - params.min_val) / denom
    elif method == "standard":
        if params.std_val == 0:
            return np.zeros_like(spec, dtype=np.float32), params
        normalized = (spec - params.mean_val) / params.std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32), params


def denormalize(
    spec_norm: np.ndarray,
    params: NormalizationParams,
    method: str,
) -> np.ndarray:
    """Reverse normalization.

    Parameters
    ----------
    spec_norm:
        Normalized spectrogram.
    params:
        Parameters used during normalization.
    method:
        "min_max" or "standard".

    Returns
    -------
    Denormalized spectrogram.
    """
    if method == "min_max":
        return (spec_norm * (params.max_val - params.min_val) + params.min_val).astype(np.float32)
    elif method == "standard":
        return (spec_norm * params.std_val + params.mean_val).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_global_norm_params(hdf5_path: str) -> NormalizationParams:
    """Compute normalization parameters across all spectrograms in an HDF5 file.

    Uses Welford's online algorithm for numerically stable mean/variance
    computation without loading all data into memory.

    Parameters
    ----------
    hdf5_path:
        Path to HDF5 file with /<stem>/spectrogram datasets.

    Returns
    -------
    Global NormalizationParams.
    """
    global_min = np.inf
    global_max = -np.inf
    # Welford's online algorithm
    count = 0
    mean = 0.0
    m2 = 0.0

    with h5py.File(hdf5_path, "r") as f:
        for stem in f:
            if "spectrogram" not in f[stem]:
                continue
            spec = f[stem]["spectrogram"][:]
            global_min = min(global_min, float(np.min(spec)))
            global_max = max(global_max, float(np.max(spec)))

            # Online mean/variance update
            for val in spec.ravel():
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                m2 += delta * delta2

    if count == 0:
        return NormalizationParams()

    variance = m2 / count if count > 0 else 0.0
    return NormalizationParams(
        min_val=global_min,
        max_val=global_max,
        mean_val=mean,
        std_val=float(np.sqrt(variance)),
    )
