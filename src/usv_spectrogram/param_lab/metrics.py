"""Metric helpers for the USV Parameter Lab."""

from __future__ import annotations

import numpy as np


def compute_metrics(
    spec_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
) -> dict[str, float]:
    """Compute summary metrics for a spectrogram segment.

    Returns
    -------
    metrics:
        Dictionary of scalar metrics describing the spectrogram segment.
    """
    if spec_db.size == 0:
        return {
            "noise_floor_db": 0.0,
            "median_db": 0.0,
            "p95_db": 0.0,
            "p5_db": 0.0,
            "contrast_db": 0.0,
            "mean_db": 0.0,
            "std_db": 0.0,
            "max_db": 0.0,
            "min_db": 0.0,
            "frames": 0.0,
            "freq_bins": 0.0,
            "duration_s": 0.0,
            "freq_min_hz": 0.0,
            "freq_max_hz": 0.0,
        }

    noise_floor = float(np.percentile(spec_db, 10.0))
    median_db = float(np.median(spec_db))
    p95_db = float(np.percentile(spec_db, 95.0))
    p5_db = float(np.percentile(spec_db, 5.0))
    contrast_db = float(p95_db - p5_db)
    duration_s = 0.0
    if times_s.size > 1:
        duration_s = float(times_s[-1] - times_s[0])

    return {
        "noise_floor_db": noise_floor,
        "median_db": median_db,
        "p95_db": p95_db,
        "p5_db": p5_db,
        "contrast_db": contrast_db,
        "mean_db": float(np.mean(spec_db)),
        "std_db": float(np.std(spec_db)),
        "max_db": float(np.max(spec_db)),
        "min_db": float(np.min(spec_db)),
        "frames": float(spec_db.shape[1]),
        "freq_bins": float(spec_db.shape[0]),
        "duration_s": duration_s,
        "freq_min_hz": float(freqs_hz[0]) if freqs_hz.size else 0.0,
        "freq_max_hz": float(freqs_hz[-1]) if freqs_hz.size else 0.0,
    }
