"""Feature-extraction utilities shared by SIS-benchmark modules (17.2–17.6).

Currently exports:
  * FilterConfig / prefilter_spectrogram — magnitude-spectrogram cleaning
    (median filter + local noise-floor mask + frequency band mask).
  * RidgeConfig / track_ridge — Viterbi-style DP ridge tracker producing
    FM (pitch) and AM (amplitude) trajectories from a cleaned spectrogram.
"""

from __future__ import annotations

from .ridge_tracker import RidgeConfig, track_ridge
from .spectrogram_filter import FilterConfig, prefilter_spectrogram

__all__ = [
    "FilterConfig",
    "RidgeConfig",
    "prefilter_spectrogram",
    "track_ridge",
]
