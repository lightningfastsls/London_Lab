"""Heuristic candidate detection for USV spectrograms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration for threshold-based candidate detection."""

    threshold_db: float = 6.0
    min_area_bins: int = 24
    min_frames: int = 3
    min_bins: int = 3


def detect_candidates(
    spec_db: np.ndarray,
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    cfg: HeuristicConfig,
) -> list[dict[str, float]]:
    """Detect candidate regions using thresholding and connected components."""
    if spec_db.size == 0:
        return []

    noise_floor = float(np.percentile(spec_db, 10.0))
    mask = spec_db >= (noise_floor + cfg.threshold_db)
    labeled, _ = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    candidates: list[dict[str, float]] = []

    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        freq_slice, time_slice = slc
        region_mask = labeled[freq_slice, time_slice] == label_id
        area = int(np.sum(region_mask))
        if area < cfg.min_area_bins:
            continue
        freq_bins = int(freq_slice.stop - freq_slice.start)
        time_bins = int(time_slice.stop - time_slice.start)
        if freq_bins < cfg.min_bins or time_bins < cfg.min_frames:
            continue

        region = spec_db[freq_slice, time_slice]
        peak_db = float(np.max(region))
        t_start_idx = int(time_slice.start)
        t_end_idx = int(min(time_slice.stop - 1, times_s.size - 1))
        f_start_idx = int(freq_slice.start)
        f_end_idx = int(min(freq_slice.stop - 1, freqs_hz.size - 1))

        t_start_s = float(times_s[t_start_idx]) if times_s.size else 0.0
        t_end_s = float(times_s[t_end_idx]) if times_s.size else t_start_s
        f_start_hz = float(freqs_hz[f_start_idx]) if freqs_hz.size else 0.0
        f_end_hz = float(freqs_hz[f_end_idx]) if freqs_hz.size else f_start_hz

        candidates.append(
            {
                "t_start_s": t_start_s,
                "t_end_s": t_end_s,
                "duration_s": max(0.0, t_end_s - t_start_s),
                "f_start_hz": f_start_hz,
                "f_end_hz": f_end_hz,
                "bandwidth_hz": max(0.0, f_end_hz - f_start_hz),
                "area_bins": float(area),
                "peak_db": peak_db,
            }
        )

    candidates.sort(key=lambda c: (c["t_start_s"], c["f_start_hz"]))
    return candidates


def summarize_candidates(candidates: list[dict[str, float]]) -> dict[str, float]:
    """Summarize candidate regions into a compact metrics dictionary."""
    if not candidates:
        return {
            "count": 0.0,
            "total_area_bins": 0.0,
            "mean_duration_s": 0.0,
            "mean_bandwidth_hz": 0.0,
            "max_peak_db": 0.0,
        }

    durations = np.array([c["duration_s"] for c in candidates], dtype=float)
    bandwidths = np.array([c["bandwidth_hz"] for c in candidates], dtype=float)
    areas = np.array([c["area_bins"] for c in candidates], dtype=float)
    peaks = np.array([c["peak_db"] for c in candidates], dtype=float)

    return {
        "count": float(len(candidates)),
        "total_area_bins": float(np.sum(areas)),
        "mean_duration_s": float(np.mean(durations)),
        "mean_bandwidth_hz": float(np.mean(bandwidths)),
        "max_peak_db": float(np.max(peaks)),
    }
