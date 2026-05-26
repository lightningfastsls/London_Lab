"""Hysteresis-based post-processing for USV detection.

Standalone batch-processing module that converts CNN per-window probabilities
into discrete USV events using dual-threshold (onset/sustain) hysteresis with
bidirectional extension from seed windows.

Unlike the interactive app's HysteresisDetector (app/core/detection_logic.py),
this module:
- Extends bidirectionally from seeds (backward AND forward)
- Works on abstract window indices, not column indices
- Returns USVEvent dataclass, not DetectedUSV
- Designed for batch pipeline use
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class HysteresisConfig:
    """Configuration for hysteresis detection.

    Attributes:
        onset_threshold: Probability threshold to seed a new event (high gate).
        sustain_threshold: Probability threshold to extend an event (low gate).
        gap_fill_windows: Merge events separated by <= this many windows.
        min_duration_windows: Drop events shorter than this many windows.
        max_duration_ms: Drop events longer than this center-to-center duration.
            Set to None to disable the long-event gate.
    """

    onset_threshold: float = 0.75
    sustain_threshold: float = 0.40
    gap_fill_windows: int = 3
    min_duration_windows: int = 5
    max_duration_ms: float | None = 600.0

    def __post_init__(self) -> None:
        if not (0 < self.sustain_threshold <= self.onset_threshold <= 1.0):
            raise ValueError(
                f"Need 0 < sustain ({self.sustain_threshold}) "
                f"<= onset ({self.onset_threshold}) <= 1.0"
            )
        if self.gap_fill_windows < 0:
            raise ValueError(f"gap_fill_windows must be >= 0, got {self.gap_fill_windows}")
        if self.min_duration_windows < 1:
            raise ValueError(f"min_duration_windows must be >= 1, got {self.min_duration_windows}")
        if self.max_duration_ms is not None and self.max_duration_ms <= 0:
            raise ValueError(f"max_duration_ms must be > 0 or None, got {self.max_duration_ms}")


@dataclass(frozen=True)
class USVEvent:
    """A detected USV event from hysteresis post-processing.

    Window indices are inclusive on both ends.
    ``duration_ms`` measures center-to-center span: for a single-window event
    it is 0.0.  To get physical duration, add one window step.
    """

    start_window: int
    end_window: int
    start_time_s: float
    end_time_s: float
    duration_ms: float  # center-to-center; 0 for single-window events
    peak_probability: float
    mean_probability: float
    window_count: int
    probabilities: np.ndarray


def hysteresis_detect(
    probabilities: np.ndarray,
    times: np.ndarray,
    config: HysteresisConfig | None = None,
) -> List[USVEvent]:
    """Detect USV events using hysteresis thresholding.

    Algorithm:
        1. Seed: find windows where prob >= onset_threshold
        2. Extend: from each seed, grow bidirectionally while prob >= sustain_threshold
        3. Extract: find contiguous marked regions
        4. Gap-fill: merge events separated by <= gap_fill_windows
        5. Duration filters: drop events that are too short or too long
        6. Build USVEvent for each surviving region

    Args:
        probabilities: 1-D array of per-window probabilities in [0, 1].
        times: 1-D array of center times (seconds) aligned with probabilities.
        config: Detection parameters. Uses defaults if None.

    Returns:
        List of USVEvent ordered by start_window.
    """
    if config is None:
        config = HysteresisConfig()

    n = len(probabilities)
    if n == 0:
        return []

    if len(probabilities) != len(times):
        raise ValueError(
            f"probabilities length ({len(probabilities)}) != times length ({len(times)})"
        )

    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "probabilities must be finite values in [0, 1] — pass sigmoid-transformed values, not raw logits"
        )

    # Step 1 — Seed: mark windows above onset threshold
    in_event = np.zeros(n, dtype=bool)
    seeds = np.where(probabilities >= config.onset_threshold)[0]
    if len(seeds) == 0:
        return []

    # Step 2 — Extend bidirectionally from each seed while >= sustain
    for seed in seeds:
        if in_event[seed]:
            continue
        in_event[seed] = True
        # Forward
        j = seed + 1
        while j < n and probabilities[j] >= config.sustain_threshold:
            in_event[j] = True
            j += 1
        # Backward
        j = seed - 1
        while j >= 0 and probabilities[j] >= config.sustain_threshold:
            in_event[j] = True
            j -= 1

    # Step 3 — Extract contiguous regions
    regions = _extract_regions(in_event)
    if not regions:
        return []

    # Step 4 — Gap-fill: merge regions separated by <= gap_fill_windows
    regions = _gap_fill(regions, config.gap_fill_windows)

    # Step 5 — Duration filters
    regions = [(s, e) for s, e in regions if (e - s + 1) >= config.min_duration_windows]
    if config.max_duration_ms is not None:
        regions = [
            (s, e)
            for s, e in regions
            if (float(times[e]) - float(times[s])) * 1000.0 <= config.max_duration_ms
        ]

    # Step 6 — Build USVEvent objects
    events: List[USVEvent] = []
    for start, end in regions:
        probs = probabilities[start : end + 1].copy()
        events.append(
            USVEvent(
                start_window=start,
                end_window=end,
                start_time_s=float(times[start]),
                end_time_s=float(times[end]),
                duration_ms=(float(times[end]) - float(times[start])) * 1000.0,
                peak_probability=float(np.max(probs)),
                mean_probability=float(np.mean(probs)),
                window_count=end - start + 1,
                probabilities=probs,
            )
        )

    return events


def convert_to_detection_format(
    events: List[USVEvent],
    column_indices: np.ndarray,
    boundary_padding_cols: int = 0,
    max_col: int | None = None,
) -> List[Dict]:
    """Convert USVEvents to ADR-010 / LabelStorage compatible dicts.

    Column indices from SlidingInference are window *centers*, so the reported
    start_col/end_col underestimate the true detection span by up to half a
    CNN window width on each side.  ``boundary_padding_cols`` compensates by
    expanding each event's reported region, then merging any that overlap as a
    result.

    Args:
        events: List of USVEvent from hysteresis_detect.
        column_indices: Array mapping window index -> spectrogram column index.
        boundary_padding_cols: Columns to add before start_col and after end_col.
            Default 0 preserves legacy behaviour.  Recommended: 25 (~10 ms at
            hop=128, sr=300 kHz) to capture USV onset/offset that falls outside
            the reported window-center span.
        max_col: Maximum valid column index (spectrogram width - 1).  Used to
            clamp padded end_col.  If None, no upper clamp is applied.

    Returns:
        List of dicts compatible with LabelStorage JSON format.
    """
    if boundary_padding_cols < 0:
        raise ValueError(f"boundary_padding_cols must be >= 0, got {boundary_padding_cols}")

    # Phase 1: build raw (padded) dicts
    raw: List[Dict] = []
    for event in events:
        if event.end_window >= len(column_indices):
            raise IndexError(
                f"event.end_window ({event.end_window}) >= "
                f"column_indices length ({len(column_indices)})"
            )
        start_col = int(column_indices[event.start_window]) - boundary_padding_cols
        end_col = int(column_indices[event.end_window]) + boundary_padding_cols

        # Clamp to valid range
        start_col = max(0, start_col)
        if max_col is not None:
            end_col = min(end_col, max_col)

        raw.append(
            {
                "start_time_s": event.start_time_s,
                "end_time_s": event.end_time_s,
                "duration_s": event.end_time_s - event.start_time_s,
                "start_col": start_col,
                "end_col": end_col,
                "max_probability": event.peak_probability,
                "mean_probability": event.mean_probability,
            }
        )

    if not raw or boundary_padding_cols == 0:
        return raw

    # Phase 2: merge overlapping detections caused by padding
    merged: List[Dict] = [raw[0]]
    for det in raw[1:]:
        prev = merged[-1]
        if det["start_col"] <= prev["end_col"]:
            # Overlap — merge by extending the earlier detection
            prev["end_col"] = max(prev["end_col"], det["end_col"])
            prev["end_time_s"] = max(prev["end_time_s"], det["end_time_s"])
            prev["duration_s"] = prev["end_time_s"] - prev["start_time_s"]
            prev["max_probability"] = max(prev["max_probability"], det["max_probability"])
            prev["mean_probability"] = (prev["mean_probability"] + det["mean_probability"]) / 2.0
        else:
            merged.append(det)

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True regions in a boolean array.

    Returns list of (start, end) inclusive tuples.
    """
    if not np.any(mask):
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    # Handle region starting at index 0
    if mask[0]:
        starts = np.concatenate([[0], starts])
    # Handle region ending at last index
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask) - 1]])

    return list(zip(starts.tolist(), ends.tolist()))


def _gap_fill(regions: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    """Merge regions separated by <= max_gap windows.

    When max_gap=0, gap-filling is skipped (no adjacent regions possible
    from _extract_regions since contiguous True regions are already merged).
    """
    if not regions or max_gap < 1:
        return regions

    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged
