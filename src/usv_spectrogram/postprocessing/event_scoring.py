"""Event-level scoring for USV detection evaluation.

Pure-function module for collar-based event matching and F-beta computation.
No I/O, no model loading — operates entirely on USVEvent lists and
ground-truth interval tuples.

Matching algorithm: greedy best-overlap-first (standard in bioacoustic
evaluation, e.g. sed_eval). Each detection and ground-truth event is used
at most once. A match requires: onset within collar OR offset within collar
OR any temporal overlap > 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .hysteresis import USVEvent


@dataclass(frozen=True)
class EventScoringConfig:
    """Configuration for event-level scoring.

    Attributes:
        onset_collar_s: Tolerance window (seconds) for onset/offset matching.
            A detection matches a ground-truth event if their onsets are
            within this collar, their offsets are within this collar, or
            they have any temporal overlap.
        min_iou: Minimum intersection-over-union for IoU-based matching.
            Not used by collar matching (collar_s takes precedence).
            Reserved for future IoU-based evaluation mode.
    """

    onset_collar_s: float = 0.200  # +/-200ms
    min_iou: float = 0.0  # Reserved; collar matching ignores this


def match_events_collar(
    detected: List[USVEvent],
    ground_truth: List[Tuple[float, float]],  # (start_s, end_s)
    collar_s: float = 0.200,
) -> Tuple[int, int, int]:
    """Match detected events to ground-truth intervals using collar tolerance.

    A detection matches a ground-truth event if ANY of these hold:
    - |det.start - gt.start| <= collar_s  (onset within collar)
    - |det.end - gt.end| <= collar_s      (offset within collar)
    - temporal overlap > 0                 (any overlap)

    Greedy assignment: build match matrix sorted by overlap descending,
    assign each pair at most once.

    Args:
        detected: List of USVEvent from hysteresis_detect.
        ground_truth: List of (start_s, end_s) tuples from labels.
        collar_s: Collar tolerance in seconds.

    Returns:
        Tuple of (TP, FP, FN).
    """
    n_det = len(detected)
    n_gt = len(ground_truth)

    if n_det == 0 and n_gt == 0:
        return (0, 0, 0)
    if n_det == 0:
        return (0, 0, n_gt)
    if n_gt == 0:
        return (0, n_det, 0)

    # Build match candidates: (overlap_or_score, det_idx, gt_idx)
    candidates = []
    for di, det in enumerate(detected):
        det_start = det.start_time_s
        det_end = det.end_time_s
        for gi, (gt_start, gt_end) in enumerate(ground_truth):
            onset_ok = abs(det_start - gt_start) <= collar_s
            offset_ok = abs(det_end - gt_end) <= collar_s
            overlap = max(0.0, min(det_end, gt_end) - max(det_start, gt_start))
            overlap_ok = overlap > 0

            if onset_ok or offset_ok or overlap_ok:
                # Score by overlap (higher = better match), then collar closeness
                score = overlap + (collar_s - min(abs(det_start - gt_start), collar_s))
                candidates.append((score, di, gi))

    # Sort by score descending (best matches first)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Greedy assignment
    matched_det = set()
    matched_gt = set()
    tp = 0

    for _score, di, gi in candidates:
        if di not in matched_det and gi not in matched_gt:
            matched_det.add(di)
            matched_gt.add(gi)
            tp += 1

    fp = n_det - tp
    fn = n_gt - tp

    return (tp, fp, fn)


def compute_f_beta(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    """Compute F-beta score from TP/FP/FN counts.

    F_beta = (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)

    With beta=2, recall is weighted ~4x over precision.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        beta: Beta parameter (default 2.0 for F2).

    Returns:
        F-beta score in [0, 1]. Returns 0.0 when TP=0 (no true positives).
    """
    if tp == 0:
        return 0.0

    beta_sq = beta * beta
    numerator = (1 + beta_sq) * tp
    denominator = numerator + beta_sq * fn + fp

    return numerator / denominator
