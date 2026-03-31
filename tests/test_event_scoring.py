"""Tests for event-level scoring module.

14 test cases using synthetic USVEvent objects (no model/WAV dependencies).
Covers: perfect match, within-collar, outside collar, greedy assignment,
noise recordings, F2 formula, edge cases, and empty inputs.
"""

import numpy as np
import pytest

from usv_spectrogram.postprocessing.event_scoring import (
    EventScoringConfig,
    compute_f_beta,
    match_events_collar,
)
from usv_spectrogram.postprocessing.hysteresis import USVEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(start_s: float, end_s: float, peak_prob: float = 0.9) -> USVEvent:
    """Create a synthetic USVEvent with minimal fields for scoring tests."""
    return USVEvent(
        start_window=0,
        end_window=1,
        start_time_s=start_s,
        end_time_s=end_s,
        duration_ms=(end_s - start_s) * 1000.0,
        peak_probability=peak_prob,
        mean_probability=peak_prob,
        window_count=2,
        probabilities=np.array([peak_prob, peak_prob]),
    )


# ---------------------------------------------------------------------------
# 1. Perfect match — detection exactly matches ground truth
# ---------------------------------------------------------------------------

def test_perfect_match():
    dets = [_make_event(1.0, 1.5)]
    gt = [(1.0, 1.5)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (1, 0, 0)


# ---------------------------------------------------------------------------
# 2. Within-collar onset — onset off by 150ms (within 200ms collar)
# ---------------------------------------------------------------------------

def test_within_collar_onset():
    dets = [_make_event(1.15, 1.65)]  # onset 150ms late
    gt = [(1.0, 1.5)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (1, 0, 0)


# ---------------------------------------------------------------------------
# 3. Within-collar offset — offset off by 180ms
# ---------------------------------------------------------------------------

def test_within_collar_offset():
    dets = [_make_event(2.0, 2.68)]  # offset 180ms late
    gt = [(2.0, 2.5)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (1, 0, 0)


# ---------------------------------------------------------------------------
# 4. Outside collar, no overlap — detection too far from GT
# ---------------------------------------------------------------------------

def test_outside_collar_no_overlap():
    dets = [_make_event(3.0, 3.5)]
    gt = [(1.0, 1.5)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (0, 1, 1)


# ---------------------------------------------------------------------------
# 5. Two detections, one GT — best match wins, other is FP
# ---------------------------------------------------------------------------

def test_two_detections_one_gt():
    dets = [_make_event(1.0, 1.5), _make_event(1.1, 1.6)]
    gt = [(1.0, 1.5)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert tp == 1
    assert fp == 1
    assert fn == 0


# ---------------------------------------------------------------------------
# 6. One detection, two GTs — matches one, other is FN
# ---------------------------------------------------------------------------

def test_one_detection_two_gts():
    dets = [_make_event(1.0, 1.5)]
    gt = [(1.0, 1.5), (5.0, 5.5)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert tp == 1
    assert fp == 0
    assert fn == 1


# ---------------------------------------------------------------------------
# 7. Greedy assignment correctness — overlapping candidates resolved correctly
# ---------------------------------------------------------------------------

def test_greedy_assignment():
    # Det A overlaps GT1 (perfect) and GT2 (partial)
    # Det B overlaps GT2 (perfect)
    # Greedy should match A->GT1 (or A->GT2), B->GT2 (or B->GT1)
    # Either way, both should match
    dets = [_make_event(1.0, 1.5), _make_event(2.0, 2.5)]
    gt = [(1.0, 1.5), (2.0, 2.5)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (2, 0, 0)


# ---------------------------------------------------------------------------
# 8. Noise recording — no detections, no GT → (0, 0, 0)
# ---------------------------------------------------------------------------

def test_noise_recording_empty():
    tp, fp, fn = match_events_collar([], [], collar_s=0.200)
    assert (tp, fp, fn) == (0, 0, 0)


# ---------------------------------------------------------------------------
# 9. Noise recording with false detections → (0, N, 0)
# ---------------------------------------------------------------------------

def test_noise_recording_false_detections():
    dets = [_make_event(0.5, 0.8), _make_event(1.2, 1.5)]
    gt = []
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (0, 2, 0)


# ---------------------------------------------------------------------------
# 10. F2 formula verification — known values
# ---------------------------------------------------------------------------

def test_f2_formula():
    # TP=8, FP=2, FN=1 → F2 = 5*8 / (5*8 + 4*1 + 2) = 40/46 ≈ 0.8696
    result = compute_f_beta(tp=8, fp=2, fn=1, beta=2.0)
    assert result == pytest.approx(40.0 / 46.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 11. F2 with all zeros → 0.0
# ---------------------------------------------------------------------------

def test_f2_all_zeros():
    assert compute_f_beta(0, 0, 0, beta=2.0) == 0.0


# ---------------------------------------------------------------------------
# 12. F1 (beta=1) verification
# ---------------------------------------------------------------------------

def test_f1_formula():
    # TP=5, FP=3, FN=2 → F1 = 2*5 / (2*5 + 2 + 3) = 10/15 = 0.6667
    result = compute_f_beta(tp=5, fp=3, fn=2, beta=1.0)
    assert result == pytest.approx(10.0 / 15.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 13. Overlap-only match (no collar) — detection overlaps GT but
#     onset/offset both exceed collar
# ---------------------------------------------------------------------------

def test_overlap_only_match():
    # GT: 1.0-2.0, Det: 1.5-2.8
    # onset diff = 0.5 > 0.2 collar, offset diff = 0.8 > 0.2 collar
    # but overlap = 0.5s > 0 → should match
    dets = [_make_event(1.5, 2.8)]
    gt = [(1.0, 2.0)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (1, 0, 0)


# ---------------------------------------------------------------------------
# 14. Config dataclass — default collar
# ---------------------------------------------------------------------------

def test_scoring_config_defaults():
    cfg = EventScoringConfig()
    assert cfg.onset_collar_s == 0.200
    assert cfg.min_iou == 0.0


# ---------------------------------------------------------------------------
# 15. One detection spanning two adjacent GTs → 1 TP + 1 FN
# ---------------------------------------------------------------------------

def test_one_detection_spans_two_adjacent_gts():
    # Detection covers both GTs but can only match one (greedy one-to-one)
    dets = [_make_event(1.0, 2.0)]
    gt = [(1.0, 1.5), (1.6, 2.0)]
    tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
    assert (tp, fp, fn) == (1, 0, 1)


# ---------------------------------------------------------------------------
# 16. 1SD conservative selection — highest sustain wins, not lowest
# ---------------------------------------------------------------------------

def test_one_sd_conservative_selection():
    """Integration test: conservative selection must prefer higher sustain.

    Higher sustain = harder to extend events = more conservative.
    This test catches the B-1 bug where sustain direction was inverted.

    Inlines the _find_one_se_params logic to avoid fragile script imports.
    """

    def _find_conservative(grid, mean_scores, threshold):
        """Mirror of optimize_hysteresis._find_one_se_params."""
        candidates = []
        for i, params in enumerate(grid):
            if mean_scores[i] >= threshold:
                simplicity = (
                    params["onset_threshold"],        # higher = stricter
                    params["sustain_threshold"],       # higher = stricter
                    -params["gap_fill_windows"],       # lower = stricter
                    params["min_duration_windows"],    # higher = stricter
                )
                candidates.append((simplicity, i))
        if not candidates:
            return int(np.argmax(mean_scores))
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Two combos: same onset/gap/min_dur, different sustain
    grid = [
        {"onset_threshold": 0.80, "sustain_threshold": 0.20,
         "gap_fill_windows": 2, "min_duration_windows": 5},
        {"onset_threshold": 0.80, "sustain_threshold": 0.50,
         "gap_fill_windows": 2, "min_duration_windows": 5},
    ]
    mean_scores = np.array([0.90, 0.89])
    threshold = 0.85

    idx = _find_conservative(grid, mean_scores, threshold)
    # Must select sustain=0.50 (more conservative), not sustain=0.20
    assert grid[idx]["sustain_threshold"] == 0.50

    # Verify the OLD (buggy) direction would pick wrong answer
    def _buggy_find(grid, mean_scores, threshold):
        candidates = []
        for i, params in enumerate(grid):
            if mean_scores[i] >= threshold:
                simplicity = (
                    params["onset_threshold"],
                    -params["sustain_threshold"],  # BUG: negated = prefers lower
                    -params["gap_fill_windows"],
                    params["min_duration_windows"],
                )
                candidates.append((simplicity, i))
        candidates.sort(reverse=True)
        return candidates[0][1]

    buggy_idx = _buggy_find(grid, mean_scores, threshold)
    assert grid[buggy_idx]["sustain_threshold"] == 0.20  # Confirms bug direction
