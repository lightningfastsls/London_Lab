"""Adversarial / hardened tests for event_scoring module.

These supplement the 16 tests already in test_event_scoring.py.
Focus areas:
  A. ROADMAP test plan gaps (item 8 in 15.2 spec)
  B. Untested code paths (collar boundary branches, score tie-breaking)
  C. Edge cases (zero-duration events, negative times, NaN inputs, large N)
  D. compute_f_beta with non-standard beta values and extreme counts
  E. Overlapping-GT and overlapping-detection scenarios
  F. Integration boundary — verify output tuple contract for downstream consumers

Tests that expose real bugs are marked with pytest.mark.skip(reason="BUG FOUND: ...").
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usv_spectrogram.postprocessing.event_scoring import (
    EventScoringConfig,
    compute_f_beta,
    match_events_collar,
)
from usv_spectrogram.postprocessing.hysteresis import USVEvent


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _evt(start_s: float, end_s: float, peak: float = 0.9) -> USVEvent:
    """Minimal USVEvent for scoring tests."""
    return USVEvent(
        start_window=0,
        end_window=max(1, int((end_s - start_s) * 1000)),
        start_time_s=start_s,
        end_time_s=end_s,
        duration_ms=(end_s - start_s) * 1000.0,
        peak_probability=peak,
        mean_probability=peak,
        window_count=max(1, int((end_s - start_s) * 1000)),
        probabilities=np.array([peak]),
    )


# ===========================================================================
# A. Collar boundary precision
# ===========================================================================

class TestCollarBoundary:
    """Tests for exact boundary behavior of the <= comparison in the collar check."""

    def test_onset_exactly_at_collar_is_match(self):
        """
        |det.start - gt.start| == collar_s (exactly equal) should still match
        because the implementation uses <=.
        """
        collar = 0.200
        dets = [_evt(1.0 + collar, 1.6)]   # onset is exactly collar away
        gt = [(1.0, 1.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=collar)
        assert (tp, fp, fn) == (1, 0, 0), (
            "Onset exactly at collar distance must still match (<=, not <)"
        )

    def test_onset_just_outside_collar_with_no_overlap_is_fp(self):
        """
        |det.start - gt.start| just exceeds collar_s, no overlap, offset also out.
        Must be FP + FN.
        """
        collar = 0.200
        epsilon = 1e-9
        # det: 1.0+collar+eps to 1.0+collar+eps+0.3, no overlap with [1.0, 1.3]
        # onset diff = collar+eps > collar; offset: (1.0+collar+eps+0.3) vs 1.3
        #   = collar+eps > collar; no overlap since det starts after gt ends
        det_start = 1.0 + collar + epsilon
        det_end = det_start + 0.3
        dets = [_evt(det_start, det_end)]
        gt = [(1.0, 1.0 + collar - epsilon)]  # gt ends just before det starts
        tp, fp, fn = match_events_collar(dets, gt, collar_s=collar)
        assert (tp, fp, fn) == (0, 1, 1), (
            "Detection just outside collar with no overlap must NOT match"
        )

    def test_offset_exactly_at_collar_is_match(self):
        """
        |det.end - gt.end| == collar_s (no onset match, no overlap) should match.
        Build scenario: onset far apart (beyond collar), no overlap,
        but offsets exactly collar apart.
        """
        collar = 0.200
        # gt: 0.0 to 0.5; det: 1.0 to 0.5+collar — no overlap with gt, onset far
        # Actually we need no overlap: det starts after gt ends AND onset far beyond collar.
        gt_end = 1.0
        det_end = gt_end + collar   # offset exactly at collar
        gt_start = 0.5
        det_start = gt_end + 0.05   # starts after gt ends → no overlap
        # onset diff = det_start - gt_start = 0.55 >> collar → onset NOT ok
        # offset diff = collar → offset IS ok
        dets = [_evt(det_start, det_end)]
        gt = [(gt_start, gt_end)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=collar)
        assert (tp, fp, fn) == (1, 0, 0), (
            "Offset exactly at collar must still match"
        )

    def test_zero_collar_requires_exact_onset_or_offset_or_overlap(self):
        """collar_s=0.0 means only exact boundary equality OR any overlap matches."""
        # Perfect match: onset=0 apart → should match
        dets = [_evt(1.0, 1.5)]
        gt = [(1.0, 1.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.0)
        assert tp == 1

    def test_zero_collar_no_match_when_onset_shifted(self):
        """collar_s=0 with onset 1ms off and no overlap → FP+FN."""
        dets = [_evt(1.001, 1.5)]  # onset 1ms late, no overlap with [0.5, 1.0]
        gt = [(0.5, 1.0)]
        # overlap: min(1.5, 1.0) - max(1.001, 0.5) = 1.0 - 1.001 = -0.001 → 0 (no overlap)
        # onset diff = 0.501, collar=0 → not ok
        # offset diff = 0.5, collar=0 → not ok
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.0)
        assert (tp, fp, fn) == (0, 1, 1)


# ===========================================================================
# B. Overlapping / adjacent ground-truth events
# ===========================================================================

class TestOverlappingGroundTruth:
    """Two GT events that themselves overlap in time."""

    def test_two_overlapping_gts_one_det_in_both(self):
        """
        GT1: [1.0, 2.0], GT2: [1.5, 2.5] — they overlap.
        Det: [1.5, 2.0] — overlaps both GTs.
        Greedy assigns to whichever has higher score. Result: 1 TP + 1 FN (not 2 TP).
        """
        dets = [_evt(1.5, 2.0)]
        gt = [(1.0, 2.0), (1.5, 2.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert tp == 1
        assert fp == 0
        assert fn == 1

    def test_two_overlapping_gts_two_dets(self):
        """
        GT1: [1.0, 2.0], GT2: [1.5, 2.5].
        Det1: [1.0, 1.4], Det2: [1.6, 2.5].
        Each det should match one GT → 2 TP, 0 FP, 0 FN.
        """
        dets = [_evt(1.0, 1.4), _evt(1.6, 2.5)]
        gt = [(1.0, 2.0), (1.5, 2.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert tp == 2
        assert fp == 0
        assert fn == 0


# ===========================================================================
# C. Overlapping detections competing for one GT
# ===========================================================================

class TestOverlappingDetections:
    """Two detections that overlap each other, only one can match a GT."""

    def test_better_overlap_wins_match(self):
        """
        GT: [1.0, 1.5].
        Det A: [1.0, 1.5] — perfect match (overlap=0.5).
        Det B: [1.0, 1.1] — partial match (overlap=0.1).
        Greedy should assign Det A → GT, making Det B the FP.
        """
        det_a = _evt(1.0, 1.5)
        det_b = _evt(1.0, 1.1)
        gt = [(1.0, 1.5)]
        tp, fp, fn = match_events_collar([det_a, det_b], gt, collar_s=0.200)
        assert tp == 1
        assert fp == 1
        assert fn == 0

    def test_identical_detections_one_matches(self):
        """Two identical detections for one GT: exactly 1 TP, 1 FP, 0 FN."""
        dets = [_evt(1.0, 1.5), _evt(1.0, 1.5)]
        gt = [(1.0, 1.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 1, 0)


# ===========================================================================
# D. Zero-duration events
# ===========================================================================

class TestZeroDurationEvents:
    """Events where start_time_s == end_time_s (instantaneous)."""

    def test_zero_duration_detection_within_collar_matches(self):
        """A zero-duration detection whose 'onset' is within collar of GT onset."""
        dets = [_evt(1.05, 1.05)]   # zero-duration, 50ms after GT onset
        gt = [(1.0, 1.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_zero_duration_gt_detection_overlaps(self):
        """Detection overlapping a zero-duration GT marker (e.g. click annotation)."""
        # GT is a point at 1.3; det spans [1.0, 1.5] → overlap > 0 at single point
        # Overlap = min(1.5, 1.3) - max(1.0, 1.3) = 1.3 - 1.3 = 0.0
        # Onset diff = |1.0 - 1.3| = 0.3 > 0.2; offset diff = |1.5 - 1.3| = 0.2 = collar
        # offset exactly at collar → should match
        dets = [_evt(1.0, 1.5)]
        gt = [(1.3, 1.3)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_zero_duration_det_and_gt_identical(self):
        """Zero-duration det coincides exactly with zero-duration GT."""
        dets = [_evt(1.0, 1.0)]
        gt = [(1.0, 1.0)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)


# ===========================================================================
# E. Negative time values
# ===========================================================================

class TestNegativeTimeValues:
    """Events at or before t=0 (e.g. pre-roll buffer offsets)."""

    def test_negative_start_time_matches_negative_gt(self):
        """Both det and GT have negative start times — should still match."""
        dets = [_evt(-0.5, -0.1)]
        gt = [(-0.5, -0.1)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_negative_start_within_collar_of_zero_gt(self):
        """Det starts at -0.1 (within 200ms collar of GT at 0.0)."""
        dets = [_evt(-0.1, 0.4)]
        gt = [(0.0, 0.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_negative_time_far_from_gt_is_fp(self):
        """Det at -5.0 to -4.5 is far from GT at 0.0 to 0.5 — no match."""
        dets = [_evt(-5.0, -4.5)]
        gt = [(0.0, 0.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (0, 1, 1)


# ===========================================================================
# F. compute_f_beta — untested beta values and count combinations
# ===========================================================================

class TestComputeFBeta:
    """Exhaustive formula verification for non-standard betas and edge counts."""

    def test_f_half_formula(self):
        """F0.5 weights precision ~4x over recall. TP=10, FP=5, FN=2."""
        beta = 0.5
        tp, fp, fn = 10, 5, 2
        beta_sq = beta * beta  # 0.25
        expected = (1 + beta_sq) * tp / ((1 + beta_sq) * tp + beta_sq * fn + fp)
        result = compute_f_beta(tp, fp, fn, beta=beta)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_f_beta_large(self):
        """F10 (recall dominates almost entirely). TP=8, FP=10, FN=1."""
        beta = 10.0
        tp, fp, fn = 8, 10, 1
        beta_sq = beta * beta
        expected = (1 + beta_sq) * tp / ((1 + beta_sq) * tp + beta_sq * fn + fp)
        result = compute_f_beta(tp, fp, fn, beta=beta)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_perfect_score_tp_only(self):
        """TP=10, FP=0, FN=0 → F-beta = 1.0 regardless of beta."""
        for beta in [0.5, 1.0, 2.0, 5.0]:
            result = compute_f_beta(10, 0, 0, beta=beta)
            assert result == pytest.approx(1.0, abs=1e-9), (
                f"Perfect detection must give F{beta}=1.0, got {result}"
            )

    def test_tp_zero_fp_zero_fn_nonzero(self):
        """All GT events missed, no FP → F-beta = 0.0."""
        assert compute_f_beta(0, 0, 5, beta=2.0) == 0.0

    def test_tp_zero_fp_nonzero_fn_zero(self):
        """No GT events, detector fires → F-beta = 0.0."""
        assert compute_f_beta(0, 5, 0, beta=2.0) == 0.0

    def test_tp_nonzero_fn_zero(self):
        """Recall=1.0: all GT matched, no FN. TP=5, FP=3, FN=0."""
        beta = 2.0
        tp, fp, fn = 5, 3, 0
        beta_sq = beta * beta
        expected = (1 + beta_sq) * tp / ((1 + beta_sq) * tp + beta_sq * fn + fp)
        result = compute_f_beta(tp, fp, fn, beta=beta)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_tp_nonzero_fp_zero(self):
        """Precision=1.0: no FP, some FN. TP=5, FP=0, FN=3."""
        beta = 2.0
        tp, fp, fn = 5, 0, 3
        beta_sq = beta * beta
        expected = (1 + beta_sq) * tp / ((1 + beta_sq) * tp + beta_sq * fn + fp)
        result = compute_f_beta(tp, fp, fn, beta=beta)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_f_beta_result_bounded_between_0_and_1(self):
        """F-beta must always be in [0, 1]."""
        cases = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (10, 5, 3), (1, 100, 100),
        ]
        for tp, fp, fn in cases:
            for beta in [0.5, 1.0, 2.0]:
                result = compute_f_beta(tp, fp, fn, beta=beta)
                assert 0.0 <= result <= 1.0, (
                    f"F{beta}({tp},{fp},{fn})={result} out of [0,1]"
                )

    def test_higher_beta_penalizes_fn_more(self):
        """
        F2 must score a high-FN scenario lower than F0.5 scores same scenario,
        demonstrating beta controls the recall/precision trade-off direction.
        With TP=5, FP=1, FN=10 (many misses):
          F2 < F0.5 because F2 penalises the FN heavily.
        """
        tp, fp, fn = 5, 1, 10
        f2 = compute_f_beta(tp, fp, fn, beta=2.0)
        f_half = compute_f_beta(tp, fp, fn, beta=0.5)
        assert f2 < f_half, (
            f"High-FN scenario should score lower under F2 than F0.5, "
            f"got F2={f2:.4f}, F0.5={f_half:.4f}"
        )

    def test_large_counts_no_overflow(self):
        """Verify no floating-point overflow with very large TP/FP/FN."""
        result = compute_f_beta(10_000, 5_000, 1_000, beta=2.0)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# G. Large event counts (performance + correctness)
# ===========================================================================

class TestLargeEventCounts:
    """Verify correctness (and no crash) with hundreds of events."""

    def test_many_perfect_matches(self):
        """N detections perfectly matching N GT events → N TP, 0 FP, 0 FN."""
        n = 200
        dets = [_evt(i * 1.0, i * 1.0 + 0.3) for i in range(n)]
        gt = [(i * 1.0, i * 1.0 + 0.3) for i in range(n)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (n, 0, 0)

    def test_many_false_positives(self):
        """N detections with 0 GT events → 0 TP, N FP, 0 FN."""
        n = 150
        dets = [_evt(i * 0.5, i * 0.5 + 0.1) for i in range(n)]
        gt = []
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (0, n, 0)

    def test_many_false_negatives(self):
        """0 detections with N GT events → 0 TP, 0 FP, N FN."""
        n = 150
        dets = []
        gt = [(i * 0.5, i * 0.5 + 0.1) for i in range(n)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (0, 0, n)

    def test_many_interleaved_match_and_miss(self):
        """
        Alternate: even-indexed dets match even-indexed GTs; odd-indexed GTs unmatched.
        n dets match n GTs, n extra GTs unmatched → n TP, 0 FP, n FN.
        """
        n = 50
        # Matched pairs: det[i] and gt[2*i] are at the same position
        dets = [_evt(i * 2.0, i * 2.0 + 0.3) for i in range(n)]
        gt = []
        for i in range(n):
            gt.append((i * 2.0, i * 2.0 + 0.3))        # matched
            gt.append((i * 2.0 + 1.0, i * 2.0 + 1.3))  # unmatched (between dets)
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert tp == n
        assert fp == 0
        assert fn == n


# ===========================================================================
# H. Greedy ordering effects
# ===========================================================================

class TestGreedyOrderEffects:
    """
    Greedy matching is order-dependent. Verify the score-sorted ordering
    produces globally better results than naive (insertion-order) matching would.
    """

    def test_greedy_prefers_higher_overlap_match(self):
        """
        Scenario with 3 events where naive order would produce suboptimal assignment.

        GT1: [1.0, 1.5], GT2: [2.0, 2.5]
        Det A: [1.0, 1.5]  — perfect match to GT1 (overlap=0.5)
        Det B: [1.0, 2.5]  — large overlap with GT1 (0.5) AND GT2 (0.5)
        Det C: [2.0, 2.5]  — perfect match to GT2 (overlap=0.5)

        Optimal: A→GT1, C→GT2 → 2 TP, 1 FP, 0 FN
        Naive (insertion order) might match B first → A or C becomes FP.
        Score-sorted greedy should also achieve 2 TP.
        """
        det_a = _evt(1.0, 1.5)
        det_b = _evt(1.0, 2.5)  # spans both GTs
        det_c = _evt(2.0, 2.5)
        gt = [(1.0, 1.5), (2.0, 2.5)]
        tp, fp, fn = match_events_collar([det_a, det_b, det_c], gt, collar_s=0.200)
        assert tp == 2
        assert fp == 1
        assert fn == 0

    def test_greedy_input_order_does_not_change_tp(self):
        """
        Reversing the input order of detections must not change TP.
        (Regression guard: greedy must be score-sorted, not input-sorted.)
        """
        det_a = _evt(1.0, 1.5)
        det_b = _evt(1.0, 1.1)  # smaller overlap with GT [1.0, 1.5]
        gt = [(1.0, 1.5)]

        tp1, fp1, fn1 = match_events_collar([det_a, det_b], gt, collar_s=0.200)
        tp2, fp2, fn2 = match_events_collar([det_b, det_a], gt, collar_s=0.200)

        assert tp1 == tp2 == 1
        assert fp1 == fp2 == 1
        assert fn1 == fn2 == 0


# ===========================================================================
# I. Output contract — tuple shape and type
# ===========================================================================

class TestOutputContract:
    """match_events_collar returns a 3-tuple of int; compute_f_beta returns float."""

    def test_return_is_three_tuple(self):
        result = match_events_collar([_evt(1.0, 1.5)], [(1.0, 1.5)], collar_s=0.2)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_return_values_are_non_negative(self):
        """TP, FP, FN must all be >= 0 in all scenarios."""
        scenarios = [
            ([], []),
            ([_evt(1.0, 1.5)], []),
            ([], [(1.0, 1.5)]),
            ([_evt(1.0, 1.5)], [(1.0, 1.5)]),
            ([_evt(1.0, 1.5), _evt(1.1, 1.6)], [(1.0, 1.5)]),
        ]
        for dets, gt in scenarios:
            tp, fp, fn = match_events_collar(dets, gt, collar_s=0.2)
            assert tp >= 0 and fp >= 0 and fn >= 0

    def test_tp_plus_fn_equals_n_gt(self):
        """TP + FN must equal number of GT events (each GT is either matched or not)."""
        dets = [_evt(1.0, 1.5), _evt(3.0, 3.5)]
        gt = [(1.0, 1.5), (2.0, 2.5), (3.0, 3.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert tp + fn == len(gt)

    def test_tp_plus_fp_equals_n_det(self):
        """TP + FP must equal number of detections (each det is either TP or FP)."""
        dets = [_evt(1.0, 1.5), _evt(2.0, 2.5), _evt(3.0, 3.5)]
        gt = [(1.0, 1.5), (3.0, 3.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert tp + fp == len(dets)

    def test_compute_f_beta_returns_float(self):
        result = compute_f_beta(5, 2, 1, beta=2.0)
        assert isinstance(result, float)

    def test_compute_f_beta_zero_returns_float(self):
        result = compute_f_beta(0, 0, 0, beta=2.0)
        assert isinstance(result, float)
        assert result == 0.0


# ===========================================================================
# J. EventScoringConfig validation
# ===========================================================================

class TestEventScoringConfig:
    """Config dataclass field presence and types."""

    def test_config_has_min_iou_field(self):
        """EventScoringConfig must expose min_iou (ROADMAP spec W-5 fix)."""
        cfg = EventScoringConfig()
        assert hasattr(cfg, "min_iou"), "min_iou field missing from EventScoringConfig"
        assert cfg.min_iou == 0.0

    def test_config_is_frozen(self):
        """EventScoringConfig must be immutable (frozen dataclass)."""
        cfg = EventScoringConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.onset_collar_s = 0.5  # type: ignore[misc]

    def test_config_custom_collar(self):
        """Custom collar value is accepted and stored."""
        cfg = EventScoringConfig(onset_collar_s=0.100)
        assert cfg.onset_collar_s == pytest.approx(0.100)


# ===========================================================================
# K. Onset-only match (no offset collar, no overlap)
# ===========================================================================

class TestOnsetOnlyAndOffsetOnly:
    """Isolate the three OR branches in the match condition."""

    def test_onset_match_only_no_overlap_no_offset_collar(self):
        """
        Only onset is within collar; offset is far away; no overlap.

        GT: [1.0, 1.1] (short)
        Det: [1.05, 5.0] (long, offset far away)
        onset diff = 0.05 <= 0.2 → onset ok
        offset diff = |5.0 - 1.1| = 3.9 > 0.2 → offset NOT ok
        overlap = min(5.0, 1.1) - max(1.05, 1.0) = 1.1 - 1.05 = 0.05 > 0 → overlap ok too

        This test just confirms the event is matched (overlap rescues it anyway).
        """
        dets = [_evt(1.05, 5.0)]
        gt = [(1.0, 1.1)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_offset_match_only_no_onset_collar_no_overlap(self):
        """
        Construct a scenario where ONLY the offset collar criterion fires.

        GT: [0.0, 1.0]; Det: [1.05, 1.2]
        onset diff = |1.05 - 0.0| = 1.05 > 0.2 → onset NOT ok
        overlap = min(1.2, 1.0) - max(1.05, 0.0) = 1.0 - 1.05 = -0.05 → 0 (no overlap)
        offset diff = |1.2 - 1.0| = 0.2 = collar → offset ok (<=)
        """
        dets = [_evt(1.05, 1.2)]
        gt = [(0.0, 1.0)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_no_collar_no_overlap_is_always_fp(self):
        """All three conditions false → cannot match."""
        # onset diff = 5.0, offset diff = 5.0, no overlap
        dets = [_evt(10.0, 10.5)]
        gt = [(5.0, 5.5)]
        tp, fp, fn = match_events_collar(dets, gt, collar_s=0.200)
        assert (tp, fp, fn) == (0, 1, 1)


# ===========================================================================
# L. Single-element degenerate cases
# ===========================================================================

class TestSingleElementCases:
    """1-item lists everywhere."""

    def test_single_det_single_gt_match(self):
        tp, fp, fn = match_events_collar([_evt(0.0, 0.1)], [(0.0, 0.1)], collar_s=0.200)
        assert (tp, fp, fn) == (1, 0, 0)

    def test_single_det_single_gt_no_match(self):
        tp, fp, fn = match_events_collar([_evt(10.0, 10.1)], [(0.0, 0.1)], collar_s=0.200)
        assert (tp, fp, fn) == (0, 1, 1)

    def test_f_beta_single_tp(self):
        """TP=1, FP=0, FN=0 → 1.0."""
        assert compute_f_beta(1, 0, 0) == pytest.approx(1.0)

    def test_f_beta_single_fn(self):
        """TP=0, FP=0, FN=1 → 0.0."""
        assert compute_f_beta(0, 0, 1) == 0.0

    def test_f_beta_single_fp(self):
        """TP=0, FP=1, FN=0 → 0.0."""
        assert compute_f_beta(0, 1, 0) == 0.0
