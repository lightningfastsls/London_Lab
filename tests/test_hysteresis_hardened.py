"""Adversarial / hardening tests for the hysteresis post-processing module.

This file supplements tests/test_hysteresis.py.  It focuses on:
- Boundary values (exact threshold values, epsilon offsets)
- Gap-fill edge cases at the exact boundary
- Config validation edge cases
- NaN/Inf numerical inputs
- convert_to_detection_format edge cases
- All-above / all-below / single-element inputs
- Three-region chain merging
- Seed-skip guard correctness when multiple seeds exist
- Output field types and constraints
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usv_spectrogram.postprocessing.hysteresis import (
    HysteresisConfig,
    USVEvent,
    _extract_regions,
    _gap_fill,
    convert_to_detection_format,
    hysteresis_detect,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _times(n: int, step: float = 0.00427) -> np.ndarray:
    return np.arange(n, dtype=float) * step


def _cfg(**kw) -> HysteresisConfig:
    defaults = dict(
        onset_threshold=0.75,
        sustain_threshold=0.40,
        gap_fill_windows=3,
        min_duration_windows=5,
    )
    defaults.update(kw)
    return HysteresisConfig(**defaults)


# ===========================================================================
# A. ROADMAP Test Plan Gaps
# ===========================================================================

class TestRoadmapGaps:
    """Items from the ROADMAP §15.1 test plan not explicitly covered."""

    def test_all_zeros_input_returns_empty(self):
        """All-zero probabilities: no seeds, must return empty list."""
        probs = np.zeros(20)
        times = _times(20)
        events = hysteresis_detect(probs, times, _cfg())
        assert events == []

    def test_all_ones_input_returns_one_event(self):
        """All probabilities == 1.0 (above onset) → single merged event."""
        probs = np.ones(10)
        times = _times(10)
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        assert len(events) == 1
        assert events[0].start_window == 0
        assert events[0].end_window == 9

    def test_event_ordered_by_start_window(self):
        """Return list is ordered by start_window (multiple events)."""
        probs = np.array([0.9] * 5 + [0.1] * 20 + [0.9] * 5)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 2
        assert events[0].start_window < events[1].start_window

    def test_adr010_duration_s_equals_end_minus_start(self):
        """convert_to_detection_format: duration_s == end_time_s - start_time_s."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 5)
        times = _times(len(probs))
        cols = np.arange(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        dicts = convert_to_detection_format(events, cols)
        for d in dicts:
            assert d["duration_s"] == pytest.approx(d["end_time_s"] - d["start_time_s"])

    def test_adr010_max_probability_is_peak(self):
        """max_probability in output dict matches event.peak_probability."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 5)
        times = _times(len(probs))
        cols = np.arange(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        dicts = convert_to_detection_format(events, cols)
        for event, d in zip(events, dicts):
            assert d["max_probability"] == pytest.approx(event.peak_probability)

    def test_adr010_mean_probability_matches_event(self):
        """mean_probability in output dict matches event.mean_probability."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 5)
        times = _times(len(probs))
        cols = np.arange(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        dicts = convert_to_detection_format(events, cols)
        for event, d in zip(events, dicts):
            assert d["mean_probability"] == pytest.approx(event.mean_probability)


# ===========================================================================
# B. Boundary Conditions
# ===========================================================================

class TestBoundaryConditions:
    """Exact threshold values, epsilon offsets, off-by-one window indices."""

    # --- Onset threshold boundary ---

    def test_probability_exactly_at_onset_threshold_seeds(self):
        """p == onset_threshold (0.75) must count as a seed (>= not >)."""
        probs = np.array([0.1] * 3 + [0.75] * 6 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 1

    def test_probability_just_below_onset_threshold_does_not_seed(self):
        """p = onset - eps should NOT seed an event (but is above sustain)."""
        eps = 1e-9
        probs = np.array([0.1] * 3 + [0.75 - eps] * 6 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        # Below onset → no seeds → no events
        assert events == []

    def test_probability_exactly_at_sustain_threshold_extends(self):
        """p == sustain_threshold (0.40) must be included in extension (>=)."""
        probs = np.array([0.1, 0.1, 0.40, 0.40, 0.9, 0.40, 0.40, 0.1, 0.1])
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        assert len(events) == 1
        assert events[0].start_window == 2
        assert events[0].end_window == 6

    def test_probability_just_below_sustain_threshold_stops_extension(self):
        """p = sustain - eps should stop extension."""
        eps = 1e-9
        probs = np.array([0.1, 0.40 - eps, 0.9, 0.40 - eps, 0.1])
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        # Extension stops immediately; single-window event at index 2
        assert len(events) == 1
        assert events[0].start_window == 2
        assert events[0].end_window == 2

    def test_onset_equals_sustain_valid_config(self):
        """sustain == onset is a valid config (boundary of <= constraint)."""
        # Should not raise
        cfg = HysteresisConfig(onset_threshold=0.8, sustain_threshold=0.8)
        probs = np.array([0.1] * 2 + [0.8] * 6 + [0.1] * 2)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, cfg)
        # Windows at exactly 0.8 seed AND sustain; the extension stops at
        # the first window below 0.8
        assert len(events) == 1
        assert events[0].window_count == 6

    def test_onset_threshold_exactly_one(self):
        """onset_threshold=1.0 is valid; only exact-1.0 probs seed."""
        cfg = HysteresisConfig(onset_threshold=1.0, sustain_threshold=0.5)
        probs = np.array([0.1] * 3 + [1.0] * 5 + [0.6] * 3 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, cfg)
        # Seeds at 3-7; extension continues into 0.6 region (above sustain)
        assert len(events) == 1
        assert events[0].start_window == 3
        assert events[0].end_window == 10

    def test_min_duration_exactly_met(self):
        """Event with window_count == min_duration_windows survives filter."""
        # 5 windows, min_duration=5 → should survive
        probs = np.array([0.1] * 3 + [0.9] * 5 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=5))
        assert len(events) == 1

    def test_min_duration_one_under_filters_out(self):
        """Event with window_count == min_duration_windows - 1 is filtered."""
        # 4 windows, min_duration=5 → should be filtered
        probs = np.array([0.1] * 3 + [0.9] * 4 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=5))
        assert events == []


# ===========================================================================
# C. Gap-Fill Edge Cases
# ===========================================================================

class TestGapFillEdgeCases:
    """Gap exactly at gap_fill_windows boundary."""

    def test_gap_exactly_at_limit_merges(self):
        """Gap == gap_fill_windows → events ARE merged."""
        # Two 6-window events separated by exactly 3 windows of noise
        probs = np.array(
            [0.1] * 2 + [0.9] * 6 + [0.1] * 3 + [0.9] * 6 + [0.1] * 2
        )
        times = _times(len(probs))
        # gap_fill=3, gap between events is exactly 3 → merge
        events = hysteresis_detect(probs, times, _cfg(gap_fill_windows=3))
        assert len(events) == 1

    def test_gap_one_over_limit_does_not_merge(self):
        """Gap == gap_fill_windows + 1 → events are NOT merged."""
        # Two 6-window events separated by exactly 4 windows of noise
        probs = np.array(
            [0.1] * 2 + [0.9] * 6 + [0.1] * 4 + [0.9] * 6 + [0.1] * 2
        )
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(gap_fill_windows=3))
        assert len(events) == 2

    def test_gap_fill_zero_no_merging(self):
        """gap_fill_windows=0 disables gap merging entirely."""
        probs = np.array(
            [0.1] * 2 + [0.9] * 6 + [0.1] * 1 + [0.9] * 6 + [0.1] * 2
        )
        times = _times(len(probs))
        # Gap of 1 window; gap_fill=0 → no merge
        events = hysteresis_detect(probs, times, _cfg(gap_fill_windows=0))
        assert len(events) == 2

    def test_gap_fill_zero_adjacent_events_stay_separate(self):
        """gap_fill=0: two back-to-back onset regions don't merge if separated by
        even one noise window."""
        probs = np.array([0.9] * 6 + [0.1] + [0.9] * 6)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(gap_fill_windows=0))
        assert len(events) == 2

    def test_three_region_chain_merge(self):
        """Three events each within gap_fill_windows of the next → all merged."""
        # Region A (6 windows), gap 2, Region B (5 windows), gap 2, Region C (5 windows)
        probs = (
            [0.1] * 2
            + [0.9] * 6   # A: indices 2-7
            + [0.1] * 2   # gap of 2
            + [0.9] * 5   # B: indices 10-14
            + [0.1] * 2   # gap of 2
            + [0.9] * 5   # C: indices 17-21
            + [0.1] * 2
        )
        probs = np.array(probs)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(gap_fill_windows=3))
        assert len(events) == 1
        # Merged event spans from start of A to end of C
        assert events[0].start_window == 2
        assert events[0].end_window == 21

    def test_gap_fill_then_min_duration_filter(self):
        """Two short events that merge via gap-fill but result is still short."""
        # Two 2-window events, gap 1 → merged to 5-window event; min_duration=6
        probs = np.array([0.1] * 2 + [0.9] * 2 + [0.1] * 1 + [0.9] * 2 + [0.1] * 2)
        times = _times(len(probs))
        events = hysteresis_detect(
            probs, times, _cfg(gap_fill_windows=3, min_duration_windows=6)
        )
        # Merged region is windows 2-6 = 5 windows; min_duration=6 → filtered
        assert events == []


# ===========================================================================
# D. Config Validation Edge Cases
# ===========================================================================

class TestConfigValidationEdgeCases:
    """Boundary values for HysteresisConfig.__post_init__."""

    def test_sustain_zero_raises(self):
        """sustain_threshold=0.0 violates 0 < sustain → ValueError."""
        with pytest.raises(ValueError, match="sustain"):
            HysteresisConfig(onset_threshold=0.5, sustain_threshold=0.0)

    def test_sustain_slightly_above_zero_valid(self):
        """sustain_threshold=1e-9 (just above 0) is valid."""
        cfg = HysteresisConfig(onset_threshold=0.5, sustain_threshold=1e-9)
        assert cfg.sustain_threshold == 1e-9

    def test_onset_above_one_raises(self):
        """onset_threshold=1.0 + eps > 1 → ValueError."""
        with pytest.raises(ValueError):
            HysteresisConfig(onset_threshold=1.0 + 1e-9, sustain_threshold=0.5)

    def test_gap_fill_zero_valid(self):
        """gap_fill_windows=0 is valid (no gap filling)."""
        cfg = HysteresisConfig(gap_fill_windows=0)
        assert cfg.gap_fill_windows == 0

    def test_min_duration_one_valid(self):
        """min_duration_windows=1 is valid (single-window events allowed)."""
        cfg = HysteresisConfig(min_duration_windows=1)
        assert cfg.min_duration_windows == 1

    def test_default_config_valid(self):
        """Default HysteresisConfig() constructs without error."""
        cfg = HysteresisConfig()
        assert cfg.onset_threshold == 0.75
        assert cfg.sustain_threshold == 0.40
        assert cfg.gap_fill_windows == 3
        assert cfg.min_duration_windows == 5

    def test_config_is_frozen(self):
        """HysteresisConfig must be immutable (frozen dataclass)."""
        cfg = HysteresisConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.onset_threshold = 0.9  # type: ignore[misc]

    def test_usvEvent_is_frozen(self):
        """USVEvent must be immutable (frozen dataclass), per S-2 fix."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        ev = events[0]
        with pytest.raises((AttributeError, TypeError)):
            ev.start_window = 999  # type: ignore[misc]


# ===========================================================================
# E. Numerical Edge Cases (NaN / Inf / exact 0.0 / exact 1.0)
# ===========================================================================

class TestNumericalEdgeCases:
    """NaN, Inf, exact 0.0 and 1.0 in probability arrays."""

    def test_nan_in_probabilities_raises(self):
        """NaN in probabilities array should raise ValueError."""
        probs = np.array([0.1, 0.5, float("nan"), 0.9])
        times = _times(len(probs))
        with pytest.raises(ValueError):
            hysteresis_detect(probs, times)

    def test_positive_inf_in_probabilities_raises(self):
        """Positive infinity (inf > 1) is caught by the > 1 range check."""
        probs = np.array([0.1, 0.5, float("inf"), 0.9])
        times = _times(len(probs))
        with pytest.raises(ValueError, match="logits"):
            hysteresis_detect(probs, times)

    def test_negative_inf_in_probabilities_raises(self):
        """Negative infinity should be caught by the < 0 range check."""
        probs = np.array([0.1, float("-inf"), 0.9, 0.5])
        times = _times(len(probs))
        with pytest.raises(ValueError, match="logits"):
            hysteresis_detect(probs, times)

    def test_exact_zero_and_one_are_valid_probabilities(self):
        """0.0 and 1.0 are in [0,1] and must not raise."""
        probs = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        times = _times(len(probs))
        # Should not raise
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 1

    def test_event_peak_prob_is_exactly_one(self):
        """peak_probability == 1.0 when some probs are exactly 1.0."""
        probs = np.array([0.1] * 2 + [1.0] * 6 + [0.1] * 2)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert events[0].peak_probability == pytest.approx(1.0)

    def test_very_small_positive_probabilities_below_sustain(self):
        """Probabilities in (0, sustain) are below sustain, treated as noise."""
        probs = np.array([0.001] * 20)
        times = _times(20)
        events = hysteresis_detect(probs, times, _cfg())
        assert events == []

    def test_probabilities_computed_window_count_matches_array_length(self):
        """USVEvent.window_count == len(USVEvent.probabilities)."""
        probs = np.array([0.1] * 3 + [0.9, 0.85, 0.8, 0.95, 0.9] + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 1
        ev = events[0]
        assert ev.window_count == len(ev.probabilities)


# ===========================================================================
# F. Single-element inputs
# ===========================================================================

class TestSingleElementInputs:
    """1-element arrays and single-window events."""

    def test_single_element_below_onset_returns_empty(self):
        """Single probability below onset → empty."""
        probs = np.array([0.5])
        times = _times(1)
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        assert events == []

    def test_single_element_at_onset_returns_event(self):
        """Single probability >= onset with min_duration=1 → 1 event."""
        probs = np.array([0.9])
        times = _times(1)
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        assert len(events) == 1
        ev = events[0]
        assert ev.start_window == 0
        assert ev.end_window == 0
        assert ev.window_count == 1
        assert ev.duration_ms == pytest.approx(0.0)
        assert ev.peak_probability == pytest.approx(0.9)
        assert ev.mean_probability == pytest.approx(0.9)

    def test_single_element_above_onset_default_min_duration_filters(self):
        """Single window event is filtered by default min_duration=5."""
        probs = np.array([0.9])
        times = _times(1)
        events = hysteresis_detect(probs, times, _cfg())  # min_duration=5
        assert events == []


# ===========================================================================
# G. convert_to_detection_format edge cases
# ===========================================================================

class TestConvertToDetectionFormatEdgeCases:
    """Boundary conditions specific to convert_to_detection_format."""

    def test_empty_events_returns_empty_list(self):
        """Empty event list → empty output list, no error."""
        col_indices = np.arange(20)
        result = convert_to_detection_format([], col_indices)
        assert result == []

    def test_end_window_exactly_at_last_column_index(self):
        """end_window == len(column_indices) - 1 is valid (last valid index)."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 1
        # column_indices exactly covers all windows (0..12)
        col_indices = np.arange(len(probs))
        # Should NOT raise IndexError
        result = convert_to_detection_format(events, col_indices)
        assert result[0]["end_col"] == int(col_indices[events[0].end_window])

    def test_end_window_one_past_column_indices_raises_index_error(self):
        """end_window >= len(column_indices) → IndexError."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        # col_indices has only 5 elements — event.end_window is 8 (> 4)
        col_indices = np.arange(5)
        with pytest.raises(IndexError):
            convert_to_detection_format(events, col_indices)

    def test_single_window_event_duration_s_is_zero(self):
        """Single-window event → duration_s == 0.0 in output dict."""
        probs = np.array([0.1] * 3 + [0.9] + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        col_indices = np.arange(len(probs))
        result = convert_to_detection_format(events, col_indices)
        assert len(result) == 1
        assert result[0]["duration_s"] == pytest.approx(0.0)
        assert result[0]["start_col"] == result[0]["end_col"]

    def test_output_start_col_less_than_end_col_for_multi_window_event(self):
        """start_col < end_col for a multi-window event with monotone col_indices."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        col_indices = np.arange(len(probs)) * 10
        result = convert_to_detection_format(events, col_indices)
        assert result[0]["start_col"] < result[0]["end_col"]

    def test_output_start_col_type_is_int(self):
        """start_col and end_col must be Python int, not numpy int."""
        probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        col_indices = np.arange(len(probs)) * 5
        result = convert_to_detection_format(events, col_indices)
        assert isinstance(result[0]["start_col"], int)
        assert isinstance(result[0]["end_col"], int)

    def test_multiple_events_convert_all(self):
        """Multiple events → output list same length as events list."""
        probs = np.array(
            [0.1] * 2 + [0.9] * 6 + [0.1] * 20 + [0.9] * 6 + [0.1] * 2
        )
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 2
        col_indices = np.arange(len(probs)) * 3
        result = convert_to_detection_format(events, col_indices)
        assert len(result) == 2
        # Second event comes after first in column space
        assert result[1]["start_col"] > result[0]["end_col"]


# ===========================================================================
# H. All-above / all-below edge cases
# ===========================================================================

class TestAllAboveAllBelow:
    """Corner cases: entire array above onset or below sustain."""

    def test_all_above_onset_produces_one_event(self):
        """All probabilities >= onset → one event spanning entire array."""
        n = 15
        probs = np.full(n, 0.9)
        times = _times(n)
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        assert len(events) == 1
        assert events[0].start_window == 0
        assert events[0].end_window == n - 1
        assert events[0].window_count == n

    def test_all_below_sustain_produces_no_events(self):
        """All probs below sustain_threshold → seeds exist if above onset but can't
        sustain; if also below onset → no seeds."""
        probs = np.full(20, 0.3)  # below sustain=0.40 and onset=0.75
        times = _times(20)
        events = hysteresis_detect(probs, times, _cfg())
        assert events == []

    def test_all_above_sustain_but_none_above_onset(self):
        """All probs above sustain but below onset → no seeds → empty."""
        probs = np.full(20, 0.6)  # above sustain=0.40, below onset=0.75
        times = _times(20)
        events = hysteresis_detect(probs, times, _cfg())
        assert events == []

    def test_all_above_onset_single_element(self):
        """Single element at onset, min_duration=1 → one single-window event."""
        probs = np.array([0.8])
        times = _times(1)
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        assert len(events) == 1
        assert events[0].window_count == 1


# ===========================================================================
# I. USVEvent output field correctness
# ===========================================================================

class TestUSVEventOutputFields:
    """Verify every field of USVEvent is computed correctly."""

    def test_window_count_is_end_minus_start_plus_one(self):
        """window_count == end_window - start_window + 1 for all events."""
        probs = np.array([0.1] * 4 + [0.9] * 7 + [0.1] * 4)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        for ev in events:
            assert ev.window_count == ev.end_window - ev.start_window + 1

    def test_start_time_s_equals_times_at_start_window(self):
        """start_time_s == times[start_window] for each event."""
        step = 0.00427
        probs = np.array([0.1] * 4 + [0.9] * 7 + [0.1] * 4)
        times = _times(len(probs), step=step)
        events = hysteresis_detect(probs, times, _cfg())
        for ev in events:
            assert ev.start_time_s == pytest.approx(times[ev.start_window])

    def test_end_time_s_equals_times_at_end_window(self):
        """end_time_s == times[end_window] for each event."""
        step = 0.00427
        probs = np.array([0.1] * 4 + [0.9] * 7 + [0.1] * 4)
        times = _times(len(probs), step=step)
        events = hysteresis_detect(probs, times, _cfg())
        for ev in events:
            assert ev.end_time_s == pytest.approx(times[ev.end_window])

    def test_duration_ms_equals_time_difference_times_1000(self):
        """duration_ms == (end_time_s - start_time_s) * 1000."""
        step = 0.00427
        probs = np.array([0.1] * 4 + [0.9] * 7 + [0.1] * 4)
        times = _times(len(probs), step=step)
        events = hysteresis_detect(probs, times, _cfg())
        for ev in events:
            expected_ms = (ev.end_time_s - ev.start_time_s) * 1000.0
            assert ev.duration_ms == pytest.approx(expected_ms)

    def test_peak_probability_is_max_of_slice(self):
        """peak_probability == max of the probabilities in the event window."""
        probs = np.array([0.1] * 3 + [0.8, 0.9, 0.95, 0.85, 0.8, 0.76] + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 1
        ev = events[0]
        assert ev.peak_probability == pytest.approx(np.max(ev.probabilities))
        assert ev.peak_probability == pytest.approx(0.95)

    def test_mean_probability_is_mean_of_slice(self):
        """mean_probability == mean of the probabilities in the event window."""
        event_probs = np.array([0.8, 0.9, 0.95, 0.85, 0.8, 0.76])
        probs = np.array([0.1] * 3 + list(event_probs) + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 1
        ev = events[0]
        assert ev.mean_probability == pytest.approx(np.mean(event_probs))

    def test_probabilities_slice_matches_original_array_values(self):
        """USVEvent.probabilities == probabilities[start_window:end_window+1]."""
        event_probs = [0.8, 0.9, 0.95, 0.85, 0.8, 0.76]
        probs = np.array([0.1] * 3 + event_probs + [0.1] * 3)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        assert len(events) == 1
        ev = events[0]
        np.testing.assert_array_almost_equal(ev.probabilities, event_probs)

    def test_start_window_is_non_negative(self):
        """start_window >= 0 for all events."""
        probs = np.array([0.9] * 6 + [0.1] * 5)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        for ev in events:
            assert ev.start_window >= 0

    def test_end_window_is_less_than_array_length(self):
        """end_window < len(probabilities) for all events."""
        probs = np.array([0.1] * 5 + [0.9] * 6)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, _cfg())
        for ev in events:
            assert ev.end_window < len(probs)


# ===========================================================================
# J. Internal helpers (_extract_regions, _gap_fill)
# ===========================================================================

class TestInternalHelpers:
    """Direct tests of private helper functions for code-path coverage."""

    # _extract_regions

    def test_extract_regions_all_false_returns_empty(self):
        mask = np.zeros(10, dtype=bool)
        assert _extract_regions(mask) == []

    def test_extract_regions_all_true_returns_full_span(self):
        mask = np.ones(5, dtype=bool)
        regions = _extract_regions(mask)
        assert regions == [(0, 4)]

    def test_extract_regions_single_true_at_start(self):
        mask = np.array([True, False, False])
        regions = _extract_regions(mask)
        assert regions == [(0, 0)]

    def test_extract_regions_single_true_at_end(self):
        mask = np.array([False, False, True])
        regions = _extract_regions(mask)
        assert regions == [(2, 2)]

    def test_extract_regions_single_true_in_middle(self):
        mask = np.array([False, True, False])
        regions = _extract_regions(mask)
        assert regions == [(1, 1)]

    def test_extract_regions_two_separate_regions(self):
        mask = np.array([True, True, False, True, True])
        regions = _extract_regions(mask)
        assert regions == [(0, 1), (3, 4)]

    def test_extract_regions_region_at_both_ends(self):
        mask = np.array([True, False, False, True])
        regions = _extract_regions(mask)
        assert regions == [(0, 0), (3, 3)]

    # _gap_fill

    def test_gap_fill_empty_input_returns_empty(self):
        assert _gap_fill([], max_gap=3) == []

    def test_gap_fill_single_region_unchanged(self):
        regions = [(2, 7)]
        assert _gap_fill(regions, max_gap=3) == [(2, 7)]

    def test_gap_fill_gap_exactly_max_merges(self):
        # gap = 8 - 5 - 1 = 2, max_gap=2 → merge
        regions = [(0, 4), (7, 10)]
        result = _gap_fill(regions, max_gap=2)
        assert result == [(0, 10)]

    def test_gap_fill_gap_one_over_max_no_merge(self):
        # gap = 8 - 4 - 1 = 3, max_gap=2 → no merge
        regions = [(0, 4), (8, 12)]
        result = _gap_fill(regions, max_gap=2)
        assert result == [(0, 4), (8, 12)]

    def test_gap_fill_max_gap_zero_no_merge(self):
        regions = [(0, 4), (6, 10)]
        result = _gap_fill(regions, max_gap=0)
        # max_gap < 1 → early return
        assert result == [(0, 4), (6, 10)]

    def test_gap_fill_max_gap_negative_no_merge(self):
        # Negative max_gap treated same as < 1 → early return
        regions = [(0, 4), (5, 10)]
        result = _gap_fill(regions, max_gap=-1)
        assert result == [(0, 4), (5, 10)]

    def test_gap_fill_three_regions_all_within_gap(self):
        # A=(0,3), B=(5,8), C=(10,13): gaps 1 and 1 both <= 2
        regions = [(0, 3), (5, 8), (10, 13)]
        result = _gap_fill(regions, max_gap=2)
        assert result == [(0, 13)]

    def test_gap_fill_three_regions_first_merges_last_does_not(self):
        # A=(0,3), B=(5,8), C=(15,18): gap A-B=1 (merge), gap merged-C=6 (no merge)
        regions = [(0, 3), (5, 8), (15, 18)]
        result = _gap_fill(regions, max_gap=2)
        assert result == [(0, 8), (15, 18)]


# ===========================================================================
# K. Integration boundary — realistic upstream input shape
# ===========================================================================

class TestIntegrationBoundaries:
    """Verify behavior with input shapes matching what SlidingInference produces."""

    def test_realistic_inference_result_shape(self):
        """Simulate a 1-second recording at 300kHz with hop=128, stride=10.
        n_windows ≈ (300000/128 - 512/128) / 10 ≈ 230 windows.
        Just check no crash and sensible output."""
        rng = np.random.default_rng(42)
        n = 230
        probs = rng.uniform(0, 0.3, size=n)
        # Inject 3 genuine USV-like events
        for start in [20, 90, 160]:
            probs[start : start + 15] = rng.uniform(0.8, 0.99, size=15)
        probs = np.clip(probs, 0.0, 1.0)
        times = np.arange(n) * 0.00427  # ~4.27ms per window
        events = hysteresis_detect(probs, times, _cfg())
        # Should detect at least the 3 injected events (may merge or split)
        assert len(events) >= 1
        # All events must have valid fields
        for ev in events:
            assert ev.start_window >= 0
            assert ev.end_window < n
            assert ev.end_window >= ev.start_window
            assert ev.peak_probability >= ev.mean_probability
            assert ev.window_count == ev.end_window - ev.start_window + 1
            assert math.isfinite(ev.duration_ms)
            assert math.isfinite(ev.peak_probability)
            assert math.isfinite(ev.mean_probability)

    def test_output_compatible_with_column_indices_from_sliding_inference(self):
        """column_indices from SlidingInference are stride-multiples (stride=10).
        Verify convert_to_detection_format produces valid col values."""
        rng = np.random.default_rng(7)
        n = 50
        probs = np.zeros(n)
        probs[10:20] = rng.uniform(0.8, 0.99, size=10)
        probs = np.clip(probs, 0, 1)
        times = np.arange(n) * 0.00427
        # Simulate column_indices: window i -> col i * hop_px (hop_px=10)
        col_indices = np.arange(n) * 10
        events = hysteresis_detect(probs, times, _cfg())
        dicts = convert_to_detection_format(events, col_indices)
        for d in dicts:
            # Column indices should be multiples of 10
            assert d["start_col"] % 10 == 0
            assert d["end_col"] % 10 == 0
            assert d["start_col"] <= d["end_col"]

    def test_none_config_uses_defaults(self):
        """Passing config=None must use HysteresisConfig() defaults, not crash."""
        probs = np.array([0.1] * 5 + [0.9] * 8 + [0.1] * 5)
        times = _times(len(probs))
        events = hysteresis_detect(probs, times, config=None)
        assert len(events) == 1


# ===========================================================================
# L. Large-input robustness (no crash / shape checks)
# ===========================================================================

class TestLargeInputRobustness:
    """Verify no crash or memory error on large inputs."""

    def test_large_all_noise_input_no_crash(self):
        n = 100_000
        probs = np.random.default_rng(0).uniform(0, 0.3, n)
        times = np.arange(n, dtype=float) * 0.00427
        events = hysteresis_detect(probs, times, _cfg())
        assert events == []

    def test_large_all_signal_input_returns_one_event(self):
        n = 10_000
        probs = np.full(n, 0.9)
        times = np.arange(n, dtype=float) * 0.00427
        events = hysteresis_detect(probs, times, _cfg(min_duration_windows=1))
        assert len(events) == 1
        assert events[0].window_count == n
