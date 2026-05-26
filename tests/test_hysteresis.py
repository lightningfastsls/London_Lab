"""Tests for hysteresis post-processing module.

Covers: single peak, two peaks (gap variants), min-duration filter,
bidirectional shoulder extension, all-noise, boundary events,
config validation, time mapping, and ADR-010 format conversion.
"""

import numpy as np
import pytest

from usv_spectrogram.postprocessing.hysteresis import (
    HysteresisConfig,
    USVEvent,
    convert_to_detection_format,
    hysteresis_detect,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_times(n: int, step_s: float = 0.00427) -> np.ndarray:
    """Simulate time array with ~4.27 ms window step."""
    return np.arange(n) * step_s


def _default_config(**overrides) -> HysteresisConfig:
    return HysteresisConfig(
        onset_threshold=overrides.get("onset_threshold", 0.75),
        sustain_threshold=overrides.get("sustain_threshold", 0.40),
        gap_fill_windows=overrides.get("gap_fill_windows", 3),
        min_duration_windows=overrides.get("min_duration_windows", 5),
        max_duration_ms=overrides.get("max_duration_ms", 600.0),
    )


# ---------------------------------------------------------------------------
# 1. Single sustained peak → 1 event
# ---------------------------------------------------------------------------

def test_single_sustained_peak():
    # 20 windows: noise-low, then 8 windows above onset, then noise
    probs = np.array([0.1] * 5 + [0.9] * 8 + [0.1] * 7)
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    ev = events[0]
    assert ev.start_window == 5
    assert ev.end_window == 12
    assert ev.window_count == 8
    assert ev.peak_probability == pytest.approx(0.9)
    assert ev.mean_probability == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 2. Two peaks, large gap → 2 events
# ---------------------------------------------------------------------------

def test_two_peaks_large_gap():
    probs = np.array(
        [0.1] * 3 + [0.9] * 6 + [0.1] * 10 + [0.85] * 7 + [0.1] * 4
    )
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 2
    assert events[0].start_window == 3
    assert events[1].start_window == 19


# ---------------------------------------------------------------------------
# 3. Two peaks, small gap → merged to 1
# ---------------------------------------------------------------------------

def test_two_peaks_small_gap_merged():
    # Two peaks separated by 2 windows of low probability (< sustain)
    # With gap_fill_windows=3, they should merge
    probs = np.array(
        [0.1] * 2 + [0.9] * 5 + [0.1] * 2 + [0.9] * 5 + [0.1] * 2
    )
    times = _make_times(len(probs))
    cfg = _default_config(gap_fill_windows=3)

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    assert events[0].start_window == 2
    assert events[0].end_window == 13


# ---------------------------------------------------------------------------
# 4. Short spike below min_duration → filtered out
# ---------------------------------------------------------------------------

def test_short_spike_filtered():
    # 3 windows above onset but min_duration is 5
    probs = np.array([0.1] * 5 + [0.9] * 3 + [0.1] * 5)
    times = _make_times(len(probs))
    cfg = _default_config(min_duration_windows=5)

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 0


# ---------------------------------------------------------------------------
# 5. Peak with sustain-level shoulders → extends through shoulders
# ---------------------------------------------------------------------------

def test_long_event_filtered_by_default_max_duration():
    # 160 windows at 4.27 ms center spacing spans ~679 ms, above the
    # default 600 ms long-event gate.
    probs = np.array([0.1] * 5 + [0.9] * 160 + [0.1] * 5)
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert events == []


def test_long_event_kept_when_max_duration_disabled():
    probs = np.array([0.1] * 5 + [0.9] * 160 + [0.1] * 5)
    times = _make_times(len(probs))
    cfg = _default_config(max_duration_ms=None)

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    assert events[0].duration_ms > 600.0


def test_sustain_shoulder_extension():
    # Shoulders at 0.5 (above sustain=0.40) around an onset peak
    probs = np.array(
        [0.1] * 2 + [0.5] * 3 + [0.9] * 4 + [0.5] * 3 + [0.1] * 3
    )
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    ev = events[0]
    # Should extend backward through the shoulders
    assert ev.start_window == 2
    assert ev.end_window == 11
    assert ev.window_count == 10


# ---------------------------------------------------------------------------
# 6. All-noise → empty list
# ---------------------------------------------------------------------------

def test_all_noise_empty():
    probs = np.array([0.1, 0.15, 0.2, 0.05, 0.12, 0.3, 0.1])
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert events == []


# ---------------------------------------------------------------------------
# 7. Peak at array start/end → correctly bounded
# ---------------------------------------------------------------------------

def test_peak_at_array_start():
    probs = np.array([0.9] * 6 + [0.1] * 10)
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    assert events[0].start_window == 0
    assert events[0].end_window == 5


def test_peak_at_array_end():
    probs = np.array([0.1] * 10 + [0.9] * 6)
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    assert events[0].start_window == 10
    assert events[0].end_window == 15


# ---------------------------------------------------------------------------
# 8. Config validation: sustain > onset → ValueError
# ---------------------------------------------------------------------------

def test_config_sustain_above_onset_raises():
    with pytest.raises(ValueError, match="sustain"):
        HysteresisConfig(onset_threshold=0.5, sustain_threshold=0.6)


def test_config_negative_gap_raises():
    with pytest.raises(ValueError, match="gap_fill_windows"):
        HysteresisConfig(gap_fill_windows=-1)


def test_config_zero_min_duration_raises():
    with pytest.raises(ValueError, match="min_duration_windows"):
        HysteresisConfig(min_duration_windows=0)


# ---------------------------------------------------------------------------
# 9. Times array → correct start_time_s / end_time_s
# ---------------------------------------------------------------------------

def test_config_invalid_max_duration_raises():
    with pytest.raises(ValueError, match="max_duration_ms"):
        HysteresisConfig(max_duration_ms=0.0)


def test_times_mapping():
    probs = np.array([0.1] * 5 + [0.9] * 6 + [0.1] * 5)
    step = 0.00427
    times = _make_times(len(probs), step_s=step)
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    ev = events[0]
    assert ev.start_time_s == pytest.approx(5 * step)
    assert ev.end_time_s == pytest.approx(10 * step)
    assert ev.duration_ms == pytest.approx((10 - 5) * step * 1000.0)


# ---------------------------------------------------------------------------
# 10. ADR-010 format conversion → valid dicts
# ---------------------------------------------------------------------------

def test_convert_to_detection_format():
    probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 5)
    times = _make_times(len(probs))
    col_indices = np.arange(len(probs)) * 10  # Simulate 10px hop
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)
    assert len(events) == 1

    dicts = convert_to_detection_format(events, col_indices)

    assert len(dicts) == 1
    d = dicts[0]
    assert "start_time_s" in d
    assert "end_time_s" in d
    assert "duration_s" in d
    assert "start_col" in d
    assert "end_col" in d
    assert "max_probability" in d
    assert "mean_probability" in d
    assert d["start_col"] == 30  # window 3 × 10
    assert d["end_col"] == 80  # window 8 × 10
    assert d["duration_s"] == pytest.approx(d["end_time_s"] - d["start_time_s"])


# ---------------------------------------------------------------------------
# 11. Empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    events = hysteresis_detect(np.array([]), np.array([]), HysteresisConfig())
    assert events == []


# ---------------------------------------------------------------------------
# 12. Length mismatch → ValueError
# ---------------------------------------------------------------------------

def test_length_mismatch_raises():
    probs = np.array([0.9] * 10)
    times = np.array([0.0] * 5)
    with pytest.raises(ValueError, match="length"):
        hysteresis_detect(probs, times)


# ---------------------------------------------------------------------------
# 13. Overlapping seed extensions produce single event
# ---------------------------------------------------------------------------

def test_overlapping_seed_extensions():
    # Two seeds separated by sustain-level windows — their bidirectional
    # extensions overlap and should produce a single contiguous event.
    probs = np.array(
        [0.1] * 2 + [0.5] * 2 + [0.9] + [0.5] * 2 + [0.9] + [0.5] * 2 + [0.1] * 2
    )
    times = _make_times(len(probs))
    cfg = _default_config(min_duration_windows=1)

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    assert events[0].start_window == 2
    assert events[0].end_window == 9


# ---------------------------------------------------------------------------
# 14. convert_to_detection_format: short column_indices → IndexError
# ---------------------------------------------------------------------------

def test_convert_short_column_indices_raises():
    probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 5)
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)
    assert len(events) == 1

    short_cols = np.arange(5)  # Way too short
    with pytest.raises(IndexError):
        convert_to_detection_format(events, short_cols)


# ---------------------------------------------------------------------------
# 15. Probabilities stored as copy, not view
# ---------------------------------------------------------------------------

def test_probabilities_are_independent_copy():
    probs = np.array([0.1] * 3 + [0.9] * 6 + [0.1] * 3)
    times = _make_times(len(probs))
    cfg = _default_config()

    events = hysteresis_detect(probs, times, cfg)
    original_peak = events[0].probabilities[0]

    # Mutate source array
    probs[3] = 0.0

    # Event's stored probabilities should be unaffected
    assert events[0].probabilities[0] == original_peak


# ---------------------------------------------------------------------------
# 16. Single-window event has duration_ms=0 (center-to-center convention)
# ---------------------------------------------------------------------------

def test_single_window_event_duration_zero():
    # min_duration_windows=1 to allow single-window events
    probs = np.array([0.1] * 3 + [0.9] + [0.1] * 3)
    times = _make_times(len(probs))
    cfg = _default_config(min_duration_windows=1)

    events = hysteresis_detect(probs, times, cfg)

    assert len(events) == 1
    ev = events[0]
    assert ev.window_count == 1
    assert ev.duration_ms == 0.0
    assert ev.start_time_s == ev.end_time_s


# ---------------------------------------------------------------------------
# 17. Probabilities outside [0,1] → ValueError
# ---------------------------------------------------------------------------

def test_probabilities_outside_range_raises():
    probs = np.array([0.1, 0.5, 1.5, 0.9])  # 1.5 is out of range
    times = _make_times(len(probs))
    with pytest.raises(ValueError, match="logits"):
        hysteresis_detect(probs, times)


def test_negative_probabilities_raises():
    probs = np.array([-0.1, 0.5, 0.9, 0.3])
    times = _make_times(len(probs))
    with pytest.raises(ValueError, match="logits"):
        hysteresis_detect(probs, times)
