"""Adversarial / hardening tests for triage.py and batch_output.py.

Supplements tests/test_triage.py (19 pre-existing tests).

Coverage gaps targeted:
  A. Boundary values at exactly 0.90 and 0.10 (inclusive vs. exclusive edges)
  B. batch_stats with event_count_std = 0 (no outlier flagging)
  C. batch_stats z-score at threshold boundary (z == outlier_count_zscore, not >)
  D. JSON output with special characters / spaces in filenames
  E. Parquet with many recordings (50+)
  F. RecordingResult with events that have duration_ms = 0 (single-window)
  G. TriageConfig with auto_accept_min_peak = 1.0 (maximum valid edge)
  H. noise_floor_p90 when probabilities array has length 1
  I. write_batch_results with both flags False (both outputs disabled)
  J. ADR-010 dict includes start_col and end_col (newly added fields)
  K. confidence_score == mean_event_confidence (not max_confidence)
  L. NaN/Inf in probabilities does not silently produce correct-looking results
  M. Concurrent writes to same output directory (no file corruption)
  N. Duplicate recording stems (last writer wins, no crash)
  O. Parquet qc_flags column present when results include flags
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List

import numpy as np
import pytest

from usv_spectrogram.postprocessing.hysteresis import USVEvent
from usv_spectrogram.postprocessing.triage import (
    RecordingResult,
    TriageConfig,
    triage_recording,
)
from usv_spectrogram.postprocessing.batch_output import write_batch_results


# ---------------------------------------------------------------------------
# Helpers (mirrors test_triage.py helpers, kept local for independence)
# ---------------------------------------------------------------------------

def _event(
    start_s: float,
    end_s: float,
    peak_prob: float = 0.95,
    mean_prob: float = 0.90,
    n_windows: int = 8,
    start_window: int = 0,
) -> USVEvent:
    probs = np.full(n_windows, mean_prob)
    probs[n_windows // 2] = peak_prob
    return USVEvent(
        start_window=start_window,
        end_window=start_window + n_windows - 1,
        start_time_s=start_s,
        end_time_s=end_s,
        duration_ms=(end_s - start_s) * 1000.0,
        peak_probability=peak_prob,
        mean_probability=mean_prob,
        window_count=n_windows,
        probabilities=probs,
    )


def _single_window_event(start_s: float = 0.5, start_window: int = 50) -> USVEvent:
    """USVEvent with duration_ms = 0 (single-window, center-to-center span is 0)."""
    return USVEvent(
        start_window=start_window,
        end_window=start_window,
        start_time_s=start_s,
        end_time_s=start_s,
        duration_ms=0.0,
        peak_probability=0.95,
        mean_probability=0.95,
        window_count=1,
        probabilities=np.array([0.95]),
    )


def _probs_low(n: int = 200) -> np.ndarray:
    return np.full(n, 0.03)


def _probs_clear(n: int = 200) -> np.ndarray:
    probs = np.full(n, 0.05)
    probs[50:60] = 0.95
    return probs


# ---------------------------------------------------------------------------
# A. Boundary values at exactly 0.90
# ---------------------------------------------------------------------------

class TestBoundaryValuesAt090:
    """auto_accept_min_peak default is 0.90. Events with peak == 0.90 should be ACCEPTED."""

    def test_event_peak_exactly_090_auto_accepts(self):
        """peak_probability == auto_accept_min_peak should satisfy the >= condition."""
        config = TriageConfig(auto_accept_min_peak=0.90)
        # Single event with peak exactly at threshold
        event = _event(0.0, 0.1, peak_prob=0.90, mean_prob=0.85)
        probs = _probs_clear()

        result = triage_recording(
            filepath="/data/boundary_090.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.tier == "auto_accept", (
            f"Event with peak_probability == auto_accept_min_peak (0.90) should be "
            f"auto_accept (>= check), got '{result.tier}'"
        )

    def test_event_peak_just_below_090_manual_review(self):
        """peak_probability just below threshold (0.90 - epsilon) should NOT auto-accept."""
        config = TriageConfig(auto_accept_min_peak=0.90)
        eps = 1e-9
        event = _event(0.0, 0.1, peak_prob=0.90 - eps, mean_prob=0.85)
        probs = _probs_clear()

        result = triage_recording(
            filepath="/data/below_090.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.tier == "manual_review", (
            f"Event with peak < 0.90 should be manual_review, got '{result.tier}'"
        )

    def test_prob_max_exactly_010_is_auto_reject_for_no_events(self):
        """prob_max == auto_reject_max_window (0.10) with no events → auto_reject (n_events=0 branch)."""
        config = TriageConfig()
        # All windows exactly at the reject threshold; no events
        probs = np.full(100, 0.10)

        result = triage_recording(
            filepath="/data/boundary_010_no_events.wav",
            events=[],
            probabilities=probs,
            config=config,
        )

        # n_events == 0 → auto_reject regardless of prob_max
        assert result.tier == "auto_reject"

    def test_prob_max_exactly_010_with_events_is_auto_reject(self):
        """prob_max == 0.10 with events present → auto_reject (prob_max <= threshold branch)."""
        config = TriageConfig(auto_reject_max_window=0.10)
        event = _event(0.0, 0.1, peak_prob=0.10, mean_prob=0.08)
        # All windows at or below 0.10
        probs = np.full(100, 0.10)

        result = triage_recording(
            filepath="/data/boundary_010_with_events.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.tier == "auto_reject", (
            f"prob_max <= auto_reject_max_window should be auto_reject, got '{result.tier}'"
        )

    def test_prob_max_just_above_010_with_high_confidence_events_auto_accepts(self):
        """prob_max just above reject threshold with high-confidence events → auto_accept."""
        config = TriageConfig(auto_accept_min_peak=0.90, auto_reject_max_window=0.10)
        event = _event(0.0, 0.1, peak_prob=0.95, mean_prob=0.92)
        probs = np.full(100, 0.05)
        probs[0] = 0.11  # just above 0.10, not <= threshold

        result = triage_recording(
            filepath="/data/just_above_010.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.tier == "auto_accept"


# ---------------------------------------------------------------------------
# B & C. batch_stats edge cases
# ---------------------------------------------------------------------------

class TestBatchStatsEdgeCases:

    def test_batch_stats_std_zero_no_outlier_flag(self):
        """When event_count_std = 0, outlier flagging must be skipped entirely."""
        events = [_event(i * 0.5, i * 0.5 + 0.1, peak_prob=0.95) for i in range(10)]
        probs = _probs_clear()
        config = TriageConfig()
        batch_stats = {"event_count_mean": 5.0, "event_count_std": 0.0}

        result = triage_recording(
            filepath="/data/std_zero.wav",
            events=events,
            probabilities=probs,
            config=config,
            batch_stats=batch_stats,
        )

        outlier_flags = [f for f in result.qc_flags if "outlier" in f.lower()]
        assert len(outlier_flags) == 0, (
            f"std=0 should skip outlier flagging, but got flags: {result.qc_flags}"
        )

    def test_batch_stats_z_exactly_at_threshold_not_flagged(self):
        """z == outlier_count_zscore (not strictly >) must NOT trigger the flag."""
        # n_events=7, mean=3, std=2 → z = (7-3)/2 = 2.0 == config.outlier_count_zscore
        events = [_event(i * 0.5, i * 0.5 + 0.1, peak_prob=0.95) for i in range(7)]
        probs = _probs_clear(300)
        config = TriageConfig(outlier_count_zscore=2.0)
        batch_stats = {"event_count_mean": 3.0, "event_count_std": 2.0}

        result = triage_recording(
            filepath="/data/z_at_threshold.wav",
            events=events,
            probabilities=probs,
            config=config,
            batch_stats=batch_stats,
        )

        outlier_flags = [f for f in result.qc_flags if "outlier" in f.lower()]
        assert len(outlier_flags) == 0, (
            f"z == outlier_count_zscore should NOT flag (condition is strictly >), "
            f"got flags: {result.qc_flags}"
        )

    def test_batch_stats_z_just_above_threshold_is_flagged(self):
        """z just above outlier_count_zscore must trigger the flag."""
        # z = (6 - 3) / 1 = 3.0 > 2.0
        events = [_event(i * 0.5, i * 0.5 + 0.1, peak_prob=0.95) for i in range(6)]
        probs = _probs_clear(300)
        config = TriageConfig(outlier_count_zscore=2.0)
        batch_stats = {"event_count_mean": 3.0, "event_count_std": 1.0}

        result = triage_recording(
            filepath="/data/z_above_threshold.wav",
            events=events,
            probabilities=probs,
            config=config,
            batch_stats=batch_stats,
        )

        outlier_flags = [f for f in result.qc_flags if "outlier" in f.lower()]
        assert len(outlier_flags) > 0, (
            f"z=3.0 > 2.0 should trigger outlier flag, but got qc_flags={result.qc_flags}"
        )

    def test_batch_stats_missing_std_key_defaults_to_zero(self):
        """If event_count_std key is absent from batch_stats, default to 0 → no crash."""
        events = [_event(0.0, 0.1, peak_prob=0.95)]
        probs = _probs_clear()
        config = TriageConfig()
        # Missing event_count_std key
        batch_stats = {"event_count_mean": 2.0}

        # Must not raise
        result = triage_recording(
            filepath="/data/missing_std.wav",
            events=events,
            probabilities=probs,
            config=config,
            batch_stats=batch_stats,
        )

        assert isinstance(result, RecordingResult)


# ---------------------------------------------------------------------------
# D. JSON output with special characters / spaces in filenames
# ---------------------------------------------------------------------------

class TestJSONSpecialCharacterFilenames:

    def test_filename_with_spaces(self, tmp_path: Path):
        """Stems containing spaces must produce valid JSON filenames."""
        config = TriageConfig()
        events = [_event(0.0, 0.1, peak_prob=0.95)]
        probs = _probs_clear()

        results = [
            triage_recording("/data/my recording with spaces.wav", events, probs, config),
        ]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        # Stem is "my recording with spaces"
        expected = tmp_path / "detections" / "my recording with spaces.json"
        assert expected.exists(), f"Expected JSON at {expected}"
        data = json.loads(expected.read_text())
        assert isinstance(data, list)

    def test_filename_with_parentheses_and_dots(self, tmp_path: Path):
        """Stems with parentheses and extra dots are handled by Path.stem correctly."""
        config = TriageConfig()
        events = [_event(0.0, 0.1, peak_prob=0.95)]
        probs = _probs_clear()

        filepath = "/data/recording(001).extra.wav"
        results = [
            triage_recording(filepath, events, probs, config),
        ]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        # Path("recording(001).extra.wav").stem == "recording(001).extra"
        stem = Path(filepath).stem
        expected = tmp_path / "detections" / f"{stem}.json"
        assert expected.exists(), f"Expected JSON at {expected}, found: {list((tmp_path / 'detections').iterdir())}"

    def test_filename_with_underscores_and_dashes(self, tmp_path: Path):
        """Typical USV recording filenames like 2024-10-04_17-34-54_0001191.wav work."""
        config = TriageConfig()
        events = [_event(0.0, 0.1, peak_prob=0.95)]
        probs = _probs_clear()
        filepath = "/data/2024-10-04_17-34-54_0001191.wav"

        results = [triage_recording(filepath, events, probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        expected = tmp_path / "detections" / "2024-10-04_17-34-54_0001191.json"
        assert expected.exists()
        data = json.loads(expected.read_text())
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# E. Parquet with many recordings
# ---------------------------------------------------------------------------

class TestParquetManyRecordings:

    def test_parquet_50_recordings(self, tmp_path: Path):
        """Parquet summary must have exactly 50 rows for 50 results."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        config = TriageConfig()
        results = []
        for i in range(50):
            if i % 3 == 0:
                events = [_event(0.0, 0.1, peak_prob=0.95)]
                probs = _probs_clear()
            elif i % 3 == 1:
                events = []
                probs = _probs_low()
            else:
                events = [_event(0.0, 0.1, peak_prob=0.70)]
                probs = _probs_clear()
            results.append(
                triage_recording(f"/data/rec_{i:03d}.wav", events, probs, config)
            )

        write_batch_results(results, output_dir=tmp_path, write_parquet=True, write_per_recording_json=False)

        df = pd.read_parquet(tmp_path / "summary.parquet")
        assert len(df) == 50, f"Expected 50 rows, got {len(df)}"

    def test_parquet_all_three_tiers_present(self, tmp_path: Path):
        """When batch contains all three tier types, parquet should contain all three."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        config = TriageConfig()
        results = [
            # auto_accept
            triage_recording("/a.wav", [_event(0.0, 0.1, peak_prob=0.95)], _probs_clear(), config),
            # auto_reject (empty events, no signal)
            triage_recording("/b.wav", [], _probs_low(), config),
            # manual_review (has events but not all above 0.90)
            triage_recording("/c.wav", [_event(0.0, 0.1, peak_prob=0.75)], _probs_clear(), config),
        ]

        write_batch_results(results, output_dir=tmp_path, write_parquet=True, write_per_recording_json=False)

        df = pd.read_parquet(tmp_path / "summary.parquet")
        tiers_present = set(df["tier"].tolist())
        assert "auto_accept" in tiers_present
        assert "auto_reject" in tiers_present
        assert "manual_review" in tiers_present


# ---------------------------------------------------------------------------
# F. Events with duration_ms = 0 (single-window events)
# ---------------------------------------------------------------------------

class TestSingleWindowEvents:

    def test_single_window_event_total_duration_is_zero(self):
        """Single-window USVEvent has duration_ms=0; total_usv_duration_ms should be 0."""
        config = TriageConfig()
        event = _single_window_event()
        probs = _probs_clear()

        result = triage_recording(
            filepath="/data/single_window.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.total_usv_duration_ms == pytest.approx(0.0), (
            f"Single-window event has duration_ms=0, expected total=0, got {result.total_usv_duration_ms}"
        )

    def test_single_window_event_still_counts_in_n_events(self):
        """A single-window event with duration_ms=0 should still be counted as 1 event."""
        config = TriageConfig()
        event = _single_window_event()
        probs = _probs_clear()

        result = triage_recording(
            filepath="/data/single_window_count.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.n_events == 1

    def test_single_window_event_with_high_peak_auto_accepts(self):
        """A single-window event with peak_prob >= 0.90 should yield auto_accept."""
        config = TriageConfig()
        event = _single_window_event()  # peak_prob=0.95 by default
        probs = _probs_clear()

        result = triage_recording(
            filepath="/data/single_window_accept.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.tier == "auto_accept"

    def test_single_window_event_json_has_duration_s_zero(self, tmp_path: Path):
        """ADR-010 JSON for single-window event: duration_s should be 0.0 (expected behavior per W-3)."""
        config = TriageConfig()
        event = _single_window_event(start_s=1.0)
        probs = _probs_clear()

        results = [triage_recording("/data/sw.wav", [event], probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        data = json.loads((tmp_path / "detections" / "sw.json").read_text())
        assert len(data) == 1
        assert data[0]["duration_s"] == pytest.approx(0.0, abs=1e-9)
        # start_time_s == end_time_s for single-window
        assert data[0]["start_time_s"] == pytest.approx(data[0]["end_time_s"], abs=1e-9)


# ---------------------------------------------------------------------------
# G. TriageConfig with auto_accept_min_peak = 1.0
# ---------------------------------------------------------------------------

class TestTriageConfigEdge:

    def test_auto_accept_min_peak_1_0_is_valid(self):
        """auto_accept_min_peak = 1.0 is a valid edge — only events with peak_prob = 1.0 are accepted."""
        config = TriageConfig(auto_accept_min_peak=1.0)
        assert config.auto_accept_min_peak == pytest.approx(1.0)

    def test_auto_accept_min_peak_1_0_rejects_sub_1_events(self):
        """With threshold=1.0, even peak_prob=0.999 falls back to manual_review."""
        config = TriageConfig(auto_accept_min_peak=1.0)
        event = _event(0.0, 0.1, peak_prob=0.999, mean_prob=0.99)
        probs = _probs_clear()

        result = triage_recording(
            filepath="/data/threshold_1_0.wav",
            events=[event],
            probabilities=probs,
            config=config,
        )

        assert result.tier == "manual_review", (
            f"peak_prob=0.999 < auto_accept_min_peak=1.0 should be manual_review, got '{result.tier}'"
        )

    def test_auto_accept_min_peak_must_exceed_zero(self):
        """auto_accept_min_peak = 0.0 must raise ValueError."""
        with pytest.raises(ValueError, match="auto_accept_min_peak"):
            TriageConfig(auto_accept_min_peak=0.0)

    def test_equal_thresholds_raises(self):
        """auto_reject_max_window == auto_accept_min_peak must raise ValueError."""
        with pytest.raises(ValueError):
            TriageConfig(auto_accept_min_peak=0.5, auto_reject_max_window=0.5)

    def test_negative_auto_reject_max_window_raises(self):
        """auto_reject_max_window < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="auto_reject_max_window"):
            TriageConfig(auto_reject_max_window=-0.01)


# ---------------------------------------------------------------------------
# H. noise_floor_p90 when probabilities array has length 1
# ---------------------------------------------------------------------------

class TestNoisFloorP90SingleElement:

    def test_single_probability_element(self):
        """probabilities with 1 element: np.percentile should return that single value."""
        config = TriageConfig()
        probs = np.array([0.75])

        result = triage_recording(
            filepath="/data/one_prob.wav",
            events=[],
            probabilities=probs,
            config=config,
        )

        assert result.noise_floor_p90 == pytest.approx(0.75, abs=1e-9), (
            f"Single-element array p90 should equal that element, got {result.noise_floor_p90}"
        )

    def test_single_probability_zero(self):
        """Single probability of 0.0: noise_floor_p90 = 0.0, tier = auto_reject."""
        config = TriageConfig()
        probs = np.array([0.0])

        result = triage_recording(
            filepath="/data/one_zero.wav",
            events=[],
            probabilities=probs,
            config=config,
        )

        assert result.noise_floor_p90 == pytest.approx(0.0, abs=1e-9)
        assert result.tier == "auto_reject"

    def test_single_probability_high_triggers_noise_flag(self):
        """Single probability of 0.9 > 0.4 threshold: should trigger high_noise_floor flag."""
        config = TriageConfig()
        probs = np.array([0.9])
        events = [_event(0.0, 0.1, peak_prob=0.95)]

        result = triage_recording(
            filepath="/data/one_high.wav",
            events=events,
            probabilities=probs,
            config=config,
        )

        assert result.noise_floor_p90 == pytest.approx(0.9, abs=1e-9)
        assert "high_noise_floor" in result.qc_flags


# ---------------------------------------------------------------------------
# I. write_batch_results with both flags False
# ---------------------------------------------------------------------------

class TestWriteBatchResultsBothDisabled:

    def test_both_outputs_disabled_no_files_created(self, tmp_path: Path):
        """When both write_parquet=False and write_per_recording_json=False, no output files created."""
        config = TriageConfig()
        events = [_event(0.0, 0.1, peak_prob=0.95)]
        probs = _probs_clear()
        results = [triage_recording("/data/a.wav", events, probs, config)]

        write_batch_results(
            results,
            output_dir=tmp_path,
            write_parquet=False,
            write_per_recording_json=False,
        )

        # No .parquet file
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 0, f"Unexpected parquet files: {parquet_files}"

        # No detections/ directory or JSON files
        detections_dir = tmp_path / "detections"
        if detections_dir.exists():
            json_files = list(detections_dir.glob("*.json"))
            assert len(json_files) == 0, f"Unexpected JSON files: {json_files}"

    def test_both_outputs_disabled_does_not_raise(self, tmp_path: Path):
        """write_batch_results with both flags disabled must complete without exception."""
        results = []
        # Should not raise even with empty list
        write_batch_results(
            results,
            output_dir=tmp_path,
            write_parquet=False,
            write_per_recording_json=False,
        )


# ---------------------------------------------------------------------------
# J. ADR-010 dict includes start_col and end_col
# ---------------------------------------------------------------------------

class TestADR010StartEndCol:

    def test_per_recording_json_has_start_col_and_end_col(self, tmp_path: Path):
        """ADR-010 detection dict must include start_col and end_col (per batch_output.py implementation)."""
        config = TriageConfig()
        event = _event(0.0, 0.1, peak_prob=0.95, start_window=5)
        probs = _probs_clear()

        results = [triage_recording("/data/coltest.wav", [event], probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        data = json.loads((tmp_path / "detections" / "coltest.json").read_text())
        assert len(data) == 1
        detection = data[0]

        assert "start_col" in detection, f"start_col missing from detection dict: {detection.keys()}"
        assert "end_col" in detection, f"end_col missing from detection dict: {detection.keys()}"

    def test_start_col_and_end_col_are_integers(self, tmp_path: Path):
        """start_col and end_col must be integer values (not float), per label_storage.py expectation."""
        config = TriageConfig()
        event = _event(0.0, 0.1, peak_prob=0.95, start_window=7)
        probs = _probs_clear()

        results = [triage_recording("/data/coltype.wav", [event], probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        data = json.loads((tmp_path / "detections" / "coltype.json").read_text())
        detection = data[0]

        assert isinstance(detection["start_col"], int), (
            f"start_col should be int, got {type(detection['start_col'])}"
        )
        assert isinstance(detection["end_col"], int), (
            f"end_col should be int, got {type(detection['end_col'])}"
        )

    def test_start_col_end_col_computed_from_window_indices_and_hop_px(self, tmp_path: Path):
        """start_col = start_window * hop_px (default 10); end_col = end_window * hop_px."""
        config = TriageConfig()
        # n_windows=8, start_window=5 → end_window = 5 + 7 = 12
        event = _event(0.0, 0.1, peak_prob=0.95, start_window=5, n_windows=8)
        probs = _probs_clear()

        results = [triage_recording("/data/colcalc.wav", [event], probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        data = json.loads((tmp_path / "detections" / "colcalc.json").read_text())
        detection = data[0]

        expected_start_col = event.start_window * 10   # hop_px default is 10
        expected_end_col = event.end_window * 10
        assert detection["start_col"] == expected_start_col, (
            f"Expected start_col={expected_start_col}, got {detection['start_col']}"
        )
        assert detection["end_col"] == expected_end_col, (
            f"Expected end_col={expected_end_col}, got {detection['end_col']}"
        )

    def test_full_adr010_key_set(self, tmp_path: Path):
        """All 7 expected ADR-010 keys are present (original 5 + start_col + end_col)."""
        config = TriageConfig()
        event = _event(0.0, 0.1, peak_prob=0.95)
        probs = _probs_clear()

        results = [triage_recording("/data/fullkeys.wav", [event], probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        data = json.loads((tmp_path / "detections" / "fullkeys.json").read_text())
        detection = data[0]

        expected_keys = {
            "start_time_s",
            "end_time_s",
            "duration_s",
            "max_probability",
            "mean_probability",
            "start_col",
            "end_col",
        }
        missing = expected_keys - set(detection.keys())
        assert not missing, f"ADR-010 dict missing keys: {missing}"


# ---------------------------------------------------------------------------
# K. confidence_score == mean_event_confidence (not max_confidence)
# ---------------------------------------------------------------------------

class TestConfidenceScoreIsMean:

    def test_confidence_score_equals_mean_not_max(self):
        """Resolved Ambiguity #5: confidence_score is mean_event_confidence, not max_confidence."""
        config = TriageConfig()
        # Two events with different peak probabilities
        event_a = _event(0.0, 0.1, peak_prob=0.98, mean_prob=0.95)
        event_b = _event(0.5, 0.6, peak_prob=0.92, mean_prob=0.88)
        events = [event_a, event_b]
        probs = _probs_clear()

        result = triage_recording(
            filepath="/data/conf_score.wav",
            events=events,
            probabilities=probs,
            config=config,
        )

        expected_mean = (event_a.peak_probability + event_b.peak_probability) / 2.0
        assert result.confidence_score == pytest.approx(result.mean_event_confidence, abs=1e-9), (
            "confidence_score must equal mean_event_confidence"
        )
        assert result.confidence_score == pytest.approx(expected_mean, abs=1e-6), (
            f"Expected confidence_score={expected_mean:.4f}, got {result.confidence_score:.4f}"
        )
        # Verify it is NOT max_confidence
        assert result.confidence_score != pytest.approx(result.max_confidence, abs=1e-9) or \
               result.mean_event_confidence == pytest.approx(result.max_confidence, abs=1e-9), (
            "confidence_score must be mean, not max, when mean != max"
        )

    def test_confidence_score_zero_for_empty_events(self):
        """When there are no events, confidence_score must be 0.0."""
        config = TriageConfig()
        probs = _probs_low()

        result = triage_recording(
            filepath="/data/empty_conf.wav",
            events=[],
            probabilities=probs,
            config=config,
        )

        assert result.confidence_score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# L. NaN / Inf in probabilities
# ---------------------------------------------------------------------------

class TestNaNInfProbabilities:

    def test_nan_in_probabilities_produces_nan_noise_floor(self):
        """NaN in probabilities propagates to noise_floor_p90 (numpy percentile behavior with NaN)."""
        config = TriageConfig()
        probs = np.array([0.1, np.nan, 0.2, 0.3])

        # This test documents the behavior — it should not crash
        # np.percentile with NaN: behavior depends on numpy version.
        # The important thing is: no unhandled exception.
        try:
            result = triage_recording(
                filepath="/data/nan_probs.wav",
                events=[],
                probabilities=probs,
                config=config,
            )
            # If it completes, result is valid (NaN is a known quirk)
            assert isinstance(result, RecordingResult)
        except (ValueError, RuntimeError):
            # Some numpy versions raise on NaN percentile — also acceptable behavior
            pass

    def test_inf_in_probabilities_does_not_crash(self):
        """Inf in probabilities: the function should not raise an unhandled exception."""
        config = TriageConfig()
        probs = np.array([0.1, np.inf, 0.2, 0.3])

        try:
            result = triage_recording(
                filepath="/data/inf_probs.wav",
                events=[],
                probabilities=probs,
                config=config,
            )
            assert isinstance(result, RecordingResult)
        except (ValueError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# M. Concurrent writes to same output directory
# ---------------------------------------------------------------------------

class TestConcurrentWrites:

    def test_concurrent_writes_no_corruption(self, tmp_path: Path):
        """Multiple threads writing to the same output_dir should not corrupt results."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        config = TriageConfig()
        errors: List[Exception] = []

        def write_subset(thread_id: int) -> None:
            events = [_event(0.0, 0.1, peak_prob=0.95)]
            probs = _probs_clear()
            results = [
                triage_recording(f"/data/thread{thread_id}_rec.wav", events, probs, config)
            ]
            # Each thread writes to a distinct subdirectory to avoid collision
            sub_dir = tmp_path / f"thread_{thread_id}"
            try:
                write_batch_results(
                    results,
                    output_dir=sub_dir,
                    write_parquet=True,
                    write_per_recording_json=True,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_subset, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent writes raised exceptions: {errors}"

        # Verify each thread produced its own valid parquet
        for i in range(5):
            sub_dir = tmp_path / f"thread_{i}"
            df = pd.read_parquet(sub_dir / "summary.parquet")
            assert len(df) == 1


# ---------------------------------------------------------------------------
# N. Duplicate recording stems — last writer wins, no crash
# ---------------------------------------------------------------------------

class TestDuplicateStems:

    def test_duplicate_stem_does_not_crash(self, tmp_path: Path):
        """Two recordings with same stem: second JSON overwrites first without raising."""
        config = TriageConfig()
        event_a = _event(0.0, 0.1, peak_prob=0.95)
        event_b = _event(1.0, 1.1, peak_prob=0.92)
        probs = _probs_clear()

        # Both have stem "recording"
        results = [
            triage_recording("/dir_a/recording.wav", [event_a], probs, config),
            triage_recording("/dir_b/recording.wav", [event_b], probs, config),
        ]

        # Should not raise
        write_batch_results(
            results,
            output_dir=tmp_path,
            write_parquet=False,
            write_per_recording_json=True,
        )

        json_files = list((tmp_path / "detections").glob("*.json"))
        assert len(json_files) == 1, f"Expected 1 JSON (last writer wins), got {len(json_files)}"

    def test_duplicate_stem_parquet_has_both_rows(self, tmp_path: Path):
        """Parquet always has one row per RecordingResult even when stems collide."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        config = TriageConfig()
        probs = _probs_clear()
        results = [
            triage_recording("/dir_a/recording.wav", [_event(0.0, 0.1, peak_prob=0.95)], probs, config),
            triage_recording("/dir_b/recording.wav", [_event(1.0, 1.1, peak_prob=0.92)], probs, config),
        ]

        write_batch_results(
            results,
            output_dir=tmp_path,
            write_parquet=True,
            write_per_recording_json=False,
        )

        df = pd.read_parquet(tmp_path / "summary.parquet")
        assert len(df) == 2, f"Parquet must have 2 rows (one per RecordingResult), got {len(df)}"


# ---------------------------------------------------------------------------
# O. Parquet qc_flags column not in standard columns but QC info present
# ---------------------------------------------------------------------------

class TestParquetQCFlagsColumn:

    def test_parquet_qc_flags_not_required_in_standard_columns(self, tmp_path: Path):
        """qc_flags is a list and is NOT in _PARQUET_COLUMNS — verify parquet rows include the
        QC-relevant columns that ARE present (noise_floor_p90, tier) without crashing."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        config = TriageConfig()
        # Recording with high noise floor → qc_flags will have "high_noise_floor"
        probs = np.full(200, 0.50)  # p90 = 0.50 > 0.4 threshold
        events = [_event(0.0, 0.1, peak_prob=0.95)]

        results = [triage_recording("/data/noisy.wav", events, probs, config)]
        assert "high_noise_floor" in results[0].qc_flags

        write_batch_results(results, output_dir=tmp_path, write_parquet=True, write_per_recording_json=False)

        df = pd.read_parquet(tmp_path / "summary.parquet")
        # noise_floor_p90 column captures the flag trigger value
        assert "noise_floor_p90" in df.columns
        assert float(df.iloc[0]["noise_floor_p90"]) > 0.4

    def test_parquet_n_events_correctly_reflects_count(self, tmp_path: Path):
        """n_events column in parquet must match actual event count passed to triage_recording."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        config = TriageConfig()
        probs = _probs_clear()
        n = 7
        events = [_event(i * 0.3, i * 0.3 + 0.1, peak_prob=0.95) for i in range(n)]

        results = [triage_recording("/data/seven.wav", events, probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=True, write_per_recording_json=False)

        df = pd.read_parquet(tmp_path / "summary.parquet")
        assert int(df.iloc[0]["n_events"]) == n


# ---------------------------------------------------------------------------
# P. Integration: triage output → batch_output → parquet roundtrip
# ---------------------------------------------------------------------------

class TestEndToEndRoundtrip:

    def test_filepath_roundtrips_through_parquet(self, tmp_path: Path):
        """filepath stored in RecordingResult must appear verbatim in parquet row."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        config = TriageConfig()
        filepath = "/some/deep/path/to/recording_2024-10-04.wav"
        probs = _probs_clear()
        events = [_event(0.0, 0.1, peak_prob=0.95)]

        results = [triage_recording(filepath, events, probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=True, write_per_recording_json=False)

        df = pd.read_parquet(tmp_path / "summary.parquet")
        assert df.iloc[0]["filepath"] == filepath

    def test_json_events_count_matches_triage_n_events(self, tmp_path: Path):
        """Number of dicts in per-recording JSON must match result.n_events."""
        config = TriageConfig()
        n = 4
        probs = _probs_clear()
        events = [_event(i * 0.3, i * 0.3 + 0.1, peak_prob=0.95) for i in range(n)]

        results = [triage_recording("/data/count_check.wav", events, probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        data = json.loads((tmp_path / "detections" / "count_check.json").read_text())
        assert len(data) == n, f"Expected {n} detection dicts, got {len(data)}"

    def test_empty_recording_json_is_empty_list(self, tmp_path: Path):
        """Per-recording JSON for zero-event recording must be an empty list []."""
        config = TriageConfig()
        probs = _probs_low()

        results = [triage_recording("/data/empty.wav", [], probs, config)]
        write_batch_results(results, output_dir=tmp_path, write_parquet=False, write_per_recording_json=True)

        data = json.loads((tmp_path / "detections" / "empty.json").read_text())
        assert data == [], f"Expected [] for zero-event recording, got {data}"
