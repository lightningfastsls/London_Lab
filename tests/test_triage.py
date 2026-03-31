"""Tests for recording-level triage and batch output — written by test-architect BEFORE implementation.

ROADMAP test plan coverage (section 15.7):
  1. All events > 0.90 peak -> auto_accept         -> test_all_high_confidence_events_auto_accept
  2. No windows > 0.10 -> auto_reject              -> test_no_signal_windows_auto_reject
  3. Mixed confidence -> manual_review             -> test_mixed_confidence_manual_review
  4. Outlier event count (z > 2) -> flagged        -> test_outlier_event_count_flagged
  5. High noise floor (p90 > 0.4) -> flagged       -> test_high_noise_floor_flagged
  6. Parquet output has expected columns/row count  -> test_parquet_output_columns_and_row_count
  7. Per-recording JSON matches ADR-010 format      -> test_per_recording_json_adr010_format
  8. Batch processes multiple recordings            -> test_batch_processes_multiple_recordings

Additional coverage (recurring gap patterns):
  - Empty events list                              -> test_triage_empty_events_list
  - TriageConfig validation (bad values)           -> test_triage_config_rejects_invalid_thresholds
  - TriageConfig validation (inverted thresholds)  -> test_triage_config_rejects_inverted_thresholds
  - RecordingResult field correctness              -> test_recording_result_fields_computed_correctly
  - write_batch_results with empty results list    -> test_write_batch_results_empty_list
  - write_batch_results creates directory structure-> test_write_batch_results_creates_detections_subdir
  - auto_reject has zero events                    -> test_auto_reject_has_no_events
  - qc_flags is a list (not None)                  -> test_qc_flags_always_a_list
  - noise_floor_p90 computed from probabilities    -> test_noise_floor_p90_computed_from_probabilities
  - batch_stats=None does not crash                -> test_triage_without_batch_stats_does_not_crash
  - tier is one of the three valid values          -> test_tier_is_one_of_three_valid_values

Total: 18 tests (8 from ROADMAP, 10 additional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Deferred imports — collected lazily so each test fails individually rather
# than the whole module failing at collection time before the module exists.
# ---------------------------------------------------------------------------

def _import_triage():
    """Return (RecordingResult, TriageConfig, triage_recording) or raise ImportError."""
    from usv_spectrogram.postprocessing.triage import (  # noqa: PLC0415
        RecordingResult,
        TriageConfig,
        triage_recording,
    )
    return RecordingResult, TriageConfig, triage_recording


def _import_batch_output():
    """Return write_batch_results or raise ImportError."""
    from usv_spectrogram.postprocessing.batch_output import write_batch_results  # noqa: PLC0415
    return write_batch_results


# USVEvent is already implemented — safe to import at module level.
from usv_spectrogram.postprocessing.hysteresis import USVEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    start_s: float,
    end_s: float,
    peak_prob: float = 0.95,
    mean_prob: float = 0.90,
    n_windows: int = 8,
) -> USVEvent:
    """Create a minimal synthetic USVEvent for triage tests."""
    probs = np.full(n_windows, mean_prob)
    probs[n_windows // 2] = peak_prob
    return USVEvent(
        start_window=0,
        end_window=n_windows - 1,
        start_time_s=start_s,
        end_time_s=end_s,
        duration_ms=(end_s - start_s) * 1000.0,
        peak_probability=peak_prob,
        mean_probability=mean_prob,
        window_count=n_windows,
        probabilities=probs,
    )


def _make_high_confidence_events(n: int = 3) -> List[USVEvent]:
    """Return n events all with peak_probability >= 0.91."""
    return [
        _make_event(
            start_s=i * 0.5,
            end_s=i * 0.5 + 0.1,
            peak_prob=0.95,
            mean_prob=0.92,
        )
        for i in range(n)
    ]


def _make_mixed_confidence_events() -> List[USVEvent]:
    """Return events with a range of confidence scores (not all above 0.90)."""
    return [
        _make_event(0.0, 0.1, peak_prob=0.95, mean_prob=0.92),   # high
        _make_event(0.5, 0.6, peak_prob=0.65, mean_prob=0.55),   # low
        _make_event(1.0, 1.1, peak_prob=0.80, mean_prob=0.72),   # mid
    ]


def _make_probs_high_confidence(n_windows: int = 200) -> np.ndarray:
    """All windows show clear signal; most are noise-level, a few are USV-level."""
    probs = np.full(n_windows, 0.05)
    probs[50:60] = 0.95
    return probs


def _make_probs_no_signal(n_windows: int = 200) -> np.ndarray:
    """All windows well below auto_reject_max_window (0.10)."""
    return np.full(n_windows, 0.03)


def _make_probs_high_noise_floor(n_windows: int = 200) -> np.ndarray:
    """90th percentile of window probabilities > 0.4 (simulates noisy recording)."""
    return np.full(n_windows, 0.50)


# ---------------------------------------------------------------------------
# ROADMAP Test 1: All events > 0.90 peak → auto_accept
# ---------------------------------------------------------------------------

def test_all_high_confidence_events_auto_accept():
    """Spec: recording with all detected events having peak_prob >= auto_accept_min_peak (0.90) is auto_accept."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    events = _make_high_confidence_events(n=4)
    # All peak probabilities are 0.95, which exceeds default auto_accept_min_peak=0.90
    probs = _make_probs_high_confidence(n_windows=300)
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/recording_001.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    assert result.tier == "auto_accept", (
        f"Expected 'auto_accept' for all-high-confidence events, got '{result.tier}'"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 2: No windows > 0.10 → auto_reject
# ---------------------------------------------------------------------------

def test_no_signal_windows_auto_reject():
    """Spec: recording where no window probability exceeds auto_reject_max_window (0.10) is auto_reject."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    probs = _make_probs_no_signal(n_windows=200)
    events: List[USVEvent] = []
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/recording_002.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    assert result.tier == "auto_reject", (
        f"Expected 'auto_reject' for all-noise recording, got '{result.tier}'"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 3: Mixed confidence → manual_review
# ---------------------------------------------------------------------------

def test_mixed_confidence_manual_review():
    """Spec: recording with events of varying confidence (not all above 0.90) → manual_review."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    events = _make_mixed_confidence_events()
    # Some events have peak_prob < 0.90 so not auto_accept; some windows > 0.10 so not auto_reject
    probs = np.array([0.05] * 100 + [0.65] * 10 + [0.95] * 10 + [0.05] * 80)
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/recording_003.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    assert result.tier == "manual_review", (
        f"Expected 'manual_review' for mixed-confidence events, got '{result.tier}'"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 4: Outlier event count (z > 2) → flagged
# ---------------------------------------------------------------------------

def test_outlier_event_count_flagged():
    """Spec: event count more than outlier_count_zscore (default 2.0) std devs above batch mean → flagged."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    # 20 events — far above a batch where mean=3, std=1 → z = (20-3)/1 = 17 >> 2.0
    events = _make_high_confidence_events(n=20)
    probs = np.concatenate(
        [np.full(10, 0.95) for _ in range(20)] + [np.full(100, 0.05)]
    )

    config = TriageConfig()
    batch_stats = {"event_count_mean": 3.0, "event_count_std": 1.0}

    result = triage_recording(
        filepath="/data/recording_004.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=batch_stats,
    )

    # Must have a flag indicating outlier event count
    assert any(
        "outlier" in flag.lower() or "count" in flag.lower()
        for flag in result.qc_flags
    ), f"Expected outlier-count QC flag but got qc_flags={result.qc_flags}"


# ---------------------------------------------------------------------------
# ROADMAP Test 5: High noise floor (p90 > 0.4) → flagged
# ---------------------------------------------------------------------------

def test_high_noise_floor_flagged():
    """Spec: 90th-percentile window probability > 0.4 triggers a noise-floor QC flag."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    probs = _make_probs_high_noise_floor(n_windows=200)
    events = _make_high_confidence_events(n=2)
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/recording_005.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    # p90 of probs is 0.50 > 0.40, so a noise-floor flag must appear
    assert result.noise_floor_p90 > 0.40, (
        f"Expected noise_floor_p90 > 0.40, got {result.noise_floor_p90}"
    )
    assert any(
        "noise" in flag.lower() or "floor" in flag.lower()
        for flag in result.qc_flags
    ), f"Expected noise-floor QC flag but got qc_flags={result.qc_flags}"


# ---------------------------------------------------------------------------
# ROADMAP Test 6: Parquet output has expected columns and row count
# ---------------------------------------------------------------------------

def test_parquet_output_columns_and_row_count(tmp_path: Path):
    """Spec: summary.parquet has one row per recording with QC metrics columns."""
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        pytest.skip("pandas not available")

    RecordingResult, TriageConfig, triage_recording = _import_triage()
    write_batch_results = _import_batch_output()

    config = TriageConfig()
    results = [
        triage_recording("/rec/a.wav", _make_high_confidence_events(3), _make_probs_high_confidence(200), config),
        triage_recording("/rec/b.wav", [], _make_probs_no_signal(200), config),
    ]

    write_batch_results(results, output_dir=tmp_path, write_parquet=True, write_per_recording_json=False)

    parquet_path = tmp_path / "summary.parquet"
    assert parquet_path.exists(), "summary.parquet was not created"

    df = pd.read_parquet(parquet_path)
    assert len(df) == 2, f"Expected 2 rows (one per recording), got {len(df)}"

    expected_columns = {
        "filepath",
        "tier",
        "n_events",
        "max_confidence",
        "mean_event_confidence",
        "total_usv_duration_ms",
        "noise_floor_p90",
        "confidence_score",
    }
    missing = expected_columns - set(df.columns)
    assert not missing, f"Parquet missing expected columns: {missing}"


# ---------------------------------------------------------------------------
# ROADMAP Test 7: Per-recording JSON matches ADR-010 format
# ---------------------------------------------------------------------------

def test_per_recording_json_adr010_format(tmp_path: Path):
    """Spec: per-recording JSON in detections/<stem>.json has ADR-010 compatible structure.

    ADR-010 detection dict keys (from convert_to_detection_format in hysteresis.py):
      start_time_s, end_time_s, duration_s, max_probability, mean_probability
    """
    RecordingResult, TriageConfig, triage_recording = _import_triage()
    write_batch_results = _import_batch_output()

    config = TriageConfig()
    events = _make_high_confidence_events(n=2)
    probs = _make_probs_high_confidence(200)

    results = [
        triage_recording("/data/rec_001.wav", events, probs, config),
    ]

    write_batch_results(
        results,
        output_dir=tmp_path,
        write_parquet=False,
        write_per_recording_json=True,
    )

    # Expect detections/rec_001.json
    json_path = tmp_path / "detections" / "rec_001.json"
    assert json_path.exists(), f"Expected per-recording JSON at {json_path}"

    with open(json_path) as f:
        data = json.load(f)

    # Must be a list of detection dicts
    assert isinstance(data, list), f"Expected list of detections, got {type(data)}"

    if len(data) > 0:
        detection = data[0]
        adr010_keys = {
            "start_time_s",
            "end_time_s",
            "duration_s",
            "max_probability",
            "mean_probability",
        }
        missing_keys = adr010_keys - set(detection.keys())
        assert not missing_keys, f"Detection dict missing ADR-010 keys: {missing_keys}"

        # Numerical sanity checks
        assert detection["start_time_s"] >= 0.0
        assert detection["end_time_s"] >= detection["start_time_s"]
        assert detection["duration_s"] == pytest.approx(
            detection["end_time_s"] - detection["start_time_s"], abs=1e-6
        )
        assert 0.0 <= detection["max_probability"] <= 1.0
        assert 0.0 <= detection["mean_probability"] <= 1.0


# ---------------------------------------------------------------------------
# ROADMAP Test 8: Batch processes multiple recordings without error
# ---------------------------------------------------------------------------

def test_batch_processes_multiple_recordings(tmp_path: Path):
    """Spec: write_batch_results handles a batch of multiple RecordingResult objects cleanly."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()
    write_batch_results = _import_batch_output()

    config = TriageConfig()
    filepaths = [f"/data/rec_{i:03d}.wav" for i in range(5)]
    results = []
    for i, fp in enumerate(filepaths):
        if i % 3 == 0:
            events = _make_high_confidence_events(n=3)
            probs = _make_probs_high_confidence(200)
        elif i % 3 == 1:
            events = []
            probs = _make_probs_no_signal(200)
        else:
            events = _make_mixed_confidence_events()
            probs = np.concatenate([np.full(80, 0.05), np.full(20, 0.65), np.full(100, 0.05)])

        results.append(triage_recording(fp, events, probs, config))

    # Must complete without exception
    write_batch_results(results, output_dir=tmp_path, write_parquet=True, write_per_recording_json=True)

    parquet_path = tmp_path / "summary.parquet"
    assert parquet_path.exists(), "summary.parquet not created during batch run"


# ---------------------------------------------------------------------------
# Additional Test: Empty events list → triage still returns RecordingResult
# ---------------------------------------------------------------------------

def test_triage_empty_events_list():
    """Edge case: zero detected events does not crash; result has n_events=0 and sensible fields."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    probs = _make_probs_no_signal(100)
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/empty.wav",
        events=[],
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    assert isinstance(result, RecordingResult)
    assert result.n_events == 0
    assert result.max_confidence == pytest.approx(0.0, abs=1e-9)
    assert result.total_usv_duration_ms == pytest.approx(0.0)
    assert result.tier in {"auto_accept", "auto_reject", "manual_review"}


# ---------------------------------------------------------------------------
# Additional Test: TriageConfig validation — bad scalar values
# ---------------------------------------------------------------------------

def test_triage_config_rejects_invalid_thresholds():
    """TriageConfig must validate that auto_accept_min_peak is in (0, 1] and
    auto_reject_max_window is non-negative."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    # auto_accept_min_peak must be > 0
    with pytest.raises((ValueError, TypeError)):
        TriageConfig(auto_accept_min_peak=0.0)

    # auto_reject_max_window must be >= 0
    with pytest.raises((ValueError, TypeError)):
        TriageConfig(auto_reject_max_window=-0.1)


# ---------------------------------------------------------------------------
# Additional Test: TriageConfig validation — inverted thresholds
# ---------------------------------------------------------------------------

def test_triage_config_rejects_inverted_thresholds():
    """auto_reject_max_window must be strictly less than auto_accept_min_peak."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    # If reject threshold exceeds accept threshold the triage logic is nonsensical
    with pytest.raises((ValueError, TypeError)):
        TriageConfig(auto_accept_min_peak=0.20, auto_reject_max_window=0.80)


# ---------------------------------------------------------------------------
# Additional Test: RecordingResult field correctness — hand-computed values
# ---------------------------------------------------------------------------

def test_recording_result_fields_computed_correctly():
    """RecordingResult computed fields must match hand-calculated expectations."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    # Two events with known peak and duration values
    # event_a: peak 0.95, duration = (0.100 - 0.0) * 1000 = 100 ms
    # event_b: peak 0.92, duration = (0.580 - 0.5) * 1000 = 80 ms
    event_a = _make_event(0.0, 0.100, peak_prob=0.95, mean_prob=0.90, n_windows=8)
    event_b = _make_event(0.5, 0.580, peak_prob=0.92, mean_prob=0.88, n_windows=6)
    events = [event_a, event_b]

    probs = _make_probs_high_confidence(300)
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/computed.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    assert result.n_events == 2
    assert result.max_confidence == pytest.approx(0.95, abs=1e-6)
    # mean_event_confidence = mean of per-event peak probabilities
    assert result.mean_event_confidence == pytest.approx((0.95 + 0.92) / 2.0, abs=1e-6)
    # total_usv_duration_ms = sum of event durations (100 ms + 80 ms = 180 ms)
    expected_total_ms = event_a.duration_ms + event_b.duration_ms
    assert result.total_usv_duration_ms == pytest.approx(expected_total_ms, abs=1e-3)
    # filepath preserved unchanged
    assert result.filepath == "/data/computed.wav"


# ---------------------------------------------------------------------------
# Additional Test: write_batch_results with empty results list
# ---------------------------------------------------------------------------

def test_write_batch_results_empty_list(tmp_path: Path):
    """write_batch_results([]) must not crash and should produce a 0-row summary.parquet."""
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        pytest.skip("pandas not available")

    write_batch_results = _import_batch_output()

    write_batch_results([], output_dir=tmp_path, write_parquet=True, write_per_recording_json=True)

    parquet_path = tmp_path / "summary.parquet"
    assert parquet_path.exists(), "summary.parquet should be created even for empty results"
    df = pd.read_parquet(parquet_path)
    assert len(df) == 0, f"Expected 0 rows for empty batch, got {len(df)}"


# ---------------------------------------------------------------------------
# Additional Test: write_batch_results creates detections/ subdirectory
# ---------------------------------------------------------------------------

def test_write_batch_results_creates_detections_subdir(tmp_path: Path):
    """write_batch_results must create detections/ subdirectory for per-recording JSONs."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()
    write_batch_results = _import_batch_output()

    config = TriageConfig()
    events = _make_high_confidence_events(n=2)
    probs = _make_probs_high_confidence(200)

    results = [
        triage_recording("/data/myrecording.wav", events, probs, config),
    ]

    write_batch_results(
        results,
        output_dir=tmp_path,
        write_parquet=False,
        write_per_recording_json=True,
    )

    detections_dir = tmp_path / "detections"
    assert detections_dir.exists(), "detections/ subdirectory was not created"
    assert detections_dir.is_dir()

    json_files = list(detections_dir.glob("*.json"))
    assert len(json_files) == 1, f"Expected 1 JSON file, found {len(json_files)}"


# ---------------------------------------------------------------------------
# Additional Test: auto_reject recording has zero events
# ---------------------------------------------------------------------------

def test_auto_reject_has_no_events():
    """A recording auto-rejected for silence should have n_events=0 and zero total duration."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    probs = _make_probs_no_signal(100)
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/silent.wav",
        events=[],
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    assert result.n_events == 0
    assert result.total_usv_duration_ms == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Additional Test: qc_flags is always a list (not None)
# ---------------------------------------------------------------------------

def test_qc_flags_always_a_list():
    """qc_flags field must always be a list, even when no flags are triggered."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    probs = _make_probs_high_confidence(200)
    config = TriageConfig()
    events = _make_high_confidence_events(n=2)

    result = triage_recording(
        filepath="/data/clean.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    assert isinstance(result.qc_flags, list), (
        f"qc_flags must be a list, got {type(result.qc_flags)}"
    )


# ---------------------------------------------------------------------------
# Additional Test: noise_floor_p90 matches hand-computed value
# ---------------------------------------------------------------------------

def test_noise_floor_p90_computed_from_probabilities():
    """noise_floor_p90 must equal numpy's 90th percentile of the passed probabilities array.

    Hand-computed: probs has 90 values at 0.05 and 10 values at 0.8.
    np.percentile(probs, 90) = 0.8 (the 90th element when sorted ascending).
    """
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    # Craft probabilities where p90 is exactly known
    probs = np.zeros(100)
    probs[:90] = 0.05   # bottom 90 values
    probs[90:] = 0.80   # top 10 values → p90 = 0.80

    config = TriageConfig()
    result = triage_recording(
        filepath="/data/p90test.wav",
        events=[],
        probabilities=probs,
        config=config,
        batch_stats=None,
    )

    expected_p90 = float(np.percentile(probs, 90))   # 0.8
    assert result.noise_floor_p90 == pytest.approx(expected_p90, abs=1e-6), (
        f"Expected noise_floor_p90 = {expected_p90:.4f}, got {result.noise_floor_p90:.4f}"
    )


# ---------------------------------------------------------------------------
# Additional Test: batch_stats=None does not crash
# ---------------------------------------------------------------------------

def test_triage_without_batch_stats_does_not_crash():
    """batch_stats=None must be handled gracefully — outlier flagging simply skipped."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    events = _make_high_confidence_events(n=5)
    probs = _make_probs_high_confidence(300)
    config = TriageConfig()

    result = triage_recording(
        filepath="/data/nobatch.wav",
        events=events,
        probabilities=probs,
        config=config,
        batch_stats=None,  # Explicitly None — should not raise
    )

    assert isinstance(result, RecordingResult)
    assert result.tier in {"auto_accept", "auto_reject", "manual_review"}


# ---------------------------------------------------------------------------
# Additional Test: tier is always one of the three valid values
# ---------------------------------------------------------------------------

def test_tier_is_one_of_three_valid_values():
    """tier must always be exactly one of: 'auto_accept', 'auto_reject', 'manual_review'."""
    RecordingResult, TriageConfig, triage_recording = _import_triage()

    valid_tiers = {"auto_accept", "auto_reject", "manual_review"}

    scenarios = [
        (_make_high_confidence_events(3), _make_probs_high_confidence(200)),
        ([], _make_probs_no_signal(200)),
        (_make_mixed_confidence_events(), np.full(200, 0.25)),
    ]

    config = TriageConfig()
    for i, (events, probs) in enumerate(scenarios):
        result = triage_recording(
            filepath=f"/data/scenario_{i}.wav",
            events=events,
            probabilities=probs,
            config=config,
            batch_stats=None,
        )
        assert result.tier in valid_tiers, (
            f"Scenario {i}: tier '{result.tier}' is not one of {valid_tiers}"
        )
