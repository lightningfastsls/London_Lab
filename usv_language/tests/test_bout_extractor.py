"""Tests for bout_extractor module."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.data.bout_extractor import Bout, BoutExtractionConfig, BoutExtractor


# ---- Config tests ----


class TestBoutExtractionConfig:
    def test_default_values(self):
        config = BoutExtractionConfig()
        assert config.bout_gap_threshold_ms == 500.0
        assert config.context_padding_ms == 200.0
        assert config.min_bout_duration_ms == 50.0
        assert config.max_bout_duration_ms == 10000.0

    def test_frozen(self):
        config = BoutExtractionConfig()
        with pytest.raises(AttributeError):
            config.bout_gap_threshold_ms = 100.0  # type: ignore[misc]

    def test_validation_gap_threshold(self):
        with pytest.raises(ValueError, match="bout_gap_threshold_ms"):
            BoutExtractionConfig(bout_gap_threshold_ms=-1.0)

    def test_validation_max_less_than_min(self):
        with pytest.raises(ValueError, match="max_bout_duration_ms"):
            BoutExtractionConfig(
                min_bout_duration_ms=100.0,
                max_bout_duration_ms=50.0,
            )


# ---- Bout grouping tests ----


class TestBoutExtractor:
    def test_groups_usvs_within_gap_threshold(self):
        """USVs closer than 500ms gap should be in the same bout."""
        extractor = BoutExtractor()
        # Three USVs within 100ms gaps
        times = [(1.0, 1.05), (1.15, 1.20), (1.30, 1.35)]
        bouts = extractor._group_into_bouts("test.wav", times)
        assert len(bouts) == 1
        assert bouts[0].usv_count == 3

    def test_starts_new_bout_when_gap_exceeds_threshold(self):
        """USVs separated by > 500ms gap should be in different bouts."""
        extractor = BoutExtractor()
        # Two clusters separated by 600ms
        times = [(1.0, 1.05), (1.10, 1.15), (1.75, 1.80), (1.85, 1.90)]
        bouts = extractor._group_into_bouts("test.wav", times)
        assert len(bouts) == 2
        assert bouts[0].usv_count == 2
        assert bouts[1].usv_count == 2

    def test_adds_padding_and_clamps_to_zero(self):
        """Bout start should be padded by 200ms but clamped to 0."""
        extractor = BoutExtractor(BoutExtractionConfig(context_padding_ms=200.0))
        times = [(0.1, 0.2)]
        bouts = extractor._group_into_bouts("test.wav", times)
        assert len(bouts) == 1
        # start = 0.1 - 0.2 = -0.1, clamped to 0
        assert bouts[0].start_time_s == 0.0
        # end = 0.2 + 0.2 = 0.4
        assert abs(bouts[0].end_time_s - 0.4) < 1e-9

    def test_splits_bouts_exceeding_max_duration(self):
        """Bouts > 10s should be split at largest gap."""
        config = BoutExtractionConfig(
            max_bout_duration_ms=1000.0,  # 1s max for easier testing
            context_padding_ms=0.0,       # no padding to simplify
            min_bout_duration_ms=10.0,
        )
        extractor = BoutExtractor(config)
        # 6 USVs spread over 2s with a 300ms gap in the middle
        times = [
            (0.0, 0.1), (0.2, 0.3), (0.4, 0.5),
            (0.8, 0.9), (1.5, 1.6), (1.7, 1.8),
        ]
        bouts = extractor._group_into_bouts("test.wav", times)
        # Should split somewhere — resulting bouts should each be <= 1s
        assert len(bouts) >= 2
        for bout in bouts:
            assert bout.duration_s <= 1.0 + 1e-9

    def test_discards_bouts_shorter_than_min_duration(self):
        """Bouts under 50ms after padding should be discarded."""
        config = BoutExtractionConfig(
            context_padding_ms=0.0,
            min_bout_duration_ms=100.0,
        )
        extractor = BoutExtractor(config)
        # Single USV of 10ms
        times = [(1.0, 1.01)]
        bouts = extractor._group_into_bouts("test.wav", times)
        assert len(bouts) == 0

    def test_handles_single_usv_bout(self):
        """A single USV should produce one bout if above min duration."""
        extractor = BoutExtractor()
        times = [(1.0, 1.1)]
        bouts = extractor._group_into_bouts("test.wav", times)
        assert len(bouts) == 1
        assert bouts[0].usv_count == 1
        assert bouts[0].usv_times == [(1.0, 1.1)]

    def test_handles_empty_detections(self):
        """Empty detection list should return no bouts."""
        extractor = BoutExtractor()
        bouts = extractor._group_into_bouts("test.wav", [])
        assert bouts == []


# ---- Input format tests ----


class TestBoutExtractorCSV:
    def test_parses_csv_format(self, tmp_path):
        """Should correctly parse CSV with wav_file, start_time_s, end_time_s."""
        csv_path = tmp_path / "detections.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wav_file", "start_time_s", "end_time_s"])
            writer.writerow(["recording_001", "1.0", "1.05"])
            writer.writerow(["recording_001", "1.10", "1.15"])
            writer.writerow(["recording_002", "2.0", "2.05"])

        extractor = BoutExtractor()
        bouts = extractor.extract_from_csv(csv_path)
        # recording_001 has 2 USVs close together -> 1 bout
        # recording_002 has 1 USV -> 1 bout
        assert len(bouts) == 2

    def test_parses_csv_with_wav_dir(self, tmp_path):
        """CSV wav_file stems should be resolved via wav_dir."""
        csv_path = tmp_path / "detections.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wav_file", "start_time_s", "end_time_s"])
            writer.writerow(["rec_001", "1.0", "1.5"])

        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()

        extractor = BoutExtractor()
        bouts = extractor.extract_from_csv(csv_path, wav_dir=wav_dir)
        assert len(bouts) == 1
        assert "rec_001.wav" in bouts[0].source_file


class TestBoutExtractorJSON:
    def test_parses_tracking_json(self, tmp_path):
        """Should parse _saved_tracking.json format."""
        json_path = tmp_path / "_saved_tracking.json"
        records = [
            {"start_time_s": 1.0, "end_time_s": 1.05,
             "output_path": "detection_000.png", "save_timestamp": "2026-01-01"},
            {"start_time_s": 1.1, "end_time_s": 1.15,
             "output_path": "detection_001.png", "save_timestamp": "2026-01-01"},
        ]
        with open(json_path, "w") as f:
            json.dump(records, f)

        extractor = BoutExtractor()
        bouts = extractor.extract_from_tracking_json(json_path, "test.wav")
        assert len(bouts) == 1
        assert bouts[0].usv_count == 2

    def test_empty_tracking_json(self, tmp_path):
        """Empty tracking JSON should return no bouts."""
        json_path = tmp_path / "_saved_tracking.json"
        with open(json_path, "w") as f:
            json.dump([], f)

        extractor = BoutExtractor()
        bouts = extractor.extract_from_tracking_json(json_path)
        assert bouts == []


# ---- Tracking JSON filtering tests ----


class TestTrackingJsonFiltering:
    """Tests for filtering deleted_by_user records from tracking JSONs."""

    def test_skips_deleted_by_user_records(self, tmp_path):
        """Records with user_action='deleted_by_user' must be excluded."""
        json_path = tmp_path / "_saved_tracking.json"
        records = [
            {"start_time_s": 1.0, "end_time_s": 1.05},
            {"start_time_s": 2.0, "end_time_s": 2.05,
             "user_action": "deleted_by_user"},
            {"start_time_s": 1.1, "end_time_s": 1.15},
        ]
        with open(json_path, "w") as f:
            json.dump(records, f)

        extractor = BoutExtractor()
        bouts = extractor.extract_from_tracking_json(json_path, "test.wav")
        # Only 2 valid records (close together) -> 1 bout with 2 USVs
        assert len(bouts) == 1
        assert bouts[0].usv_count == 2

    def test_all_deleted_returns_empty(self, tmp_path):
        """If all records are deleted, should return empty list."""
        json_path = tmp_path / "_saved_tracking.json"
        records = [
            {"start_time_s": 1.0, "end_time_s": 1.05,
             "user_action": "deleted_by_user"},
            {"start_time_s": 2.0, "end_time_s": 2.05,
             "user_action": "deleted_by_user"},
        ]
        with open(json_path, "w") as f:
            json.dump(records, f)

        extractor = BoutExtractor()
        bouts = extractor.extract_from_tracking_json(json_path, "test.wav")
        assert bouts == []

    def test_no_user_action_field_included(self, tmp_path):
        """Records without user_action field should be included (normal case)."""
        json_path = tmp_path / "_saved_tracking.json"
        records = [
            {"start_time_s": 1.0, "end_time_s": 1.05,
             "output_path": "detection_000.png"},
        ]
        with open(json_path, "w") as f:
            json.dump(records, f)

        extractor = BoutExtractor()
        bouts = extractor.extract_from_tracking_json(json_path, "test.wav")
        assert len(bouts) == 1
        assert bouts[0].usv_count == 1

    def test_parse_tracking_json_filters_deleted(self, tmp_path):
        """Internal _parse_tracking_json should also filter deleted records."""
        json_path = tmp_path / "_saved_tracking.json"
        records = [
            {"start_time_s": 1.0, "end_time_s": 1.05},
            {"start_time_s": 2.0, "end_time_s": 2.05,
             "user_action": "deleted_by_user"},
        ]
        with open(json_path, "w") as f:
            json.dump(records, f)

        extractor = BoutExtractor()
        times = extractor._parse_tracking_json(json_path)
        assert len(times) == 1
        assert times[0] == (1.0, 1.05)

    def test_detection_dir_filters_deleted(self, tmp_path):
        """extract_from_detection_dir should skip deleted records via _parse_tracking_json."""
        # Create a detection subdir with tracking JSON
        subdir = tmp_path / "rec_001"
        subdir.mkdir()
        records = [
            {"start_time_s": 1.0, "end_time_s": 1.05},
            {"start_time_s": 1.1, "end_time_s": 1.15,
             "user_action": "deleted_by_user"},
            {"start_time_s": 1.2, "end_time_s": 1.25},
        ]
        with open(subdir / "_saved_tracking.json", "w") as f:
            json.dump(records, f)

        extractor = BoutExtractor()
        bouts = extractor.extract_from_detection_dir(tmp_path)
        assert len(bouts) == 1
        assert bouts[0].usv_count == 2  # Only 2 valid records


# ---- Recursive WAV lookup tests ----


class TestRecursiveWavLookup:
    """Tests for recursive WAV path resolution with lazy index."""

    def test_flat_dir_fast_path(self, tmp_path):
        """Flat directory should resolve without building index."""
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        (wav_dir / "rec_001.wav").write_bytes(b"RIFF")

        extractor = BoutExtractor()
        result = extractor._resolve_wav_path("rec_001", wav_dir)
        assert result == str(wav_dir / "rec_001.wav")
        # Index should NOT have been built (fast path hit)
        assert extractor._wav_index is None

    def test_nested_dir_builds_index(self, tmp_path):
        """Nested directory should trigger index build and find file."""
        wav_dir = tmp_path / "wavs"
        nested = wav_dir / "session_1" / "batch_2"
        nested.mkdir(parents=True)
        (nested / "rec_001.wav").write_bytes(b"RIFF")

        extractor = BoutExtractor()
        result = extractor._resolve_wav_path("rec_001", wav_dir)
        assert result == str(nested / "rec_001.wav")
        # Index should have been built
        assert extractor._wav_index is not None
        assert "rec_001" in extractor._wav_index

    def test_index_cached_across_calls(self, tmp_path):
        """Index should be built once and reused."""
        wav_dir = tmp_path / "wavs"
        nested = wav_dir / "sub"
        nested.mkdir(parents=True)
        (nested / "rec_001.wav").write_bytes(b"RIFF")
        (nested / "rec_002.wav").write_bytes(b"RIFF")

        extractor = BoutExtractor()
        # First call builds index
        extractor._resolve_wav_path("rec_001", wav_dir)
        index_id = id(extractor._wav_index)

        # Second call reuses same index object
        extractor._resolve_wav_path("rec_002", wav_dir)
        assert id(extractor._wav_index) == index_id

    def test_missing_wav_returns_flat_path(self, tmp_path):
        """Missing WAV should fall back to flat path string."""
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()

        extractor = BoutExtractor()
        result = extractor._resolve_wav_path("nonexistent", wav_dir)
        assert result == str(wav_dir / "nonexistent.wav")
