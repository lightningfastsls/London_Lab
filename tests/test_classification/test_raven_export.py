"""Tests for Raven selection-table export adapter.

Covers: config validation, JSON loading, skip logic, WAV↔detection mapping,
DataFrame conversion, TSV output formatting, and end-to-end integration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Bootstrap sys.path — tests live in tests/test_classification/
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classification.raven_export import (
    ExportSummary,
    RavenExportConfig,
    _is_detection_json,
    detections_to_raven_table,
    discover_wav_detection_mapping,
    export_raven_tables,
    load_detection_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detection_json_factory(tmp_path: Path):
    """Factory that creates detection JSON files on disk.

    Returns a callable:
        (subdir_name, start_s, end_s, index=0) -> Path
    """

    def _create(
        subdir_name: str,
        start_s: float,
        end_s: float,
        index: int = 0,
    ) -> Path:
        det_dir = tmp_path / "detections" / subdir_name
        det_dir.mkdir(parents=True, exist_ok=True)

        duration_ms = (end_s - start_s) * 1000
        filename = f"detection_{index:03d}_{start_s:.3f}s-{end_s:.3f}s.json"
        data = {
            "detection_index": index,
            "core_time": {
                "start_s": start_s,
                "end_s": end_s,
                "duration_ms": duration_ms,
            },
            "saved_region": {
                "start_s": start_s - 0.05,
                "end_s": end_s + 0.05,
                "context_ms": 50.0,
            },
            "probabilities": {
                "max": 0.95,
                "mean": 0.85,
                "original_cnn_probability": 0.90,
            },
            "spectrogram_columns": {"start_col": 10, "end_col": 20},
            "user_action": None,
            "session": {
                "session_id": "test-session",
                "threshold_preset": "default",
                "threshold_high": 0.7,
                "threshold_low": 0.3,
            },
            "timestamp": "2026-01-01T00:00:00",
        }

        out = det_dir / filename
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out

    return _create


@pytest.fixture
def wav_dir(tmp_path: Path) -> Path:
    """Create a temporary wav_dir with dummy .wav files (0-byte placeholders)."""
    d = tmp_path / "wavs"
    d.mkdir()
    for name in ("rec_001.wav", "rec_002.wav", "rec_003.wav"):
        (d / name).write_bytes(b"")
    return d


@pytest.fixture
def populated_dirs(tmp_path: Path, detection_json_factory, wav_dir: Path):
    """Set up a realistic directory structure with multiple WAVs + detections.

    Returns (detections_dir, wav_dir, output_dir).
    """
    # rec_001: 3 detections
    detection_json_factory("rec_001", 0.10, 0.15, index=0)
    detection_json_factory("rec_001", 0.30, 0.38, index=1)
    detection_json_factory("rec_001", 0.50, 0.55, index=2)

    # rec_002: 1 detection
    detection_json_factory("rec_002", 1.00, 1.05, index=0)

    # rec_003: empty dir (no JSONs — should be skipped)
    (tmp_path / "detections" / "rec_003").mkdir(parents=True, exist_ok=True)

    output_dir = tmp_path / "output"
    return tmp_path / "detections", wav_dir, output_dir


# ===================================================================
# Config validation
# ===================================================================


class TestRavenExportConfig:
    def test_valid_defaults(self, tmp_path: Path) -> None:
        cfg = RavenExportConfig(
            detections_dir=tmp_path,
            wav_dir=tmp_path,
            output_dir=tmp_path / "out",
        )
        assert cfg.low_freq_hz == 25_000.0
        assert cfg.high_freq_hz == 125_000.0

    def test_low_ge_high_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be less than"):
            RavenExportConfig(
                detections_dir=tmp_path,
                wav_dir=tmp_path,
                output_dir=tmp_path / "out",
                low_freq_hz=130_000,
                high_freq_hz=125_000,
            )

    def test_negative_freq_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RavenExportConfig(
                detections_dir=tmp_path,
                wav_dir=tmp_path,
                output_dir=tmp_path / "out",
                low_freq_hz=-1000,
            )

    def test_string_paths_converted(self, tmp_path: Path) -> None:
        cfg = RavenExportConfig(
            detections_dir=str(tmp_path),
            wav_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        assert isinstance(cfg.detections_dir, Path)
        assert isinstance(cfg.wav_dir, Path)
        assert isinstance(cfg.output_dir, Path)


# ===================================================================
# JSON loading
# ===================================================================


class TestLoadDetectionJson:
    def test_valid_load(self, detection_json_factory) -> None:
        path = detection_json_factory("test_wav", 0.14165, 0.20139, index=0)
        result = load_detection_json(path)
        assert result["start_s"] == pytest.approx(0.14165)
        assert result["end_s"] == pytest.approx(0.20139)
        assert "duration_ms" in result

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_detection_json(bad)

    def test_missing_core_time_raises(self, tmp_path: Path) -> None:
        no_core = tmp_path / "no_core.json"
        no_core.write_text(
            json.dumps({"detection_index": 0, "timestamp": "2026-01-01"}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Missing 'core_time'"):
            load_detection_json(no_core)


# ===================================================================
# Skip logic
# ===================================================================


class TestSkipLogic:
    def test_saved_tracking_skipped(self) -> None:
        assert not _is_detection_json(Path("_saved_tracking.json"))

    def test_png_skipped(self) -> None:
        assert not _is_detection_json(Path("spectrogram.png"))

    def test_csv_skipped(self) -> None:
        assert not _is_detection_json(Path("detections_summary.csv"))

    def test_valid_json_accepted(self) -> None:
        assert _is_detection_json(Path("detection_000_0.142s-0.201s.json"))

    def test_non_json_skipped(self) -> None:
        assert not _is_detection_json(Path("notes.txt"))


# ===================================================================
# Discovery / mapping
# ===================================================================


class TestDiscoverMapping:
    def test_finds_matching_dirs(
        self, populated_dirs, wav_dir: Path
    ) -> None:
        det_dir, wav_dir, _ = populated_dirs
        mapping = discover_wav_detection_mapping(det_dir, wav_dir)
        assert "rec_001" in mapping
        assert "rec_002" in mapping
        assert len(mapping["rec_001"]) == 3
        assert len(mapping["rec_002"]) == 1

    def test_empty_dir_not_in_mapping(
        self, populated_dirs, wav_dir: Path
    ) -> None:
        det_dir, wav_dir, _ = populated_dirs
        mapping = discover_wav_detection_mapping(det_dir, wav_dir)
        # rec_003 has an empty detection dir — should NOT appear.
        assert "rec_003" not in mapping

    def test_unmatched_dir_logged(
        self, tmp_path: Path, wav_dir: Path, caplog
    ) -> None:
        det_dir = tmp_path / "detections"
        orphan = det_dir / "orphan_recording"
        orphan.mkdir(parents=True)
        # Put a JSON in the orphan dir so it's not empty.
        (orphan / "det.json").write_text(
            json.dumps({"core_time": {"start_s": 0, "end_s": 1, "duration_ms": 1000}}),
            encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            mapping = discover_wav_detection_mapping(det_dir, wav_dir)
        assert "orphan_recording" not in mapping
        assert "Unmapped" in caplog.text

    def test_missing_detections_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_wav_detection_mapping(
                tmp_path / "nonexistent", tmp_path
            )

    def test_missing_wav_dir_raises(self, tmp_path: Path) -> None:
        det_dir = tmp_path / "detections"
        det_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="WAV directory"):
            discover_wav_detection_mapping(det_dir, tmp_path / "nonexistent_wavs")

    def test_prefix_match_prefers_longest_stem(
        self, tmp_path: Path, detection_json_factory
    ) -> None:
        """Longer WAV stem wins prefix match over shorter one."""
        wav_d = tmp_path / "wavs"
        wav_d.mkdir()
        (wav_d / "rec_001.wav").write_bytes(b"")
        (wav_d / "rec_001_retry.wav").write_bytes(b"")

        # Subdir name matches the longer stem exactly.
        detection_json_factory("rec_001_retry", 0.10, 0.15, index=0)

        det_dir = tmp_path / "detections"
        mapping = discover_wav_detection_mapping(det_dir, wav_d)
        assert "rec_001_retry" in mapping
        assert "rec_001" not in mapping


# ===================================================================
# Conversion
# ===================================================================


class TestDetectionsToRavenTable:
    def test_sorted_by_begin_time(self) -> None:
        dets = [
            {"start_s": 0.50, "end_s": 0.55},
            {"start_s": 0.10, "end_s": 0.15},
            {"start_s": 0.30, "end_s": 0.35},
        ]
        df = detections_to_raven_table(dets)
        times = df["Begin Time (s)"].tolist()
        assert times == [0.10, 0.30, 0.50]

    def test_one_indexed_selections(self) -> None:
        dets = [
            {"start_s": 0.1, "end_s": 0.2},
            {"start_s": 0.3, "end_s": 0.4},
        ]
        df = detections_to_raven_table(dets)
        assert df["Selection"].tolist() == [1, 2]

    def test_freq_bounds_applied(self) -> None:
        dets = [{"start_s": 0.1, "end_s": 0.2}]
        df = detections_to_raven_table(dets, low_freq_hz=30_000, high_freq_hz=110_000)
        assert df["Low Freq (Hz)"].iloc[0] == 30_000
        assert df["High Freq (Hz)"].iloc[0] == 110_000

    def test_correct_columns(self) -> None:
        dets = [{"start_s": 0.1, "end_s": 0.2}]
        df = detections_to_raven_table(dets)
        expected = [
            "Selection",
            "View",
            "Channel",
            "Begin Time (s)",
            "End Time (s)",
            "Low Freq (Hz)",
            "High Freq (Hz)",
        ]
        assert list(df.columns) == expected

    def test_view_and_channel_values(self) -> None:
        dets = [{"start_s": 0.1, "end_s": 0.2}]
        df = detections_to_raven_table(dets)
        assert df["View"].iloc[0] == "Spectrogram 1"
        assert df["Channel"].iloc[0] == 1

    def test_time_rounding(self) -> None:
        dets = [{"start_s": 0.141650001, "end_s": 0.201390009}]
        df = detections_to_raven_table(dets)
        assert df["Begin Time (s)"].iloc[0] == 0.1417
        assert df["End Time (s)"].iloc[0] == 0.2014


# ===================================================================
# TSV output
# ===================================================================


class TestTsvOutput:
    def test_tab_delimited_header(self, tmp_path: Path) -> None:
        dets = [{"start_s": 0.1417, "end_s": 0.2014}]
        df = detections_to_raven_table(dets)
        out = tmp_path / "test.Table.1.selections.txt"
        df.to_csv(out, sep="\t", index=False, lineterminator="\n")

        content = out.read_text(encoding="utf-8")
        lines = content.split("\n")
        # Header line should be tab-separated.
        header_fields = lines[0].split("\t")
        assert header_fields[0] == "Selection"
        assert header_fields[3] == "Begin Time (s)"
        assert len(header_fields) == 7

    def test_data_row_values(self, tmp_path: Path) -> None:
        dets = [{"start_s": 0.1417, "end_s": 0.2014}]
        df = detections_to_raven_table(dets)
        out = tmp_path / "test.Table.1.selections.txt"
        df.to_csv(out, sep="\t", index=False, lineterminator="\n")

        content = out.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        fields = lines[1].split("\t")
        assert fields[0] == "1"  # Selection
        assert fields[1] == "Spectrogram 1"  # View
        assert fields[2] == "1"  # Channel
        assert fields[3] == "0.1417"  # Begin Time
        assert fields[4] == "0.2014"  # End Time
        assert fields[5] == "25000"  # Low Freq
        assert fields[6] == "125000"  # High Freq


# ===================================================================
# Integration
# ===================================================================


class TestExportIntegration:
    def test_end_to_end_export(self, populated_dirs) -> None:
        det_dir, wav_dir, output_dir = populated_dirs
        config = RavenExportConfig(
            detections_dir=det_dir,
            wav_dir=wav_dir,
            output_dir=output_dir,
        )
        created = export_raven_tables(config)

        # Two WAVs with detections -> two selection tables.
        assert len(created) == 2

        # Check filenames follow convention.
        names = {p.name for p in created}
        assert "rec_001.Table.1.selections.txt" in names
        assert "rec_002.Table.1.selections.txt" in names

        # Verify rec_001 has 3 rows.
        rec1 = output_dir / "rec_001.Table.1.selections.txt"
        df = pd.read_csv(rec1, sep="\t")
        assert len(df) == 3
        assert df["Selection"].tolist() == [1, 2, 3]

        # Verify summary file was written.
        summary_path = output_dir / "export_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["total_tables_written"] == 2
        assert summary["total_detections"] == 4
        assert summary["unmapped_count"] == 0
        assert summary["unmapped_dirs"] == []

    def test_output_dir_created_if_missing(self, populated_dirs) -> None:
        """Verify that export_raven_tables creates output_dir when it doesn't exist."""
        det_dir, wav_dir, output_dir = populated_dirs
        assert not output_dir.exists()

        config = RavenExportConfig(
            detections_dir=det_dir,
            wav_dir=wav_dir,
            output_dir=output_dir,
        )
        export_raven_tables(config)
        assert output_dir.exists()

    def test_malformed_json_skipped_gracefully(
        self, populated_dirs, tmp_path: Path
    ) -> None:
        det_dir, wav_dir, output_dir = populated_dirs

        # Add a malformed JSON to rec_001's directory.
        bad_json = det_dir / "rec_001" / "bad_detection.json"
        bad_json.write_text("{{corrupt", encoding="utf-8")

        config = RavenExportConfig(
            detections_dir=det_dir,
            wav_dir=wav_dir,
            output_dir=output_dir,
        )
        # Should not raise — the bad file is skipped.
        created = export_raven_tables(config)
        assert len(created) == 2

        # rec_001 should still have its 3 valid detections.
        rec1 = output_dir / "rec_001.Table.1.selections.txt"
        df = pd.read_csv(rec1, sep="\t")
        assert len(df) == 3

    def test_detections_dir_not_found_raises(self, tmp_path: Path) -> None:
        config = RavenExportConfig(
            detections_dir=tmp_path / "nonexistent",
            wav_dir=tmp_path,
            output_dir=tmp_path / "out",
        )
        with pytest.raises(FileNotFoundError):
            export_raven_tables(config)


# ===================================================================
# ExportSummary
# ===================================================================


class TestExportSummary:
    def test_to_dict_roundtrip(self) -> None:
        summary = ExportSummary(
            total_wav_files=5,
            total_detections=42,
            total_tables_written=4,
            unmapped_dirs=["orphan_1"],
            empty_detection_dirs=["rec_003"],
            per_wav_counts={"rec_001": 10, "rec_002": 32},
        )
        d = summary.to_dict()
        assert d["total_wav_files"] == 5
        assert d["total_detections"] == 42
        assert d["unmapped_count"] == 1
        assert d["unmapped_dirs"] == ["orphan_1"]
        assert d["empty_detection_dirs"] == ["rec_003"]
        assert d["per_wav_counts"]["rec_001"] == 10

    def test_empty_matched_dirs_not_in_unmapped(self, populated_dirs) -> None:
        """rec_003 has a matching WAV but no detections — should appear in
        empty_detection_dirs, NOT unmapped_dirs."""
        det_dir, wav_dir, output_dir = populated_dirs
        config = RavenExportConfig(
            detections_dir=det_dir,
            wav_dir=wav_dir,
            output_dir=output_dir,
        )
        export_raven_tables(config)

        summary_path = output_dir / "export_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        assert summary["unmapped_dirs"] == []
        assert "rec_003" in summary["empty_detection_dirs"]


# ===================================================================
# CLI dry-run
# ===================================================================


class TestCliDryRun:
    def test_dry_run_writes_no_files(self, populated_dirs, monkeypatch, capsys) -> None:
        """--dry-run should report mapping but write no output files."""
        det_dir, wav_dir, output_dir = populated_dirs

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export_raven_tables.py",
                "--detections-dir", str(det_dir),
                "--wav-dir", str(wav_dir),
                "--output-dir", str(output_dir),
                "--dry-run",
            ],
        )

        # Import the CLI main function.
        import importlib
        script_path = REPO_ROOT / "scripts"
        if str(script_path) not in sys.path:
            sys.path.insert(0, str(script_path))
        import export_raven_tables as cli_module
        importlib.reload(cli_module)  # Ensure fresh import after monkeypatch.

        rc = cli_module.main()
        assert rc == 0

        # No output files should have been written.
        assert not output_dir.exists() or not any(output_dir.glob("*.txt"))

        captured = capsys.readouterr()
        assert "2 WAV files matched" in captured.out
