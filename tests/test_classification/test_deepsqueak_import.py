"""Tests for DeepSqueak results import and merge adapter.

Covers: column normalization, exact timestamp matching, fuzzy matching,
tolerance rejection, unmatched reporting, multi-file concatenation,
empty file handling, and output column completeness.
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

from usv_spectrogram.classification.deepsqueak_import import (
    DeepSqueakImportConfig,
    ImportSummary,
    _DS_OUTPUT_COLUMNS,
    _extract_wav_stem,
    _normalize_columns,
    export_classified_detections,
    import_deepsqueak_results,
    load_all_deepsqueak_results,
    load_deepsqueak_excel,
    load_detections_for_merge,
    merge_with_detections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ds_excel_row(
    call_id: int = 1,
    label: str = "USV",
    begin_time: float = 0.100,
    end_time: float = 0.150,
) -> dict:
    """Create a single row of DeepSqueak Excel data with raw column names."""
    return {
        "ID": call_id,
        "Label": label,
        "Begin Time (s)": begin_time,
        "End Time (s)": end_time,
        "Call Length (s)": end_time - begin_time,
        "Principal Frequency (kHz)": 65.0,
        "Low Freq (kHz)": 40.0,
        "High Freq (kHz)": 90.0,
        "Delta Freq (kHz)": 50.0,
        "Frequency Standard Deviation (kHz)": 8.5,
        "Slope (kHz/s)": 120.0,
        "Sinuosity": 1.3,
        "Mean Power (dB/Hz)": -45.0,
        "Tonality": 0.85,
        "Peak Frequency (kHz)": 70.0,
    }


@pytest.fixture
def ds_excel_factory(tmp_path: Path):
    """Factory that creates mock DeepSqueak Excel files on disk.

    Returns a callable:
        (filename, rows) -> Path
    where rows is a list of dicts with raw DeepSqueak column names.
    """
    results_dir = tmp_path / "ds_results"
    results_dir.mkdir()

    def _create(filename: str, rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        path = results_dir / filename
        df.to_excel(path, index=False, engine="openpyxl")
        return path

    return _create


@pytest.fixture
def detection_json_factory(tmp_path: Path):
    """Factory that creates detection JSON files with extended fields.

    Returns a callable:
        (subdir_name, start_s, end_s, index=0, prob_max=0.95) -> Path
    """

    def _create(
        subdir_name: str,
        start_s: float,
        end_s: float,
        index: int = 0,
        prob_max: float = 0.95,
        prob_mean: float = 0.85,
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
                "max": prob_max,
                "mean": prob_mean,
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


# ===================================================================
# Config validation
# ===================================================================


class TestDeepSqueakImportConfig:
    def test_valid_defaults(self, tmp_path: Path) -> None:
        cfg = DeepSqueakImportConfig(
            results_dir=tmp_path,
            detections_dir=tmp_path,
            output_path=tmp_path / "out.csv",
        )
        assert cfg.tolerance_ms == 5.0

    def test_zero_tolerance_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            DeepSqueakImportConfig(
                results_dir=tmp_path,
                detections_dir=tmp_path,
                output_path=tmp_path / "out.csv",
                tolerance_ms=0.0,
            )

    def test_negative_tolerance_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            DeepSqueakImportConfig(
                results_dir=tmp_path,
                detections_dir=tmp_path,
                output_path=tmp_path / "out.csv",
                tolerance_ms=-1.0,
            )

    def test_string_paths_converted(self, tmp_path: Path) -> None:
        cfg = DeepSqueakImportConfig(
            results_dir=str(tmp_path),
            detections_dir=str(tmp_path),
            output_path=str(tmp_path / "out.csv"),
        )
        assert isinstance(cfg.results_dir, Path)
        assert isinstance(cfg.detections_dir, Path)
        assert isinstance(cfg.output_path, Path)


# ===================================================================
# 1. Column normalization
# ===================================================================


class TestColumnNormalization:
    def test_known_columns_normalized(self) -> None:
        """Excel loading normalizes column names correctly."""
        raw = pd.DataFrame([_make_ds_excel_row()])
        normalized = _normalize_columns(raw)

        assert "begin_time_s" in normalized.columns
        assert "end_time_s" in normalized.columns
        assert "label" in normalized.columns
        assert "principal_freq_hz" in normalized.columns
        assert "mean_power_db" in normalized.columns
        assert "sinuosity" in normalized.columns

    def test_unknown_columns_get_basic_normalization(self) -> None:
        """Columns not in _COLUMN_MAP still get snake_case treatment."""
        raw = pd.DataFrame({"Custom Score (Hz)": [1.0]})
        normalized = _normalize_columns(raw)
        assert "custom_score_hz" in normalized.columns

    def test_type_column_maps_to_label(self) -> None:
        """Some DS versions use 'Type' instead of 'Label'."""
        raw = pd.DataFrame({"Type": ["chirp"]})
        normalized = _normalize_columns(raw)
        assert "label" in normalized.columns
        assert normalized["label"].iloc[0] == "chirp"


# ===================================================================
# 2. Exact timestamp match (0ms difference)
# ===================================================================


class TestExactMatch:
    def test_exact_match_zero_distance(
        self, tmp_path: Path, detection_json_factory
    ) -> None:
        """Timestamp matching finds exact match (0ms difference)."""
        # Detection at exactly 0.100s
        detection_json_factory("rec_001", 0.100, 0.150, index=0)

        # DS call at exactly 0.100s
        ds_df = pd.DataFrame([_make_ds_excel_row(begin_time=0.100)])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        det_dir = tmp_path / "detections"
        dets = load_detections_for_merge(det_dir)

        merged_df, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)

        assert summary.matched == 1
        assert summary.unmatched_ds == 0
        assert merged_df["match_quality"].iloc[0] == "exact"
        assert merged_df["match_distance_ms"].iloc[0] == 0.0


# ===================================================================
# 3. Fuzzy match within tolerance
# ===================================================================


class TestFuzzyMatch:
    def test_fuzzy_match_within_tolerance(
        self, tmp_path: Path, detection_json_factory
    ) -> None:
        """Timestamp matching finds fuzzy match within tolerance."""
        # Detection at 0.100s
        detection_json_factory("rec_001", 0.100, 0.150, index=0)

        # DS call at 0.103s (3ms offset — within 5ms tolerance)
        ds_df = pd.DataFrame([_make_ds_excel_row(begin_time=0.103)])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        det_dir = tmp_path / "detections"
        dets = load_detections_for_merge(det_dir)

        merged_df, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)

        assert summary.matched == 1
        assert merged_df["match_quality"].iloc[0] == "fuzzy"
        assert merged_df["match_distance_ms"].iloc[0] == pytest.approx(3.0, abs=0.1)


# ===================================================================
# 4. Beyond tolerance rejection
# ===================================================================


class TestBeyondTolerance:
    def test_rejects_match_beyond_tolerance(
        self, tmp_path: Path, detection_json_factory
    ) -> None:
        """Timestamp matching rejects match beyond tolerance."""
        # Detection at 0.100s
        detection_json_factory("rec_001", 0.100, 0.150, index=0)

        # DS call at 0.200s (100ms offset — way beyond 5ms tolerance)
        ds_df = pd.DataFrame([_make_ds_excel_row(begin_time=0.200)])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        det_dir = tmp_path / "detections"
        dets = load_detections_for_merge(det_dir)

        merged_df, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)

        assert summary.matched == 0
        assert summary.unmatched_ds == 1
        assert summary.unmatched_det == 1

        # Find the unmatched DS row.
        ds_rows = merged_df[merged_df["match_quality"] == "unmatched_ds"]
        assert len(ds_rows) == 1

        # Find the unmatched detection row.
        det_rows = merged_df[merged_df["match_quality"] == "unmatched_det"]
        assert len(det_rows) == 1


# ===================================================================
# 5. Unmatched from both sides reported
# ===================================================================


class TestUnmatchedReporting:
    def test_unmatched_from_both_sides(
        self, tmp_path: Path, detection_json_factory
    ) -> None:
        """Unmatched detections from both sides reported as warnings."""
        # Two detections: one matchable, one not
        detection_json_factory("rec_001", 0.100, 0.150, index=0)
        detection_json_factory("rec_001", 0.500, 0.550, index=1)

        # Two DS calls: one matchable, one for a different time
        rows = [
            _make_ds_excel_row(call_id=1, begin_time=0.100),  # matches det 0
            _make_ds_excel_row(call_id=2, begin_time=0.900),  # no match
        ]
        ds_df = pd.DataFrame(rows)
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        det_dir = tmp_path / "detections"
        dets = load_detections_for_merge(det_dir)

        merged_df, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)

        assert summary.matched == 1
        assert summary.unmatched_ds == 1  # DS call at 0.900s
        assert summary.unmatched_det == 1  # Detection at 0.500s

    def test_prefix_matched_detection_dir_round_trips(
        self, tmp_path: Path, detection_json_factory
    ) -> None:
        """Import side mirrors Raven export prefix matching for detection dirs."""
        detection_json_factory("rec_001_retry", 0.100, 0.150, index=0)

        ds_df = pd.DataFrame([_make_ds_excel_row(begin_time=0.100)])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        det_dir = tmp_path / "detections"
        dets = load_detections_for_merge(det_dir)

        merged_df, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)

        assert summary.matched == 1
        assert summary.unmatched_ds == 0
        assert summary.unmatched_det == 0
        assert merged_df["match_quality"].iloc[0] == "exact"
        assert merged_df["det_json_path"].iloc[0].endswith("rec_001_retry\\detection_000_0.100s-0.150s.json")


# ===================================================================
# 6. Multiple Excel files concatenated
# ===================================================================


class TestMultipleExcelFiles:
    def test_multiple_files_concatenated_with_source(
        self, ds_excel_factory, tmp_path: Path
    ) -> None:
        """Multiple Excel files concatenated with source_file column."""
        rows1 = [_make_ds_excel_row(call_id=1, begin_time=0.10)]
        rows2 = [
            _make_ds_excel_row(call_id=1, begin_time=0.20),
            _make_ds_excel_row(call_id=2, begin_time=0.30),
        ]
        ds_excel_factory("rec_001.xlsx", rows1)
        ds_excel_factory("rec_002.xlsx", rows2)

        results_dir = tmp_path / "ds_results"
        combined = load_all_deepsqueak_results(results_dir)

        assert len(combined) == 3
        assert "source_file" in combined.columns
        assert set(combined["source_file"].unique()) == {"rec_001.xlsx", "rec_002.xlsx"}
        assert set(combined["wav_stem"].unique()) == {"rec_001", "rec_002"}


# ===================================================================
# 7. Empty Excel file handled gracefully
# ===================================================================


class TestEmptyExcelFile:
    def test_empty_excel_raises_value_error(self, tmp_path: Path) -> None:
        """Empty Excel file handled gracefully (raises ValueError)."""
        # Create an Excel with headers but no data rows.
        empty_df = pd.DataFrame(columns=["ID", "Label", "Begin Time (s)"])
        path = tmp_path / "empty.xlsx"
        empty_df.to_excel(path, index=False, engine="openpyxl")

        with pytest.raises(ValueError, match="Empty Excel file"):
            load_deepsqueak_excel(path)

    def test_empty_file_skipped_in_batch(self, tmp_path: Path) -> None:
        """Batch loading skips empty (header-only) files and continues with valid ones."""
        results_dir = tmp_path / "ds_results"
        results_dir.mkdir()

        # One header-only (zero data rows) Excel file.
        empty_df = pd.DataFrame(columns=["ID", "Label", "Begin Time (s)"])
        empty_df.to_excel(results_dir / "empty.xlsx", index=False, engine="openpyxl")

        # One valid file.
        valid_df = pd.DataFrame([_make_ds_excel_row()])
        valid_df.to_excel(results_dir / "valid.xlsx", index=False, engine="openpyxl")

        combined = load_all_deepsqueak_results(results_dir)
        assert len(combined) == 1
        assert combined["source_file"].iloc[0] == "valid.xlsx"


# ===================================================================
# 8. All 16 DeepSqueak columns preserved
# ===================================================================


class TestOutputCompleteness:
    def test_all_ds_columns_preserved(
        self, tmp_path: Path, detection_json_factory
    ) -> None:
        """Merged output preserves all DeepSqueak acoustic features (16 columns)."""
        detection_json_factory("rec_001", 0.100, 0.150, index=0)

        ds_df = pd.DataFrame([_make_ds_excel_row(begin_time=0.100)])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        det_dir = tmp_path / "detections"
        dets = load_detections_for_merge(det_dir)

        merged_df, _ = merge_with_detections(ds_df, dets, tolerance_ms=5.0)

        # All 15 DS output columns should be present (id through peak_freq_hz).
        for col in _DS_OUTPUT_COLUMNS:
            assert col in merged_df.columns, f"Missing DS column: {col}"

        # Plus detection columns.
        det_cols = [
            "det_start_s", "det_end_s", "det_duration_ms",
            "det_index", "det_prob_max", "det_prob_mean",
            "det_user_action", "det_json_path",
        ]
        for col in det_cols:
            assert col in merged_df.columns, f"Missing detection column: {col}"

        # Plus merge metadata.
        assert "match_quality" in merged_df.columns
        assert "match_distance_ms" in merged_df.columns
        assert "wav_stem" in merged_df.columns


# ===================================================================
# WAV stem extraction
# ===================================================================


class TestWavStemExtraction:
    def test_plain_stem(self) -> None:
        assert _extract_wav_stem("rec_001.xlsx") == "rec_001"

    def test_detections_suffix_stripped(self) -> None:
        assert _extract_wav_stem("rec_001_Detections.xlsx") == "rec_001"

    def test_calls_suffix_stripped(self) -> None:
        assert _extract_wav_stem("rec_001_calls.xlsx") == "rec_001"

    def test_classified_suffix_stripped(self) -> None:
        assert _extract_wav_stem("rec_001_classified.xlsx") == "rec_001"


# ===================================================================
# ImportSummary
# ===================================================================


class TestImportSummary:
    def test_to_dict_includes_match_rate(self) -> None:
        summary = ImportSummary(
            total_ds_calls=100,
            total_detections=100,
            matched=95,
            unmatched_ds=5,
            unmatched_det=5,
            files_processed=3,
            per_file_counts={"rec_001.xlsx": 50, "rec_002.xlsx": 50},
        )
        d = summary.to_dict()
        assert d["match_rate"] == 0.95
        assert d["files_processed"] == 3

    def test_zero_calls_no_division_error(self) -> None:
        summary = ImportSummary()
        d = summary.to_dict()
        assert d["match_rate"] == 0.0


# ===================================================================
# Integration: end-to-end pipeline
# ===================================================================


class TestIntegration:
    def test_end_to_end_import(
        self, tmp_path: Path, ds_excel_factory, detection_json_factory
    ) -> None:
        """Full pipeline: Excel -> load -> match -> CSV output."""
        # Create detections.
        detection_json_factory("rec_001", 0.100, 0.150, index=0)
        detection_json_factory("rec_001", 0.300, 0.350, index=1)

        # Create DS Excel with matching timestamps.
        rows = [
            _make_ds_excel_row(call_id=1, label="chirp", begin_time=0.100),
            _make_ds_excel_row(call_id=2, label="trill", begin_time=0.300),
        ]
        ds_excel_factory("rec_001.xlsx", rows)

        config = DeepSqueakImportConfig(
            results_dir=tmp_path / "ds_results",
            detections_dir=tmp_path / "detections",
            output_path=tmp_path / "output" / "merged.csv",
        )

        summary = import_deepsqueak_results(config)

        assert summary.matched == 2
        assert summary.unmatched_ds == 0
        assert summary.unmatched_det == 0
        assert summary.files_processed == 1

        # Verify CSV was written.
        output_csv = tmp_path / "output" / "merged.csv"
        assert output_csv.exists()
        result_df = pd.read_csv(output_csv)
        assert len(result_df) == 2
        assert set(result_df["label"]) == {"chirp", "trill"}

        # Verify summary JSON alongside CSV.
        summary_json = tmp_path / "output" / "import_summary.json"
        assert summary_json.exists()
