"""Adversarial / edge-case tests for the DeepSqueak import module.

Supplements test_deepsqueak_import.py with coverage of:

- Column name variations and collision edge cases (S3 from review)
- NaN/Inf in acoustic feature columns
- Zero-length calls and zero-bandwidth calls
- Empty and non-xlsx files mixed in results directory
- Error paths: missing results dir, all-bad files, corrupt JSON
- Tolerance boundary (exactly at tolerance, epsilon inside/outside)
- Multiple Excel files sharing the same WAV stem (overlap)
- "Unclassified" label string from new MATLAB script
- Categorical vs string label dtypes
- Detection JSON with missing optional fields (probabilities, user_action)
- Detections directory with no valid JSON subdirectories
- Non-directory entries at detections_dir root
- Malformed / corrupt Excel file raises ValueError
- Empty DataFrame passed to merge_with_detections
- DS stem with no matching detection stem
- _detections / _Calls suffix variants of _extract_wav_stem
- Performance: 1000-call Excel loads without crash
- Pre-existing path-separator bug marked explicitly

All tests must be green (or marked skip with BUG FOUND) without touching
the implementation or the original test file.
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Bootstrap sys.path
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
    load_all_deepsqueak_results,
    load_deepsqueak_excel,
    load_detections_for_merge,
    merge_with_detections,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_raw_row(
    call_id: int = 1,
    label: str = "USV",
    begin_time: float = 0.100,
    end_time: float = 0.150,
    low_freq: float = 40.0,
    high_freq: float = 90.0,
    delta_freq: float = 50.0,
) -> dict:
    """Minimal DeepSqueak row with raw column names."""
    return {
        "ID": call_id,
        "Label": label,
        "Begin Time (s)": begin_time,
        "End Time (s)": end_time,
        "Call Length (s)": end_time - begin_time,
        "Principal Frequency (kHz)": 65.0,
        "Low Freq (kHz)": low_freq,
        "High Freq (kHz)": high_freq,
        "Delta Freq (kHz)": delta_freq,
        "Frequency Standard Deviation (kHz)": 8.5,
        "Slope (kHz/s)": 120.0,
        "Sinuosity": 1.3,
        "Mean Power (dB/Hz)": -45.0,
        "Tonality": 0.85,
        "Peak Frequency (kHz)": 70.0,
    }


def _write_excel(path: Path, rows: list[dict]) -> None:
    """Write rows as Excel via openpyxl."""
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")


def _make_detection_json(
    det_dir: Path,
    start_s: float,
    end_s: float,
    index: int = 0,
    prob_max: float = 0.95,
    prob_mean: float = 0.85,
    include_probs: bool = True,
    include_user_action: bool = True,
    include_detection_index: bool = True,
) -> Path:
    """Write a detection JSON to det_dir, returning the path."""
    det_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = (end_s - start_s) * 1000
    filename = f"detection_{index:03d}_{start_s:.3f}s-{end_s:.3f}s.json"
    data: dict = {
        "core_time": {
            "start_s": start_s,
            "end_s": end_s,
            "duration_ms": duration_ms,
        },
    }
    if include_detection_index:
        data["detection_index"] = index
    if include_probs:
        data["probabilities"] = {"max": prob_max, "mean": prob_mean}
    if include_user_action:
        data["user_action"] = None
    out = det_dir / filename
    out.write_text(json.dumps(data), encoding="utf-8")
    return out



# ---------------------------------------------------------------------------
# A. WAV stem suffix variants not yet tested
# ---------------------------------------------------------------------------

class TestWavStemExtractionAdversarial:
    """Cover the two _DS_SUFFIXES variants missing from the original tests."""

    def test_lowercase_detections_suffix_stripped(self) -> None:
        # '_detections' (lowercase d)
        assert _extract_wav_stem("rec_001_detections.xlsx") == "rec_001"

    def test_calls_capitalized_suffix_stripped(self) -> None:
        # '_Calls' (capital C)
        assert _extract_wav_stem("rec_001_Calls.xlsx") == "rec_001"

    def test_path_with_subdirectory_component(self) -> None:
        # Filename may be constructed from a full path string; only the stem matters.
        assert _extract_wav_stem("subdir/rec_001_Detections.xlsx") == "rec_001"

    def test_stem_ends_with_number_no_false_strip(self) -> None:
        # A stem like "rec_001_session2" should NOT be stripped.
        assert _extract_wav_stem("rec_001_session2.xlsx") == "rec_001_session2"

    def test_suffix_at_middle_of_stem_not_stripped(self) -> None:
        # '_Detections' appearing mid-name is not at the end so must not be stripped.
        assert _extract_wav_stem("rec_Detections_001.xlsx") == "rec_Detections_001"


# ---------------------------------------------------------------------------
# B. Column normalization edge cases
# ---------------------------------------------------------------------------

class TestColumnNormalizationAdversarial:
    """Cover whitespace, special chars, and the Label+Type collision."""

    def test_column_with_leading_trailing_whitespace(self) -> None:
        # DeepSqueak may export column names with extra spaces.
        raw = pd.DataFrame({" Label ": ["USV"]})
        normalized = _normalize_columns(raw)
        assert "label" in normalized.columns, (
            "Column with surrounding whitespace should still map to 'label'"
        )

    def test_fallback_normalization_hyphen_and_dot(self) -> None:
        # Fallback path: hyphens and dots should become underscores.
        raw = pd.DataFrame({"My-Score.v2": [1.0]})
        normalized = _normalize_columns(raw)
        assert "my_score_v2" in normalized.columns

    def test_fallback_normalization_slash(self) -> None:
        # Slash in unknown column (not in _COLUMN_MAP) becomes underscore.
        raw = pd.DataFrame({"Custom Hz/s": [1.0]})
        normalized = _normalize_columns(raw)
        assert "custom_hz_s" in normalized.columns

    def test_both_label_and_type_present_no_crash(self) -> None:
        """S3 deferred in review: if both Label and Type exist, no exception."""
        # Both map to "label" in _COLUMN_MAP — this can produce a duplicate column
        # name. The test just verifies we don't crash, and that at least one "label"
        # column exists in the output.
        raw = pd.DataFrame({"Label": ["USV"], "Type": ["chirp"]})
        # May produce duplicate columns; must not raise.
        try:
            normalized = _normalize_columns(raw)
            assert "label" in normalized.columns
        except Exception as exc:
            pytest.fail(
                f"_normalize_columns crashed when both Label and Type present: {exc}"
            )

    def test_empty_dataframe_normalizes_without_crash(self) -> None:
        # Zero-row DataFrame with canonical columns should normalize fine.
        raw = pd.DataFrame(columns=["ID", "Label", "Begin Time (s)"])
        normalized = _normalize_columns(raw)
        assert "label" in normalized.columns
        assert len(normalized) == 0


# ---------------------------------------------------------------------------
# C. NaN / Inf in acoustic feature columns
# ---------------------------------------------------------------------------

class TestNaNInfAcousticFeatures:
    """DeepSqueak's CalculateStats can output NaN/Inf for some features."""

    def test_nan_in_numeric_columns_passes_through(self, tmp_path: Path) -> None:
        """NaN values in acoustic features are preserved, not dropped."""
        row = _make_raw_row()
        row["Slope (kHz/s)"] = float("nan")
        row["Sinuosity"] = float("nan")
        path = tmp_path / "nan_test.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert not df.empty
        assert math.isnan(df["slope"].iloc[0])
        assert math.isnan(df["sinuosity"].iloc[0])

    def test_inf_in_numeric_column_passes_through(self, tmp_path: Path) -> None:
        """Inf values (e.g. from division by zero in MATLAB) are preserved."""
        row = _make_raw_row()
        row["Slope (kHz/s)"] = float("inf")
        path = tmp_path / "inf_test.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert not df.empty
        assert math.isinf(df["slope"].iloc[0])

    def test_nan_in_begin_time_does_not_crash_merge(self, tmp_path: Path) -> None:
        """A NaN begin_time_s row should be handled gracefully, not crash."""
        row = _make_raw_row()
        row["Begin Time (s)"] = float("nan")
        path = tmp_path / "nan_begin.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        df["wav_stem"] = "rec_nan"
        df["source_file"] = "nan_begin.xlsx"

        # No detections for this stem — just ensure we don't get an exception.
        try:
            result_df, summary = merge_with_detections(df, {}, tolerance_ms=5.0)
        except Exception as exc:
            pytest.fail(f"merge_with_detections raised on NaN begin_time_s: {exc}")

    def test_all_nan_features_row_survives_load(self, tmp_path: Path) -> None:
        """Row where every acoustic feature is NaN still loads (call exists)."""
        row = {
            "ID": 1,
            "Label": "Unclassified",
            "Begin Time (s)": 0.1,
            "End Time (s)": 0.15,
            "Call Length (s)": 0.05,
            "Principal Frequency (kHz)": float("nan"),
            "Low Freq (kHz)": float("nan"),
            "High Freq (kHz)": float("nan"),
            "Delta Freq (kHz)": float("nan"),
            "Frequency Standard Deviation (kHz)": float("nan"),
            "Slope (kHz/s)": float("nan"),
            "Sinuosity": float("nan"),
            "Mean Power (dB/Hz)": float("nan"),
            "Tonality": float("nan"),
            "Peak Frequency (kHz)": float("nan"),
        }
        path = tmp_path / "all_nan.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert len(df) == 1
        assert df["label"].iloc[0] == "Unclassified"


# ---------------------------------------------------------------------------
# D. Zero-length calls and zero-bandwidth calls
# ---------------------------------------------------------------------------

class TestZeroLengthAndZeroBandwidth:
    def test_zero_length_call_loads_without_error(self, tmp_path: Path) -> None:
        """Call where begin_time == end_time (call_length == 0) should load."""
        row = _make_raw_row(begin_time=0.100, end_time=0.100)  # zero duration
        path = tmp_path / "zero_len.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert len(df) == 1
        assert df["call_length_s"].iloc[0] == 0.0

    def test_zero_bandwidth_call_loads_without_error(self, tmp_path: Path) -> None:
        """Call where Delta Freq == 0 (pure tone) should load."""
        row = _make_raw_row(low_freq=65.0, high_freq=65.0, delta_freq=0.0)
        path = tmp_path / "zero_bw.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert len(df) == 1
        assert df["bandwidth_hz"].iloc[0] == 0.0

    def test_negative_call_length_passes_through(self, tmp_path: Path) -> None:
        """Negative call length (MATLAB bug artifact) should load, not crash."""
        row = _make_raw_row(begin_time=0.150, end_time=0.100)  # inverted
        row["Call Length (s)"] = -0.05
        path = tmp_path / "neg_len.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert len(df) == 1
        assert df["call_length_s"].iloc[0] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# E. Label string variants from new MATLAB script
# ---------------------------------------------------------------------------

class TestLabelVariants:
    def test_unclassified_label_string(self) -> None:
        """'Unclassified' label from new MATLAB script is preserved as-is."""
        raw = pd.DataFrame([_make_raw_row(label="Unclassified")])
        norm = _normalize_columns(raw)
        assert norm["label"].iloc[0] == "Unclassified"

    def test_categorical_label_dtype_normalized(self, tmp_path: Path) -> None:
        """Categorical dtype for label column (pandas read_excel may produce this)
        should survive normalization and merge without type errors."""
        row = _make_raw_row(label="USV")
        path = tmp_path / "cat_label.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        # Force label to categorical dtype to simulate what some Excel readers do.
        df["label"] = df["label"].astype("category")

        df["wav_stem"] = "rec_cat"
        df["source_file"] = "cat_label.xlsx"

        # Merge should not crash on categorical dtype.
        try:
            result_df, _ = merge_with_detections(df, {}, tolerance_ms=5.0)
            assert len(result_df) == 1
        except Exception as exc:
            pytest.fail(f"merge_with_detections crashed on categorical label: {exc}")

    def test_numeric_label_value_preserved(self, tmp_path: Path) -> None:
        """DeepSqueak sometimes uses numeric IDs as labels (e.g., cluster numbers)."""
        row = _make_raw_row()
        row["Label"] = 42  # numeric label
        path = tmp_path / "num_label.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        # Value should be preserved (42), not silently dropped.
        assert df["label"].iloc[0] == 42


# ---------------------------------------------------------------------------
# F. Tolerance boundary tests (exactly at / epsilon inside / epsilon outside)
# ---------------------------------------------------------------------------

class TestToleranceBoundary:
    """The matching uses `best_dist <= tolerance_s`, so the boundary is inclusive."""

    def _make_ds_df(self, begin_time: float, wav_stem: str = "rec_tol") -> pd.DataFrame:
        row = _make_raw_row(begin_time=begin_time)
        df = pd.DataFrame([row])
        df = _normalize_columns(df)
        df["wav_stem"] = wav_stem
        df["source_file"] = f"{wav_stem}.xlsx"
        return df

    @pytest.mark.skip(
        reason=(
            "BUG FOUND: floating-point tolerance boundary. 0.100 + 0.005 = 0.10500000000000001 in IEEE 754, so abs(dist - tolerance_s) = 4.3e-18 and dist <= tolerance_s evaluates to False. The implementation needs an epsilon guard (e.g. tolerance_s + 1e-9) or use round() before comparison."
        )
    )
    def test_exactly_at_tolerance_boundary_matches(self, tmp_path: Path) -> None:
        """Distance == tolerance (5.000 ms) must match (inclusive boundary)."""
        det_dir = tmp_path / "detections" / "rec_tol"
        _make_detection_json(det_dir, start_s=0.100, end_s=0.150)

        dets = load_detections_for_merge(tmp_path / "detections")
        ds_df = self._make_ds_df(begin_time=0.100 + 0.005)  # exactly 5ms later

        merged, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)
        assert summary.matched == 1, (
            "Distance exactly at tolerance should be accepted (<=)"
        )
        assert merged["match_quality"].iloc[0] == "fuzzy"
        assert merged["match_distance_ms"].iloc[0] == pytest.approx(5.0, abs=0.001)

    def test_one_microsecond_inside_tolerance_matches(self, tmp_path: Path) -> None:
        """Distance slightly under tolerance (4.999 ms) must match."""
        det_dir = tmp_path / "detections" / "rec_tol"
        _make_detection_json(det_dir, start_s=0.100, end_s=0.150)

        dets = load_detections_for_merge(tmp_path / "detections")
        ds_df = self._make_ds_df(begin_time=0.100 + 0.004999)

        merged, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)
        assert summary.matched == 1
        assert merged["match_quality"].iloc[0] == "fuzzy"

    def test_one_microsecond_outside_tolerance_rejected(self, tmp_path: Path) -> None:
        """Distance 5.001 ms beyond tolerance must be rejected."""
        det_dir = tmp_path / "detections" / "rec_tol"
        _make_detection_json(det_dir, start_s=0.100, end_s=0.150)

        dets = load_detections_for_merge(tmp_path / "detections")
        ds_df = self._make_ds_df(begin_time=0.100 + 0.005001)

        merged, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)
        assert summary.matched == 0
        assert summary.unmatched_ds == 1
        assert summary.unmatched_det == 1


# ---------------------------------------------------------------------------
# G. Multiple Excel files with overlapping WAV stems
# ---------------------------------------------------------------------------

class TestOverlappingWavStems:
    """Two Excel files that resolve to the same WAV stem: last-file wins or both rows kept."""

    def test_two_files_same_stem_rows_both_appear(self, tmp_path: Path) -> None:
        """If two Excel files map to the same WAV stem, all calls from both appear."""
        results_dir = tmp_path / "ds_results"
        results_dir.mkdir()

        # rec_001.xlsx and rec_001_Detections.xlsx both extract to stem "rec_001"
        _write_excel(results_dir / "rec_001.xlsx", [_make_raw_row(call_id=1, begin_time=0.1)])
        _write_excel(results_dir / "rec_001_Detections.xlsx", [_make_raw_row(call_id=2, begin_time=0.2)])

        combined = load_all_deepsqueak_results(results_dir)
        # Both rows must be present.
        assert len(combined) == 2
        # Both report wav_stem == "rec_001"
        assert (combined["wav_stem"] == "rec_001").all()
        # source_file distinguishes them
        assert set(combined["source_file"].unique()) == {
            "rec_001.xlsx",
            "rec_001_Detections.xlsx",
        }

    def test_overlapping_stems_all_calls_matched(self, tmp_path: Path) -> None:
        """With two detection files for the same WAV, both DS calls compete for matches."""
        results_dir = tmp_path / "ds_results"
        results_dir.mkdir()
        _write_excel(results_dir / "rec_001.xlsx", [_make_raw_row(call_id=1, begin_time=0.1)])
        _write_excel(results_dir / "rec_001_Detections.xlsx", [_make_raw_row(call_id=2, begin_time=0.2)])

        det_dir = tmp_path / "detections" / "rec_001"
        _make_detection_json(det_dir, start_s=0.1, end_s=0.15, index=0)
        _make_detection_json(det_dir, start_s=0.2, end_s=0.25, index=1)

        ds_df = load_all_deepsqueak_results(results_dir)
        dets = load_detections_for_merge(tmp_path / "detections")

        merged, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)
        assert summary.matched == 2
        assert summary.unmatched_ds == 0
        assert summary.unmatched_det == 0


# ---------------------------------------------------------------------------
# H. Error paths not covered by original tests
# ---------------------------------------------------------------------------

class TestErrorPaths:
    def test_load_all_missing_results_dir_raises(self, tmp_path: Path) -> None:
        """Missing results directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Results directory not found"):
            load_all_deepsqueak_results(tmp_path / "nonexistent_dir")

    def test_load_all_no_xlsx_raises(self, tmp_path: Path) -> None:
        """Directory with no .xlsx files raises FileNotFoundError."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No .xlsx files found"):
            load_all_deepsqueak_results(empty_dir)

    def test_load_all_only_bad_files_raises_value_error(self, tmp_path: Path) -> None:
        """Directory where every .xlsx is invalid raises ValueError after skipping all."""
        results_dir = tmp_path / "bad_only"
        results_dir.mkdir()
        # Write a zero-byte file — openpyxl will reject it.
        (results_dir / "corrupt.xlsx").write_bytes(b"")
        with pytest.raises(ValueError, match="No valid Excel files could be loaded"):
            load_all_deepsqueak_results(results_dir)

    def test_load_excel_corrupt_file_raises_value_error(self, tmp_path: Path) -> None:
        """Zero-byte .xlsx raises ValueError, not an unhandled openpyxl exception."""
        path = tmp_path / "zero.xlsx"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="Cannot read Excel file"):
            load_deepsqueak_excel(path)

    def test_load_all_non_xlsx_files_skipped(self, tmp_path: Path) -> None:
        """Non-xlsx files in results directory are silently skipped."""
        results_dir = tmp_path / "mixed"
        results_dir.mkdir()
        (results_dir / "notes.txt").write_text("not an excel file")
        (results_dir / "image.png").write_bytes(b"\x89PNG")
        # Valid Excel alongside junk.
        _write_excel(results_dir / "valid.xlsx", [_make_raw_row()])
        combined = load_all_deepsqueak_results(results_dir)
        assert len(combined) == 1

    def test_load_detections_missing_dir_raises(self, tmp_path: Path) -> None:
        """Missing detections directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Detections directory not found"):
            load_detections_for_merge(tmp_path / "nonexistent_detections")

    def test_load_detections_skips_non_directory_entries(self, tmp_path: Path) -> None:
        """Files at the detections root (not subdirs) are silently skipped."""
        det_root = tmp_path / "detections"
        det_root.mkdir()
        # A stray file at the root level
        (det_root / "readme.txt").write_text("this is not a detection")
        # One valid subdir
        _make_detection_json(det_root / "rec_001", start_s=0.1, end_s=0.15)

        result = load_detections_for_merge(det_root)
        assert "rec_001" in result
        assert "readme.txt" not in result

    def test_load_detections_empty_subdir_not_in_result(self, tmp_path: Path) -> None:
        """A subdirectory with no valid JSON files is omitted from the result dict."""
        det_root = tmp_path / "detections"
        det_root.mkdir()
        empty_subdir = det_root / "rec_empty"
        empty_subdir.mkdir()
        (empty_subdir / "notes.txt").write_text("not a json")

        result = load_detections_for_merge(det_root)
        assert "rec_empty" not in result

    def test_load_detections_malformed_json_skipped(self, tmp_path: Path) -> None:
        """Malformed JSON is skipped with a warning, valid JSONs in same dir still load."""
        det_dir = tmp_path / "detections" / "rec_partial"
        det_dir.mkdir(parents=True)
        # Corrupt JSON
        (det_dir / "detection_000_0.100s-0.150s.json").write_text(
            "{invalid json{{", encoding="utf-8"
        )
        # Valid JSON
        _make_detection_json(det_dir, start_s=0.200, end_s=0.250, index=1)

        result = load_detections_for_merge(tmp_path / "detections")
        assert "rec_partial" in result
        assert len(result["rec_partial"]) == 1  # only the valid one

    def test_load_detections_json_missing_core_time_skipped(self, tmp_path: Path) -> None:
        """JSON without 'core_time' key is skipped, not crash."""
        det_dir = tmp_path / "detections" / "rec_notime"
        det_dir.mkdir(parents=True)
        (det_dir / "detection_000_0.100s-0.150s.json").write_text(
            json.dumps({"detection_index": 0}), encoding="utf-8"
        )

        result = load_detections_for_merge(tmp_path / "detections")
        # The stem has no valid detections, so it should be absent.
        assert "rec_notime" not in result


# ---------------------------------------------------------------------------
# I. Detection JSON with missing optional fields
# ---------------------------------------------------------------------------

class TestDetectionJsonMissingOptionalFields:
    def test_missing_probabilities_key_defaults_to_none(self, tmp_path: Path) -> None:
        """JSON without 'probabilities' key sets det_prob_max/mean to None."""
        det_dir = tmp_path / "detections" / "rec_noprob"
        _make_detection_json(det_dir, 0.1, 0.15, include_probs=False)

        dets = load_detections_for_merge(tmp_path / "detections")
        assert "rec_noprob" in dets
        det = dets["rec_noprob"][0]
        assert det["prob_max"] is None
        assert det["prob_mean"] is None

    def test_missing_detection_index_defaults_to_minus_one(self, tmp_path: Path) -> None:
        """JSON without 'detection_index' sets detection_index to -1."""
        det_dir = tmp_path / "detections" / "rec_noidx"
        _make_detection_json(det_dir, 0.1, 0.15, include_detection_index=False)

        dets = load_detections_for_merge(tmp_path / "detections")
        assert dets["rec_noidx"][0]["detection_index"] == -1

    def test_missing_user_action_defaults_to_none(self, tmp_path: Path) -> None:
        """JSON without 'user_action' key sets user_action to None."""
        det_dir = tmp_path / "detections" / "rec_noaction"
        _make_detection_json(det_dir, 0.1, 0.15, include_user_action=False)

        dets = load_detections_for_merge(tmp_path / "detections")
        assert dets["rec_noaction"][0]["user_action"] is None


# ---------------------------------------------------------------------------
# J. Merge edge cases
# ---------------------------------------------------------------------------

class TestMergeEdgeCases:
    def _normalized_ds_df(
        self, begin_time: float, wav_stem: str, call_id: int = 1
    ) -> pd.DataFrame:
        row = _make_raw_row(call_id=call_id, begin_time=begin_time)
        df = pd.DataFrame([row])
        df = _normalize_columns(df)
        df["wav_stem"] = wav_stem
        df["source_file"] = f"{wav_stem}.xlsx"
        return df

    def test_ds_stem_with_no_matching_detection_stem(self) -> None:
        """All DS calls unmatched when no detection stem matches the WAV stem."""
        ds_df = self._normalized_ds_df(begin_time=0.1, wav_stem="unknown_wav")
        # detections_by_stem has an entirely different stem
        det = {
            "start_s": 0.1, "end_s": 0.15, "duration_ms": 50.0,
            "detection_index": 0, "prob_max": 0.9, "prob_mean": 0.8,
            "user_action": None, "json_path": "/fake/path.json",
        }
        detections_by_stem = {"completely_different_stem": [det]}

        merged, summary = merge_with_detections(ds_df, detections_by_stem, tolerance_ms=5.0)
        assert summary.unmatched_ds == 1
        assert summary.unmatched_det == 1
        assert summary.matched == 0

    def test_empty_ds_dataframe_returns_only_unmatched_dets(self, tmp_path: Path) -> None:
        """Empty DS DataFrame: no DS rows, but detections become unmatched_det rows."""
        det_dir = tmp_path / "detections" / "rec_001"
        _make_detection_json(det_dir, 0.1, 0.15)
        dets = load_detections_for_merge(tmp_path / "detections")

        empty_ds = pd.DataFrame(
            columns=["wav_stem", "source_file", "begin_time_s", "label", "id"]
        )
        # Groupby on empty DF should work without crash.
        merged, summary = merge_with_detections(empty_ds, dets, tolerance_ms=5.0)
        assert summary.total_ds_calls == 0
        assert summary.unmatched_det == 1
        assert summary.matched == 0

    def test_greedy_matching_prevents_double_assignment(self, tmp_path: Path) -> None:
        """Two DS calls at same timestamp: second one must not steal the already-matched detection."""
        det_dir = tmp_path / "detections" / "rec_greed"
        _make_detection_json(det_dir, start_s=0.100, end_s=0.150, index=0)

        dets = load_detections_for_merge(tmp_path / "detections")

        # Two DS calls with identical begin_time — only one can match
        row1 = _make_raw_row(call_id=1, begin_time=0.100)
        row2 = _make_raw_row(call_id=2, begin_time=0.100)
        ds_df = pd.DataFrame([row1, row2])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_greed"
        ds_df["source_file"] = "rec_greed.xlsx"

        merged, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)
        assert summary.matched == 1
        assert summary.unmatched_ds == 1
        # Total rows: 2 DS rows
        assert len(merged) == 2

    def test_match_distance_ms_none_when_no_candidates(self) -> None:
        """match_distance_ms is None (not inf) when there are zero detections for the stem."""
        row = _make_raw_row(begin_time=0.1)
        ds_df = pd.DataFrame([row])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_empty_det"
        ds_df["source_file"] = "rec_empty_det.xlsx"

        merged, summary = merge_with_detections(ds_df, {}, tolerance_ms=5.0)
        # When available_dets is empty, the loop sets match_distance_ms = None
        assert summary.unmatched_ds == 1
        row_out = merged[merged["match_quality"] == "unmatched_ds"].iloc[0]
        assert row_out["match_distance_ms"] is None

    def test_detections_sorted_by_start_time_in_result(self, tmp_path: Path) -> None:
        """Detections loaded by stem are sorted by start_s ascending."""
        det_dir = tmp_path / "detections" / "rec_sort"
        # Write in reverse order
        _make_detection_json(det_dir, start_s=0.500, end_s=0.550, index=1)
        _make_detection_json(det_dir, start_s=0.100, end_s=0.150, index=0)
        _make_detection_json(det_dir, start_s=0.300, end_s=0.350, index=2)

        dets = load_detections_for_merge(tmp_path / "detections")
        starts = [d["start_s"] for d in dets["rec_sort"]]
        assert starts == sorted(starts), "Detections must be sorted by start_s"


# ---------------------------------------------------------------------------
# K. Excel with missing or extra columns
# ---------------------------------------------------------------------------

class TestExcelColumnStructure:
    def test_missing_optional_columns_load_gracefully(self, tmp_path: Path) -> None:
        """Excel missing some optional columns (e.g. Sinuosity) still loads."""
        row = {
            "ID": 1,
            "Label": "USV",
            "Begin Time (s)": 0.1,
            "End Time (s)": 0.15,
            # All other columns missing — they simply won't appear in output
        }
        path = tmp_path / "minimal.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert len(df) == 1
        assert "begin_time_s" in df.columns
        # Missing columns are absent but that's not an error
        assert "sinuosity" not in df.columns

    def test_extra_unknown_column_passes_through_normalized(self, tmp_path: Path) -> None:
        """Extra column from a newer DS version appears in output with normalized name."""
        row = _make_raw_row()
        row["Custom Score (AU)"] = 0.77
        path = tmp_path / "extra_col.xlsx"
        _write_excel(path, [row])

        df = load_deepsqueak_excel(path)
        assert "custom_score_au" in df.columns
        assert df["custom_score_au"].iloc[0] == pytest.approx(0.77)

    def test_all_canonical_columns_present_after_normalization(self, tmp_path: Path) -> None:
        """Full 15-column DeepSqueak Excel produces all expected snake_case names."""
        path = tmp_path / "full.xlsx"
        _write_excel(path, [_make_raw_row()])
        df = load_deepsqueak_excel(path)
        for col in _DS_OUTPUT_COLUMNS:
            assert col in df.columns, f"Canonical column missing after normalization: {col}"


# ---------------------------------------------------------------------------
# L. Stem resolution adversarial cases
# ---------------------------------------------------------------------------

class TestStemResolutionAdversarial:
    def test_exact_match_preferred_over_prefix_match(self, tmp_path: Path) -> None:
        """When both exact and prefix match exist, exact wins."""
        det_dir_exact = tmp_path / "detections" / "rec_001"
        det_dir_prefix = tmp_path / "detections" / "rec_001_extra"
        _make_detection_json(det_dir_exact, 0.1, 0.15, index=0)
        _make_detection_json(det_dir_prefix, 0.1, 0.15, index=0)

        dets = load_detections_for_merge(tmp_path / "detections")

        row = _make_raw_row(begin_time=0.1)
        ds_df = pd.DataFrame([row])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        merged, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)
        assert summary.matched == 1
        # The matched detection must come from the exact dir, not the prefix dir.
        assert "rec_001" in merged["det_json_path"].iloc[0]
        # Specifically not the _extra path
        assert "_extra" not in merged["det_json_path"].iloc[0]

    def test_multiple_prefix_candidates_shortest_wins(self, tmp_path: Path) -> None:
        """Among prefix-matched candidates, the shortest name wins."""
        for suffix in ["_a", "_ab", "_abc"]:
            det_dir = tmp_path / "detections" / f"rec_001{suffix}"
            _make_detection_json(det_dir, 0.1, 0.15, index=0)

        dets = load_detections_for_merge(tmp_path / "detections")

        row = _make_raw_row(begin_time=0.1)
        ds_df = pd.DataFrame([row])
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_001"
        ds_df["source_file"] = "rec_001.xlsx"

        merged, summary = merge_with_detections(ds_df, dets, tolerance_ms=5.0)
        assert summary.matched == 1
        # The shortest prefix match is rec_001_a
        assert "rec_001_a" + "/" in merged["det_json_path"].iloc[0] or \
               "rec_001_a" + "\\" in merged["det_json_path"].iloc[0]


# ---------------------------------------------------------------------------
# M. Export
# ---------------------------------------------------------------------------

class TestExportEdgeCases:
    def test_export_empty_dataframe_writes_header_only_csv(self, tmp_path: Path) -> None:
        """Exporting an empty DataFrame writes a file with column headers but no rows."""
        empty_df = pd.DataFrame(columns=["wav_stem", "label", "match_quality"])
        out = tmp_path / "sub" / "out.csv"
        result_path = export_classified_detections(empty_df, out)
        assert result_path.exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == 0
        assert set(loaded.columns) == {"wav_stem", "label", "match_quality"}

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """export_classified_detections creates nested parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "output.csv"
        df = pd.DataFrame([{"col": 1}])
        export_classified_detections(df, deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# N. ImportSummary.to_dict completeness
# ---------------------------------------------------------------------------

class TestImportSummaryAdversarial:
    def test_all_keys_present_in_to_dict(self) -> None:
        """to_dict must include all documented keys."""
        s = ImportSummary(
            total_ds_calls=10,
            total_detections=10,
            matched=8,
            unmatched_ds=2,
            unmatched_det=2,
            files_processed=2,
            per_file_counts={"a.xlsx": 5, "b.xlsx": 5},
        )
        d = s.to_dict()
        required_keys = {
            "total_ds_calls",
            "total_detections",
            "matched",
            "unmatched_ds",
            "unmatched_det",
            "files_processed",
            "match_rate",
            "per_file_counts",
        }
        missing = required_keys - set(d.keys())
        assert not missing, f"to_dict missing keys: {missing}"

    def test_match_rate_rounds_to_three_decimal_places(self) -> None:
        """match_rate is rounded to 3 decimal places."""
        s = ImportSummary(total_ds_calls=3, matched=1)
        d = s.to_dict()
        # 1/3 = 0.333...  rounded to 0.333
        assert d["match_rate"] == 0.333


# ---------------------------------------------------------------------------
# O. Performance: 1000 calls, no crash
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_1000_calls_load_without_crash(self, tmp_path: Path) -> None:
        """Loading and merging 1000-row Excel completes without error."""
        rows = [_make_raw_row(call_id=i, begin_time=i * 0.01) for i in range(1000)]
        path = tmp_path / "big.xlsx"
        _write_excel(path, rows)

        df = load_deepsqueak_excel(path)
        assert len(df) == 1000

    def test_1000_calls_merge_completes_in_reasonable_time(self, tmp_path: Path) -> None:
        """Greedy matching of 1000 calls against 1000 detections runs in < 10 seconds."""
        n = 1000
        rows = [_make_raw_row(call_id=i, begin_time=i * 0.01) for i in range(n)]
        ds_df = pd.DataFrame(rows)
        ds_df = _normalize_columns(ds_df)
        ds_df["wav_stem"] = "rec_big"
        ds_df["source_file"] = "rec_big.xlsx"

        dets = []
        for i in range(n):
            dets.append({
                "start_s": i * 0.01,
                "end_s": i * 0.01 + 0.05,
                "duration_ms": 50.0,
                "detection_index": i,
                "prob_max": 0.9,
                "prob_mean": 0.8,
                "user_action": None,
                "json_path": f"/fake/{i}.json",
            })

        detections_by_stem = {"rec_big": dets}

        t0 = time.monotonic()
        merged, summary = merge_with_detections(ds_df, detections_by_stem, tolerance_ms=5.0)
        elapsed = time.monotonic() - t0

        assert summary.matched == n
        assert elapsed < 10.0, (
            f"1000-call merge took {elapsed:.1f}s — may need algorithmic improvement"
        )
