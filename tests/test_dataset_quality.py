"""Tests for dataset quality checks and metadata modules."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import csv
import pytest

from usv_spectrogram.dataset.quality_checks import (
    CheckResult,
    QualityReport,
    check_no_recording_leakage,
    check_class_balance,
    check_no_duplicate_ids,
    check_population_coverage,
    check_split_sizes,
)
from usv_spectrogram.dataset.metadata import (
    RecordingMetadata,
    extract_source_file_from_candidate_id,
    extract_mouse_id_from_source_file,
    load_metadata,
    generate_metadata_csv,
)
from usv_spectrogram.dataset.splits import Sample


# ============================================================================
# Module 1: quality_checks.py Tests
# ============================================================================


class TestCheckNoRecordingLeakage:
    """Tests for check_no_recording_leakage function."""

    def test_no_leakage_different_recordings(self):
        """No leakage when splits have different recordings."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
            ],
            "val": [
                Sample("id3", "rec3.wav", "Not USV", "/path/id3.png"),
            ],
            "test": [
                Sample("id4", "rec4.wav", "USV", "/path/id4.png"),
            ],
        }

        # Act
        result = check_no_recording_leakage(splits)

        # Assert
        assert result.passed
        assert "No recordings appear in multiple splits" in result.message
        assert len(result.details) == 0

    def test_leakage_same_recording_in_train_and_test(self):
        """Detects leakage when same recording in train and test."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
            ],
            "test": [
                Sample("id2", "rec1.wav", "Not USV", "/path/id2.png"),
            ],
        }

        # Act
        result = check_no_recording_leakage(splits)

        # Assert
        assert not result.passed
        assert "1 recordings in multiple splits" in result.message
        assert any("rec1.wav" in detail for detail in result.details)

    def test_leakage_multiple_recordings_across_splits(self):
        """Detects multiple recordings leaked across splits."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
            ],
            "val": [
                Sample("id3", "rec1.wav", "USV", "/path/id3.png"),
            ],
            "test": [
                Sample("id4", "rec2.wav", "Not USV", "/path/id4.png"),
            ],
        }

        # Act
        result = check_no_recording_leakage(splits)

        # Assert
        assert not result.passed
        assert "2 recordings in multiple splits" in result.message
        assert len(result.details) == 2


class TestCheckClassBalance:
    """Tests for check_class_balance function."""

    def test_balanced_splits_default_ratio(self):
        """Passes when class ratio is within default 3.0 threshold."""
        # Arrange - 2:1 ratio, well within 3.0
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
                Sample("id3", "rec3.wav", "Not USV", "/path/id3.png"),
            ],
        }

        # Act
        result = check_class_balance(splits)

        # Assert
        assert result.passed
        assert "acceptable" in result.message
        assert len(result.details) > 0  # Should have summary

    def test_imbalanced_splits_exceeds_ratio(self):
        """Fails when class ratio exceeds threshold."""
        # Arrange - 4:1 ratio, exceeds default 3.0
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
                Sample("id3", "rec3.wav", "USV", "/path/id3.png"),
                Sample("id4", "rec4.wav", "USV", "/path/id4.png"),
                Sample("id5", "rec5.wav", "Not USV", "/path/id5.png"),
            ],
        }

        # Act
        result = check_class_balance(splits, max_ratio=3.0)

        # Assert
        assert not result.passed
        assert "Imbalanced classes" in result.message
        assert any("train" in detail for detail in result.details)

    def test_missing_class_fails(self):
        """Fails when a split has only one class."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
            ],
        }

        # Act
        result = check_class_balance(splits)

        # Assert
        assert not result.passed
        assert "missing class" in result.details[0]

    def test_custom_ratio_threshold(self):
        """Uses custom max_ratio parameter."""
        # Arrange - 2:1 ratio
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
                Sample("id3", "rec3.wav", "Not USV", "/path/id3.png"),
            ],
        }

        # Act - stricter ratio
        result = check_class_balance(splits, max_ratio=1.5)

        # Assert
        assert not result.passed


class TestCheckNoDuplicateIds:
    """Tests for check_no_duplicate_ids function."""

    def test_all_unique_ids(self):
        """Passes when all candidate_ids are unique."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
            ],
            "test": [
                Sample("id3", "rec3.wav", "Not USV", "/path/id3.png"),
            ],
        }

        # Act
        result = check_no_duplicate_ids(splits)

        # Assert
        assert result.passed
        assert "All 3 candidate_ids are unique" in result.message

    def test_duplicate_ids_within_split(self):
        """Detects duplicates within same split."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),  # Duplicate
            ],
        }

        # Act
        result = check_no_duplicate_ids(splits)

        # Assert
        assert not result.passed
        assert "1 duplicate candidate_ids found" in result.message
        assert "id1" in result.details

    def test_duplicate_ids_across_splits(self):
        """Detects duplicates across different splits."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
            ],
            "test": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),  # Duplicate
            ],
        }

        # Act
        result = check_no_duplicate_ids(splits)

        # Assert
        assert not result.passed
        assert "id1" in result.details


class TestCheckPopulationCoverage:
    """Tests for check_population_coverage function."""

    def test_no_metadata_skips_check(self):
        """Skips check when no metadata provided."""
        # Arrange
        splits = {
            "train": [Sample("id1", "rec1.wav", "USV", "/path/id1.png")],
            "test": [Sample("id2", "rec2.wav", "USV", "/path/id2.png")],
        }

        # Act
        result = check_population_coverage(splits, metadata=None)

        # Assert
        assert result.passed
        assert "No metadata provided" in result.message

    def test_all_populations_in_test_passes(self):
        """Passes when test set includes all populations from training."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
            ],
            "test": [
                Sample("id3", "rec3.wav", "USV", "/path/id3.png"),
                Sample("id4", "rec4.wav", "USV", "/path/id4.png"),
            ],
        }
        metadata = {
            "rec1.wav": RecordingMetadata("rec1.wav", "001", "PopA"),
            "rec2.wav": RecordingMetadata("rec2.wav", "002", "PopB"),
            "rec3.wav": RecordingMetadata("rec3.wav", "003", "PopA"),
            "rec4.wav": RecordingMetadata("rec4.wav", "004", "PopB"),
        }

        # Act
        result = check_population_coverage(splits, metadata)

        # Assert
        assert result.passed
        assert "PopA" in result.message or "PopB" in result.message

    def test_missing_population_in_test_fails(self):
        """Fails when test set missing a population."""
        # Arrange
        splits = {
            "train": [
                Sample("id1", "rec1.wav", "USV", "/path/id1.png"),
                Sample("id2", "rec2.wav", "USV", "/path/id2.png"),
            ],
            "test": [
                Sample("id3", "rec3.wav", "USV", "/path/id3.png"),
            ],
        }
        metadata = {
            "rec1.wav": RecordingMetadata("rec1.wav", "001", "PopA"),
            "rec2.wav": RecordingMetadata("rec2.wav", "002", "PopB"),
            "rec3.wav": RecordingMetadata("rec3.wav", "003", "PopA"),
        }

        # Act
        result = check_population_coverage(splits, metadata)

        # Assert
        assert not result.passed
        assert "PopB" in result.message

    def test_all_unknown_population_passes(self):
        """Passes when all recordings have unknown population."""
        # Arrange
        splits = {
            "train": [Sample("id1", "rec1.wav", "USV", "/path/id1.png")],
            "test": [Sample("id2", "rec2.wav", "USV", "/path/id2.png")],
        }
        metadata = {
            "rec1.wav": RecordingMetadata("rec1.wav", "001", "unknown"),
            "rec2.wav": RecordingMetadata("rec2.wav", "002", "unknown"),
        }

        # Act
        result = check_population_coverage(splits, metadata)

        # Assert
        assert result.passed
        assert "stratification not applicable" in result.message


class TestCheckSplitSizes:
    """Tests for check_split_sizes function."""

    def test_exact_split_ratios(self):
        """Passes when split sizes match expected ratios exactly."""
        # Arrange - 100 samples: 70/15/15 split
        splits = {
            "train": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(70)],
            "val": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(70, 85)],
            "test": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(85, 100)],
        }

        # Act
        result = check_split_sizes(splits)

        # Assert
        assert result.passed
        assert "within tolerance" in result.message

    def test_split_sizes_within_tolerance(self):
        """Passes when split sizes are within tolerance."""
        # Arrange - 100 samples: 68/17/15 split (within 0.10 tolerance)
        splits = {
            "train": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(68)],
            "val": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(68, 85)],
            "test": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(85, 100)],
        }

        # Act
        result = check_split_sizes(splits, tolerance=0.10)

        # Assert
        assert result.passed

    def test_split_sizes_exceed_tolerance(self):
        """Fails when split sizes exceed tolerance."""
        # Arrange - 100 samples: 50/30/20 split (deviates too much from 70/15/15)
        splits = {
            "train": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(50)],
            "val": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(50, 80)],
            "test": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(80, 100)],
        }

        # Act
        result = check_split_sizes(splits, tolerance=0.10)

        # Assert
        assert not result.passed
        assert "deviate from expected ratios" in result.message
        assert any("train" in detail for detail in result.details)

    def test_empty_splits_fails(self):
        """Fails when no samples in any split."""
        # Arrange
        splits = {"train": [], "val": [], "test": []}

        # Act
        result = check_split_sizes(splits)

        # Assert
        assert not result.passed
        assert "No samples" in result.message

    def test_custom_split_ratios(self):
        """Uses custom expected ratios."""
        # Arrange - 100 samples: 80/10/10 split
        splits = {
            "train": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(80)],
            "val": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(80, 90)],
            "test": [Sample(f"id{i}", f"rec{i}.wav", "USV", f"/path/{i}.png") for i in range(90, 100)],
        }

        # Act
        result = check_split_sizes(
            splits,
            expected_train_ratio=0.80,
            expected_val_ratio=0.10,
            expected_test_ratio=0.10,
        )

        # Assert
        assert result.passed


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_all_passed_true_when_all_checks_pass(self):
        """all_passed is True when all checks passed."""
        # Arrange
        report = QualityReport(checks=[
            CheckResult("Check1", True, "OK"),
            CheckResult("Check2", True, "OK"),
        ])

        # Act & Assert
        assert report.all_passed is True

    def test_all_passed_false_when_any_check_fails(self):
        """all_passed is False when any check failed."""
        # Arrange
        report = QualityReport(checks=[
            CheckResult("Check1", True, "OK"),
            CheckResult("Check2", False, "Failed"),
        ])

        # Act & Assert
        assert report.all_passed is False

    def test_n_passed_counts_passed_checks(self):
        """n_passed counts number of passing checks."""
        # Arrange
        report = QualityReport(checks=[
            CheckResult("Check1", True, "OK"),
            CheckResult("Check2", False, "Failed"),
            CheckResult("Check3", True, "OK"),
        ])

        # Act & Assert
        assert report.n_passed == 2

    def test_n_failed_counts_failed_checks(self):
        """n_failed counts number of failing checks."""
        # Arrange
        report = QualityReport(checks=[
            CheckResult("Check1", True, "OK"),
            CheckResult("Check2", False, "Failed"),
            CheckResult("Check3", False, "Failed"),
        ])

        # Act & Assert
        assert report.n_failed == 2

    def test_summary_contains_check_status(self):
        """summary() includes status and message for each check."""
        # Arrange
        report = QualityReport(checks=[
            CheckResult("Check1", True, "Everything fine"),
            CheckResult("Check2", False, "Something wrong", ["detail1", "detail2"]),
        ])

        # Act
        summary = report.summary()

        # Assert
        assert "2 passed" in summary or "1/2 passed" in summary
        assert "PASS" in summary
        assert "FAIL" in summary
        assert "Check1" in summary
        assert "Check2" in summary
        assert "Everything fine" in summary
        assert "Something wrong" in summary

    def test_summary_limits_details_to_five(self):
        """summary() limits details shown to 5."""
        # Arrange
        many_details = [f"detail{i}" for i in range(10)]
        report = QualityReport(checks=[
            CheckResult("Check1", False, "Many issues", many_details),
        ])

        # Act
        summary = report.summary()

        # Assert
        assert "and 5 more" in summary


# ============================================================================
# Module 2: metadata.py Tests
# ============================================================================


class TestExtractSourceFileFromCandidateId:
    """Tests for extract_source_file_from_candidate_id function."""

    def test_regular_candidate_id(self):
        """Extracts source file from regular candidate_id."""
        # Arrange
        candidate_id = "2024-09-30_11-18-17_0000001_00000764"

        # Act
        result = extract_source_file_from_candidate_id(candidate_id)

        # Assert
        assert result == "2024-09-30_11-18-17_0000001.wav"

    def test_noise_candidate_id(self):
        """Extracts source file from noise candidate_id."""
        # Arrange
        candidate_id = "2024-09-30_11-18-17_0000001_noise_abc123"

        # Act
        result = extract_source_file_from_candidate_id(candidate_id)

        # Assert
        assert result == "2024-09-30_11-18-17_0000001.wav"

    def test_different_date_time_format(self):
        """Handles different date/time values."""
        # Arrange
        candidate_id = "2025-12-31_23-59-59_0000999_12345678"

        # Act
        result = extract_source_file_from_candidate_id(candidate_id)

        # Assert
        assert result == "2025-12-31_23-59-59_0000999.wav"

    def test_fallback_for_malformed_id(self):
        """Falls back for malformed candidate_id."""
        # Arrange
        candidate_id = "invalid"

        # Act
        result = extract_source_file_from_candidate_id(candidate_id)

        # Assert
        assert result == "invalid.wav"


class TestExtractMouseIdFromSourceFile:
    """Tests for extract_mouse_id_from_source_file function."""

    def test_standard_source_file(self):
        """Extracts mouse_id from standard source file."""
        # Arrange
        source_file = "2024-09-30_11-18-17_0000001.wav"

        # Act
        result = extract_mouse_id_from_source_file(source_file)

        # Assert
        assert result == "0000001"

    def test_different_mouse_id(self):
        """Extracts different mouse_id values."""
        # Arrange
        source_file = "2024-09-30_11-18-17_0000999.wav"

        # Act
        result = extract_mouse_id_from_source_file(source_file)

        # Assert
        assert result == "0000999"

    def test_malformed_source_file_returns_unknown(self):
        """Returns 'unknown' for malformed source file."""
        # Arrange
        source_file = "invalid.wav"

        # Act
        result = extract_mouse_id_from_source_file(source_file)

        # Assert
        assert result == "unknown"


class TestLoadMetadata:
    """Tests for load_metadata function."""

    def test_load_valid_metadata_csv(self, tmp_path):
        """Loads metadata from valid CSV file."""
        # Arrange
        csv_path = tmp_path / "metadata.csv"
        csv_content = """source_file,mouse_id,population,notes
rec1.wav,001,PopA,Test recording 1
rec2.wav,002,PopB,Test recording 2
"""
        csv_path.write_text(csv_content)

        # Act
        result = load_metadata(csv_path)

        # Assert
        assert len(result) == 2
        assert "rec1.wav" in result
        assert result["rec1.wav"].mouse_id == "001"
        assert result["rec1.wav"].population == "PopA"
        assert result["rec1.wav"].notes == "Test recording 1"
        assert result["rec2.wav"].mouse_id == "002"

    def test_load_metadata_with_missing_columns(self, tmp_path):
        """Handles CSV with missing optional columns."""
        # Arrange
        csv_path = tmp_path / "metadata.csv"
        csv_content = """source_file,mouse_id
rec1.wav,001
"""
        csv_path.write_text(csv_content)

        # Act
        result = load_metadata(csv_path)

        # Assert
        assert len(result) == 1
        assert result["rec1.wav"].population == "unknown"
        assert result["rec1.wav"].notes == ""

    def test_load_nonexistent_file_returns_empty_dict(self, tmp_path):
        """Returns empty dict when file doesn't exist."""
        # Arrange
        csv_path = tmp_path / "nonexistent.csv"

        # Act
        result = load_metadata(csv_path)

        # Assert
        assert result == {}

    def test_load_empty_csv_returns_empty_dict(self, tmp_path):
        """Returns empty dict for CSV with only headers."""
        # Arrange
        csv_path = tmp_path / "metadata.csv"
        csv_content = """source_file,mouse_id,population,notes
"""
        csv_path.write_text(csv_content)

        # Act
        result = load_metadata(csv_path)

        # Assert
        assert result == {}


class TestGenerateMetadataCsv:
    """Tests for generate_metadata_csv function."""

    def test_generate_from_labels_csv(self, tmp_path):
        """Generates metadata from labels.csv."""
        # Arrange
        labels_csv = tmp_path / "labels.csv"
        labels_content = """candidate_id,label,labeled_at
2024-09-30_11-18-17_0000001_00000764,USV,2024-01-01
2024-09-30_11-18-17_0000002_00001234,Not USV,2024-01-02
"""
        labels_csv.write_text(labels_content)

        noise_csv = tmp_path / "noise.csv"
        noise_csv.write_text("source_file\n")  # Empty

        output_csv = tmp_path / "output.csv"

        # Act
        result = generate_metadata_csv(labels_csv, noise_csv, output_csv)

        # Assert
        assert len(result) == 2
        assert any(m.source_file == "2024-09-30_11-18-17_0000001.wav" for m in result)
        assert any(m.source_file == "2024-09-30_11-18-17_0000002.wav" for m in result)
        assert output_csv.exists()

    def test_generate_from_noise_samples_csv(self, tmp_path):
        """Generates metadata from noise_samples_final.csv."""
        # Arrange
        labels_csv = tmp_path / "labels.csv"
        labels_csv.write_text("candidate_id,label,labeled_at\n")  # Empty

        noise_csv = tmp_path / "noise.csv"
        noise_content = """source_file,candidate_id,spectrogram_path
2024-09-30_11-18-17_0000003.wav,noise_id_1,/path/noise1.png
2024-09-30_11-18-17_0000004,noise_id_2,/path/noise2.png
"""
        noise_csv.write_text(noise_content)

        output_csv = tmp_path / "output.csv"

        # Act
        result = generate_metadata_csv(labels_csv, noise_csv, output_csv)

        # Assert
        assert len(result) == 2
        assert any(m.source_file == "2024-09-30_11-18-17_0000003.wav" for m in result)
        assert any(m.source_file == "2024-09-30_11-18-17_0000004.wav" for m in result)

    def test_generate_extracts_mouse_ids(self, tmp_path):
        """Extracts mouse_id correctly from source files."""
        # Arrange
        labels_csv = tmp_path / "labels.csv"
        labels_content = """candidate_id,label,labeled_at
2024-09-30_11-18-17_0000999_00000001,USV,2024-01-01
"""
        labels_csv.write_text(labels_content)

        noise_csv = tmp_path / "noise.csv"
        noise_csv.write_text("source_file\n")

        output_csv = tmp_path / "output.csv"

        # Act
        result = generate_metadata_csv(labels_csv, noise_csv, output_csv)

        # Assert
        assert len(result) == 1
        assert result[0].mouse_id == "0000999"

    def test_generate_writes_csv_with_correct_format(self, tmp_path):
        """Writes CSV with correct headers and format."""
        # Arrange
        labels_csv = tmp_path / "labels.csv"
        labels_content = """candidate_id,label,labeled_at
2024-09-30_11-18-17_0000001_00000001,USV,2024-01-01
"""
        labels_csv.write_text(labels_content)

        noise_csv = tmp_path / "noise.csv"
        noise_csv.write_text("source_file\n")

        output_csv = tmp_path / "output.csv"

        # Act
        generate_metadata_csv(labels_csv, noise_csv, output_csv)

        # Assert
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert "source_file" in rows[0]
            assert "mouse_id" in rows[0]
            assert "population" in rows[0]
            assert "notes" in rows[0]
            assert rows[0]["population"] == "unknown"

    def test_generate_deduplicates_source_files(self, tmp_path):
        """Deduplicates when same recording appears multiple times."""
        # Arrange
        labels_csv = tmp_path / "labels.csv"
        labels_content = """candidate_id,label,labeled_at
2024-09-30_11-18-17_0000001_00000001,USV,2024-01-01
2024-09-30_11-18-17_0000001_00000002,Not USV,2024-01-02
"""
        labels_csv.write_text(labels_content)

        noise_csv = tmp_path / "noise.csv"
        noise_csv.write_text("source_file\n")

        output_csv = tmp_path / "output.csv"

        # Act
        result = generate_metadata_csv(labels_csv, noise_csv, output_csv)

        # Assert
        assert len(result) == 1  # Only one unique source file
        assert result[0].source_file == "2024-09-30_11-18-17_0000001.wav"

    def test_generate_creates_output_directory(self, tmp_path):
        """Creates output directory if it doesn't exist."""
        # Arrange
        labels_csv = tmp_path / "labels.csv"
        labels_csv.write_text("candidate_id,label,labeled_at\n")

        noise_csv = tmp_path / "noise.csv"
        noise_csv.write_text("source_file\n")

        output_csv = tmp_path / "subdir" / "output.csv"

        # Act
        generate_metadata_csv(labels_csv, noise_csv, output_csv)

        # Assert
        assert output_csv.exists()
        assert output_csv.parent.exists()
