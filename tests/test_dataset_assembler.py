"""Tests for the training data assembly pipeline.

Tests cover:
1. Full assembly produces non-empty CSVs for all splits
2. No recording leakage across splits (ADR-004)
3. Negatives from all 3 sources (random, gap, low-energy)
4. Jittered count matches expected (originals * jitter_n_samples)
5. Dry-run produces no output files
6. Report statistics match actual sample counts
7. Empty labels directory raises ValueError
8. Spectrogram paths in CSVs point to existing files

Fixtures create 3 synthetic WAV files with 55kHz tones at known positions
and matching LabelStorage JSON files.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.dataset.assembler import (
    AssemblyConfig,
    AssemblyReport,
    DatasetAssembler,
)


# ==================== Fixtures ====================


SAMPLE_RATE = 300_000
TONE_FREQ = 55_000  # 55 kHz — well within USV band (20-120 kHz)

# Detection locations (in seconds) for each of 3 recordings
# Durations 15-30ms (realistic USV lengths, fit within 40ms jitter window)
# Gaps between detections >= 100ms for gap-negative testing
RECORDING_DETECTIONS = {
    "rec_001": [(0.5, 0.520), (1.0, 1.025), (2.0, 2.015)],
    "rec_002": [(0.3, 0.318), (1.5, 1.530)],
    "rec_003": [(0.8, 0.825), (1.2, 1.220), (2.5, 2.518)],
}


def _create_tone_wav(wav_path: Path, duration_s: float, tone_intervals: list[tuple[float, float]]):
    """Create a WAV file with 55kHz tones at specified intervals.

    Background is silence; tones simulate USV-like signals.
    """
    n_samples = int(duration_s * SAMPLE_RATE)
    audio = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples) / SAMPLE_RATE

    for start_s, end_s in tone_intervals:
        start_idx = int(start_s * SAMPLE_RATE)
        end_idx = min(int(end_s * SAMPLE_RATE), n_samples)
        audio[start_idx:end_idx] = 0.5 * np.sin(
            2 * np.pi * TONE_FREQ * t[start_idx:end_idx]
        ).astype(np.float32)

    sf.write(str(wav_path), audio, SAMPLE_RATE)


def _create_label_json(json_path: Path, wav_path: Path, detections: list[tuple[float, float]]):
    """Create a LabelStorage JSON file matching ADR-010 format."""
    det_list = []
    for start_s, end_s in detections:
        det_list.append({
            "start_time_s": start_s,
            "end_time_s": end_s,
            "duration_s": end_s - start_s,
            "start_col": int(start_s * SAMPLE_RATE / 128),
            "end_col": int(end_s * SAMPLE_RATE / 128),
            "max_probability": 0.95,
            "mean_probability": 0.85,
            "user_adjusted": False,
            "original_start_time_s": start_s,
            "original_end_time_s": end_s,
            "user_action": None,
            "original_cnn_probability": None,
        })

    data = {
        "metadata": {
            "wav_file": str(wav_path.resolve()),
            "model_file": "dummy_model.pt",
            "created_at": "2026-02-21T12:00:00Z",
            "duration_s": 4.0,
            "sample_rate": SAMPLE_RATE,
            "n_detections": len(detections),
            "file_label": None,
        },
        "detections": det_list,
        "detection_params": {
            "high_threshold": 0.40,
            "low_threshold": 0.28,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@pytest.fixture
def mock_wav_dir(tmp_path):
    """Create 3 WAV files with synthetic 55kHz tones at known times."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()

    for rec_name, detections in RECORDING_DETECTIONS.items():
        wav_path = wav_dir / f"{rec_name}.wav"
        _create_tone_wav(wav_path, duration_s=4.0, tone_intervals=detections)

    return wav_dir


@pytest.fixture
def mock_labels_dir(tmp_path, mock_wav_dir):
    """Create 3 LabelStorage JSONs matching the WAV files."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    for rec_name, detections in RECORDING_DETECTIONS.items():
        wav_path = mock_wav_dir / f"{rec_name}.wav"
        json_path = labels_dir / f"{rec_name}.json"
        _create_label_json(json_path, wav_path, detections)

    return labels_dir


@pytest.fixture
def assembly_config(tmp_path, mock_wav_dir, mock_labels_dir):
    """Default AssemblyConfig using mock data."""
    return AssemblyConfig(
        labels_dir=mock_labels_dir,
        wav_dir=mock_wav_dir,
        output_dir=tmp_path / "output",
        jitter_n_samples=3,  # Fewer for test speed
        neg_ratio=1.0,
        seed=42,
    )


# ==================== Test Cases ====================


class TestAssemblyProducesSplits:
    """Test 1: Full assembly with 3 recordings produces non-empty CSVs."""

    def test_assembly_produces_splits(self, assembly_config):
        assembler = DatasetAssembler(assembly_config)
        report = assembler.assemble(dry_run=False)

        # All three CSVs should exist and be non-empty
        for split_name in ["train", "val", "test"]:
            csv_path = assembly_config.output_dir / f"{split_name}.csv"
            assert csv_path.exists(), f"{split_name}.csv not created"

        # Total samples should be positives + negatives
        total = report.train_count + report.val_count + report.test_count
        assert total == report.total_positives + report.total_negatives
        assert total > 0

        # Report file should exist
        report_path = assembly_config.output_dir / "assembly_report.json"
        assert report_path.exists()


class TestNoRecordingLeakage:
    """Test 2: No recording appears in multiple splits (ADR-004)."""

    def test_no_recording_leakage(self, assembly_config):
        assembler = DatasetAssembler(assembly_config)
        assembler.assemble(dry_run=False)

        # Read all CSVs and collect recordings per split
        recordings_by_split: dict[str, set[str]] = {}
        for split_name in ["train", "val", "test"]:
            csv_path = assembly_config.output_dir / f"{split_name}.csv"
            recordings = set()
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    recordings.add(row["source_file"])
            recordings_by_split[split_name] = recordings

        # No overlap between any pair of splits
        for s1 in ["train", "val", "test"]:
            for s2 in ["train", "val", "test"]:
                if s1 >= s2:
                    continue
                overlap = recordings_by_split[s1] & recordings_by_split[s2]
                assert not overlap, (
                    f"Recording leakage: {overlap} in both {s1} and {s2}"
                )


class TestNegativeSourceTypes:
    """Test 3: Negatives come from all 3 sources (check candidate_id prefixes)."""

    def test_negative_source_types(self, assembly_config):
        assembler = DatasetAssembler(assembly_config)
        report = assembler.assemble(dry_run=False)

        # Collect all negative candidate IDs from CSVs
        neg_ids: list[str] = []
        for split_name in ["train", "val", "test"]:
            csv_path = assembly_config.output_dir / f"{split_name}.csv"
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["label"] == "Not USV":
                        neg_ids.append(row["candidate_id"])

        assert len(neg_ids) > 0, "No negatives generated"

        # Check for all 3 source types via candidate_id naming convention
        has_random = any("_neg_rand_" in cid for cid in neg_ids)
        has_gap = any("_neg_gap_" in cid for cid in neg_ids)
        has_low_energy = any("_neg_lowE_" in cid for cid in neg_ids)

        assert has_random, "No random negatives found (expected _neg_rand_ prefix)"
        assert has_gap, "No gap negatives found (expected _neg_gap_ prefix)"
        assert has_low_energy, "No low-energy negatives found (expected _neg_lowE_ prefix)"


class TestJitteredCount:
    """Test 4: Jittered count = n_originals * jitter_n_samples."""

    def test_jittered_count(self, assembly_config):
        assembler = DatasetAssembler(assembly_config)
        report = assembler.assemble(dry_run=True)

        # Total detections across all recordings
        n_originals = sum(len(dets) for dets in RECORDING_DETECTIONS.values())
        n_jitter = assembly_config.jitter_n_samples

        # Total positives = originals + (originals * jitter_n_samples)
        expected = n_originals * (1 + n_jitter)
        assert report.total_positives == expected, (
            f"Expected {expected} positives ({n_originals} originals + "
            f"{n_originals * n_jitter} jittered), got {report.total_positives}"
        )


class TestDryRunNoOutput:
    """Test 5: Dry-run produces no files."""

    def test_dry_run_no_output(self, assembly_config):
        assembler = DatasetAssembler(assembly_config)
        report = assembler.assemble(dry_run=True)

        # Output dir should not exist (or be empty)
        output = assembly_config.output_dir
        if output.exists():
            files = list(output.rglob("*"))
            assert len(files) == 0, f"Dry run created files: {files}"

        # But report should have statistics
        assert report.total_positives > 0
        assert report.total_negatives > 0
        assert report.n_recordings == 3


class TestReportAccuracy:
    """Test 6: Report statistics match actual sample counts."""

    def test_report_accuracy(self, assembly_config):
        assembler = DatasetAssembler(assembly_config)
        report = assembler.assemble(dry_run=False)

        # Count actual samples from CSVs
        actual_train = actual_val = actual_test = 0
        actual_pos = actual_neg = 0

        for split_name in ["train", "val", "test"]:
            csv_path = assembly_config.output_dir / f"{split_name}.csv"
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if split_name == "train":
                        actual_train += 1
                    elif split_name == "val":
                        actual_val += 1
                    else:
                        actual_test += 1

                    if row["label"] == "USV":
                        actual_pos += 1
                    else:
                        actual_neg += 1

        assert report.train_count == actual_train
        assert report.val_count == actual_val
        assert report.test_count == actual_test
        assert report.total_positives == actual_pos
        assert report.total_negatives == actual_neg


class TestEmptyLabelsDir:
    """Test 7: Empty directory raises ValueError with clear message."""

    def test_empty_labels_dir(self, tmp_path, mock_wav_dir):
        empty_dir = tmp_path / "empty_labels"
        empty_dir.mkdir()

        config = AssemblyConfig(
            labels_dir=empty_dir,
            wav_dir=mock_wav_dir,
            output_dir=tmp_path / "output",
        )
        assembler = DatasetAssembler(config)

        with pytest.raises(ValueError, match="No JSON files found"):
            assembler.assemble()


class TestSpectrogramPathsExist:
    """Test 8: All spectrogram_path values in CSVs point to existing files."""

    def test_spectrogram_paths_exist(self, assembly_config):
        assembler = DatasetAssembler(assembly_config)
        assembler.assemble(dry_run=False)

        missing: list[str] = []
        for split_name in ["train", "val", "test"]:
            csv_path = assembly_config.output_dir / f"{split_name}.csv"
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    spec_path = row.get("spectrogram_path", "")
                    if spec_path and not Path(spec_path).exists():
                        missing.append(f"{split_name}: {row['candidate_id']} -> {spec_path}")

        assert not missing, (
            f"{len(missing)} spectrogram files missing:\n" +
            "\n".join(missing[:10])
        )


class TestJitterFailureForLongUSVs:
    """Test 9: USVs >= 40ms get no jitter (window too small for overlap constraint)."""

    def test_long_usv_gets_no_jitter(self, tmp_path, mock_wav_dir):
        """With default 40ms window and 0.5 overlap, a 50ms USV cannot be jittered."""
        # Create a label JSON with one long detection (50ms)
        labels_dir = tmp_path / "labels_long"
        labels_dir.mkdir()

        long_detection = {
            "metadata": {
                "wav_file": str(mock_wav_dir / "rec_001.wav"),
                "duration_s": 4.0,
                "sample_rate": SAMPLE_RATE,
                "n_detections": 1,
            },
            "detections": [{
                "start_time_s": 1.0,
                "end_time_s": 1.05,  # 50ms — exceeds the 40ms jitter threshold
                "duration_s": 0.05,
                "max_probability": 0.9,
                "user_action": None,
            }],
        }

        with open(labels_dir / "rec_001.json", "w", encoding="utf-8") as f:
            json.dump(long_detection, f)

        config = AssemblyConfig(
            labels_dir=labels_dir,
            wav_dir=mock_wav_dir,
            output_dir=tmp_path / "output_long",
            jitter_n_samples=5,
            seed=42,
        )
        assembler = DatasetAssembler(config)
        report = assembler.assemble(dry_run=True)

        # Should have exactly 1 positive (the original, no jittered copies)
        assert report.total_positives == 1, (
            f"Expected 1 positive (long USV, no jitter), got {report.total_positives}"
        )


class TestAllLabelsDeleted:
    """Test 10: JSON where all detections are deleted raises ValueError."""

    def test_all_labels_deleted(self, tmp_path, mock_wav_dir):
        labels_dir = tmp_path / "labels_deleted"
        labels_dir.mkdir()

        deleted_json = {
            "metadata": {
                "wav_file": str(mock_wav_dir / "rec_001.wav"),
                "duration_s": 4.0,
                "sample_rate": SAMPLE_RATE,
                "n_detections": 2,
            },
            "detections": [
                {
                    "start_time_s": 0.5,
                    "end_time_s": 0.52,
                    "duration_s": 0.02,
                    "max_probability": 0.9,
                    "user_action": "deleted_by_user",
                },
                {
                    "start_time_s": 1.0,
                    "end_time_s": 1.02,
                    "duration_s": 0.02,
                    "max_probability": 0.85,
                    "user_action": "deleted_by_user",
                },
            ],
        }

        with open(labels_dir / "rec_001.json", "w", encoding="utf-8") as f:
            json.dump(deleted_json, f)

        config = AssemblyConfig(
            labels_dir=labels_dir,
            wav_dir=mock_wav_dir,
            output_dir=tmp_path / "output_deleted",
        )
        assembler = DatasetAssembler(config)

        with pytest.raises(ValueError, match="No valid detections found"):
            assembler.assemble()
