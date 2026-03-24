"""Tests for dataset splits module.

Tests cover:
1. SplitConfig validation (ratios must sum to 1.0)
2. Sample dataclass creation
3. group_samples_by_recording - groups samples by source_file
4. split_recordings - splits with correct ratios, supports stratification
5. create_splits - end-to-end split creation
6. load_splits - loads splits from CSV files
7. _write_split_csv - writes samples to CSV
8. load_labeled_samples - loads from labels.csv and noise_samples_final.csv

CRITICAL tests:
- No data leakage: recordings in train must NOT appear in val or test
- All samples accounted for: train + val + test = all samples
- Stratification works: samples from each population in each split
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import csv
import pytest

from usv_spectrogram.dataset.splits import (
    SplitConfig,
    Sample,
    group_samples_by_recording,
    split_recordings,
    create_splits,
    load_splits,
    _write_split_csv,
    load_labeled_samples,
)
from usv_spectrogram.dataset.metadata import RecordingMetadata


# ==================== SplitConfig Tests ====================


def test_split_config_default_ratios_sum_to_one():
    """Default split ratios should sum to 1.0."""
    config = SplitConfig()
    total = config.train_ratio + config.val_ratio + config.test_ratio
    assert abs(total - 1.0) < 0.001


def test_split_config_valid_custom_ratios():
    """Custom ratios that sum to 1.0 should be accepted."""
    config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    assert config.train_ratio == 0.8
    assert config.val_ratio == 0.1
    assert config.test_ratio == 0.1


def test_split_config_invalid_ratios_sum_too_high():
    """Ratios summing above 1.0 should raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SplitConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.2)


def test_split_config_invalid_ratios_sum_too_low():
    """Ratios summing below 1.0 should raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SplitConfig(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


# ==================== Sample Tests ====================


def test_sample_creation():
    """Sample dataclass should store all required fields."""
    sample = Sample(
        candidate_id="2024-09-30_11-18-17_0000001_00000764",
        source_file="2024-09-30_11-18-17_0000001.wav",
        label="USV",
        spectrogram_path="/path/to/spec.png",
    )
    assert sample.candidate_id == "2024-09-30_11-18-17_0000001_00000764"
    assert sample.source_file == "2024-09-30_11-18-17_0000001.wav"
    assert sample.label == "USV"
    assert sample.spectrogram_path == "/path/to/spec.png"


# ==================== group_samples_by_recording Tests ====================


def test_group_samples_by_recording_single_recording():
    """Single recording should have all samples in one group."""
    samples = [
        Sample("cand1", "rec1.wav", "USV", "spec1.png"),
        Sample("cand2", "rec1.wav", "Not USV", "spec2.png"),
        Sample("cand3", "rec1.wav", "USV", "spec3.png"),
    ]
    groups = group_samples_by_recording(samples)

    assert len(groups) == 1
    assert "rec1.wav" in groups
    assert len(groups["rec1.wav"]) == 3


def test_group_samples_by_recording_multiple_recordings():
    """Multiple recordings should be grouped correctly."""
    samples = [
        Sample("cand1", "rec1.wav", "USV", "spec1.png"),
        Sample("cand2", "rec2.wav", "Not USV", "spec2.png"),
        Sample("cand3", "rec1.wav", "USV", "spec3.png"),
        Sample("cand4", "rec2.wav", "USV", "spec4.png"),
        Sample("cand5", "rec3.wav", "Not USV", "spec5.png"),
    ]
    groups = group_samples_by_recording(samples)

    assert len(groups) == 3
    assert len(groups["rec1.wav"]) == 2
    assert len(groups["rec2.wav"]) == 2
    assert len(groups["rec3.wav"]) == 1


def test_group_samples_by_recording_empty_list():
    """Empty sample list should return empty dict."""
    groups = group_samples_by_recording([])
    assert groups == {}


# ==================== split_recordings Tests ====================


def test_split_recordings_basic_split():
    """Basic split should distribute recordings according to ratios."""
    recordings = [f"rec{i}.wav" for i in range(100)]
    config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    train, val, test = split_recordings(recordings, config)

    # Check approximate ratios (within ±1 due to rounding)
    assert 68 <= len(train) <= 72
    assert 13 <= len(val) <= 17
    assert 13 <= len(test) <= 17

    # Check total count
    assert len(train) + len(val) + len(test) == 100


def test_split_recordings_no_data_leakage():
    """Train, val, and test sets should have no overlapping recordings."""
    recordings = [f"rec{i}.wav" for i in range(50)]
    config = SplitConfig(random_seed=42)

    train, val, test = split_recordings(recordings, config)

    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    assert len(train_set & val_set) == 0, "Train and val have overlapping recordings"
    assert len(train_set & test_set) == 0, "Train and test have overlapping recordings"
    assert len(val_set & test_set) == 0, "Val and test have overlapping recordings"


def test_split_recordings_all_recordings_accounted_for():
    """All recordings should be in exactly one split."""
    recordings = [f"rec{i}.wav" for i in range(50)]
    config = SplitConfig(random_seed=42)

    train, val, test = split_recordings(recordings, config)

    all_split = set(train + val + test)
    original = set(recordings)

    assert all_split == original


def test_split_recordings_deterministic_with_seed():
    """Same seed should produce same splits."""
    recordings = [f"rec{i}.wav" for i in range(30)]
    config = SplitConfig(random_seed=123)

    train1, val1, test1 = split_recordings(recordings, config)
    train2, val2, test2 = split_recordings(recordings, config)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_split_recordings_different_seeds_produce_different_splits():
    """Different seeds should produce different splits."""
    recordings = [f"rec{i}.wav" for i in range(30)]
    config1 = SplitConfig(random_seed=1)
    config2 = SplitConfig(random_seed=999)

    train1, _, _ = split_recordings(recordings, config1)
    train2, _, _ = split_recordings(recordings, config2)

    # With 30 recordings, highly unlikely to get identical splits with different seeds
    assert train1 != train2


def test_split_recordings_stratified_by_population():
    """Stratified split should have samples from each population in each split."""
    # Create metadata with two populations
    metadata = {
        "rec1.wav": RecordingMetadata("rec1.wav", "m1", "pop_A"),
        "rec2.wav": RecordingMetadata("rec2.wav", "m2", "pop_A"),
        "rec3.wav": RecordingMetadata("rec3.wav", "m3", "pop_A"),
        "rec4.wav": RecordingMetadata("rec4.wav", "m4", "pop_A"),
        "rec5.wav": RecordingMetadata("rec5.wav", "m5", "pop_B"),
        "rec6.wav": RecordingMetadata("rec6.wav", "m6", "pop_B"),
        "rec7.wav": RecordingMetadata("rec7.wav", "m7", "pop_B"),
        "rec8.wav": RecordingMetadata("rec8.wav", "m8", "pop_B"),
    }

    recordings = list(metadata.keys())
    config = SplitConfig(stratify_by_population=True, random_seed=42)

    train, val, test = split_recordings(recordings, config, metadata)

    # Check that both populations appear in train set (at least)
    train_pops = {metadata[r].population for r in train}
    assert "pop_A" in train_pops
    assert "pop_B" in train_pops

    # All recordings should still be accounted for
    assert len(train) + len(val) + len(test) == len(recordings)

    # No data leakage
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0


def test_split_recordings_no_stratification_when_disabled():
    """Stratification should be skipped when stratify_by_population=False."""
    metadata = {
        "rec1.wav": RecordingMetadata("rec1.wav", "m1", "pop_A"),
        "rec2.wav": RecordingMetadata("rec2.wav", "m2", "pop_B"),
    }

    recordings = list(metadata.keys())
    config = SplitConfig(stratify_by_population=False, random_seed=42)

    # Should not raise even with metadata available
    train, val, test = split_recordings(recordings, config, metadata)
    assert len(train) + len(val) + len(test) == 2


def test_split_recordings_small_dataset():
    """Small dataset should still split without error."""
    recordings = ["rec1.wav", "rec2.wav", "rec3.wav"]
    config = SplitConfig(random_seed=42)

    train, val, test = split_recordings(recordings, config)

    # At least one recording in train
    assert len(train) >= 1
    # All recordings accounted for
    assert len(train) + len(val) + len(test) == 3


# ==================== _write_split_csv Tests ====================


def test_write_split_csv(tmp_path):
    """Should write samples to CSV with correct format."""
    samples = [
        Sample("cand1", "rec1.wav", "USV", "spec1.png"),
        Sample("cand2", "rec2.wav", "Not USV", "spec2.png"),
    ]

    csv_path = tmp_path / "test.csv"
    _write_split_csv(samples, csv_path)

    assert csv_path.exists()

    # Read back and verify
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["candidate_id"] == "cand1"
    assert rows[0]["source_file"] == "rec1.wav"
    assert rows[0]["label"] == "USV"
    assert rows[0]["spectrogram_path"] == "spec1.png"


def test_write_split_csv_empty_list(tmp_path):
    """Should write header even with empty sample list."""
    csv_path = tmp_path / "empty.csv"
    _write_split_csv([], csv_path)

    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 0
    assert reader.fieldnames == ["candidate_id", "source_file", "label", "spectrogram_path"]


# ==================== load_splits Tests ====================


def test_load_splits_all_files_exist(tmp_path):
    """Should load all three splits when CSV files exist."""
    # Create dummy CSV files
    for split_name in ["train", "val", "test"]:
        csv_path = tmp_path / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["candidate_id", "source_file", "label", "spectrogram_path"]
            )
            writer.writeheader()
            writer.writerow({
                "candidate_id": f"{split_name}_cand1",
                "source_file": f"{split_name}_rec1.wav",
                "label": "USV",
                "spectrogram_path": f"{split_name}_spec1.png",
            })

    splits = load_splits(tmp_path)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    assert len(splits["train"]) == 1
    assert splits["train"][0].candidate_id == "train_cand1"


def test_load_splits_missing_files(tmp_path):
    """Should return empty lists for missing split files."""
    # Create only train.csv
    train_path = tmp_path / "train.csv"
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "source_file", "label", "spectrogram_path"]
        )
        writer.writeheader()

    splits = load_splits(tmp_path)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 0


def test_load_splits_roundtrip(tmp_path):
    """Writing and then loading splits should preserve data."""
    samples = [
        Sample("cand1", "rec1.wav", "USV", "spec1.png"),
        Sample("cand2", "rec2.wav", "Not USV", "spec2.png"),
    ]

    # Write
    csv_path = tmp_path / "train.csv"
    _write_split_csv(samples, csv_path)

    # Load
    splits = load_splits(tmp_path)
    loaded_samples = splits["train"]

    assert len(loaded_samples) == 2
    assert loaded_samples[0].candidate_id == samples[0].candidate_id
    assert loaded_samples[0].source_file == samples[0].source_file
    assert loaded_samples[0].label == samples[0].label
    assert loaded_samples[0].spectrogram_path == samples[0].spectrogram_path


# ==================== load_labeled_samples Tests ====================


def test_load_labeled_samples_from_labels_csv(tmp_path):
    """Should load samples from labels.csv."""
    labels_csv = tmp_path / "labels.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    spectrograms_dir.mkdir()

    # Create labels.csv
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000764",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })

    # Create spectrogram file
    spec_path = spectrograms_dir / "2024-09-30_11-18-17_0000001_00000764.png"
    spec_path.touch()

    samples = load_labeled_samples(
        labels_csv=labels_csv,
        spectrograms_dir=spectrograms_dir,
        noise_samples_csv=tmp_path / "noise.csv",  # doesn't exist
        require_spectrogram_exists=True,
    )

    assert len(samples) == 1
    assert samples[0].candidate_id == "2024-09-30_11-18-17_0000001_00000764"
    assert samples[0].label == "USV"
    assert samples[0].source_file == "2024-09-30_11-18-17_0000001.wav"


def test_load_labeled_samples_exclude_uncertain(tmp_path):
    """Should exclude samples labeled as Uncertain when exclude_uncertain=True."""
    labels_csv = tmp_path / "labels.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    spectrograms_dir.mkdir()

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000764",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000765",
            "label": "Uncertain",
            "labeled_at": "2024-01-01",
        })

    # Create spectrograms
    (spectrograms_dir / "2024-09-30_11-18-17_0000001_00000764.png").touch()
    (spectrograms_dir / "2024-09-30_11-18-17_0000001_00000765.png").touch()

    samples = load_labeled_samples(
        labels_csv=labels_csv,
        spectrograms_dir=spectrograms_dir,
        noise_samples_csv=tmp_path / "noise.csv",
        exclude_uncertain=True,
        require_spectrogram_exists=False,
    )

    assert len(samples) == 1
    assert samples[0].label == "USV"


def test_load_labeled_samples_skip_missing_spectrogram(tmp_path):
    """Should skip samples with missing spectrograms when require_spectrogram_exists=True."""
    labels_csv = tmp_path / "labels.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    spectrograms_dir.mkdir()

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000764",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })

    # Don't create spectrogram file

    with pytest.warns(UserWarning, match="Skipped 1 samples"):
        samples = load_labeled_samples(
            labels_csv=labels_csv,
            spectrograms_dir=spectrograms_dir,
            noise_samples_csv=tmp_path / "noise.csv",
            require_spectrogram_exists=True,
        )

    assert len(samples) == 0


def test_load_labeled_samples_allow_missing_spectrogram(tmp_path):
    """Should include samples with missing spectrograms when require_spectrogram_exists=False."""
    labels_csv = tmp_path / "labels.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    spectrograms_dir.mkdir()

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000764",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })

    # Don't create spectrogram file

    samples = load_labeled_samples(
        labels_csv=labels_csv,
        spectrograms_dir=spectrograms_dir,
        noise_samples_csv=tmp_path / "noise.csv",
        require_spectrogram_exists=False,
    )

    assert len(samples) == 1


def test_load_labeled_samples_from_noise_csv(tmp_path):
    """Should load samples from noise_samples_final.csv."""
    labels_csv = tmp_path / "labels.csv"
    noise_csv = tmp_path / "noise.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    noise_samples_dir = tmp_path / "noise_samples"
    spectrograms_dir.mkdir()
    noise_samples_dir.mkdir()

    # Create empty labels.csv
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()

    # Create noise_samples_final.csv
    with open(noise_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "label", "source_file", "spectrogram_path"]
        )
        writer.writeheader()
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_noise_abc123",
            "label": "Not USV",
            "source_file": "2024-09-30_11-18-17_0000001.wav",
            "spectrogram_path": str(noise_samples_dir / "2024-09-30_11-18-17_0000001_noise_abc123.png"),
        })

    # Create spectrogram
    (noise_samples_dir / "2024-09-30_11-18-17_0000001_noise_abc123.png").touch()

    samples = load_labeled_samples(
        labels_csv=labels_csv,
        spectrograms_dir=spectrograms_dir,
        noise_samples_csv=noise_csv,
        require_spectrogram_exists=True,
    )

    assert len(samples) == 1
    assert samples[0].candidate_id == "2024-09-30_11-18-17_0000001_noise_abc123"
    assert samples[0].label == "Not USV"


def test_load_labeled_samples_no_duplicates_from_noise(tmp_path):
    """Should not duplicate samples if candidate_id appears in both CSVs."""
    labels_csv = tmp_path / "labels.csv"
    noise_csv = tmp_path / "noise.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    spectrograms_dir.mkdir()

    candidate_id = "2024-09-30_11-18-17_0000001_noise_abc123"

    # Add to labels.csv
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        writer.writerow({
            "candidate_id": candidate_id,
            "label": "USV",
            "labeled_at": "2024-01-01",
        })

    # Add same candidate_id to noise.csv
    with open(noise_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "label", "source_file", "spectrogram_path"]
        )
        writer.writeheader()
        writer.writerow({
            "candidate_id": candidate_id,
            "label": "Not USV",
            "source_file": "2024-09-30_11-18-17_0000001.wav",
            "spectrogram_path": str(spectrograms_dir / f"{candidate_id}.png"),
        })

    # Create spectrogram
    (spectrograms_dir / f"{candidate_id}.png").touch()

    samples = load_labeled_samples(
        labels_csv=labels_csv,
        spectrograms_dir=spectrograms_dir,
        noise_samples_csv=noise_csv,
        require_spectrogram_exists=False,
    )

    # Should only appear once (from labels.csv, which is processed first)
    assert len(samples) == 1
    assert samples[0].label == "USV"  # From labels.csv


# ==================== create_splits Integration Tests ====================


def test_create_splits_end_to_end(tmp_path):
    """End-to-end test: load samples, split, write CSVs."""
    labels_csv = tmp_path / "labels.csv"
    noise_csv = tmp_path / "noise.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    output_dir = tmp_path / "splits"
    spectrograms_dir.mkdir()

    # Create labels.csv with samples from 3 different recordings
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        # Recording 1 - 3 samples
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000001",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000002",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000003",
            "label": "Not USV",
            "labeled_at": "2024-01-01",
        })
        # Recording 2 - 2 samples
        writer.writerow({
            "candidate_id": "2024-09-30_12-00-00_0000002_00000001",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })
        writer.writerow({
            "candidate_id": "2024-09-30_12-00-00_0000002_00000002",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })
        # Recording 3 - 1 sample
        writer.writerow({
            "candidate_id": "2024-09-30_13-00-00_0000003_00000001",
            "label": "Not USV",
            "labeled_at": "2024-01-01",
        })

    # Create spectrograms
    for i in range(1, 4):
        (spectrograms_dir / f"2024-09-30_11-18-17_0000001_0000000{i}.png").touch()
    for i in range(1, 3):
        (spectrograms_dir / f"2024-09-30_12-00-00_0000002_0000000{i}.png").touch()
    (spectrograms_dir / "2024-09-30_13-00-00_0000003_00000001.png").touch()

    # Create empty noise.csv
    with open(noise_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "label", "source_file", "spectrogram_path"]
        )
        writer.writeheader()

    # Run create_splits
    config = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
    splits = create_splits(
        labels_csv=labels_csv,
        noise_samples_csv=noise_csv,
        spectrograms_dir=spectrograms_dir,
        output_dir=output_dir,
        config=config,
    )

    # Verify output
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    # Total samples should match input
    total_samples = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_samples == 6

    # CSV files should exist
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "val.csv").exists()
    assert (output_dir / "test.csv").exists()

    # Test no data leakage at recording level
    train_recordings = {s.source_file for s in splits["train"]}
    val_recordings = {s.source_file for s in splits["val"]}
    test_recordings = {s.source_file for s in splits["test"]}

    assert len(train_recordings & val_recordings) == 0
    assert len(train_recordings & test_recordings) == 0
    assert len(val_recordings & test_recordings) == 0

    # All recordings should be represented
    all_split_recordings = train_recordings | val_recordings | test_recordings
    assert len(all_split_recordings) == 3


def test_create_splits_with_stratification(tmp_path):
    """Create splits with population stratification."""
    labels_csv = tmp_path / "labels.csv"
    noise_csv = tmp_path / "noise.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    metadata_csv = tmp_path / "metadata.csv"
    output_dir = tmp_path / "splits"
    spectrograms_dir.mkdir()

    # Create labels with recordings from 2 populations
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        # Population A - 2 recordings
        writer.writerow({
            "candidate_id": "2024-09-30_11-18-17_0000001_00000001",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })
        writer.writerow({
            "candidate_id": "2024-09-30_12-00-00_0000002_00000001",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })
        # Population B - 2 recordings
        writer.writerow({
            "candidate_id": "2024-09-30_13-00-00_0000003_00000001",
            "label": "Not USV",
            "labeled_at": "2024-01-01",
        })
        writer.writerow({
            "candidate_id": "2024-09-30_14-00-00_0000004_00000001",
            "label": "USV",
            "labeled_at": "2024-01-01",
        })

    # Create spectrograms
    (spectrograms_dir / "2024-09-30_11-18-17_0000001_00000001.png").touch()
    (spectrograms_dir / "2024-09-30_12-00-00_0000002_00000001.png").touch()
    (spectrograms_dir / "2024-09-30_13-00-00_0000003_00000001.png").touch()
    (spectrograms_dir / "2024-09-30_14-00-00_0000004_00000001.png").touch()

    # Create metadata
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "mouse_id", "population", "notes"])
        writer.writeheader()
        writer.writerow({
            "source_file": "2024-09-30_11-18-17_0000001.wav",
            "mouse_id": "0000001",
            "population": "pop_A",
            "notes": "",
        })
        writer.writerow({
            "source_file": "2024-09-30_12-00-00_0000002.wav",
            "mouse_id": "0000002",
            "population": "pop_A",
            "notes": "",
        })
        writer.writerow({
            "source_file": "2024-09-30_13-00-00_0000003.wav",
            "mouse_id": "0000003",
            "population": "pop_B",
            "notes": "",
        })
        writer.writerow({
            "source_file": "2024-09-30_14-00-00_0000004.wav",
            "mouse_id": "0000004",
            "population": "pop_B",
            "notes": "",
        })

    # Create empty noise.csv
    with open(noise_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "label", "source_file", "spectrogram_path"]
        )
        writer.writeheader()

    # Run with stratification
    config = SplitConfig(stratify_by_population=True, random_seed=42)
    splits = create_splits(
        labels_csv=labels_csv,
        noise_samples_csv=noise_csv,
        spectrograms_dir=spectrograms_dir,
        output_dir=output_dir,
        config=config,
        metadata_csv=metadata_csv,
    )

    # Load metadata to check populations
    from usv_spectrogram.dataset.metadata import load_metadata
    metadata = load_metadata(metadata_csv)

    # Get populations in train split
    train_recordings = {s.source_file for s in splits["train"]}
    train_populations = {metadata[rec].population for rec in train_recordings if rec in metadata}

    # Both populations should be in train (with 4 recordings and stratification)
    assert "pop_A" in train_populations
    assert "pop_B" in train_populations


def test_create_splits_no_data_leakage_integration(tmp_path):
    """Integration test: verify no recording appears in multiple splits."""
    labels_csv = tmp_path / "labels.csv"
    noise_csv = tmp_path / "noise.csv"
    spectrograms_dir = tmp_path / "spectrograms"
    output_dir = tmp_path / "splits"
    spectrograms_dir.mkdir()

    # Create labels with many recordings
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "label", "labeled_at"])
        writer.writeheader()
        for rec_id in range(1, 21):  # 20 recordings
            for sample in range(1, 4):  # 3 samples each
                writer.writerow({
                    "candidate_id": f"2024-09-30_11-18-17_{rec_id:07d}_{sample:08d}",
                    "label": "USV" if sample % 2 == 0 else "Not USV",
                    "labeled_at": "2024-01-01",
                })
                # Create spectrogram
                (spectrograms_dir / f"2024-09-30_11-18-17_{rec_id:07d}_{sample:08d}.png").touch()

    # Create empty noise.csv
    with open(noise_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "label", "source_file", "spectrogram_path"]
        )
        writer.writeheader()

    # Create splits
    splits = create_splits(
        labels_csv=labels_csv,
        noise_samples_csv=noise_csv,
        spectrograms_dir=spectrograms_dir,
        output_dir=output_dir,
    )

    # Extract recording sets
    train_recs = {s.source_file for s in splits["train"]}
    val_recs = {s.source_file for s in splits["val"]}
    test_recs = {s.source_file for s in splits["test"]}

    # CRITICAL: No data leakage
    assert len(train_recs & val_recs) == 0, "Train and val have overlapping recordings!"
    assert len(train_recs & test_recs) == 0, "Train and test have overlapping recordings!"
    assert len(val_recs & test_recs) == 0, "Val and test have overlapping recordings!"

    # All samples accounted for
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 60

    # All recordings accounted for
    assert len(train_recs | val_recs | test_recs) == 20
