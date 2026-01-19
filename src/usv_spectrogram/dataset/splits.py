"""Dataset splitting for USV classification training.

Key principle: Split by RECORDING, not by candidate, to prevent data leakage
from temporal correlation within recordings.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .metadata import (
    RecordingMetadata,
    extract_source_file_from_candidate_id,
    load_metadata,
)


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    stratify_by_population: bool = True
    exclude_uncertain: bool = True

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.3f} "
                f"({self.train_ratio} + {self.val_ratio} + {self.test_ratio})"
            )


@dataclass
class Sample:
    """A single sample for the dataset."""

    candidate_id: str
    source_file: str
    label: str  # "USV" or "Not USV"
    spectrogram_path: str


def load_labeled_samples(
    labels_csv: Path,
    spectrograms_dir: Path,
    noise_samples_csv: Path,
    exclude_uncertain: bool = True,
    require_spectrogram_exists: bool = True,
) -> list[Sample]:
    """Load all labeled samples from labels.csv and noise_samples_final.csv.

    Parameters
    ----------
    labels_csv : Path
        Path to labels.csv with candidate_id,label,labeled_at columns.
    spectrograms_dir : Path
        Directory containing spectrogram PNGs for labeled candidates.
    noise_samples_csv : Path
        Path to noise_samples_final.csv.
    exclude_uncertain : bool
        If True, exclude samples labeled as "Uncertain".
    require_spectrogram_exists : bool
        If True, only include samples where the spectrogram file exists.

    Returns
    -------
    List of Sample objects with valid spectrogram paths.
    """
    samples = []
    skipped_missing = 0
    labels_csv = Path(labels_csv)
    spectrograms_dir = Path(spectrograms_dir)
    noise_samples_csv = Path(noise_samples_csv)
    noise_samples_dir = spectrograms_dir.parent / "noise_samples"

    # Load from labels.csv (candidate detections)
    if labels_csv.exists():
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidate_id = row.get("candidate_id", "")
                label = row.get("label", "")

                if not candidate_id or not label:
                    continue

                if exclude_uncertain and label == "Uncertain":
                    continue

                # Determine spectrogram path
                # Check if it's a noise sample (already reviewed via labeling tool)
                if "_noise_" in candidate_id:
                    # Noise samples - check multiple locations
                    spec_path = noise_samples_dir / f"{candidate_id}.png"
                else:
                    # Regular candidates are in spectrograms_review
                    spec_path = spectrograms_dir / f"{candidate_id}.png"

                # Skip if spectrogram doesn't exist
                if require_spectrogram_exists and not spec_path.exists():
                    skipped_missing += 1
                    continue

                source_file = extract_source_file_from_candidate_id(candidate_id)

                samples.append(Sample(
                    candidate_id=candidate_id,
                    source_file=source_file,
                    label=label,
                    spectrogram_path=str(spec_path),
                ))

    # Load from noise_samples_final.csv (additional noise samples)
    if noise_samples_csv.exists():
        # First, collect candidate_ids already in samples to avoid duplicates
        existing_ids = {s.candidate_id for s in samples}

        with open(noise_samples_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidate_id = row.get("candidate_id", "")
                label = row.get("label", "Not USV")
                orig_spec_path = row.get("spectrogram_path", "")
                source_file = row.get("source_file", "")

                if not candidate_id or candidate_id in existing_ids:
                    continue

                if not source_file:
                    source_file = extract_source_file_from_candidate_id(candidate_id)

                # Normalize spectrogram path - try local noise_samples dir first
                spec_path = noise_samples_dir / f"{candidate_id}.png"
                if not spec_path.exists():
                    # Try the original path if it exists
                    orig_path = Path(orig_spec_path)
                    if orig_path.exists():
                        spec_path = orig_path
                    elif require_spectrogram_exists:
                        skipped_missing += 1
                        continue

                samples.append(Sample(
                    candidate_id=candidate_id,
                    source_file=source_file,
                    label=label,
                    spectrogram_path=str(spec_path),
                ))

    if skipped_missing > 0:
        import warnings
        warnings.warn(
            f"Skipped {skipped_missing} samples with missing spectrogram files. "
            "Consider regenerating spectrograms for noise samples."
        )

    return samples


def group_samples_by_recording(
    samples: list[Sample],
) -> dict[str, list[Sample]]:
    """Group samples by their source recording.

    Returns dict mapping source_file -> list of samples.
    """
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        if sample.source_file not in groups:
            groups[sample.source_file] = []
        groups[sample.source_file].append(sample)
    return groups


def split_recordings(
    recordings: list[str],
    config: SplitConfig,
    metadata: Optional[dict[str, RecordingMetadata]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split recordings into train/val/test sets.

    Parameters
    ----------
    recordings : list[str]
        List of recording source_files to split.
    config : SplitConfig
        Split configuration.
    metadata : dict, optional
        Recording metadata for stratification.

    Returns
    -------
    Tuple of (train_recordings, val_recordings, test_recordings).
    """
    random.seed(config.random_seed)

    # Try stratified split if metadata available and stratification requested
    if config.stratify_by_population and metadata:
        # Group recordings by population
        by_population: dict[str, list[str]] = {}
        for rec in recordings:
            meta = metadata.get(rec)
            pop = meta.population if meta else "unknown"
            if pop not in by_population:
                by_population[pop] = []
            by_population[pop].append(rec)

        # Check if we have enough diversity to stratify
        populations = [p for p in by_population.keys() if p != "unknown"]
        if len(populations) >= 2:
            # Stratified split
            train, val, test = [], [], []
            for pop, pop_recordings in by_population.items():
                random.shuffle(pop_recordings)
                n = len(pop_recordings)
                n_train = max(1, int(n * config.train_ratio))
                n_val = max(1, int(n * config.val_ratio)) if n > 2 else 0
                n_test = n - n_train - n_val

                train.extend(pop_recordings[:n_train])
                val.extend(pop_recordings[n_train:n_train + n_val])
                test.extend(pop_recordings[n_train + n_val:])

            return train, val, test

    # Random split (no stratification or all unknown population)
    recordings = list(recordings)
    random.shuffle(recordings)

    n = len(recordings)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train = recordings[:n_train]
    val = recordings[n_train:n_train + n_val]
    test = recordings[n_train + n_val:]

    return train, val, test


def create_splits(
    labels_csv: Path,
    noise_samples_csv: Path,
    spectrograms_dir: Path,
    output_dir: Path,
    config: Optional[SplitConfig] = None,
    metadata_csv: Optional[Path] = None,
) -> dict[str, list[Sample]]:
    """Create train/val/test splits and save to CSV files.

    Parameters
    ----------
    labels_csv : Path
        Path to labels.csv.
    noise_samples_csv : Path
        Path to noise_samples_final.csv.
    spectrograms_dir : Path
        Directory containing spectrogram PNGs.
    output_dir : Path
        Directory to write split CSVs.
    config : SplitConfig, optional
        Split configuration. Defaults to standard 70/15/15.
    metadata_csv : Path, optional
        Path to recordings_metadata.csv for stratification.

    Returns
    -------
    Dict with keys "train", "val", "test" containing sample lists.
    """
    config = config or SplitConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata if available
    metadata = None
    if metadata_csv and Path(metadata_csv).exists():
        metadata = load_metadata(metadata_csv)

    # Load all samples
    samples = load_labeled_samples(
        labels_csv=labels_csv,
        spectrograms_dir=spectrograms_dir,
        noise_samples_csv=noise_samples_csv,
        exclude_uncertain=config.exclude_uncertain,
    )

    # Group by recording
    by_recording = group_samples_by_recording(samples)
    all_recordings = list(by_recording.keys())

    # Split recordings
    train_recs, val_recs, test_recs = split_recordings(
        all_recordings, config, metadata
    )

    # Collect samples for each split
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for rec in train_recs:
        splits["train"].extend(by_recording[rec])
    for rec in val_recs:
        splits["val"].extend(by_recording[rec])
    for rec in test_recs:
        splits["test"].extend(by_recording[rec])

    # Write CSVs
    for split_name, split_samples in splits.items():
        csv_path = output_dir / f"{split_name}.csv"
        _write_split_csv(split_samples, csv_path)

    return splits


def _write_split_csv(samples: list[Sample], output_path: Path) -> None:
    """Write samples to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["candidate_id", "source_file", "label", "spectrogram_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "candidate_id": sample.candidate_id,
                "source_file": sample.source_file,
                "label": sample.label,
                "spectrogram_path": sample.spectrogram_path,
            })


def load_splits(splits_dir: Path) -> dict[str, list[Sample]]:
    """Load train/val/test splits from CSV files.

    Returns dict with keys "train", "val", "test" containing sample lists.
    """
    splits_dir = Path(splits_dir)
    result = {}

    for split_name in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split_name}.csv"
        samples = []

        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(Sample(
                        candidate_id=row.get("candidate_id", ""),
                        source_file=row.get("source_file", ""),
                        label=row.get("label", ""),
                        spectrogram_path=row.get("spectrogram_path", ""),
                    ))

        result[split_name] = samples

    return result
