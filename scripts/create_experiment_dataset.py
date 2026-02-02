"""Create experiment dataset by combining original training data with random negatives.

This script combines the original training/validation CSVs with the newly generated
random negative samples, creating a merged dataset for the CNN retraining experiment.

Key Operations:
1. Load original train.csv and val.csv from spectrograms_training/
2. Load random_negatives_metadata.csv
3. Copy all spectrograms to experiment directory
4. Update spectrogram paths to point to experiment directory
5. Combine train + random negatives, shuffle
6. Save experiment CSVs

Output Format:
- data/experiment_dataset/spectrograms/ (all PNGs)
- data/experiment_dataset/train_experiment.csv (combined & shuffled)
- data/experiment_dataset/val_experiment.csv (unchanged copy)

Expected Row Counts:
- Original train: 852 (376 USV + 476 Not USV)
- Random negatives: 100 (all Not USV)
- Combined train: 952 (376 USV + 576 Not USV)

Usage:
    python scripts/create_experiment_dataset.py \\
        --train-csv spectrograms_training/train.csv \\
        --val-csv spectrograms_training/val.csv \\
        --random-negatives-csv data/experiment_negatives/random_negatives_metadata.csv \\
        --random-negatives-dir data/experiment_negatives \\
        --output-dir data/experiment_dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# Add src to path (not used in this script, but consistent with others)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def copy_spectrograms(
    df: pd.DataFrame,
    source_base: Path,
    dest_dir: Path,
    path_column: str = "spectrogram_path",
    is_filename_only: bool = False,
) -> None:
    """Copy spectrograms from source to destination directory.

    Parameters
    ----------
    df:
        DataFrame with spectrogram paths
    source_base:
        Base directory for resolving relative paths (usually repo root)
    dest_dir:
        Destination directory for spectrograms
    path_column:
        Column name containing spectrogram paths
    is_filename_only:
        If True, paths are just filenames and should be resolved from source_base directly
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        src_path_str = row[path_column]

        if is_filename_only:
            # Path is just a filename - construct full path from source_base
            src = source_base / src_path_str
        else:
            # Path may be relative or absolute - handle accordingly
            src = Path(src_path_str)
            # If path is relative, resolve from source_base
            if not src.is_absolute():
                src = source_base / src

        if not src.exists():
            print(f"  Warning: Source file not found: {src}")
            continue

        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)


def create_experiment_dataset(
    train_csv: Path,
    val_csv: Path,
    random_negatives_csv: Path,
    random_negatives_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Create experiment dataset by combining original data with random negatives.

    Parameters
    ----------
    train_csv:
        Path to original train.csv (spectrograms_training/train.csv)
    val_csv:
        Path to original val.csv (spectrograms_training/val.csv)
    random_negatives_csv:
        Path to random_negatives_metadata.csv
    random_negatives_dir:
        Directory containing random negative spectrograms
    output_dir:
        Output directory for experiment dataset
    seed:
        Random seed for shuffling (default: 42)
    """
    output_dir = Path(output_dir)
    train_csv = Path(train_csv)
    val_csv = Path(val_csv)
    random_negatives_csv = Path(random_negatives_csv)
    random_negatives_dir = Path(random_negatives_dir)

    # Determine base path (repo root) for resolving relative paths
    # Assume this script is in scripts/ directory
    repo_root = Path(__file__).resolve().parents[1]

    print("Loading CSVs...")
    print(f"  Train: {train_csv}")
    print(f"  Val: {val_csv}")
    print(f"  Random negatives: {random_negatives_csv}")

    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    random_neg_df = pd.read_csv(random_negatives_csv)

    # Verify CSV formats
    required_columns = {"candidate_id", "spectrogram_path", "label", "source_file"}

    for name, df in [("train", train_df), ("val", val_df), ("random_negatives", random_neg_df)]:
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"{name}.csv missing columns: {missing}")

    # Verify random negatives have "Not USV" label
    if not (random_neg_df["label"] == "Not USV").all():
        unexpected = random_neg_df[random_neg_df["label"] != "Not USV"]["label"].unique()
        raise ValueError(f"Random negatives CSV has unexpected labels: {unexpected}")

    # Print statistics
    print("\nOriginal dataset statistics:")
    print(f"  Train total: {len(train_df)}")
    print(f"    USV: {(train_df['label'] == 'USV').sum()}")
    print(f"    Not USV: {(train_df['label'] == 'Not USV').sum()}")
    print(f"  Val total: {len(val_df)}")
    print(f"    USV: {(val_df['label'] == 'USV').sum()}")
    print(f"    Not USV: {(val_df['label'] == 'Not USV').sum()}")
    print(f"  Random negatives: {len(random_neg_df)} (all Not USV)")

    # Create experiment spectrograms directory
    exp_spec_dir = output_dir / "spectrograms"
    print(f"\nCreating experiment spectrograms directory: {exp_spec_dir}")

    # Copy original training spectrograms
    print("Copying original training spectrograms...")
    copy_spectrograms(train_df, repo_root, exp_spec_dir)

    # Copy original validation spectrograms
    print("Copying original validation spectrograms...")
    copy_spectrograms(val_df, repo_root, exp_spec_dir)

    # Copy random negative spectrograms
    print("Copying random negative spectrograms...")
    # Random negatives CSV has filename-only paths
    copy_spectrograms(random_neg_df, random_negatives_dir, exp_spec_dir, is_filename_only=True)

    # Update spectrogram paths to point to experiment directory
    print("\nUpdating spectrogram paths...")

    # For original data, use candidate_id to construct new path
    train_df = train_df.copy()
    train_df["spectrogram_path"] = train_df["candidate_id"].apply(
        lambda cid: f"data/experiment_dataset/spectrograms/{cid}.png"
    )

    val_df = val_df.copy()
    val_df["spectrogram_path"] = val_df["candidate_id"].apply(
        lambda cid: f"data/experiment_dataset/spectrograms/{cid}.png"
    )

    # For random negatives, use candidate_id as well
    random_neg_df = random_neg_df.copy()
    random_neg_df["spectrogram_path"] = random_neg_df["candidate_id"].apply(
        lambda cid: f"data/experiment_dataset/spectrograms/{cid}.png"
    )

    # Verify all spectrograms exist
    print("\nVerifying spectrograms exist...")
    all_paths = (
        list(train_df["spectrogram_path"])
        + list(val_df["spectrogram_path"])
        + list(random_neg_df["spectrogram_path"])
    )

    missing_count = 0
    for path_str in all_paths:
        path = repo_root / path_str
        if not path.exists():
            print(f"  Warning: Missing {path}")
            missing_count += 1

    if missing_count > 0:
        print(f"  WARNING: {missing_count} spectrograms missing!")
    else:
        print("  All spectrograms verified ✓")

    # Combine training data with random negatives
    print("\nCombining training data...")
    train_combined = pd.concat([train_df, random_neg_df], ignore_index=True)

    # Shuffle combined training data
    print("Shuffling combined training data...")
    train_combined = train_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Print combined statistics
    print("\nCombined dataset statistics:")
    print(f"  Train total: {len(train_combined)}")
    print(f"    USV: {(train_combined['label'] == 'USV').sum()}")
    print(f"    Not USV: {(train_combined['label'] == 'Not USV').sum()}")
    print(f"  Val total: {len(val_df)}")
    print(f"    USV: {(val_df['label'] == 'USV').sum()}")
    print(f"    Not USV: {(val_df['label'] == 'Not USV').sum()}")

    # Save experiment CSVs
    train_exp_csv = output_dir / "train_experiment.csv"
    val_exp_csv = output_dir / "val_experiment.csv"

    print(f"\nSaving experiment CSVs...")
    print(f"  Train: {train_exp_csv}")
    print(f"  Val: {val_exp_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_combined.to_csv(train_exp_csv, index=False)
    val_df.to_csv(val_exp_csv, index=False)

    print("\nExperiment dataset creation complete!")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── spectrograms/  ({len(all_paths)} PNGs)")
    print(f"  ├── train_experiment.csv ({len(train_combined)} samples)")
    print(f"  └── val_experiment.csv ({len(val_df)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Create experiment dataset by combining original data with random negatives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Path to original train.csv (e.g., spectrograms_training/train.csv)",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        required=True,
        help="Path to original val.csv (e.g., spectrograms_training/val.csv)",
    )
    parser.add_argument(
        "--random-negatives-csv",
        type=Path,
        required=True,
        help="Path to random_negatives_metadata.csv",
    )
    parser.add_argument(
        "--random-negatives-dir",
        type=Path,
        required=True,
        help="Directory containing random negative spectrograms",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for experiment dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()

    create_experiment_dataset(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        random_negatives_csv=args.random_negatives_csv,
        random_negatives_dir=args.random_negatives_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
