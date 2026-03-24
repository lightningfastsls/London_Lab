"""Create full training dataset with comprehensive negatives.

Combines:
- Original training data (376 USV + 476 Not USV = 852 samples)
- Comprehensive negatives (1000 Not USV samples: 500 random + 300 gaps + 200 low-energy)

Output:
- data/full_training_dataset/train.csv (~1852 samples: 376 USV + 1476 Not USV)
- data/full_training_dataset/val.csv (unchanged copy of original val.csv)
- data/full_training_dataset/test.csv (unchanged copy of original test.csv)
- data/full_training_dataset/spectrograms/ (all PNGs in one directory)
- data/full_training_dataset/class_weights.csv (3.0x USV boost → pos_weight ≈ 11.8)

Key Operations:
1. Load original train/val/test CSVs from spectrograms_training/
2. Load comprehensive_negatives_metadata.csv
3. Copy all spectrograms to unified output directory
4. Combine train + comprehensive negatives (add ONLY to training, not val/test)
5. Update paths to point to new directory
6. Calculate class weights with 3.0x USV boost
7. Save CSVs

Usage:
    python scripts/create_full_training_dataset.py \\
        --original-train spectrograms_training/train.csv \\
        --original-val spectrograms_training/val.csv \\
        --original-test spectrograms_training/test.csv \\
        --negatives-csv data/comprehensive_negatives/comprehensive_negatives_metadata.csv \\
        --negatives-dir data/comprehensive_negatives \\
        --output-dir data/full_training_dataset \\
        --seed 42
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def copy_spectrograms(
    df: pd.DataFrame,
    source_base: Path,
    dest_dir: Path,
    path_column: str = "spectrogram_path",
    is_filename_only: bool = False,
) -> int:
    """Copy spectrograms from source to destination directory.

    Returns number of files copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for idx, row in df.iterrows():
        src_path_str = row[path_column]

        if is_filename_only:
            src = source_base / src_path_str
        else:
            src = Path(src_path_str)
            if not src.is_absolute():
                src = source_base / src

        if not src.exists():
            print(f"  Warning: Source file not found: {src}")
            continue

        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1

    return copied


def create_full_training_dataset(
    original_train_csv: Path,
    original_val_csv: Path,
    original_test_csv: Path,
    negatives_csv: Path,
    negatives_dir: Path,
    output_dir: Path,
    seed: int = 42,
    usv_boost: float = 3.0,
) -> None:
    """Create full training dataset with comprehensive negatives."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectrograms_dir = output_dir / "spectrograms"
    spectrograms_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CREATING FULL TRAINING DATASET")
    print("=" * 60)

    # Load original data
    print("\n1. Loading original data...")
    train_df = pd.read_csv(original_train_csv)
    val_df = pd.read_csv(original_val_csv)
    test_df = pd.read_csv(original_test_csv)

    print(f"   Train: {len(train_df)} samples")
    print(f"     USV: {(train_df['label'] == 'USV').sum()}")
    print(f"     Not USV: {(train_df['label'] == 'Not USV').sum()}")
    print(f"   Val: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")

    # Load comprehensive negatives
    print("\n2. Loading comprehensive negatives...")
    neg_df = pd.read_csv(negatives_csv)
    print(f"   Total negatives: {len(neg_df)}")

    # Verify sample types
    if 'sample_type' in neg_df.columns:
        print("   Breakdown by type:")
        for sample_type in neg_df['sample_type'].unique():
            count = (neg_df['sample_type'] == sample_type).sum()
            print(f"     {sample_type}: {count}")

    # Verify all negatives have correct label
    if 'label' not in neg_df.columns:
        print("   ERROR: negatives CSV missing 'label' column")
        return

    if not (neg_df['label'] == 'Not USV').all():
        wrong_labels = neg_df[neg_df['label'] != 'Not USV']['label'].unique()
        print(f"   ERROR: Some negatives have wrong labels: {wrong_labels}")
        print("   All negatives should be labeled 'Not USV'")
        return

    # Copy spectrograms
    print("\n3. Copying spectrograms to unified directory...")

    # Get base directory (repo root)
    repo_root = Path(__file__).parent.parent

    # Copy original training spectrograms
    print("   Copying original train spectrograms...")
    copied_train = copy_spectrograms(
        train_df,
        source_base=repo_root,
        dest_dir=spectrograms_dir,
        is_filename_only=False,
    )
    print(f"     Copied {copied_train} files")

    # Copy original validation spectrograms
    print("   Copying original val spectrograms...")
    copied_val = copy_spectrograms(
        val_df,
        source_base=repo_root,
        dest_dir=spectrograms_dir,
        is_filename_only=False,
    )
    print(f"     Copied {copied_val} files")

    # Copy original test spectrograms
    print("   Copying original test spectrograms...")
    copied_test = copy_spectrograms(
        test_df,
        source_base=repo_root,
        dest_dir=spectrograms_dir,
        is_filename_only=False,
    )
    print(f"     Copied {copied_test} files")

    # Copy comprehensive negatives
    print("   Copying comprehensive negatives...")
    copied_neg = copy_spectrograms(
        neg_df,
        source_base=negatives_dir,
        dest_dir=spectrograms_dir,
        is_filename_only=True,  # negatives CSV has just filenames
    )
    print(f"     Copied {copied_neg} files")

    # Update paths to point to new directory
    print("\n4. Updating spectrogram paths...")

    # Get relative path from repo root to spectrograms directory
    relative_spec_dir = output_dir.relative_to(repo_root) / "spectrograms"

    # For original data, extract filename and prepend new directory
    train_df['spectrogram_path'] = train_df['spectrogram_path'].apply(
        lambda p: str(relative_spec_dir / Path(p).name)
    )
    val_df['spectrogram_path'] = val_df['spectrogram_path'].apply(
        lambda p: str(relative_spec_dir / Path(p).name)
    )
    test_df['spectrogram_path'] = test_df['spectrogram_path'].apply(
        lambda p: str(relative_spec_dir / Path(p).name)
    )

    # For negatives, prepend directory (they have just filenames)
    neg_df['spectrogram_path'] = neg_df['spectrogram_path'].apply(
        lambda p: str(relative_spec_dir / p)
    )

    # Combine train + negatives
    print("\n5. Combining training data with comprehensive negatives...")

    # Ensure both have sample_type column
    if 'sample_type' not in train_df.columns:
        train_df['sample_type'] = 'original'

    # Concatenate
    train_combined = pd.concat([train_df, neg_df], ignore_index=True)

    # Shuffle
    train_combined = train_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"   Combined train: {len(train_combined)} samples")
    print(f"     USV: {(train_combined['label'] == 'USV').sum()} ({100*(train_combined['label'] == 'USV').sum()/len(train_combined):.1f}%)")
    print(f"     Not USV: {(train_combined['label'] == 'Not USV').sum()} ({100*(train_combined['label'] == 'Not USV').sum()/len(train_combined):.1f}%)")

    # Calculate class weights
    print(f"\n6. Calculating class weights (USV boost: {usv_boost}x)...")

    n_usv = (train_combined['label'] == 'USV').sum()
    n_not_usv = (train_combined['label'] == 'Not USV').sum()
    total = len(train_combined)

    # Standard inverse frequency weighting
    usv_weight_base = total / (2 * n_usv)
    not_usv_weight = total / (2 * n_not_usv)

    # Apply boost to USV weight
    usv_weight = usv_weight_base * usv_boost

    # pos_weight for BCEWithLogitsLoss
    pos_weight = usv_weight / not_usv_weight

    print(f"   USV weight: {usv_weight:.3f} (base: {usv_weight_base:.3f} × {usv_boost})")
    print(f"   Not USV weight: {not_usv_weight:.3f}")
    print(f"   pos_weight for BCEWithLogitsLoss: {pos_weight:.3f}")

    # Save class weights
    weights_df = pd.DataFrame([{
        'usv_weight': usv_weight,
        'not_usv_weight': not_usv_weight,
        'pos_weight': pos_weight,
        'usv_boost': usv_boost,
    }])
    weights_csv = output_dir / "class_weights.csv"
    weights_df.to_csv(weights_csv, index=False)
    print(f"   Saved to {weights_csv}")

    # Save CSVs
    print("\n7. Saving dataset CSVs...")

    train_csv = output_dir / "train.csv"
    train_combined.to_csv(train_csv, index=False)
    print(f"   Train: {train_csv} ({len(train_combined)} samples)")

    val_csv = output_dir / "val.csv"
    val_df.to_csv(val_csv, index=False)
    print(f"   Val: {val_csv} ({len(val_df)} samples)")

    test_csv = output_dir / "test.csv"
    test_df.to_csv(test_csv, index=False)
    print(f"   Test: {test_csv} ({len(test_df)} samples)")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Spectrograms: {spectrograms_dir} ({copied_train + copied_val + copied_test + copied_neg} files)")
    print(f"\nDataset composition:")
    print(f"  Train: {len(train_combined)} samples ({100*(train_combined['label'] == 'USV').sum()/len(train_combined):.1f}% USV)")
    print(f"  Val: {len(val_df)} samples (unchanged)")
    print(f"  Test: {len(test_df)} samples (unchanged)")
    print(f"\nClass weights (for training):")
    print(f"  pos_weight = {pos_weight:.3f}")
    print(f"\nNext step: Train CNN with:")
    print(f"  python scripts/train_cnn.py \\")
    print(f"    --train-csv {train_csv} \\")
    print(f"    --val-csv {val_csv} \\")
    print(f"    --batch-size 32 \\")
    print(f"    --num-epochs 50 \\")
    print(f"    --patience 15 \\")
    print(f"    --use-class-weights \\")
    print(f"    --output-dir models/full_retrained_cnn")


def main():
    parser = argparse.ArgumentParser(
        description="Create full training dataset with comprehensive negatives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--original-train",
        type=Path,
        required=True,
        help="Path to original train.csv (e.g., spectrograms_training/train.csv)",
    )
    parser.add_argument(
        "--original-val",
        type=Path,
        required=True,
        help="Path to original val.csv (e.g., spectrograms_training/val.csv)",
    )
    parser.add_argument(
        "--original-test",
        type=Path,
        required=True,
        help="Path to original test.csv (e.g., spectrograms_training/test.csv)",
    )
    parser.add_argument(
        "--negatives-csv",
        type=Path,
        required=True,
        help="Path to comprehensive_negatives_metadata.csv",
    )
    parser.add_argument(
        "--negatives-dir",
        type=Path,
        required=True,
        help="Directory containing comprehensive negative spectrograms",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for full training dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--usv-boost",
        type=float,
        default=3.0,
        help="Class weight boost for USV samples (default: 3.0)",
    )

    args = parser.parse_args()

    create_full_training_dataset(
        original_train_csv=args.original_train,
        original_val_csv=args.original_val,
        original_test_csv=args.original_test,
        negatives_csv=args.negatives_csv,
        negatives_dir=args.negatives_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        usv_boost=args.usv_boost,
    )


if __name__ == "__main__":
    main()
