#!/usr/bin/env python3
"""CLI script for preparing USV classification dataset.

Usage:
    # Generate metadata template
    python scripts/prepare_dataset.py --generate-metadata

    # Create train/val/test splits
    python scripts/prepare_dataset.py --create-splits

    # Run quality checks
    python scripts/prepare_dataset.py --check

    # All steps
    python scripts/prepare_dataset.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from usv_spectrogram.dataset import (
    generate_metadata_csv,
    create_splits,
    run_quality_checks,
    SplitConfig,
)


def get_default_paths() -> dict[str, Path]:
    """Get default paths relative to project root."""
    return {
        "labels_csv": project_root / "labels.csv",
        "noise_samples_csv": project_root / "noise_samples" / "noise_samples_final.csv",
        "spectrograms_dir": project_root / "spectrograms_review",
        "metadata_csv": project_root / "recordings_metadata.csv",
        "splits_dir": project_root / "splits",
    }


def cmd_generate_metadata(args: argparse.Namespace) -> int:
    """Generate recordings_metadata.csv template."""
    paths = get_default_paths()

    print("Generating recordings metadata template...")
    print(f"  Labels CSV: {paths['labels_csv']}")
    print(f"  Noise samples CSV: {paths['noise_samples_csv']}")
    print(f"  Output: {paths['metadata_csv']}")

    metadata = generate_metadata_csv(
        labels_csv=paths["labels_csv"],
        noise_samples_csv=paths["noise_samples_csv"],
        output_csv=paths["metadata_csv"],
    )

    print(f"\nGenerated metadata for {len(metadata)} recordings.")
    print(f"Output saved to: {paths['metadata_csv']}")
    print("\nNext step: Edit recordings_metadata.csv to fill in population info (lab/wild)")
    print("  If unknown, leave as 'unknown' - splits will still work without stratification.")

    return 0


def cmd_create_splits(args: argparse.Namespace) -> int:
    """Create train/val/test splits."""
    paths = get_default_paths()

    print("Creating dataset splits...")
    print(f"  Labels CSV: {paths['labels_csv']}")
    print(f"  Noise samples CSV: {paths['noise_samples_csv']}")
    print(f"  Spectrograms dir: {paths['spectrograms_dir']}")
    print(f"  Output dir: {paths['splits_dir']}")

    config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify_by_population=not args.no_stratify,
        exclude_uncertain=True,
    )

    # Check for metadata
    metadata_csv = paths["metadata_csv"] if paths["metadata_csv"].exists() else None
    if metadata_csv and not args.no_stratify:
        print(f"  Metadata CSV: {metadata_csv}")
    else:
        print("  No metadata (random split without population stratification)")

    splits = create_splits(
        labels_csv=paths["labels_csv"],
        noise_samples_csv=paths["noise_samples_csv"],
        spectrograms_dir=paths["spectrograms_dir"],
        output_dir=paths["splits_dir"],
        config=config,
        metadata_csv=metadata_csv,
    )

    print("\nSplit summary:")
    total = sum(len(s) for s in splits.values())
    for split_name, samples in splits.items():
        n = len(samples)
        pct = n / total * 100 if total > 0 else 0
        n_usv = sum(1 for s in samples if s.label == "USV")
        n_not_usv = sum(1 for s in samples if s.label == "Not USV")
        print(f"  {split_name:5}: {n:4} samples ({pct:5.1f}%) - {n_usv} USV, {n_not_usv} Not USV")

    print(f"\nOutputs saved to: {paths['splits_dir']}")
    print("  train.csv, val.csv, test.csv")

    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run quality checks on the dataset."""
    paths = get_default_paths()

    print("Running quality checks...")
    print(f"  Splits dir: {paths['splits_dir']}")

    metadata_csv = paths["metadata_csv"] if paths["metadata_csv"].exists() else None

    report = run_quality_checks(
        splits_dir=paths["splits_dir"],
        metadata_csv=metadata_csv,
    )

    print()
    print(report.summary())

    if not report.all_passed:
        print("\nSome checks failed. Please review and fix issues before training.")
        return 1

    print("\nAll checks passed! Dataset is ready for training.")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run all steps: generate metadata, create splits, run checks."""
    print("=" * 60)
    print("Step 1: Generate Metadata")
    print("=" * 60)
    ret = cmd_generate_metadata(args)
    if ret != 0:
        return ret

    print("\n" + "=" * 60)
    print("Step 2: Create Splits")
    print("=" * 60)
    ret = cmd_create_splits(args)
    if ret != 0:
        return ret

    print("\n" + "=" * 60)
    print("Step 3: Quality Checks")
    print("=" * 60)
    ret = cmd_check(args)

    return ret


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare USV classification dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate-metadata",
        action="store_true",
        help="Generate recordings_metadata.csv template",
    )
    group.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/val/test splits",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Run quality checks on splits",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all steps (metadata, splits, checks)",
    )

    # Split options
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable population stratification even if metadata available",
    )

    args = parser.parse_args()

    if args.generate_metadata:
        return cmd_generate_metadata(args)
    elif args.create_splits:
        return cmd_create_splits(args)
    elif args.check:
        return cmd_check(args)
    elif args.all:
        return cmd_all(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
