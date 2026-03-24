#!/usr/bin/env python3
"""Assemble training data from LabelStorage JSON files.

Unified pipeline that replaces the manual multi-script workflow:
  generate_comprehensive_negatives.py -> create_full_training_dataset.py -> manual combination

Usage:
    python scripts/assemble_training_data.py --labels-dir data/labels --wav-dir "5970 USV"
    python scripts/assemble_training_data.py --labels-dir data/labels --wav-dir "5970 USV" --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.dataset.assembler import AssemblyConfig, DatasetAssembler


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble training data from LabelStorage JSON files."
    )
    parser.add_argument(
        "--labels-dir", type=Path, required=True,
        help="Directory containing LabelStorage JSON files",
    )
    parser.add_argument(
        "--wav-dir", type=Path, required=True,
        help="Directory containing WAV files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/training/assembled"),
        help="Output directory for CSVs and spectrograms (default: data/training/assembled)",
    )
    parser.add_argument(
        "--jitter-samples", type=int, default=5,
        help="Number of jittered versions per positive (default: 5)",
    )
    parser.add_argument(
        "--neg-ratio", type=float, default=1.0,
        help="Ratio of negatives to positives (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute statistics without extracting spectrograms or writing files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        config = AssemblyConfig(
            labels_dir=args.labels_dir,
            wav_dir=args.wav_dir,
            output_dir=args.output_dir,
            jitter_n_samples=args.jitter_samples,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
        )

        assembler = DatasetAssembler(config)
        report = assembler.assemble(dry_run=args.dry_run)

        # Print summary
        print("\n=== Assembly Report ===")
        print(f"Recordings:  {report.n_recordings}")
        print(f"Positives:   {report.total_positives}")
        print(f"Negatives:   {report.total_negatives}")
        print(f"Train:       {report.train_count}")
        print(f"Val:         {report.val_count}")
        print(f"Test:        {report.test_count}")
        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for w in report.warnings:
                print(f"  {w}")
        else:
            print("\nAll quality checks passed.")

        if args.dry_run:
            print("\n(Dry run — no files written)")
        else:
            print(f"\nOutput: {report.output_dir}")

        return 0

    except Exception as e:
        logging.error("Assembly failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
