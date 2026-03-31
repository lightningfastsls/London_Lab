#!/usr/bin/env python3
"""Assemble training data for USV CNN classifier.

Supports two pipelines:
  - Global-MAD (default): Matched to inference — fixed 100-column windows from
    globally MAD-normalized spectrograms. Requires --unified-labels.
  - Legacy (--no-global-mad): Per-candidate extraction with variable widths.
    Requires --labels-dir.

Usage (new pipeline):
    python scripts/assemble_training_data.py \
        --unified-labels data/unified_labels.json \
        --wav-dir "5970 USV" \
        --output-dir data/training/matched_windows \
        --dry-run

Usage (legacy pipeline):
    python scripts/assemble_training_data.py \
        --no-global-mad \
        --labels-dir data/labels \
        --wav-dir "5970 USV" \
        --dry-run
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
        description="Assemble training data for USV CNN classifier."
    )

    # Label sources (mutually exclusive in practice)
    parser.add_argument(
        "--unified-labels", type=Path, default=None,
        help="Path to unified_labels.json (new global-MAD pipeline)",
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=None,
        help="Directory containing LabelStorage JSON files (legacy pipeline)",
    )

    # WAV directory (required for legacy, optional fallback for global-MAD)
    parser.add_argument(
        "--wav-dir", type=Path, default=None,
        help="WAV directory (required for legacy; fallback for global-MAD if wav_path not in unified labels)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/training/matched_windows"),
        help="Output directory for CSVs and spectrograms (default: data/training/matched_windows)",
    )

    # Global-MAD pipeline options
    parser.add_argument(
        "--no-global-mad", action="store_true",
        help="Use legacy per-candidate pipeline instead of global-MAD",
    )
    parser.add_argument(
        "--window-columns", type=int, default=100,
        help="STFT columns per window (default: 100, matches inference)",
    )
    parser.add_argument(
        "--rolling-stride-ms", type=float, default=10.0,
        help="Rolling window stride for long USVs in ms (default: 10.0)",
    )

    # Noise recording options
    parser.add_argument(
        "--noise-wav-dir", type=Path, default=None,
        help="Directory containing noise-labeled WAV files",
    )
    parser.add_argument(
        "--noise-stride-ms", type=float, default=42.7,
        help="Stride for systematic noise slicing in ms (default: 42.7)",
    )
    parser.add_argument(
        "--noise-max-ratio", type=float, default=2.0,
        help="Cap noise negatives at this multiple of positives (default: 2.0)",
    )

    # Common options
    parser.add_argument(
        "--min-usv-duration-ms", type=float, default=5.0,
        help="Minimum USV duration in ms; shorter detections are filtered (default: 5.0)",
    )
    parser.add_argument(
        "--jitter-samples", type=int, default=2,
        help="Number of jittered versions per positive (default: 2)",
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

    use_global_mad = not args.no_global_mad

    # Validate label source
    if use_global_mad and args.unified_labels is None:
        parser.error("Global-MAD pipeline requires --unified-labels")
    if not use_global_mad and args.labels_dir is None:
        parser.error("Legacy pipeline requires --labels-dir")
    if not use_global_mad and args.wav_dir is None:
        parser.error("Legacy pipeline requires --wav-dir")

    try:
        config = AssemblyConfig(
            wav_dir=args.wav_dir,
            labels_dir=args.labels_dir,
            use_global_mad=use_global_mad,
            unified_labels_path=args.unified_labels,
            window_columns=args.window_columns,
            rolling_stride_ms=args.rolling_stride_ms,
            noise_wav_dir=args.noise_wav_dir,
            noise_stride_ms=args.noise_stride_ms,
            noise_max_ratio=args.noise_max_ratio,
            output_dir=args.output_dir,
            min_usv_duration_ms=args.min_usv_duration_ms,
            jitter_n_samples=args.jitter_samples,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
        )

        assembler = DatasetAssembler(config)
        report = assembler.assemble(dry_run=args.dry_run)

        # Print summary
        pipeline = "global-MAD (matched)" if use_global_mad else "legacy (per-candidate)"
        print(f"\n=== Assembly Report ({pipeline}) ===")
        print(f"Recordings:  {report.n_recordings}")
        print(f"Positives:   {report.total_positives}")
        print(f"Negatives:   {report.total_negatives}")
        print(f"Total:       {report.total_positives + report.total_negatives}")
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
        logging.error("Assembly failed: %s", e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
