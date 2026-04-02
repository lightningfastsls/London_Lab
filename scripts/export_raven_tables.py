"""Export USV detections as Raven Pro selection tables.

Reads individual detection JSON files from USV_Detections/, maps them back to
their source WAV files, and writes one tab-delimited Raven selection table per
WAV.  The output files (*.Table.1.selections.txt) can be opened directly in
Raven Pro, DeepSqueak, or any tool that accepts the Raven selection format.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classification.raven_export import (
    RavenExportConfig,
    export_raven_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export USV detection JSONs as Raven Pro selection tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard export with default frequency bounds
  python export_raven_tables.py \\
      --detections-dir USV_Detections \\
      --wav-dir "5970 USV" \\
      --output-dir raven_tables

  # Dry run — show what would be exported without writing files
  python export_raven_tables.py \\
      --detections-dir USV_Detections \\
      --wav-dir "5970 USV" \\
      --output-dir raven_tables \\
      --dry-run -v

  # Custom frequency bounds
  python export_raven_tables.py \\
      --detections-dir USV_Detections \\
      --wav-dir "5970 USV" \\
      --output-dir raven_tables \\
      --low-freq 30000 --high-freq 110000
        """,
    )
    parser.add_argument(
        "--detections-dir",
        default=str(REPO_ROOT / "USV_Detections"),
        help="Root directory containing detection subdirectories. "
        "Default: USV_Detections/",
    )
    parser.add_argument(
        "--wav-dir",
        default=None,
        help='Directory containing source .wav files. Required unless --batch-format.',
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "raven_tables"),
        help="Where to write Raven .txt files. Default: raven_tables/",
    )
    parser.add_argument(
        "--batch-format",
        action="store_true",
        help="Read flat batch detection JSONs (one per WAV, each a list of "
        "detections) instead of per-detection subdirectories.",
    )
    parser.add_argument(
        "--low-freq",
        type=float,
        default=25_000,
        help="Low frequency bound in Hz. Default: 25000",
    )
    parser.add_argument(
        "--high-freq",
        type=float,
        default=125_000,
        help="High frequency bound in Hz. Default: 125000",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show mapping and counts without writing any files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Configure logging.
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    # Validate input directories exist early.
    detections_dir = Path(args.detections_dir)
    wav_dir = Path(args.wav_dir) if args.wav_dir else None

    if not detections_dir.is_dir():
        print(
            f"Error: Detections directory not found: {detections_dir}",
            file=sys.stderr,
        )
        return 1

    if not args.batch_format and (wav_dir is None or not wav_dir.is_dir()):
        print(
            f"Error: WAV directory not found: {wav_dir}  "
            f"(use --batch-format if reading flat batch JSONs)",
            file=sys.stderr,
        )
        return 1

    try:
        config = RavenExportConfig(
            detections_dir=detections_dir,
            wav_dir=wav_dir,
            output_dir=Path(args.output_dir),
            low_freq_hz=args.low_freq,
            high_freq_hz=args.high_freq,
            batch_format=args.batch_format,
        )
    except ValueError as exc:
        print(f"Error: Invalid configuration: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        if args.batch_format:
            from usv_spectrogram.classification.raven_export import (
                discover_batch_detection_mapping,
            )
            mapping = discover_batch_detection_mapping(config.detections_dir)
            print(f"Dry run (batch) — {len(mapping)} WAV files with detections:\n")
            total = 0
            for stem, dets in sorted(mapping.items()):
                print(f"  {stem}: {len(dets)} detections")
                total += len(dets)
            print(f"\nTotal detections: {total}")
        else:
            from usv_spectrogram.classification.raven_export import (
                discover_wav_detection_mapping,
            )
            mapping = discover_wav_detection_mapping(
                config.detections_dir, config.wav_dir
            )
            print(f"Dry run — {len(mapping)} WAV files matched:\n")
            total = 0
            for stem, paths in sorted(mapping.items()):
                print(f"  {stem}: {len(paths)} detections")
                total += len(paths)
            print(f"\nTotal detections: {total}")

        print(f"Output would go to: {config.output_dir}")
        return 0

    created = export_raven_tables(config)

    print(f"\nExported {len(created)} Raven selection tables to {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
