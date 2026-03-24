"""Import DeepSqueak classification results and merge with detection metadata.

Reads DeepSqueak Excel outputs (.xlsx), matches each classified call back to
its detection JSON via timestamp proximity, and writes a merged CSV containing
both DeepSqueak acoustic features and our detection metadata.

Round-trip: detections -> Raven tables -> DeepSqueak -> Excel -> THIS -> CSV
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

from usv_spectrogram.classification.deepsqueak_import import (
    DeepSqueakImportConfig,
    import_deepsqueak_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import DeepSqueak Excel results and merge with detection JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard import with default tolerance
  python import_deepsqueak_results.py \\
      --results-dir deepsqueak_output \\
      --detections-dir USV_Detections \\
      --output classified_detections.csv

  # Tighter matching tolerance
  python import_deepsqueak_results.py \\
      --results-dir deepsqueak_output \\
      --detections-dir USV_Detections \\
      --output classified_detections.csv \\
      --tolerance-ms 2.0

  # Dry run — show summary without writing output
  python import_deepsqueak_results.py \\
      --results-dir deepsqueak_output \\
      --detections-dir USV_Detections \\
      --output classified_detections.csv \\
      --dry-run -v
        """,
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing DeepSqueak Excel output files (.xlsx).",
    )
    parser.add_argument(
        "--detections-dir",
        default=str(REPO_ROOT / "USV_Detections"),
        help="Root directory containing detection subdirectories. "
        "Default: USV_Detections/",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "classified_detections.csv"),
        help="Output CSV path. Default: classified_detections.csv",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=5.0,
        help="Timestamp matching tolerance in ms. Default: 5.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and match but don't write output files.",
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
    results_dir = Path(args.results_dir)
    detections_dir = Path(args.detections_dir)

    if not results_dir.is_dir():
        print(
            f"Error: Results directory not found: {results_dir}",
            file=sys.stderr,
        )
        return 1

    if not detections_dir.is_dir():
        print(
            f"Error: Detections directory not found: {detections_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        config = DeepSqueakImportConfig(
            results_dir=results_dir,
            detections_dir=detections_dir,
            output_path=Path(args.output),
            tolerance_ms=args.tolerance_ms,
        )
    except ValueError as exc:
        print(f"Error: Invalid configuration: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        from usv_spectrogram.classification.deepsqueak_import import (
            load_all_deepsqueak_results,
            load_detections_for_merge,
            merge_with_detections,
        )

        try:
            ds_df = load_all_deepsqueak_results(config.results_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading results: {exc}", file=sys.stderr)
            return 1

        det_by_stem = load_detections_for_merge(config.detections_dir)
        _, summary = merge_with_detections(
            ds_df, det_by_stem, tolerance_ms=config.tolerance_ms
        )

        print(f"Dry run — {summary.files_processed} Excel files loaded:")
        for fname, count in sorted(summary.per_file_counts.items()):
            print(f"  {fname}: {count} calls")
        print(f"\nTotal DS calls: {summary.total_ds_calls}")
        print(f"Total detections: {summary.total_detections}")
        print(f"Matched: {summary.matched}")
        print(f"Unmatched DS calls: {summary.unmatched_ds}")
        print(f"Unmatched detections: {summary.unmatched_det}")
        print(f"Output would go to: {config.output_path}")
        return 0

    try:
        summary = import_deepsqueak_results(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nImport complete:")
    print(f"  Files processed: {summary.files_processed}")
    print(f"  DS calls: {summary.total_ds_calls}")
    print(f"  Detections: {summary.total_detections}")
    print(f"  Matched: {summary.matched}")
    print(f"  Unmatched DS: {summary.unmatched_ds}")
    print(f"  Unmatched det: {summary.unmatched_det}")
    print(f"  Output: {config.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
