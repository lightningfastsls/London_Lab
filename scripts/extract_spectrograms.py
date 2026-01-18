"""Extract spectrogram images from detected USV candidates.

Converts candidate segments from detection CSV into standardized spectrogram
PNG images for human labeling and CNN training.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.io_wav import get_default_wav_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract spectrogram images from candidate CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract spectrograms for all candidates
  python extract_spectrograms.py --candidates candidates.csv --wav-dir data/ --output-dir spectrograms/

  # Use training mode (no axes, raw images for CNN)
  python extract_spectrograms.py --candidates c.csv --wav-dir data/ --output-dir spec/ --mode training

  # Custom frequency range and colormap
  python extract_spectrograms.py --candidates c.csv --wav-dir data/ --output-dir spec/ --freq-min 15000 --freq-max 130000 --colormap viridis
        """,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to input CSV file with candidates.",
    )
    parser.add_argument(
        "--wav-dir",
        required=True,
        help="Directory containing source WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save spectrogram PNG files.",
    )
    parser.add_argument(
        "--output-csv",
        help="Path to save updated CSV with spectrogram paths. Defaults to input CSV path with '_extracted' suffix.",
    )
    parser.add_argument(
        "--mode",
        choices=["review", "training"],
        default="review",
        help="Render mode. 'review' has axes/labels, 'training' is raw images. Default: review",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=300000,
        help="Expected sample rate in Hz. Default: 300000",
    )
    parser.add_argument(
        "--freq-min",
        type=int,
        default=20000,
        help="Minimum frequency to display in Hz. Default: 20000",
    )
    parser.add_argument(
        "--freq-max",
        type=int,
        default=120000,
        help="Maximum frequency to display in Hz. Default: 120000",
    )
    parser.add_argument(
        "--colormap",
        default="magma",
        help="Matplotlib colormap name. Default: magma (dark background)",
    )
    parser.add_argument(
        "--db-floor",
        type=float,
        default=-80.0,
        help="Minimum dB value (black level). Default: -80",
    )
    parser.add_argument(
        "--db-ceiling",
        type=float,
        default=-10.0,
        help="Maximum dB value (white level). Default: -10",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit extraction to first N candidates (0 = all). Default: 0",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve paths
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: Candidates CSV not found: {candidates_path}", file=sys.stderr)
        return 1

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_absolute():
        # Try relative to default WAV dir
        if not wav_dir.exists():
            default_wav_dir = get_default_wav_dir()
            wav_dir = default_wav_dir / args.wav_dir
            if not wav_dir.exists():
                wav_dir = default_wav_dir

    if not wav_dir.exists():
        print(f"Error: WAV directory not found: {wav_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output CSV path
    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        output_csv = candidates_path.with_stem(candidates_path.stem + "_extracted")

    # Create config
    try:
        config = ExtractionConfig(
            sample_rate=args.sample_rate,
            freq_min_hz=args.freq_min,
            freq_max_hz=args.freq_max,
            db_floor=args.db_floor,
            db_ceiling=args.db_ceiling,
            colormap=args.colormap,
            default_render_mode=args.mode,
        )
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        return 1

    extractor = SpectrogramExtractor(config)

    if args.verbose:
        print(f"Input CSV: {candidates_path}")
        print(f"WAV directory: {wav_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Render mode: {args.mode}")
        print()

    t0 = time.perf_counter()

    # Count total candidates
    with open(candidates_path, "r", encoding="utf-8") as f:
        total_candidates = sum(1 for _ in f) - 1  # Subtract header

    # Apply limit if specified
    process_count = args.limit if args.limit > 0 else total_candidates

    if args.verbose:
        if args.limit > 0:
            print(f"Processing {process_count} of {total_candidates} candidates (limited)...")
        else:
            print(f"Processing {total_candidates} candidates...")

    # Extract spectrograms
    results = []
    success_count = 0
    fail_count = 0

    for i, (candidate, output_path) in enumerate(
        extractor.extract_batch(candidates_path, wav_dir, output_dir, args.mode), 1
    ):
        results.append((candidate, output_path))
        if output_path is not None:
            success_count += 1
        else:
            fail_count += 1

        if args.verbose and i % 50 == 0:
            print(f"  Processed {i}/{process_count}...")

        # Stop if limit reached
        if args.limit > 0 and i >= args.limit:
            break

    # Save updated CSV
    extractor.save_updated_csv(results, output_csv)

    elapsed = time.perf_counter() - t0

    # Print summary
    if args.verbose:
        print()
        print(f"--- Summary ---")
        print(f"Total candidates: {total_candidates}")
        print(f"Successfully extracted: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Output directory: {output_dir}")
        print(f"Updated CSV: {output_csv}")
        print(f"Time elapsed: {elapsed:.2f}s")

    if fail_count > 0:
        print(f"\nWarning: {fail_count} candidates failed to extract", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
