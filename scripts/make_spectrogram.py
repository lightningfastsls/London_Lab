"""Generate a USV spectrogram PNG from a WAV file (in-memory path)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import get_default_wav_dir, load_wav_mono
from usv_spectrogram.render_tiles import render_png, render_tiled_pages
from usv_spectrogram.spectrogram import compute_spectrogram_db

DEFAULT_WAV_DIR = get_default_wav_dir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a USV spectrogram PNG.")
    parser.add_argument("--input", required=True, help="Path to input WAV file.")
    parser.add_argument(
        "--output",
        help="Path to output PNG. Defaults to <input>_spectrogram.png",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Render tiled pages instead of a single PNG.",
    )
    parser.add_argument(
        "--tile-dir",
        help="Output directory for tiled pages. Defaults to input directory.",
    )
    parser.add_argument(
        "--tile-base",
        help="Base name for tiled pages. Defaults to input file stem.",
    )
    parser.add_argument(
        "--auto-sample-rate",
        action="store_true",
        help="Use the WAV file's sample rate instead of enforcing 250 kHz.",
    )
    parser.add_argument("--title", help="Optional plot title.")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = DEFAULT_WAV_DIR / input_path

    # Validate input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not input_path.suffix.lower() == ".wav":
        print(f"Error: Input must be a .wav file: {input_path}", file=sys.stderr)
        return 1

    # Determine output path
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_suffix("").with_name(f"{input_path.stem}_spectrogram.png")
    )

    try:
        # Load WAV
        if args.verbose:
            print(f"Loading: {input_path}")
        t0 = time.perf_counter()
        samples, sample_rate_hz = load_wav_mono(input_path)
        if args.verbose:
            print(f"  Loaded {len(samples):,} samples at {sample_rate_hz} Hz")

        # Configure
        if args.auto_sample_rate:
            cfg = SpectrogramConfig(
                enforce_sample_rate=False,
                expected_sample_rate_hz=sample_rate_hz,
            )
            if args.verbose:
                print(f"  Using detected sample rate: {sample_rate_hz} Hz")
        else:
            cfg = SpectrogramConfig()

        # Compute spectrogram
        if args.verbose:
            print("Computing spectrogram...")
        spec_db, freqs_hz, times_s = compute_spectrogram_db(samples, sample_rate_hz, cfg)
        if args.verbose:
            print(f"  Shape: {spec_db.shape[0]} freq bins x {spec_db.shape[1]} time frames")

        # Render output
        if args.tiled:
            tile_dir = Path(args.tile_dir) if args.tile_dir else input_path.parent
            tile_base = args.tile_base or input_path.stem
            if args.verbose:
                print(f"Rendering tiled pages to: {tile_dir}/{tile_base}_*.png")
            render_tiled_pages(
                spec_db,
                freqs_hz,
                times_s,
                tile_dir,
                tile_base,
                cfg,
                title=args.title,
            )
        else:
            if args.verbose:
                print(f"Rendering PNG: {output_path}")
            render_png(spec_db, freqs_hz, times_s, output_path, cfg, title=args.title)

        elapsed = time.perf_counter() - t0
        if args.verbose:
            print(f"Done in {elapsed:.2f}s")

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
