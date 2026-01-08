"""Generate a USV spectrogram PNG from a WAV file (in-memory path)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram.render_tiles import render_png, render_tiled_pages
from usv_spectrogram.spectrogram import compute_spectrogram_db

DEFAULT_WAV_DIR = Path(r"C:\Users\shach\PycharmProjects\mickey_london_lab\5970 USV")


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = DEFAULT_WAV_DIR / input_path
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_suffix("").with_name(f"{input_path.stem}_spectrogram.png")
    )

    samples, sample_rate_hz = load_wav_mono(input_path)
    if args.auto_sample_rate:
        cfg = SpectrogramConfig(
            enforce_sample_rate=False,
            expected_sample_rate_hz=sample_rate_hz,
        )
        print(f"Detected sample rate: {sample_rate_hz} Hz")
    else:
        cfg = SpectrogramConfig()
    spec_db, freqs_hz, times_s = compute_spectrogram_db(samples, sample_rate_hz, cfg)
    if args.tiled:
        tile_dir = Path(args.tile_dir) if args.tile_dir else input_path.parent
        tile_base = args.tile_base or input_path.stem
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
        render_png(spec_db, freqs_hz, times_s, output_path, cfg, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
