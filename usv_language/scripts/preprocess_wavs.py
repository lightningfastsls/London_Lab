"""CLI: Preprocess WAV files into HDF5 spectrograms.

Usage:
    python preprocess_wavs.py --wav-dir <path> --output <path.h5> [--config <data.yaml>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Bootstrap imports
# Add project root so 'usv_language' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import usv_language.src  # noqa: F401

from usv_language.src.data.preprocessing import STFTConfig, preprocess_all_wavs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess WAVs into HDF5 spectrograms")
    parser.add_argument("--wav-dir", type=str, required=True, help="Directory of WAV files")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--config", type=str, default=None, help="Path to data.yaml config")
    parser.add_argument("--glob-pattern", type=str, default="*.wav", help="WAV file glob pattern")
    args = parser.parse_args()

    config = STFTConfig()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        stft_cfg = cfg.get("stft", {})
        config = STFTConfig(**{k: v for k, v in stft_cfg.items() if k in STFTConfig.__dataclass_fields__})

    summary = preprocess_all_wavs(
        wav_dir=args.wav_dir,
        output_hdf5=args.output,
        config=config,
        glob_pattern=args.glob_pattern,
    )

    print(f"\nDone! Processed {summary['n_files']} files, {summary['total_frames']} total frames")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
