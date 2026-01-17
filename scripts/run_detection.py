"""Run USV candidate detection on WAV files.

Detects candidate USV segments using energy thresholding and saves results to CSV.
Optimized for HIGH RECALL - precision is handled downstream by labeling.
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

from usv_spectrogram.detection.config import DetectionConfig
from usv_spectrogram.detection.energy_detector import EnergyDetector
from usv_spectrogram.io_wav import get_default_wav_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect USV candidates in WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python run_detection.py --input recording.wav --output candidates.csv

  # Process a directory of WAV files
  python run_detection.py --input data/recordings/ --output all_candidates.csv

  # Use a more sensitive threshold
  python run_detection.py --input rec.wav --output out.csv --threshold -60
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input WAV file or directory containing WAV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file for candidates.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-50.0,
        help="Energy threshold in dB (relative to max). Lower = more candidates. Default: -50",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum candidate duration in ms. Default: 10",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=500.0,
        help="Maximum candidate duration in ms. Default: 500",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=5.0,
        help="Merge detections closer than this (ms). Default: 5",
    )
    parser.add_argument(
        "--freq-min",
        type=int,
        default=25000,
        help="Minimum frequency band in Hz. Default: 25000",
    )
    parser.add_argument(
        "--freq-max",
        type=int,
        default=110000,
        help="Maximum frequency band in Hz. Default: 110000",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=250000,
        help="Expected sample rate in Hz. Default: 250000",
    )
    parser.add_argument(
        "--energy-mode",
        choices=["peak", "mean"],
        default="peak",
        help="Energy computation mode. 'peak' better for narrow-band USVs. Default: peak",
    )
    parser.add_argument(
        "--max-bandwidth",
        type=int,
        default=20000,
        help="Max bandwidth in Hz (0 to disable). Rejects broadband noise. Default: 20000",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern for WAV files when input is directory. Default: *.wav",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # Check if it exists relative to current dir first
        if not input_path.exists():
            # Try relative to default WAV dir
            default_wav_dir = get_default_wav_dir()
            input_path = default_wav_dir / args.input

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        return 1

    # Determine if input is file or directory
    is_single_file = input_path.is_file()
    if is_single_file and not input_path.suffix.lower() == ".wav":
        print(f"Error: Input file must be .wav: {input_path}", file=sys.stderr)
        return 1

    # Resolve output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create config
    try:
        config = DetectionConfig(
            sample_rate=args.sample_rate,
            energy_threshold_db=args.threshold,
            energy_mode=args.energy_mode,
            max_bandwidth_hz=args.max_bandwidth,
            min_duration_ms=args.min_duration,
            max_duration_ms=args.max_duration,
            merge_gap_ms=args.merge_gap,
            freq_min_hz=args.freq_min,
            freq_max_hz=args.freq_max,
        )
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        return 1

    detector = EnergyDetector(config)
    all_candidates = []

    t0 = time.perf_counter()

    if is_single_file:
        # Process single file
        if args.verbose:
            print(f"Processing: {input_path}")
        try:
            candidates = detector.detect(input_path)
            all_candidates.extend(candidates)
            if args.verbose:
                print(f"  Found {len(candidates)} candidates")
        except Exception as e:
            print(f"Error processing {input_path}: {e}", file=sys.stderr)
            return 1
    else:
        # Process directory
        wav_files = sorted(input_path.glob(args.pattern))
        if not wav_files:
            print(f"Error: No files matching '{args.pattern}' in {input_path}", file=sys.stderr)
            return 1

        if args.verbose:
            print(f"Found {len(wav_files)} WAV files in {input_path}")

        for i, wav_path in enumerate(wav_files, 1):
            if args.verbose:
                print(f"[{i}/{len(wav_files)}] Processing: {wav_path.name}")
            try:
                candidates = detector.detect(wav_path)
                all_candidates.extend(candidates)
                if args.verbose:
                    print(f"  Found {len(candidates)} candidates")
            except Exception as e:
                print(f"  Warning: Failed to process {wav_path.name}: {e}", file=sys.stderr)
                continue

    # Save results
    if all_candidates:
        detector.save_candidates_csv(all_candidates, output_path)
        if args.verbose:
            print(f"\nSaved {len(all_candidates)} candidates to: {output_path}")
    else:
        # Create empty CSV with headers
        output_path.write_text(
            "candidate_id,source_file,start_ms,end_ms,duration_ms,"
            "context_start_ms,context_end_ms,peak_freq_hz,peak_energy_db,"
            "interference_flag,spectrogram_path\n"
        )
        if args.verbose:
            print(f"\nNo candidates found. Created empty CSV: {output_path}")

    elapsed = time.perf_counter() - t0

    # Print summary
    if args.verbose:
        print(f"\n--- Summary ---")
        if is_single_file:
            print(f"Files processed: 1")
        else:
            print(f"Files processed: {len(wav_files)}")
        print(f"Total candidates: {len(all_candidates)}")

        if all_candidates:
            # Calculate statistics
            durations = [c.duration_ms for c in all_candidates]
            freqs = [c.peak_freq_hz for c in all_candidates]
            interference_count = sum(1 for c in all_candidates if c.interference_flag)

            print(f"Duration range: {min(durations):.1f} - {max(durations):.1f} ms")
            print(f"Frequency range: {min(freqs)/1000:.1f} - {max(freqs)/1000:.1f} kHz")
            print(f"Interference flagged: {interference_count}")

        print(f"Time elapsed: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
