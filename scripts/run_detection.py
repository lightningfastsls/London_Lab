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
        default=3.0,
        help="Merge detections closer than this (ms). Default: 3 (tuned)",
    )
    parser.add_argument(
        "--no-segment-continuity",
        dest="segment_continuity",
        action="store_false",
        default=True,
        help="Disable continuity-based extension/merging (enabled by default).",
    )
    parser.add_argument(
        "--segment-continuity-gap-ms",
        type=float,
        default=5.0,
        help="Max gap (ms) to extend/bridge when continuity is enabled. Default: 5 (tuned)",
    )
    parser.add_argument(
        "--segment-continuity-freq-tol",
        type=float,
        default=1500.0,
        help="Peak frequency tolerance (Hz) for continuity. Default: 1500 (tuned)",
    )
    parser.add_argument(
        "--segment-continuity-energy-tol",
        type=float,
        default=15.0,
        help="Peak energy tolerance (dB) for continuity. Default: 15 (tuned middle-ground)",
    )
    parser.add_argument(
        "--segment-continuity-gap-match",
        type=float,
        default=0.6,
        help="Fraction of gap frames that must match continuity (0-1). Default: 0.6",
    )
    parser.add_argument(
        "--segment-continuity-bandwidth",
        type=float,
        default=6000.0,
        help="Band half-width in Hz around reference freq for continuity. Default: 6000",
    )
    parser.add_argument(
        "--segment-continuity-band-energy-tol",
        type=float,
        default=6.0,
        help="Band energy tolerance (dB) for continuity. Default: 6",
    )
    parser.add_argument(
        "--segment-continuity-band-match",
        type=float,
        default=0.6,
        help="Fraction of gap frames that must match band energy (0-1). Default: 0.6",
    )
    parser.add_argument(
        "--segment-continuity-kernel-size",
        type=int,
        default=5,
        help="Odd kernel size for continuity smoothing in detection. Default: 5",
    )
    parser.add_argument(
        "--segment-continuity-weight-center",
        type=float,
        default=1.0,
        help="Continuity smoothing center weight. Default: 1.0",
    )
    parser.add_argument(
        "--segment-continuity-weight-time",
        type=float,
        default=0.5,
        help="Continuity smoothing time-axis weight (left/right). Default: 0.5",
    )
    parser.add_argument(
        "--segment-continuity-weight-freq",
        type=float,
        default=0.2,
        help="Continuity smoothing freq-axis weight (up/down). Default: 0.2",
    )
    parser.add_argument(
        "--segment-continuity-weight-diag",
        type=float,
        default=0.8,
        help="Continuity smoothing diagonal weight. Default: 0.8",
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
        default=300000,
        help="Expected sample rate in Hz (used when auto sample-rate is disabled). Default: 300000",
    )
    auto_sr_group = parser.add_mutually_exclusive_group()
    auto_sr_group.add_argument(
        "--auto-sample-rate",
        dest="auto_sample_rate",
        action="store_true",
        default=True,
        help="Use the WAV file's sample rate during detection (default).",
    )
    auto_sr_group.add_argument(
        "--no-auto-sample-rate",
        dest="auto_sample_rate",
        action="store_false",
        help="Disable auto sample-rate detection and enforce --sample-rate.",
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
            auto_sample_rate=args.auto_sample_rate,
            energy_threshold_db=args.threshold,
            energy_mode=args.energy_mode,
            max_bandwidth_hz=args.max_bandwidth,
            min_duration_ms=args.min_duration,
            max_duration_ms=args.max_duration,
            merge_gap_ms=args.merge_gap,
            segment_continuity_enabled=args.segment_continuity,
            segment_continuity_max_gap_ms=args.segment_continuity_gap_ms,
            segment_continuity_freq_tolerance_hz=args.segment_continuity_freq_tol,
            segment_continuity_energy_tolerance_db=args.segment_continuity_energy_tol,
            segment_continuity_gap_match_fraction=args.segment_continuity_gap_match,
            segment_continuity_bandwidth_hz=args.segment_continuity_bandwidth,
            segment_continuity_band_energy_tolerance_db=args.segment_continuity_band_energy_tol,
            segment_continuity_band_match_fraction=args.segment_continuity_band_match,
            segment_continuity_kernel_size=args.segment_continuity_kernel_size,
            segment_continuity_weight_center=args.segment_continuity_weight_center,
            segment_continuity_weight_time=args.segment_continuity_weight_time,
            segment_continuity_weight_freq=args.segment_continuity_weight_freq,
            segment_continuity_weight_diag=args.segment_continuity_weight_diag,
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
