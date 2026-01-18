"""Extract random non-candidate segments as noise examples.

Creates spectrogram images from time regions NOT flagged by the energy detector.
These are guaranteed noise examples to balance the USV/Not-USV dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.io_wav import get_default_wav_dir


def get_wav_duration_ms(wav_path: Path) -> float:
    """Get duration of WAV file in milliseconds."""
    with sf.SoundFile(str(wav_path)) as wav:
        return (wav.frames / wav.samplerate) * 1000.0


def load_candidates_by_file(
    candidates_csv: Path,
) -> dict[str, list[tuple[float, float]]]:
    """Load candidates and group by source file.

    Returns dict mapping source_file name to list of (start_ms, end_ms) tuples.
    """
    candidates_by_file: dict[str, list[tuple[float, float]]] = {}

    with open(candidates_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_file = row["source_file"]
            start_ms = float(row["start_ms"])
            end_ms = float(row["end_ms"])

            if source_file not in candidates_by_file:
                candidates_by_file[source_file] = []
            candidates_by_file[source_file].append((start_ms, end_ms))

    # Sort each file's candidates by start time
    for source_file in candidates_by_file:
        candidates_by_file[source_file].sort(key=lambda x: x[0])

    return candidates_by_file


def find_non_candidate_regions(
    wav_path: Path,
    candidates_for_file: list[tuple[float, float]],
    min_gap_ms: float = 100.0,
) -> list[tuple[float, float]]:
    """Find time regions not covered by any candidate.

    Parameters
    ----------
    wav_path:
        Path to WAV file.
    candidates_for_file:
        List of (start_ms, end_ms) tuples for candidates in this file.
    min_gap_ms:
        Minimum gap size to consider (must be able to extract a segment).

    Returns
    -------
    List of (start_ms, end_ms) tuples representing gaps.
    """
    duration_ms = get_wav_duration_ms(wav_path)

    # Sort candidates by start time
    sorted_candidates = sorted(candidates_for_file, key=lambda c: c[0])

    gaps: list[tuple[float, float]] = []
    prev_end = 0.0

    for start_ms, end_ms in sorted_candidates:
        if start_ms - prev_end >= min_gap_ms:
            gaps.append((prev_end, start_ms))
        prev_end = max(prev_end, end_ms)

    # Add final gap to end of file
    if duration_ms - prev_end >= min_gap_ms:
        gaps.append((prev_end, duration_ms))

    return gaps


def sample_segments_from_gaps(
    gaps_by_file: dict[str, list[tuple[float, float]]],
    count: int,
    min_duration_ms: float,
    max_duration_ms: float,
    context_ms: float,
    seed: Optional[int] = None,
) -> list[tuple[str, float, float, float, float]]:
    """Randomly sample segments from gaps.

    Uses weighted sampling based on available gap time per file.

    Returns list of (source_file, start_ms, end_ms, context_start_ms, context_end_ms) tuples.
    """
    if seed is not None:
        random.seed(seed)

    # Build pool of potential sampling points with weights
    # Each entry: (source_file, gap_start, gap_end, gap_duration)
    weighted_pool: list[tuple[str, float, float, float]] = []

    for source_file, gaps in gaps_by_file.items():
        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start
            # Only include gaps that can fit a segment + context
            min_required = min_duration_ms + 2 * context_ms
            if gap_duration >= min_required:
                weighted_pool.append((source_file, gap_start, gap_end, gap_duration))

    if not weighted_pool:
        return []

    # Calculate weights (proportional to gap duration)
    total_duration = sum(item[3] for item in weighted_pool)
    weights = [item[3] / total_duration for item in weighted_pool]

    samples: list[tuple[str, float, float, float, float]] = []
    attempts = 0
    max_attempts = count * 10  # Avoid infinite loops

    while len(samples) < count and attempts < max_attempts:
        attempts += 1

        # Weighted random selection of a gap
        gap_entry = random.choices(weighted_pool, weights=weights, k=1)[0]
        source_file, gap_start, gap_end, _ = gap_entry

        # Random duration within range
        duration_ms = random.uniform(min_duration_ms, max_duration_ms)

        # Random start position within gap (accounting for context and duration)
        # Segment must fit: gap_start + context <= start <= gap_end - duration - context
        min_start = gap_start + context_ms
        max_start = gap_end - duration_ms - context_ms

        if max_start <= min_start:
            continue

        start_ms = random.uniform(min_start, max_start)
        end_ms = start_ms + duration_ms
        context_start_ms = max(0.0, start_ms - context_ms)
        context_end_ms = end_ms + context_ms

        samples.append((source_file, start_ms, end_ms, context_start_ms, context_end_ms))

    return samples


def create_noise_candidate(
    source_file: str,
    start_ms: float,
    end_ms: float,
    context_start_ms: float,
    context_end_ms: float,
    wav_dir: Path,
) -> Candidate:
    """Create a Candidate object for a noise segment."""
    duration_ms = end_ms - start_ms
    random_id = uuid.uuid4().hex[:8]
    source_stem = Path(source_file).stem
    candidate_id = f"{source_stem}_noise_{random_id}"

    return Candidate(
        source_file=wav_dir / source_file,
        candidate_id=candidate_id,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        context_start_ms=context_start_ms,
        context_end_ms=context_end_ms,
        peak_freq_hz=0.0,  # Not applicable for noise
        peak_energy_db=0.0,  # Not applicable for noise
        interference_flag=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract random non-candidate segments as noise examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 350 noise samples
  python extract_noise_samples.py --candidates candidates_optimized.csv --wav-dir "5970 USV" --output-dir noise_samples --count 350 -v

  # Use specific duration range
  python extract_noise_samples.py --candidates c.csv --wav-dir data/ --output-dir noise/ --count 100 --min-duration-ms 30 --max-duration-ms 80
        """,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to input CSV file with detected candidates.",
    )
    parser.add_argument(
        "--wav-dir",
        required=True,
        help="Directory containing source WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save noise spectrogram PNG files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=350,
        help="Number of noise samples to extract. Default: 350",
    )
    parser.add_argument(
        "--min-gap-ms",
        type=float,
        default=100.0,
        help="Minimum gap size to consider in ms. Default: 100",
    )
    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=20.0,
        help="Minimum segment duration in ms. Default: 20",
    )
    parser.add_argument(
        "--max-duration-ms",
        type=float,
        default=100.0,
        help="Maximum segment duration in ms. Default: 100",
    )
    parser.add_argument(
        "--context-ms",
        type=float,
        default=50.0,
        help="Context window before/after segment in ms. Default: 50",
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: None (random)",
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

    # Resolve paths
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: Candidates CSV not found: {candidates_path}", file=sys.stderr)
        return 1

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_absolute():
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

    if args.verbose:
        print(f"Candidates CSV: {candidates_path}")
        print(f"WAV directory: {wav_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Target count: {args.count}")
        print(f"Duration range: {args.min_duration_ms}-{args.max_duration_ms} ms")
        print(f"Context window: {args.context_ms} ms")
        print(f"Minimum gap: {args.min_gap_ms} ms")
        if args.seed is not None:
            print(f"Random seed: {args.seed}")
        print()

    t0 = time.perf_counter()

    # Step 1: Load candidates grouped by file
    if args.verbose:
        print("Loading candidates...")
    candidates_by_file = load_candidates_by_file(candidates_path)
    total_candidates = sum(len(c) for c in candidates_by_file.values())
    if args.verbose:
        print(f"  Found {total_candidates} candidates across {len(candidates_by_file)} files")

    # Step 2: Find gaps for each file
    if args.verbose:
        print("Finding non-candidate regions...")
    gaps_by_file: dict[str, list[tuple[float, float]]] = {}
    total_gap_time_ms = 0.0

    for source_file in candidates_by_file:
        wav_path = wav_dir / source_file
        if not wav_path.exists():
            if args.verbose:
                print(f"  Warning: WAV not found: {wav_path}")
            continue

        gaps = find_non_candidate_regions(
            wav_path, candidates_by_file[source_file], args.min_gap_ms
        )
        if gaps:
            gaps_by_file[source_file] = gaps
            gap_time = sum(g[1] - g[0] for g in gaps)
            total_gap_time_ms += gap_time

    if args.verbose:
        print(f"  Found {sum(len(g) for g in gaps_by_file.values())} gaps")
        print(f"  Total gap time: {total_gap_time_ms/1000:.1f}s")

    if not gaps_by_file:
        print("Error: No valid gaps found for sampling", file=sys.stderr)
        return 1

    # Step 3: Sample random segments from gaps
    if args.verbose:
        print(f"Sampling {args.count} segments...")
    samples = sample_segments_from_gaps(
        gaps_by_file,
        args.count,
        args.min_duration_ms,
        args.max_duration_ms,
        args.context_ms,
        args.seed,
    )
    if args.verbose:
        print(f"  Sampled {len(samples)} segments")

    if len(samples) < args.count:
        print(
            f"Warning: Only {len(samples)} samples available (requested {args.count})",
            file=sys.stderr,
        )

    # Step 4: Create Candidate objects and extract spectrograms
    if args.verbose:
        print("Extracting spectrograms...")

    config = ExtractionConfig(
        sample_rate=args.sample_rate,
        default_render_mode=args.mode,
    )
    extractor = SpectrogramExtractor(config)

    results: list[tuple[Candidate, Optional[Path]]] = []
    success_count = 0
    fail_count = 0

    for i, (source_file, start_ms, end_ms, ctx_start, ctx_end) in enumerate(samples, 1):
        candidate = create_noise_candidate(
            source_file, start_ms, end_ms, ctx_start, ctx_end, wav_dir
        )
        output_path = extractor.extract_single(candidate, wav_dir, output_dir, args.mode)

        results.append((candidate, output_path))
        if output_path is not None:
            success_count += 1
        else:
            fail_count += 1

        if args.verbose and i % 50 == 0:
            print(f"  Processed {i}/{len(samples)}...")

    # Step 5: Save CSV with metadata
    output_csv = output_dir / "noise_samples.csv"
    if results:
        fieldnames = [
            "candidate_id",
            "source_file",
            "start_ms",
            "end_ms",
            "duration_ms",
            "context_start_ms",
            "context_end_ms",
            "peak_freq_hz",
            "peak_energy_db",
            "interference_flag",
            "spectrogram_path",
            "label",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate, spec_path in results:
                if spec_path is not None:
                    candidate.spectrogram_path = spec_path
                row = candidate.to_dict()
                row["label"] = "Not USV"
                writer.writerow(row)

    elapsed = time.perf_counter() - t0

    if args.verbose:
        print()
        print("--- Summary ---")
        print(f"Target samples: {args.count}")
        print(f"Successfully extracted: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Output directory: {output_dir}")
        print(f"Output CSV: {output_csv}")
        print(f"Time elapsed: {elapsed:.2f}s")

    if fail_count > 0:
        print(f"\nWarning: {fail_count} segments failed to extract", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
