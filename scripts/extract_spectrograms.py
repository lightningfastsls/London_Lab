"""Extract spectrogram images from detected USV candidates.

Converts candidate segments from detection CSV into standardized spectrogram
PNG images for human labeling and CNN training.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

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
        help="Minimum dB value (black level, used with --dynamic-range=fixed). Default: -80",
    )
    parser.add_argument(
        "--db-ceiling",
        type=float,
        default=0.0,
        help="Maximum dB value (white level, used with --dynamic-range=fixed). Default: 0",
    )
    parser.add_argument(
        "--dynamic-range",
        choices=["fixed", "percentile", "std", "mad"],
        default="mad",
        help="Dynamic range method: 'fixed' (use db-floor/ceiling), 'percentile', 'std', or 'mad' (most robust). Default: mad",
    )
    parser.add_argument(
        "--mad-vmin-scale",
        type=float,
        default=2.0,
        help="Scale factor for vmin with MAD method (median - scale*MAD). Default: 2.0",
    )
    parser.add_argument(
        "--mad-vmax-scale",
        type=float,
        default=4.0,
        help="Scale factor for vmax with MAD method (median + scale*MAD). Default: 4.0",
    )
    parser.add_argument(
        "--continuity-filter",
        action="store_true",
        help="Enable continuity filter for review spectrograms (diagonal-emphasis smoothing).",
    )
    parser.add_argument(
        "--continuity-kernel-size",
        type=int,
        default=3,
        help="Odd kernel size for continuity filter (e.g., 3 or 5). Default: 3",
    )
    parser.add_argument(
        "--continuity-weight-center",
        type=float,
        default=1.0,
        help="Continuity filter center weight. Default: 1.0",
    )
    parser.add_argument(
        "--continuity-weight-axis",
        type=float,
        default=0.5,
        help="Continuity filter axis-neighbor weight. Default: 0.5",
    )
    parser.add_argument(
        "--continuity-weight-diag",
        type=float,
        default=0.8,
        help="Continuity filter diagonal-neighbor weight. Default: 0.8",
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
    parser.add_argument(
        "--overlap-prune",
        type=float,
        default=0.7,
        help=(
            "Prune overlapping candidates when overlap >= fraction of shorter duration (0-1). "
            "Set to 0 to disable. Default: 0.7"
        ),
    )
    parser.add_argument(
        "--overlap-unique-fraction",
        type=float,
        default=0.2,
        help=(
            "Keep the shorter candidate if its unique (non-overlap) portion exceeds this fraction "
            "(0-1). Default: 0.2"
        ),
    )
    parser.add_argument(
        "--overlap-compare-dir",
        default="spectrograms_review_overlap_compare",
        help="Directory to store kept/removed PNGs when overlap pruning is enabled.",
    )
    return parser.parse_args()


def _dedupe_results(
    results: list[tuple[Candidate, Optional[Path]]],
) -> tuple[list[tuple[Candidate, Optional[Path]]], list[Path]]:
    """Deduplicate results by candidate_id, keeping the most complete entry."""
    last_by_id: dict[str, tuple[Candidate, Optional[Path]]] = {}
    duplicate_paths: list[Path] = []

    for candidate, output_path in results:
        candidate_id = candidate.candidate_id
        if candidate_id in last_by_id:
            prev_candidate, prev_path = last_by_id[candidate_id]
            if prev_path is None and output_path is not None:
                last_by_id[candidate_id] = (candidate, output_path)
            elif prev_path is not None and output_path is None:
                continue
            else:
                if prev_path and output_path and Path(prev_path) != Path(output_path):
                    duplicate_paths.append(Path(prev_path))
                last_by_id[candidate_id] = (candidate, output_path)
        else:
            last_by_id[candidate_id] = (candidate, output_path)

    return list(last_by_id.values()), duplicate_paths


def _overlap_duration_ms(a: Candidate, b: Candidate) -> float:
    """Return overlap duration in ms between two candidates."""
    start = max(a.start_ms, b.start_ms)
    end = min(a.end_ms, b.end_ms)
    return max(0.0, end - start)


def _prune_overlapping_candidates(
    results: Iterable[tuple[Candidate, Optional[Path]]],
    overlap_fraction_threshold: float,
    unique_fraction_threshold: float,
) -> tuple[list[tuple[Candidate, Optional[Path]]], list[tuple[Candidate, Optional[Path]]]]:
    """Prune overlapping candidates based on overlap fraction and unique portion."""
    by_file: dict[str, list[tuple[Candidate, Optional[Path]]]] = {}
    for candidate, output_path in results:
        key = str(candidate.source_file)
        by_file.setdefault(key, []).append((candidate, output_path))

    kept: list[tuple[Candidate, Optional[Path]]] = []
    removed: list[tuple[Candidate, Optional[Path]]] = []

    for items in by_file.values():
        items.sort(key=lambda item: (item[0].start_ms, item[0].end_ms))
        kept_local: list[tuple[Candidate, Optional[Path]]] = []

        for candidate, output_path in items:
            drop_current = False
            while kept_local:
                prev_candidate, prev_path = kept_local[-1]
                overlap_ms = _overlap_duration_ms(prev_candidate, candidate)
                if overlap_ms <= 0:
                    break

                shorter = (
                    prev_candidate
                    if prev_candidate.duration_ms <= candidate.duration_ms
                    else candidate
                )
                shorter_duration = max(shorter.duration_ms, 1e-9)
                overlap_fraction = overlap_ms / shorter_duration
                unique_fraction = (shorter_duration - overlap_ms) / shorter_duration

                if overlap_fraction < overlap_fraction_threshold or unique_fraction > unique_fraction_threshold:
                    break

                if shorter is prev_candidate:
                    removed.append((prev_candidate, prev_path))
                    kept_local.pop()
                    continue

                removed.append((candidate, output_path))
                drop_current = True
                break

            if drop_current:
                continue

            kept_local.append((candidate, output_path))

        kept.extend(kept_local)

    return kept, removed


def _write_overlap_compare_folders(
    kept: Iterable[tuple[Candidate, Optional[Path]]],
    removed: Iterable[tuple[Candidate, Optional[Path]]],
    compare_dir: Path,
) -> None:
    """Write kept/removed PNGs into compare folder and remove pruned PNGs."""
    kept_dir = compare_dir / "kept"
    removed_dir = compare_dir / "removed"
    kept_dir.mkdir(parents=True, exist_ok=True)
    removed_dir.mkdir(parents=True, exist_ok=True)

    for candidate, output_path in kept:
        if output_path is None:
            continue
        src = Path(output_path)
        if not src.exists():
            continue
        dst = kept_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)

    for candidate, output_path in removed:
        if output_path is None:
            continue
        src = Path(output_path)
        if not src.exists():
            continue
        dst = removed_dir / src.name
        if dst.exists():
            src.unlink(missing_ok=True)
            continue
        shutil.move(str(src), str(dst))


def _write_dedupe_removed(
    duplicate_paths: Iterable[Path],
    compare_dir: Path,
) -> int:
    """Move dedupe-removed PNGs into a compare folder."""
    removed_dir = compare_dir / "dedupe_removed"
    removed_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    for dup_path in duplicate_paths:
        src = Path(dup_path)
        if not src.exists():
            continue
        dst = removed_dir / src.name
        if dst.exists():
            src.unlink(missing_ok=True)
            continue
        shutil.move(str(src), str(dst))
        moved += 1

    return moved


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.overlap_prune <= 1.0:
        print("Error: --overlap-prune must be between 0 and 1.", file=sys.stderr)
        return 1
    if not 0.0 <= args.overlap_unique_fraction <= 1.0:
        print("Error: --overlap-unique-fraction must be between 0 and 1.", file=sys.stderr)
        return 1

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

    # Clean existing PNG files from output directory
    existing_pngs = list(output_dir.glob("*.png"))
    if existing_pngs:
        if args.verbose:
            print(f"Cleaning {len(existing_pngs)} existing PNG files from {output_dir}...")
        for png_file in existing_pngs:
            png_file.unlink()

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
            dynamic_range_method=args.dynamic_range,
            mad_vmin_scale=args.mad_vmin_scale,
            mad_vmax_scale=args.mad_vmax_scale,
            continuity_filter_enabled=args.continuity_filter,
            continuity_kernel_size=args.continuity_kernel_size,
            continuity_weight_center=args.continuity_weight_center,
            continuity_weight_axis=args.continuity_weight_axis,
            continuity_weight_diag=args.continuity_weight_diag,
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
        print(f"Dynamic range: {args.dynamic_range}")
        if args.dynamic_range == "mad":
            print(f"  MAD scales: vmin={args.mad_vmin_scale}, vmax={args.mad_vmax_scale}")
        if args.continuity_filter:
            print("Continuity filter: enabled")
            print(
                "  Kernel size: {size} | weights: center={center}, axis={axis}, diag={diag}".format(
                    size=args.continuity_kernel_size,
                    center=args.continuity_weight_center,
                    axis=args.continuity_weight_axis,
                    diag=args.continuity_weight_diag,
                )
            )
        else:
            print("Continuity filter: disabled")
        if args.overlap_prune > 0:
            print(
                "Overlap prune: enabled (fraction={fraction}, unique>{unique})".format(
                    fraction=args.overlap_prune,
                    unique=args.overlap_unique_fraction,
                )
            )
            print(f"  Compare dir: {args.overlap_compare_dir}")
        else:
            print("Overlap prune: disabled")
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

    results, duplicate_paths = _dedupe_results(results)
    dedupe_removed_count = 0
    if args.overlap_prune > 0 and duplicate_paths:
        dedupe_removed_count = _write_dedupe_removed(
            duplicate_paths,
            compare_dir=Path(args.overlap_compare_dir),
        )
    else:
        for dup_path in duplicate_paths:
            try:
                dup_path.unlink()
            except FileNotFoundError:
                continue

    pruned_count = 0
    if args.overlap_prune > 0:
        results, pruned = _prune_overlapping_candidates(
            results,
            overlap_fraction_threshold=args.overlap_prune,
            unique_fraction_threshold=args.overlap_unique_fraction,
        )
        _write_overlap_compare_folders(
            results,
            pruned,
            compare_dir=Path(args.overlap_compare_dir),
        )
        pruned_count = len(pruned)

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
        if args.overlap_prune > 0:
            print(f"Dedupe removed: {dedupe_removed_count}")
        if args.overlap_prune > 0:
            print(f"Overlap pruned: {pruned_count}")
        print(f"Output directory: {output_dir}")
        print(f"Updated CSV: {output_csv}")
        print(f"Time elapsed: {elapsed:.2f}s")

    if fail_count > 0:
        print(f"\nWarning: {fail_count} candidates failed to extract", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
