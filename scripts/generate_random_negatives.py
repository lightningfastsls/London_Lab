"""Generate random negative samples for CNN retraining experiment.

This script extracts random chunks from WAV files, avoiding known USV regions,
to create negative training samples. The goal is to teach the CNN what "no USV"
looks like, addressing the batch detection problem where the model predicts
high probability on random chunks.

Key Requirements:
- Extract ~40ms random chunks from WAV files
- Avoid known USV regions (with 50ms buffer)
- Use render_mode="training" for CNN-compatible spectrograms
- Match ExtractionConfig defaults (sr=300000, n_fft=512, hop=128, freq=20-120kHz)
- Label as "Not USV" (NOT "noise")
- Use Candidate.create() for consistent metadata format

Usage:
    python scripts/generate_random_negatives.py \\
        --wav-dir "5970 USV" \\
        --labels-csv labels.csv \\
        --output-dir data/experiment_negatives \\
        --n-samples 100 \\
        --duration-ms 40 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.io_wav import load_wav_mono


class USVRegion(NamedTuple):
    """Represents a USV region to avoid during random sampling."""

    start_ms: float
    end_ms: float


def load_usv_regions(labels_csv: Path, buffer_ms: float = 50.0) -> dict[str, list[USVRegion]]:
    """Load USV regions from labels.csv with buffer.

    Parameters
    ----------
    labels_csv:
        Path to labels.csv file with columns: candidate_id, label, labeled_at, expand_ms
    buffer_ms:
        Buffer to add around each USV region (default: 50ms on each side)

    Returns
    -------
    Dictionary mapping WAV filename to list of USVRegion tuples
    """
    usv_regions: dict[str, list[USVRegion]] = {}

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only process USV labels
            if row["label"] != "USV":
                continue

            # Parse candidate_id to extract source file and start time
            # Format: {source_stem}_{start_ms:08.0f}
            candidate_id = row["candidate_id"]
            parts = candidate_id.rsplit("_", 1)
            if len(parts) != 2:
                continue

            source_stem = parts[0]
            try:
                start_ms = float(parts[1])
            except ValueError:
                continue

            # We don't have end_ms in labels.csv, so we need to estimate from candidates.csv
            # For now, use a conservative estimate of 100ms duration (typical USV duration)
            # This will be refined when we load the actual candidates
            # TODO: Consider loading candidates.csv to get actual end_ms
            # For now, use conservative 100ms duration + expand_ms
            expand_ms = float(row.get("expand_ms", 0.0))
            estimated_duration_ms = 100.0  # Conservative estimate

            # Calculate region with buffer
            region_start = max(0.0, start_ms - buffer_ms + expand_ms)
            region_end = start_ms + estimated_duration_ms + buffer_ms + expand_ms

            # Add .wav extension to source stem
            wav_filename = f"{source_stem}.wav"

            if wav_filename not in usv_regions:
                usv_regions[wav_filename] = []

            usv_regions[wav_filename].append(USVRegion(region_start, region_end))

    # Merge overlapping regions for efficiency
    for wav_filename in usv_regions:
        regions = sorted(usv_regions[wav_filename], key=lambda r: r.start_ms)
        merged = []
        for region in regions:
            if merged and region.start_ms <= merged[-1].end_ms:
                # Overlapping or adjacent - merge
                merged[-1] = USVRegion(merged[-1].start_ms, max(merged[-1].end_ms, region.end_ms))
            else:
                merged.append(region)
        usv_regions[wav_filename] = merged

    return usv_regions


def overlaps_usv(start_ms: float, end_ms: float, usv_regions: list[USVRegion]) -> bool:
    """Check if a time range overlaps with any USV region.

    Parameters
    ----------
    start_ms:
        Start time in milliseconds
    end_ms:
        End time in milliseconds
    usv_regions:
        List of USVRegion tuples to check against

    Returns
    -------
    True if overlaps with any USV region, False otherwise
    """
    for region in usv_regions:
        # Check for any overlap
        if not (end_ms <= region.start_ms or start_ms >= region.end_ms):
            return True
    return False


def generate_random_negatives(
    wav_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    n_samples: int,
    duration_ms: float,
    seed: int,
    buffer_ms: float = 50.0,
    max_attempts_per_file: int = 100,
) -> None:
    """Generate random negative samples avoiding USV regions.

    Parameters
    ----------
    wav_dir:
        Directory containing WAV files
    labels_csv:
        Path to labels.csv with USV regions
    output_dir:
        Directory to save spectrograms and metadata CSV
    n_samples:
        Number of random negatives to generate
    duration_ms:
        Duration of each random chunk in milliseconds
    seed:
        Random seed for reproducibility
    buffer_ms:
        Buffer around USV regions to avoid (default: 50ms)
    max_attempts_per_file:
        Maximum attempts to find non-USV region per file before giving up
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load USV regions to avoid
    print(f"Loading USV regions from {labels_csv}...")
    usv_regions_map = load_usv_regions(labels_csv, buffer_ms)
    print(f"Loaded USV regions for {len(usv_regions_map)} WAV files")

    # Find all WAV files
    wav_dir = Path(wav_dir)
    wav_files = list(wav_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")
    print(f"Found {len(wav_files)} WAV files")

    # Create extraction config matching training pipeline
    # CRITICAL: Use same parameters as training (sr=300000, n_fft=512, hop=128)
    extraction_config = ExtractionConfig(
        sample_rate=300_000,
        n_fft=512,
        hop_length=128,
        freq_min_hz=20_000,
        freq_max_hz=120_000,
        colormap="magma",
        dynamic_range_method="mad",
        default_render_mode="training",  # Set default to training
    )

    # Create spectrogram extractor
    extractor = SpectrogramExtractor(config=extraction_config)

    # Stratified sampling: distribute samples evenly across WAV files
    samples_per_file = n_samples // len(wav_files)
    remainder = n_samples % len(wav_files)

    metadata_rows = []
    generated_count = 0

    for file_idx, wav_path in enumerate(wav_files):
        # Determine number of samples for this file
        n_samples_this_file = samples_per_file
        if file_idx < remainder:
            n_samples_this_file += 1

        if n_samples_this_file == 0:
            continue

        print(f"\nProcessing {wav_path.name} (target: {n_samples_this_file} samples)...")

        # Load WAV file to get duration
        try:
            samples, sample_rate = load_wav_mono(wav_path)
        except Exception as e:
            print(f"  Error loading {wav_path.name}: {e}")
            continue

        # Validate sample rate
        if sample_rate != extraction_config.sample_rate:
            print(f"  Warning: Sample rate {sample_rate} Hz doesn't match config {extraction_config.sample_rate} Hz")
            print(f"  Skipping {wav_path.name}")
            continue

        total_duration_ms = len(samples) / sample_rate * 1000.0
        print(f"  Duration: {total_duration_ms:.0f} ms")

        # Get USV regions for this file
        usv_regions = usv_regions_map.get(wav_path.name, [])
        print(f"  USV regions to avoid: {len(usv_regions)}")

        # Generate random samples for this file
        file_samples_generated = 0
        attempts = 0

        while file_samples_generated < n_samples_this_file and attempts < max_attempts_per_file * n_samples_this_file:
            attempts += 1

            # Random start time
            max_start_ms = total_duration_ms - duration_ms
            if max_start_ms <= 0:
                print(f"  File too short for {duration_ms}ms chunks")
                break

            start_ms = random.uniform(0, max_start_ms)
            end_ms = start_ms + duration_ms

            # Check for USV overlap
            if overlaps_usv(start_ms, end_ms, usv_regions):
                continue

            # Create candidate using factory method
            # Use placeholder values for peak_freq_hz and peak_energy_db since this is random background
            candidate = Candidate.create(
                source_file=wav_path,
                start_ms=start_ms,
                end_ms=end_ms,
                peak_freq_hz=0.0,  # Placeholder - not meaningful for random chunks
                peak_energy_db=0.0,  # Placeholder - not meaningful for random chunks
                context_before_ms=0.0,  # No context needed for random chunks
                context_after_ms=0.0,
                interference_flag=False,
            )

            # Extract spectrogram with render_mode="training"
            try:
                spec_path = extractor.extract_single(
                    candidate=candidate,
                    wav_dir=wav_dir,
                    output_dir=output_dir,
                    render_mode="training",  # CRITICAL: Use training mode
                )
            except Exception as e:
                print(f"  Error extracting spectrogram: {e}")
                continue

            if spec_path is None:
                continue

            # Rename to random_neg_XXXXX.png format
            random_neg_id = f"random_neg_{generated_count + 1:05d}"
            new_spec_path = output_dir / f"{random_neg_id}.png"
            spec_path.rename(new_spec_path)

            # Add metadata row
            # CRITICAL: Save only filename (not full path) to match training CSV format
            metadata_rows.append({
                "candidate_id": random_neg_id,
                "spectrogram_path": new_spec_path.name,  # Just filename, not full path
                "label": "Not USV",  # CRITICAL: Use "Not USV" not "noise"
                "source_file": wav_path.name,
            })

            file_samples_generated += 1
            generated_count += 1
            print(f"  Generated {file_samples_generated}/{n_samples_this_file} (total: {generated_count}/{n_samples})")

        if file_samples_generated < n_samples_this_file:
            print(f"  Warning: Only generated {file_samples_generated}/{n_samples_this_file} samples (reached max attempts)")

    # Save metadata CSV
    metadata_csv = output_dir / "random_negatives_metadata.csv"
    print(f"\nSaving metadata to {metadata_csv}...")

    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["candidate_id", "spectrogram_path", "label", "source_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nGeneration complete!")
    print(f"  Generated: {generated_count}/{n_samples} samples")
    print(f"  Spectrograms: {output_dir}")
    print(f"  Metadata: {metadata_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random negative samples for CNN retraining experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help='Directory containing WAV files (e.g., "5970 USV")',
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="Path to labels.csv with USV regions to avoid",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save spectrograms and metadata CSV",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of random negatives to generate (default: 100)",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=40.0,
        help="Duration of each random chunk in milliseconds (default: 40.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--buffer-ms",
        type=float,
        default=50.0,
        help="Buffer around USV regions to avoid in milliseconds (default: 50.0)",
    )

    args = parser.parse_args()

    generate_random_negatives(
        wav_dir=args.wav_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        duration_ms=args.duration_ms,
        seed=args.seed,
        buffer_ms=args.buffer_ms,
    )


if __name__ == "__main__":
    main()
