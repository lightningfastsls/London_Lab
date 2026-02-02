"""Generate comprehensive negative samples for CNN retraining.

Creates three types of negatives:
1. Random positions - arbitrary chunks from anywhere in recordings
2. Inter-USV gaps - silence/noise between known USVs
3. Low-energy regions - quiet parts with minimal acoustic activity

All samples avoid overlap with labeled USV regions (with 50ms buffer).

Key Requirements:
- Use SpectrogramExtractor with render_mode="training" (NOT scipy.signal)
- Match ExtractionConfig defaults (sr=300000, n_fft=512, hop=128, freq=20-120kHz)
- Label as "Not USV" (NOT "noise")
- Add sample_type field to track negative type
- Validate spectrogram dimensions after generation

Usage:
    python scripts/generate_comprehensive_negatives.py \\
        --wav-dir "5970 USV" \\
        --labels-csv labels.csv \\
        --output-dir data/comprehensive_negatives \\
        --n-random 500 \\
        --n-inter-usv 300 \\
        --n-low-energy 200 \\
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
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram._stft_core import compute_stft_frames_db


class USVRegion(NamedTuple):
    """Represents a USV region to avoid during random sampling."""
    start_ms: float
    end_ms: float


def load_usv_regions(labels_csv: Path, buffer_ms: float = 50.0) -> dict[str, list[USVRegion]]:
    """Load USV regions from labels.csv with buffer.

    Reused from generate_random_negatives.py
    """
    usv_regions: dict[str, list[USVRegion]] = {}

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only process USV labels
            if row["label"] != "USV":
                continue

            # Parse candidate_id to extract source file and start time
            candidate_id = row["candidate_id"]
            parts = candidate_id.rsplit("_", 1)
            if len(parts) != 2:
                continue

            source_stem = parts[0]
            try:
                start_ms = float(parts[1])
            except ValueError:
                continue

            # Estimate duration (conservative 100ms) + buffer
            expand_ms = float(row.get("expand_ms", 0.0))
            estimated_duration_ms = 100.0

            region_start = max(0.0, start_ms - buffer_ms + expand_ms)
            region_end = start_ms + estimated_duration_ms + buffer_ms + expand_ms

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
                merged[-1] = USVRegion(merged[-1].start_ms, max(merged[-1].end_ms, region.end_ms))
            else:
                merged.append(region)
        usv_regions[wav_filename] = merged

    return usv_regions


def overlaps_usv(start_ms: float, end_ms: float, usv_regions: list[USVRegion]) -> bool:
    """Check if a time range overlaps with any USV region.

    Reused from generate_random_negatives.py
    """
    for region in usv_regions:
        if not (end_ms <= region.start_ms or start_ms >= region.end_ms):
            return True
    return False


def generate_random_negatives(
    wav_files: list[Path],
    usv_regions_map: dict[str, list[USVRegion]],
    extractor: SpectrogramExtractor,
    wav_dir: Path,
    output_dir: Path,
    n_samples: int,
    duration_ms: float,
    seed: int,
) -> list[dict]:
    """Generate random position negatives.

    Pattern reused from generate_random_negatives.py
    """
    random.seed(seed)
    np.random.seed(seed)

    metadata_rows = []
    samples_per_file = max(1, n_samples // len(wav_files))
    generated_count = 0

    for wav_path in tqdm(wav_files, desc="Random negatives"):
        # Load WAV to get duration
        try:
            samples, sample_rate = load_wav_mono(wav_path)
        except Exception as e:
            print(f"  Error loading {wav_path.name}: {e}")
            continue

        if sample_rate != 300_000:
            print(f"  Skipping {wav_path.name} (sample rate {sample_rate} != 300000)")
            continue

        total_duration_ms = len(samples) / sample_rate * 1000.0
        usv_regions = usv_regions_map.get(wav_path.name, [])

        # Generate random samples for this file
        file_samples = 0
        attempts = 0
        max_attempts = samples_per_file * 20

        while file_samples < samples_per_file and attempts < max_attempts:
            attempts += 1

            # Random start time
            max_start_ms = total_duration_ms - duration_ms
            if max_start_ms <= 0:
                break

            start_ms = random.uniform(0, max_start_ms)
            end_ms = start_ms + duration_ms

            # Check for USV overlap
            if overlaps_usv(start_ms, end_ms, usv_regions):
                continue

            # Create candidate
            candidate = Candidate.create(
                source_file=wav_path,
                start_ms=start_ms,
                end_ms=end_ms,
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
                context_before_ms=0.0,
                context_after_ms=0.0,
                interference_flag=False,
            )

            # Extract spectrogram
            try:
                spec_path = extractor.extract_single(
                    candidate=candidate,
                    wav_dir=wav_dir,
                    output_dir=output_dir,
                    render_mode="training",
                )
            except Exception as e:
                continue

            if spec_path is None:
                continue

            # Rename with type prefix
            neg_id = f"neg_random_{generated_count:05d}"
            new_spec_path = output_dir / f"{neg_id}.png"
            spec_path.rename(new_spec_path)

            metadata_rows.append({
                "candidate_id": neg_id,
                "spectrogram_path": new_spec_path.name,
                "label": "Not USV",
                "source_file": wav_path.name,
                "sample_type": "random",
            })

            file_samples += 1
            generated_count += 1

            if generated_count >= n_samples:
                break

        if generated_count >= n_samples:
            break

    return metadata_rows


def generate_inter_usv_gap_negatives(
    wav_files: list[Path],
    usv_regions_map: dict[str, list[USVRegion]],
    extractor: SpectrogramExtractor,
    wav_dir: Path,
    output_dir: Path,
    n_samples: int,
    duration_ms: float,
    min_gap_ms: float,
    seed: int,
) -> list[dict]:
    """Generate negatives from gaps between consecutive USVs."""
    random.seed(seed + 1)  # Different seed than random
    np.random.seed(seed + 1)

    metadata_rows = []

    # Find all gaps across all files
    all_gaps = []
    for wav_path in wav_files:
        regions = usv_regions_map.get(wav_path.name, [])
        if len(regions) < 2:
            continue

        for i in range(len(regions) - 1):
            gap_start = regions[i].end_ms
            gap_end = regions[i + 1].start_ms
            gap_duration = gap_end - gap_start

            if gap_duration >= min_gap_ms:
                all_gaps.append((wav_path, gap_start, gap_end))

    if not all_gaps:
        print("Warning: No inter-USV gaps found")
        return []

    # Sample from gaps
    random.shuffle(all_gaps)
    generated_count = 0

    for wav_path, gap_start, gap_end in tqdm(all_gaps, desc="Inter-USV gaps"):
        if generated_count >= n_samples:
            break

        # Sample multiple positions within this gap if it's large enough
        gap_duration = gap_end - gap_start
        n_from_gap = min(3, int(gap_duration / (duration_ms * 2)))  # Up to 3 samples per gap

        for _ in range(n_from_gap):
            if generated_count >= n_samples:
                break

            # Random position within gap (with margin)
            margin = duration_ms / 2
            if gap_duration < duration_ms + 2 * margin:
                start_ms = gap_start + margin
            else:
                start_ms = random.uniform(gap_start + margin, gap_end - duration_ms - margin)

            end_ms = start_ms + duration_ms

            # Create candidate
            candidate = Candidate.create(
                source_file=wav_path,
                start_ms=start_ms,
                end_ms=end_ms,
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
                context_before_ms=0.0,
                context_after_ms=0.0,
                interference_flag=False,
            )

            # Extract spectrogram
            try:
                spec_path = extractor.extract_single(
                    candidate=candidate,
                    wav_dir=wav_dir,
                    output_dir=output_dir,
                    render_mode="training",
                )
            except Exception as e:
                continue

            if spec_path is None:
                continue

            # Rename with type prefix
            neg_id = f"neg_inter_usv_gap_{generated_count:05d}"
            new_spec_path = output_dir / f"{neg_id}.png"
            spec_path.rename(new_spec_path)

            metadata_rows.append({
                "candidate_id": neg_id,
                "spectrogram_path": new_spec_path.name,
                "label": "Not USV",
                "source_file": wav_path.name,
                "sample_type": "inter_usv_gap",
            })

            generated_count += 1

    return metadata_rows


def compute_energy_in_chunk(
    wav_path: Path,
    start_ms: float,
    end_ms: float,
    sample_rate: int,
    freq_min_hz: int = 20_000,
    freq_max_hz: int = 120_000,
) -> float:
    """Compute energy in USV frequency band for a chunk."""
    try:
        samples, sr = load_wav_mono(wav_path)

        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)

        if end_sample > len(samples):
            return float('inf')

        chunk = samples[start_sample:end_sample]

        # Compute STFT to get energy in frequency band
        spec_db, _ = compute_stft_frames_db(
            samples=chunk,
            sample_rate=sr,
            n_fft=512,
            hop_length=128,
            normalize_magnitude=False,
        )

        # Crop to USV frequency band
        freqs = np.fft.rfftfreq(512, 1 / sr)
        freq_mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
        spec_band = spec_db[freq_mask, :]

        # Return mean energy in band
        return float(np.mean(spec_band))

    except Exception:
        return float('inf')


def generate_low_energy_negatives(
    wav_files: list[Path],
    usv_regions_map: dict[str, list[USVRegion]],
    extractor: SpectrogramExtractor,
    wav_dir: Path,
    output_dir: Path,
    n_samples: int,
    duration_ms: float,
    energy_percentile: float,
    seed: int,
) -> list[dict]:
    """Generate negatives from low-energy regions."""
    random.seed(seed + 2)  # Different seed
    np.random.seed(seed + 2)

    metadata_rows = []
    generated_count = 0

    # Sample candidate positions and compute energy
    print("Computing energy for candidate positions...")
    candidates_per_file = 200

    for wav_path in tqdm(wav_files, desc="Low-energy negatives"):
        if generated_count >= n_samples:
            break

        # Load WAV to get duration
        try:
            samples, sample_rate = load_wav_mono(wav_path)
        except Exception:
            continue

        if sample_rate != 300_000:
            continue

        total_duration_ms = len(samples) / sample_rate * 1000.0
        usv_regions = usv_regions_map.get(wav_path.name, [])

        # Sample many positions and compute energy
        candidate_positions = []
        for _ in range(candidates_per_file):
            max_start_ms = total_duration_ms - duration_ms
            if max_start_ms <= 0:
                break

            start_ms = random.uniform(0, max_start_ms)
            end_ms = start_ms + duration_ms

            if not overlaps_usv(start_ms, end_ms, usv_regions):
                energy = compute_energy_in_chunk(wav_path, start_ms, end_ms, sample_rate)
                if energy != float('inf'):
                    candidate_positions.append((start_ms, end_ms, energy))

        if not candidate_positions:
            continue

        # Sort by energy and take lowest percentile
        candidate_positions.sort(key=lambda x: x[2])
        n_low_energy = max(1, int(len(candidate_positions) * energy_percentile / 100))
        low_energy_positions = candidate_positions[:n_low_energy]

        # Extract spectrograms for low-energy positions
        for start_ms, end_ms, _ in low_energy_positions:
            if generated_count >= n_samples:
                break

            # Create candidate
            candidate = Candidate.create(
                source_file=wav_path,
                start_ms=start_ms,
                end_ms=end_ms,
                peak_freq_hz=0.0,
                peak_energy_db=0.0,
                context_before_ms=0.0,
                context_after_ms=0.0,
                interference_flag=False,
            )

            # Extract spectrogram
            try:
                spec_path = extractor.extract_single(
                    candidate=candidate,
                    wav_dir=wav_dir,
                    output_dir=output_dir,
                    render_mode="training",
                )
            except Exception:
                continue

            if spec_path is None:
                continue

            # Rename with type prefix
            neg_id = f"neg_low_energy_{generated_count:05d}"
            new_spec_path = output_dir / f"{neg_id}.png"
            spec_path.rename(new_spec_path)

            metadata_rows.append({
                "candidate_id": neg_id,
                "spectrogram_path": new_spec_path.name,
                "label": "Not USV",
                "source_file": wav_path.name,
                "sample_type": "low_energy",
            })

            generated_count += 1

    return metadata_rows


def validate_spectrograms(output_dir: Path, all_metadata: list[dict]) -> None:
    """Validate generated spectrograms for consistency."""
    print("\nValidating generated spectrograms...")

    failed_checks = []
    widths = []
    heights = []

    for row in all_metadata:
        filename = row["spectrogram_path"]
        img_path = output_dir / filename

        if not img_path.exists():
            failed_checks.append(f"{filename}: File not found")
            continue

        try:
            img = Image.open(img_path)
            width, height = img.size

            # Check dimensions
            if height != 256:
                failed_checks.append(f"{filename}: Height {height} != 256")
            if width < 100 or width > 800:
                failed_checks.append(f"{filename}: Width {width} out of range [100, 800]")

            # Check mode (should be RGB after colormap)
            if img.mode != 'RGB':
                failed_checks.append(f"{filename}: Mode {img.mode} != RGB")

            widths.append(width)
            heights.append(height)

        except Exception as e:
            failed_checks.append(f"{filename}: Error loading - {e}")

    # Print results
    if failed_checks:
        print(f"  WARNINGS ({len(failed_checks)} issues):")
        for check in failed_checks[:10]:  # Show first 10
            print(f"    - {check}")
        if len(failed_checks) > 10:
            print(f"    ... and {len(failed_checks) - 10} more")
    else:
        print("  All spectrograms valid ✓")

    if widths:
        print(f"  Width range: {min(widths)}-{max(widths)}px (median: {int(np.median(widths))}px)")
        print(f"  Height range: {min(heights)}-{max(heights)}px")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive negative samples for CNN retraining",
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
        "--n-random",
        type=int,
        default=500,
        help="Number of random position negatives (default: 500)",
    )
    parser.add_argument(
        "--n-inter-usv",
        type=int,
        default=300,
        help="Number of inter-USV gap negatives (default: 300)",
    )
    parser.add_argument(
        "--n-low-energy",
        type=int,
        default=200,
        help="Number of low-energy negatives (default: 200)",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=40.0,
        help="Duration of each negative sample in ms (default: 40)",
    )
    parser.add_argument(
        "--min-gap-ms",
        type=float,
        default=100.0,
        help="Minimum gap between USVs for inter-USV sampling (default: 100)",
    )
    parser.add_argument(
        "--energy-percentile",
        type=float,
        default=20.0,
        help="Percentile for low-energy selection (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load USV regions to avoid
    print("Loading USV regions from labels.csv...")
    usv_regions_map = load_usv_regions(args.labels_csv, buffer_ms=50.0)
    print(f"  Loaded regions for {len(usv_regions_map)} files")

    # Get WAV files
    wav_dir = Path(args.wav_dir)
    wav_files = list(wav_dir.glob("*.wav"))
    if not wav_files:
        print(f"Error: No WAV files found in {wav_dir}")
        return
    print(f"  Found {len(wav_files)} WAV files")

    # Create spectrogram extractor
    extraction_config = ExtractionConfig(
        sample_rate=300_000,
        n_fft=512,
        hop_length=128,
        freq_min_hz=20_000,
        freq_max_hz=120_000,
        colormap="magma",
        dynamic_range_method="mad",
        default_render_mode="training",
    )
    extractor = SpectrogramExtractor(config=extraction_config)

    # Generate all negative types
    all_metadata = []

    print(f"\n1. Generating {args.n_random} random position negatives...")
    random_metadata = generate_random_negatives(
        wav_files=wav_files,
        usv_regions_map=usv_regions_map,
        extractor=extractor,
        wav_dir=wav_dir,
        output_dir=output_dir,
        n_samples=args.n_random,
        duration_ms=args.duration_ms,
        seed=args.seed,
    )
    all_metadata.extend(random_metadata)
    print(f"   Generated {len(random_metadata)} random samples")

    print(f"\n2. Generating {args.n_inter_usv} inter-USV gap negatives...")
    gap_metadata = generate_inter_usv_gap_negatives(
        wav_files=wav_files,
        usv_regions_map=usv_regions_map,
        extractor=extractor,
        wav_dir=wav_dir,
        output_dir=output_dir,
        n_samples=args.n_inter_usv,
        duration_ms=args.duration_ms,
        min_gap_ms=args.min_gap_ms,
        seed=args.seed,
    )
    all_metadata.extend(gap_metadata)
    print(f"   Generated {len(gap_metadata)} inter-USV gap samples")

    print(f"\n3. Generating {args.n_low_energy} low-energy negatives...")
    low_energy_metadata = generate_low_energy_negatives(
        wav_files=wav_files,
        usv_regions_map=usv_regions_map,
        extractor=extractor,
        wav_dir=wav_dir,
        output_dir=output_dir,
        n_samples=args.n_low_energy,
        duration_ms=args.duration_ms,
        energy_percentile=args.energy_percentile,
        seed=args.seed,
    )
    all_metadata.extend(low_energy_metadata)
    print(f"   Generated {len(low_energy_metadata)} low-energy samples")

    # Validate spectrograms
    validate_spectrograms(output_dir, all_metadata)

    # Save metadata CSV
    metadata_csv = output_dir / "comprehensive_negatives_metadata.csv"
    print(f"\nSaving metadata to {metadata_csv}...")

    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["candidate_id", "spectrogram_path", "label", "source_file", "sample_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata)

    # Print summary
    print(f"\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(all_metadata)}")
    print(f"  - Random: {len(random_metadata)}")
    print(f"  - Inter-USV gaps: {len(gap_metadata)}")
    print(f"  - Low-energy: {len(low_energy_metadata)}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Metadata CSV: {metadata_csv}")


if __name__ == "__main__":
    main()
