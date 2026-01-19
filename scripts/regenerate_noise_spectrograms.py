#!/usr/bin/env python3
"""Regenerate noise sample spectrograms from noise_samples_final.csv.

This script reads the metadata from noise_samples_final.csv and generates
spectrogram PNG files for each noise sample using the SpectrogramExtractor.

Usage:
    python scripts/regenerate_noise_spectrograms.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from usv_spectrogram.detection.candidate import Candidate
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor
from usv_spectrogram.detection.extraction_config import ExtractionConfig


def main() -> int:
    # Paths
    noise_samples_csv = project_root / "noise_samples" / "noise_samples_final.csv"
    output_dir = project_root / "noise_samples"

    # Find WAV directory
    wav_dir = None
    env_wav_dir = os.environ.get("USV_WAV_DIR")
    if env_wav_dir and Path(env_wav_dir).exists():
        wav_dir = Path(env_wav_dir)
    else:
        # Try default location
        default_wav_dir = project_root / "5970 USV"
        if default_wav_dir.exists():
            wav_dir = default_wav_dir

    if not wav_dir:
        print("ERROR: WAV directory not found.")
        print("Set USV_WAV_DIR environment variable or ensure '5970 USV' directory exists.")
        return 1

    print(f"WAV directory: {wav_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Noise samples CSV: {noise_samples_csv}")

    if not noise_samples_csv.exists():
        print(f"ERROR: {noise_samples_csv} not found")
        return 1

    # Configure extractor - use same config as original extraction
    config = ExtractionConfig(
        sample_rate=300_000,  # Actual sample rate of recordings
        default_render_mode="review",
    )
    extractor = SpectrogramExtractor(config)

    # Read noise samples CSV
    with open(noise_samples_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nFound {len(rows)} noise samples to process")

    # Process each noise sample
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, row in enumerate(rows, 1):
        candidate_id = row["candidate_id"]
        output_path = output_dir / f"{candidate_id}.png"

        # Skip if already exists
        if output_path.exists():
            skip_count += 1
            continue

        # Create Candidate from CSV row
        try:
            candidate = Candidate.from_dict(row, base_path=wav_dir)
        except Exception as e:
            print(f"  [{i}/{len(rows)}] FAIL: {candidate_id} - Error creating candidate: {e}")
            fail_count += 1
            continue

        # Extract spectrogram
        try:
            result_path = extractor.extract_single(
                candidate=candidate,
                wav_dir=wav_dir,
                output_dir=output_dir,
                render_mode="review",
            )

            if result_path:
                success_count += 1
                if success_count % 50 == 0:
                    print(f"  [{i}/{len(rows)}] Generated {success_count} spectrograms...")
            else:
                print(f"  [{i}/{len(rows)}] FAIL: {candidate_id} - Extraction returned None")
                fail_count += 1
        except Exception as e:
            print(f"  [{i}/{len(rows)}] FAIL: {candidate_id} - {e}")
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Generated: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {success_count + skip_count + fail_count}")

    # Update the CSV with new paths
    if success_count > 0:
        print(f"\nUpdating {noise_samples_csv} with local paths...")
        updated_rows = []
        for row in rows:
            candidate_id = row["candidate_id"]
            local_path = output_dir / f"{candidate_id}.png"
            row["spectrogram_path"] = str(local_path)
            updated_rows.append(row)

        # Write updated CSV
        fieldnames = list(rows[0].keys())
        with open(noise_samples_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        print("CSV updated with local spectrogram paths.")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
