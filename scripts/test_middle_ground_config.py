"""Test middle-ground config on priority recordings."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.config import DetectionConfig
from usv_spectrogram.detection.energy_detector import EnergyDetector


def main():
    priority_files = [
        '2024-09-30_11-21-26_0000040.wav',
        '2024-09-30_11-20-32_0000025.wav',
        '2024-09-30_11-20-56_0000033.wav',
        '2024-09-30_11-20-41_0000029.wav',
    ]

    wav_dir = Path('5970 USV')

    # Middle-ground config
    config = DetectionConfig(
        segment_continuity_enabled=True,
        segment_continuity_max_gap_ms=5.0,
        segment_continuity_freq_tolerance_hz=1500.0,
        segment_continuity_energy_tolerance_db=15.0,  # Middle ground
        merge_gap_ms=3.0
    )

    detector = EnergyDetector(config)

    all_candidates = []

    print("Testing middle-ground config (energy_tol=15dB)")
    print("=" * 60)

    for filename in priority_files:
        wav_path = wav_dir / filename
        print(f"Processing: {filename}")

        candidates = detector.detect(wav_path)
        print(f"  Found {len(candidates)} candidates")

        all_candidates.extend(candidates)

    print()
    print(f"Total candidates: {len(all_candidates)}")

    # Save to CSV
    if all_candidates:
        import csv
        output_path = Path('candidates_middle_ground.csv')

        fieldnames = list(all_candidates[0].to_dict().keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in all_candidates:
                writer.writerow(candidate.to_dict())

        print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
