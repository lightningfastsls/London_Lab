"""Extract training-mode spectrograms for train/val/test splits.

This script:
1. Loads split CSV (candidate_id, final_label, source_file)
2. Merges with full candidate metadata (timing, frequency, etc.)
3. Extracts spectrograms in training mode (no axes)
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.extraction_config import ExtractionConfig
from usv_spectrogram.detection.spectrogram_extractor import SpectrogramExtractor


def load_split_with_metadata(split_path: Path) -> pd.DataFrame:
    """Load split and merge with full candidate metadata."""
    # Load split
    split_df = pd.read_csv(split_path)
    print(f"Loaded {len(split_df)} candidates from {split_path.name}")

    # Load all candidate metadata sources
    usv_candidates = pd.read_csv('candidates_final.csv')
    noise_old = pd.read_csv('noise_samples/noise_samples.csv')
    noise_new = pd.read_csv('noise_samples/noise_samples_final.csv')

    # Combine all candidates
    all_candidates = pd.concat([usv_candidates, noise_old, noise_new], ignore_index=True)
    all_candidates = all_candidates.drop_duplicates(subset=['candidate_id'], keep='first')

    print(f"  Loaded {len(all_candidates)} total candidates from all sources")

    # Merge split with full metadata
    merged = split_df.merge(all_candidates, on='candidate_id', how='left', suffixes=('', '_meta'))

    # Check for missing metadata
    missing = merged[merged['start_ms'].isna()]
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} candidates missing metadata!")
        print(f"  Sample missing: {missing['candidate_id'].head().tolist()}")

    # Use source_file from split (it's the same)
    merged['source_file'] = merged['source_file']

    return merged


def extract_spectrograms(split_name: str, split_path: Path, output_dir: Path, wav_dir: Path):
    """Extract spectrograms for a given split."""
    print()
    print("=" * 80)
    print(f"EXTRACTING {split_name.upper()} SET SPECTROGRAMS")
    print("=" * 80)
    print()

    # Load split with metadata
    candidates = load_split_with_metadata(split_path)

    # Filter out any with missing metadata
    candidates = candidates[candidates['start_ms'].notna()].copy()
    print(f"Extracting {len(candidates)} spectrograms...")
    print()

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create extraction config (training mode - no axes)
    config = ExtractionConfig(
        default_render_mode='training',  # No axes, clean images
        freq_min_hz=25_000,
        freq_max_hz=110_000,
        colormap='inferno'
    )

    # Create extractor
    extractor = SpectrogramExtractor(config=config)

    # Save candidates to temp CSV for extractor
    temp_csv = output_dir / f'{split_name}_candidates.csv'
    candidates.to_csv(temp_csv, index=False)

    # Extract using batch iterator
    results = []
    success_count = 0
    fail_count = 0

    for i, (candidate, output_path) in enumerate(extractor.extract_batch(
        temp_csv, wav_dir, output_dir, render_mode='training'
    ), 1):
        results.append((candidate, output_path))
        if output_path is not None:
            success_count += 1
        else:
            fail_count += 1

        if i % 100 == 0:
            print(f"  Processed {i}/{len(candidates)}...")

    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")

    # Save updated CSV with spectrogram paths
    extracted_csv = output_dir / f'{split_name}_extracted.csv'
    extractor.save_updated_csv(results, extracted_csv)

    # Also create a training CSV with label column for data loader
    # Data loader expects: candidate_id, spectrogram_path, label, source_file
    training_csv = output_dir.parent / f'{split_name}.csv'
    training_data = []
    for candidate, spec_path in results:
        if spec_path is not None:
            # Find the label from original split data
            label_row = candidates[candidates['candidate_id'] == candidate.candidate_id]
            if not label_row.empty:
                final_label = label_row['final_label'].iloc[0]
                source_file = label_row['source_file'].iloc[0]
                # Convert final_label to label format expected by data loader
                label = 'USV' if final_label == 'USV' else 'Not USV'
                training_data.append({
                    'candidate_id': candidate.candidate_id,
                    'spectrogram_path': str(spec_path),
                    'label': label,
                    'source_file': source_file
                })

    training_df = pd.DataFrame(training_data)
    training_df.to_csv(training_csv, index=False)
    print(f"Saved training CSV to: {training_csv}")

    print()
    print(f"Saved extracted spectrograms to: {output_dir}/")
    print(f"Saved metadata to: {extracted_csv}")
    print()

    # Clean up temp file
    temp_csv.unlink()


def main():
    print("=" * 80)
    print("EXTRACTING TRAINING-MODE SPECTROGRAMS FOR ALL SPLITS")
    print("=" * 80)
    print()

    # Paths
    wav_dir = Path("5970 USV")
    splits_dir = Path("splits")

    if not wav_dir.exists():
        print(f"Error: WAV directory not found: {wav_dir}")
        return

    # Extract for each split
    splits = [
        ('train', splits_dir / 'train.csv', Path('spectrograms_training/train')),
        ('val', splits_dir / 'val.csv', Path('spectrograms_training/val')),
        ('test', splits_dir / 'test.csv', Path('spectrograms_training/test')),
    ]

    for split_name, split_path, output_dir in splits:
        if not split_path.exists():
            print(f"Error: Split not found: {split_path}")
            continue

        extract_spectrograms(split_name, split_path, output_dir, wav_dir)

    print("=" * 80)
    print("ALL SPECTROGRAMS EXTRACTED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Next step: Train CNN classifier")
    print()


if __name__ == '__main__':
    main()
