"""Create stratified train/val/test splits for USV classification.

Splits data by recording (source_file) to prevent data leakage.
Uses 70/15/15 split ratio (train/val/test).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_labeled_data(labels_path: Path, candidates_path: Path, noise_candidates_paths: list) -> pd.DataFrame:
    """Load labels and merge with candidate metadata from both USV and noise sources."""
    labels = pd.read_csv(labels_path)

    # Load USV candidates
    usv_candidates = pd.read_csv(candidates_path)

    # Load noise candidates from all provided paths
    noise_candidates_list = []
    for noise_path in noise_candidates_paths:
        if noise_path.exists():
            noise_candidates_list.append(pd.read_csv(noise_path))

    # Combine all noise candidates and remove duplicates
    noise_candidates = pd.concat(noise_candidates_list, ignore_index=True)
    noise_candidates = noise_candidates.drop_duplicates(subset=['candidate_id'], keep='first')

    # Combine USV and noise candidates (only need candidate_id and source_file)
    all_candidates = pd.concat([
        usv_candidates[['candidate_id', 'source_file']],
        noise_candidates[['candidate_id', 'source_file']]
    ], ignore_index=True)

    # Remove duplicates (some candidates appear in both USV and noise)
    all_candidates = all_candidates.drop_duplicates(subset=['candidate_id'], keep='first')

    # Merge with labels to get source_file for stratification
    data = labels.merge(all_candidates[['candidate_id', 'source_file']], on='candidate_id', how='left')

    return data


def create_splits(data: pd.DataFrame, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """Create train/val/test splits stratified by recording.

    Uses GroupShuffleSplit to ensure all samples from same recording
    stay in same split (prevents data leakage).
    """
    # Get unique recordings
    recordings = data['source_file'].unique()
    print(f"Total unique recordings: {len(recordings)}")
    print()

    # Create recording-level labels for stratification
    # Count USVs per recording to ensure balanced split
    recording_stats = data.groupby('source_file').agg({
        'final_label': lambda x: (x == 'USV').sum(),
        'candidate_id': 'count'
    }).reset_index()
    recording_stats.columns = ['source_file', 'usv_count', 'total_count']
    recording_stats['usv_ratio'] = recording_stats['usv_count'] / recording_stats['total_count']

    # First split: separate test set
    test_ratio = test_size
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)

    for train_val_idx, test_idx in gss_test.split(data, groups=data['source_file']):
        train_val_data = data.iloc[train_val_idx]
        test_data = data.iloc[test_idx]

    # Second split: separate train and val from remaining data
    val_ratio = val_size / (train_size + val_size)  # Adjust ratio for remaining data
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)

    for train_idx, val_idx in gss_val.split(train_val_data, groups=train_val_data['source_file']):
        train_data = train_val_data.iloc[train_idx]
        val_data = train_val_data.iloc[val_idx]

    return train_data, val_data, test_data


def print_split_stats(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame):
    """Print statistics for each split."""
    splits = {
        'Train': train_data,
        'Val': val_data,
        'Test': test_data
    }

    print("=" * 80)
    print("SPLIT STATISTICS")
    print("=" * 80)
    print()

    for split_name, split_data in splits.items():
        usv_count = (split_data['final_label'] == 'USV').sum()
        noise_count = (split_data['final_label'] == 'noise').sum()
        recordings = split_data['source_file'].nunique()

        print(f"{split_name} Set:")
        print(f"  Total samples: {len(split_data)}")
        print(f"  USV: {usv_count} ({100*usv_count/len(split_data):.1f}%)")
        print(f"  Noise: {noise_count} ({100*noise_count/len(split_data):.1f}%)")
        print(f"  Unique recordings: {recordings}")
        print()


def save_splits(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: Path):
    """Save splits to CSV files."""
    output_dir.mkdir(exist_ok=True, parents=True)

    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'

    # Save only necessary columns
    columns = ['candidate_id', 'final_label', 'source_file']

    train_data[columns].to_csv(train_path, index=False)
    val_data[columns].to_csv(val_path, index=False)
    test_data[columns].to_csv(test_path, index=False)

    print(f"Saved splits to {output_dir}/")
    print(f"  - train.csv: {len(train_data)} samples")
    print(f"  - val.csv: {len(val_data)} samples")
    print(f"  - test.csv: {len(test_data)} samples")
    print()


def main():
    print("=" * 80)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 80)
    print()

    # Load data
    labels_path = Path('labels_combined.csv')
    candidates_path = Path('candidates_final.csv')
    noise_candidates_paths = [
        Path('noise_samples/noise_samples.csv'),          # Old noise batch
        Path('noise_samples/noise_samples_final.csv'),    # New noise batch
    ]

    if not labels_path.exists():
        print(f"Error: {labels_path} not found!")
        print("Run prepare_labeled_dataset.py first")
        return

    if not candidates_path.exists():
        print(f"Error: {candidates_path} not found!")
        return

    print("Loading labeled data...")
    data = load_labeled_data(labels_path, candidates_path, noise_candidates_paths)
    print(f"Loaded {len(data)} labeled samples")
    print()

    # Create splits
    print("Creating splits (70% train, 15% val, 15% test)...")
    print("Stratifying by recording to prevent data leakage...")
    print()

    train_data, val_data, test_data = create_splits(data)

    # Print statistics
    print_split_stats(train_data, val_data, test_data)

    # Save splits
    output_dir = Path('splits')
    save_splits(train_data, val_data, test_data, output_dir)

    print("=" * 80)
    print("SPLITS CREATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Extract training-mode spectrograms (no axes)")
    print("2. Train CNN classifier")
    print()


if __name__ == '__main__':
    main()
