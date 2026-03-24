"""Combine USV candidate labels and noise sample reviews into final training dataset.

This script:
1. Loads USV candidate labels (labels.csv)
2. Loads noise sample reviews (noise_reviews.csv)
3. Combines them into unified dataset
4. Filters out uncertain/skip samples
5. Creates final labeled dataset ready for train/val/test split
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_usv_labels(labels_path: Path) -> pd.DataFrame:
    """Load USV candidate labels."""
    df = pd.read_csv(labels_path)

    # Standardize label names
    label_map = {
        'USV': 'USV',
        'Not USV': 'noise',
        'Uncertain': 'uncertain'
    }

    df['final_label'] = df['label'].map(label_map)
    df['source'] = 'candidates'

    return df[['candidate_id', 'final_label', 'labeled_at', 'source']]


def load_noise_reviews(reviews_path: Path) -> pd.DataFrame:
    """Load noise sample reviews and convert to labels."""
    df = pd.read_csv(reviews_path)

    # Map review status to labels
    # Clean = confirmed noise
    # Trimmed = noise (USV was removed)
    # Skip = exclude from training
    status_map = {
        'Clean': 'noise',
        'Trimmed': 'noise',
        'Skip': 'skip'
    }

    df['final_label'] = df['status'].map(status_map)
    df['source'] = 'noise_samples'

    # Rename reviewed_at to labeled_at for consistency
    if 'reviewed_at' in df.columns:
        df = df.rename(columns={'reviewed_at': 'labeled_at'})

    return df[['candidate_id', 'final_label', 'labeled_at', 'source']]


def combine_labels(usv_labels: pd.DataFrame, noise_labels: pd.DataFrame) -> pd.DataFrame:
    """Combine USV and noise labels into single dataset."""
    combined = pd.concat([usv_labels, noise_labels], ignore_index=True)
    return combined


def filter_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to keep only clean labels for training.

    Keeps:
    - USV
    - noise

    Removes:
    - uncertain
    - skip
    """
    valid_labels = ['USV', 'noise']
    filtered = df[df['final_label'].isin(valid_labels)].copy()

    return filtered


def generate_summary(combined: pd.DataFrame, filtered: pd.DataFrame) -> dict:
    """Generate summary statistics."""
    summary = {
        'total_labeled': len(combined),
        'usv_candidates_labeled': len(combined[combined['source'] == 'candidates']),
        'noise_samples_labeled': len(combined[combined['source'] == 'noise_samples']),
        'excluded_uncertain': len(combined[combined['final_label'] == 'uncertain']),
        'excluded_skip': len(combined[combined['final_label'] == 'skip']),
        'final_training_samples': len(filtered),
        'final_usv_count': len(filtered[filtered['final_label'] == 'USV']),
        'final_noise_count': len(filtered[filtered['final_label'] == 'noise']),
    }

    return summary


def main():
    print("=" * 80)
    print("PREPARING LABELED DATASET FOR TRAINING")
    print("=" * 80)
    print()

    # Load labels
    labels_path = Path('labels.csv')
    noise_reviews_path = Path('noise_samples/noise_reviews.csv')

    if not labels_path.exists():
        print(f"Error: {labels_path} not found!")
        return

    if not noise_reviews_path.exists():
        print(f"Error: {noise_reviews_path} not found!")
        return

    print("Loading USV candidate labels...")
    usv_labels = load_usv_labels(labels_path)
    print(f"  Loaded {len(usv_labels)} USV candidate labels")
    print(f"  Distribution: {usv_labels['final_label'].value_counts().to_dict()}")
    print()

    print("Loading noise sample reviews...")
    noise_labels = load_noise_reviews(noise_reviews_path)
    print(f"  Loaded {len(noise_labels)} noise sample reviews")
    print(f"  Distribution: {noise_labels['final_label'].value_counts().to_dict()}")
    print()

    # Combine
    print("Combining labels...")
    combined = combine_labels(usv_labels, noise_labels)
    print(f"  Total combined: {len(combined)}")
    print()

    # Filter for training
    print("Filtering for training (excluding uncertain/skip)...")
    filtered = filter_for_training(combined)
    print(f"  Training samples: {len(filtered)}")
    print()

    # Generate summary
    summary = generate_summary(combined, filtered)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total labeled samples: {summary['total_labeled']}")
    print(f"  - USV candidates: {summary['usv_candidates_labeled']}")
    print(f"  - Noise samples: {summary['noise_samples_labeled']}")
    print()
    print(f"Excluded from training:")
    print(f"  - Uncertain: {summary['excluded_uncertain']}")
    print(f"  - Skip: {summary['excluded_skip']}")
    print()
    print(f"Final training dataset: {summary['final_training_samples']}")
    print(f"  - USV: {summary['final_usv_count']} ({100*summary['final_usv_count']/summary['final_training_samples']:.1f}%)")
    print(f"  - Noise: {summary['final_noise_count']} ({100*summary['final_noise_count']/summary['final_training_samples']:.1f}%)")
    print()

    # Save filtered dataset
    output_path = Path('labels_combined.csv')
    filtered.to_csv(output_path, index=False)
    print(f"Saved combined labels to: {output_path}")
    print()

    # Create a backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f'labels_combined_backup_{timestamp}.csv')
    filtered.to_csv(backup_path, index=False)
    print(f"Backup saved to: {backup_path}")
    print()

    # Save summary
    summary_path = Path('analysis/labeling_summary.txt')
    summary_path.parent.mkdir(exist_ok=True, parents=True)

    with open(summary_path, 'w') as f:
        f.write("LABELING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"Summary saved to: {summary_path}")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review labels_combined.csv to verify correctness")
    print("2. Create train/val/test splits (stratified by recording)")
    print("3. Extract training-mode spectrograms (without axes)")
    print("4. Train CNN classifier")
    print()
    print(f"Dataset ready with {summary['final_training_samples']} samples!")
    print()


if __name__ == '__main__':
    main()
