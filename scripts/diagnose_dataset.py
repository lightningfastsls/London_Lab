"""
Diagnostic script for investigating train/val/test performance discrepancies.

Checks for:
1. Data leakage between splits
2. Statistical differences in spectrograms across splits
3. Recording-level distribution patterns
4. Model confidence analysis (if predictions available)
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_splits():
    """Load train/val/test splits."""
    splits_dir = Path(__file__).parent.parent / "splits"

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    return train_df, val_df, test_df


def check_data_leakage(train_df, val_df, test_df):
    """Check if any recordings appear in multiple splits."""
    print("=" * 80)
    print("1. DATA LEAKAGE CHECK")
    print("=" * 80)

    train_recordings = set(train_df['source_file'].unique())
    val_recordings = set(val_df['source_file'].unique())
    test_recordings = set(test_df['source_file'].unique())

    train_val_overlap = train_recordings & val_recordings
    train_test_overlap = train_recordings & test_recordings
    val_test_overlap = val_recordings & test_recordings

    if train_val_overlap:
        print(f"[LEAKAGE] Train/Val overlap: {len(train_val_overlap)} recordings")
        print(f"  {sorted(train_val_overlap)[:5]}...")
    else:
        print("[OK] No Train/Val overlap")

    if train_test_overlap:
        print(f"[LEAKAGE] Train/Test overlap: {len(train_test_overlap)} recordings")
        print(f"  {sorted(train_test_overlap)[:5]}...")
    else:
        print("[OK] No Train/Test overlap")

    if val_test_overlap:
        print(f"[LEAKAGE] Val/Test overlap: {len(val_test_overlap)} recordings")
        print(f"  {sorted(val_test_overlap)[:5]}...")
    else:
        print("[OK] No Val/Test overlap")

    # Print all unique recordings and their splits
    print(f"\nTotal unique recordings:")
    print(f"  Train: {len(train_recordings)}")
    print(f"  Val: {len(val_recordings)}")
    print(f"  Test: {len(test_recordings)}")

    all_recordings = train_recordings | val_recordings | test_recordings
    print(f"  All: {len(all_recordings)}")

    return len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap) == 0


def compute_spectrogram_stats(df, split_name, sample_size=None):
    """Compute statistics for spectrograms in a split."""
    print(f"\nAnalyzing {split_name} split ({len(df)} samples)...")

    # Sample if dataset is large
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"  (sampling {sample_size} images for speed)")
    else:
        df_sample = df

    means = []
    stds = []
    shapes = []

    for idx, row in df_sample.iterrows():
        img_path = Path(row['spectrogram_path'])
        if not img_path.exists():
            continue

        # Load image and convert to grayscale numpy array
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0

        means.append(img_array.mean())
        stds.append(img_array.std())
        shapes.append(img_array.shape)

    if not means:
        print("  [ERROR] No valid images found!")
        return None

    return {
        'mean_of_means': np.mean(means),
        'mean_of_stds': np.mean(stds),
        'std_of_means': np.std(means),
        'std_of_stds': np.std(stds),
        'min_shape': min(shapes),
        'max_shape': max(shapes),
        'unique_shapes': len(set(shapes)),
    }


def compare_spectrogram_statistics(train_df, val_df, test_df, sample_size=200):
    """Compare spectrogram statistics across splits."""
    print("\n" + "=" * 80)
    print("2. SPECTROGRAM STATISTICS COMPARISON")
    print("=" * 80)

    train_stats = compute_spectrogram_stats(train_df, "Train", sample_size)
    val_stats = compute_spectrogram_stats(val_df, "Val", sample_size)
    test_stats = compute_spectrogram_stats(test_df, "Test", sample_size)

    if not all([train_stats, val_stats, test_stats]):
        print("[ERROR] Could not compute stats for all splits")
        return

    print("\n" + "-" * 80)
    print(f"{'Metric':<30} {'Train':>15} {'Val':>15} {'Test':>15}")
    print("-" * 80)

    metrics = [
        ('Mean of pixel means', 'mean_of_means'),
        ('Std of pixel means', 'std_of_means'),
        ('Mean of pixel stds', 'mean_of_stds'),
        ('Std of pixel stds', 'std_of_stds'),
    ]

    for label, key in metrics:
        train_val = train_stats[key]
        val_val = val_stats[key]
        test_val = test_stats[key]
        print(f"{label:<30} {train_val:>15.4f} {val_val:>15.4f} {test_val:>15.4f}")

    print("-" * 80)
    print(f"{'Min shape (H, W)':<30} {str(train_stats['min_shape']):>15} "
          f"{str(val_stats['min_shape']):>15} {str(test_stats['min_shape']):>15}")
    print(f"{'Max shape (H, W)':<30} {str(train_stats['max_shape']):>15} "
          f"{str(val_stats['max_shape']):>15} {str(test_stats['max_shape']):>15}")
    print(f"{'Unique shapes':<30} {train_stats['unique_shapes']:>15} "
          f"{val_stats['unique_shapes']:>15} {test_stats['unique_shapes']:>15}")
    print("-" * 80)

    # Check for significant differences
    mean_diff_train_test = abs(train_stats['mean_of_means'] - test_stats['mean_of_means'])
    mean_diff_val_test = abs(val_stats['mean_of_means'] - test_stats['mean_of_means'])

    print("\nDISTRIBUTION SHIFT ANALYSIS:")
    if mean_diff_train_test > 0.1:
        print(f"  [WARNING] Large mean difference between Train and Test: {mean_diff_train_test:.4f}")
    if mean_diff_val_test > 0.1:
        print(f"  [WARNING] Large mean difference between Val and Test: {mean_diff_val_test:.4f}")

    if mean_diff_train_test < 0.05 and mean_diff_val_test < 0.05:
        print("  [OK] Train/Val/Test have similar distributions")


def analyze_recording_distribution(train_df, val_df, test_df):
    """Analyze recording-level distribution patterns."""
    print("\n" + "=" * 80)
    print("3. RECORDING-LEVEL DISTRIBUTION")
    print("=" * 80)

    # Combine all data
    all_df = pd.concat([train_df, val_df, test_df])

    # Group by recording
    recording_stats = []
    for recording in sorted(all_df['source_file'].unique()):
        rec_df = all_df[all_df['source_file'] == recording]

        split = rec_df['split'].iloc[0]
        n_samples = len(rec_df)

        # Class distribution
        class_counts = rec_df['label'].value_counts().to_dict()
        n_usv = class_counts.get('USV', 0)
        n_not_usv = class_counts.get('Not USV', 0)

        # Sample some spectrograms for mean/std
        sample_df = rec_df.sample(n=min(10, len(rec_df)), random_state=42)
        pixel_means = []
        pixel_stds = []

        for idx, row in sample_df.iterrows():
            img_path = Path(row['spectrogram_path'])
            if img_path.exists():
                img = Image.open(img_path).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                pixel_means.append(img_array.mean())
                pixel_stds.append(img_array.std())

        avg_mean = np.mean(pixel_means) if pixel_means else 0
        avg_std = np.mean(pixel_stds) if pixel_stds else 0

        recording_stats.append({
            'recording': recording,
            'split': split,
            'n_samples': n_samples,
            'n_usv': n_usv,
            'n_not_usv': n_not_usv,
            'usv_ratio': n_usv / n_samples if n_samples > 0 else 0,
            'avg_pixel_mean': avg_mean,
            'avg_pixel_std': avg_std,
        })

    stats_df = pd.DataFrame(recording_stats)

    # Print summary by split
    print("\nSamples per recording by split:")
    print(stats_df.groupby('split')['n_samples'].describe())

    print("\nUSV ratio per recording by split:")
    print(stats_df.groupby('split')['usv_ratio'].describe())

    print("\nAverage pixel mean by split:")
    print(stats_df.groupby('split')['avg_pixel_mean'].describe())

    # Find outlier recordings
    print("\n" + "-" * 80)
    print("POTENTIAL OUTLIER RECORDINGS:")
    print("-" * 80)

    # Recordings with very high or low pixel means
    overall_mean = stats_df['avg_pixel_mean'].mean()
    overall_std = stats_df['avg_pixel_mean'].std()

    outliers = stats_df[
        (stats_df['avg_pixel_mean'] < overall_mean - 2 * overall_std) |
        (stats_df['avg_pixel_mean'] > overall_mean + 2 * overall_std)
    ]

    if len(outliers) > 0:
        print(f"\nRecordings with unusual pixel intensity (>2 std from mean):")
        for _, row in outliers.iterrows():
            print(f"  {row['recording']:<40} [{row['split']:>5}]  "
                  f"pixel_mean={row['avg_pixel_mean']:.3f}  samples={row['n_samples']}")
    else:
        print("  [OK] No recordings with unusual pixel intensity")

    # Recordings with extreme class imbalance
    extreme_imbalance = stats_df[
        (stats_df['usv_ratio'] < 0.1) | (stats_df['usv_ratio'] > 0.9)
    ]

    if len(extreme_imbalance) > 0:
        print(f"\nRecordings with extreme class imbalance (<10% or >90% USV):")
        for _, row in extreme_imbalance.iterrows():
            print(f"  {row['recording']:<40} [{row['split']:>5}]  "
                  f"USV ratio={row['usv_ratio']:.1%}  ({row['n_usv']}/{row['n_samples']})")

    # Test set recordings
    print("\n" + "-" * 80)
    print("TEST SET RECORDINGS (detailed):")
    print("-" * 80)
    test_recs = stats_df[stats_df['split'] == 'test'].sort_values('recording')
    for _, row in test_recs.iterrows():
        print(f"{row['recording']:<40}  samples={row['n_samples']:>3}  "
              f"USV={row['n_usv']:>3}  NotUSV={row['n_not_usv']:>3}  "
              f"ratio={row['usv_ratio']:.1%}  "
              f"pixel_mean={row['avg_pixel_mean']:.3f}")


def analyze_class_distribution(train_df, val_df, test_df):
    """Analyze class distribution across splits."""
    print("\n" + "=" * 80)
    print("4. CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        counts = df['label'].value_counts()
        total = len(df)
        print(f"\n{name}:")
        for label, count in counts.items():
            print(f"  {label}: {count} ({count/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose dataset issues causing test set performance problems"
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=200,
        help='Number of images to sample per split for statistics (default: 200)'
    )
    args = parser.parse_args()

    print("\nUSV DATASET DIAGNOSTIC TOOL")
    print("Investigating train/val/test performance discrepancy\n")

    # Load splits
    print("Loading splits...")
    train_df, val_df, test_df = load_splits()
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Run diagnostics
    check_data_leakage(train_df, val_df, test_df)
    compare_spectrogram_statistics(train_df, val_df, test_df, args.sample_size)
    analyze_recording_distribution(train_df, val_df, test_df)
    analyze_class_distribution(train_df, val_df, test_df)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
