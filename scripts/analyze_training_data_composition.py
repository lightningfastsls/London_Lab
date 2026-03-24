"""Analyze training data composition for multi-syllable USV bias.

Investigates:
1. Duration distribution of USV labels
2. Potential multi-syllable patterns
3. Train/val/test split comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_labels_with_splits():
    """Load labels and merge with split and candidate information."""
    labels_df = pd.read_csv('labels.csv')
    candidates_df = pd.read_csv('candidates.csv')

    # Merge labels with candidate metadata (for duration, etc.)
    data = labels_df.merge(candidates_df, on='candidate_id', how='left')

    # Load split files
    train_df = pd.read_csv('splits/train.csv')[['candidate_id']]
    val_df = pd.read_csv('splits/val.csv')[['candidate_id']]
    test_df = pd.read_csv('splits/test.csv')[['candidate_id']]

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Merge
    splits_df = pd.concat([train_df, val_df, test_df])
    data_with_splits = data.merge(splits_df, on='candidate_id', how='left')

    return data_with_splits


def analyze_duration_distribution(df):
    """Analyze USV duration distribution."""
    print("=" * 80)
    print("USV DURATION DISTRIBUTION")
    print("=" * 80)

    # Filter USVs only
    usvs = df[df['label'] == 'USV'].copy()

    # Check if duration_ms exists, otherwise calculate it
    if 'duration_ms' not in usvs.columns:
        print("Warning: duration_ms not found, calculating from start_ms and end_ms")
        usvs['duration_ms'] = usvs['end_ms'] - usvs['start_ms']

    print(f"\nTotal USV labels: {len(usvs)}")
    print(f"\nDuration statistics (ms):")
    print(usvs['duration_ms'].describe())

    # Categorize by duration
    usvs['duration_category'] = pd.cut(
        usvs['duration_ms'],
        bins=[0, 50, 100, 150, 200, 300, 1000],
        labels=['Very Short (0-50ms)', 'Short (50-100ms)', 'Medium (100-150ms)',
                'Long (150-200ms)', 'Very Long (200-300ms)', 'Extremely Long (300ms+)']
    )

    print("\nDuration categories:")
    print(usvs['duration_category'].value_counts().sort_index())

    # By split
    if 'split' in usvs.columns:
        print("\n" + "=" * 80)
        print("DURATION BY SPLIT")
        print("=" * 80)
        for split in ['train', 'val', 'test']:
            split_usvs = usvs[usvs['split'] == split]
            if len(split_usvs) > 0:
                print(f"\n{split.upper()} set ({len(split_usvs)} USVs):")
                print(f"  Mean: {split_usvs['duration_ms'].mean():.1f} ms")
                print(f"  Median: {split_usvs['duration_ms'].median():.1f} ms")
                print(f"  Range: [{split_usvs['duration_ms'].min():.1f}, {split_usvs['duration_ms'].max():.1f}] ms")
                print(f"  > 100ms: {(split_usvs['duration_ms'] > 100).sum()} ({(split_usvs['duration_ms'] > 100).mean()*100:.1f}%)")
                print(f"  > 150ms: {(split_usvs['duration_ms'] > 150).sum()} ({(split_usvs['duration_ms'] > 150).mean()*100:.1f}%)")

    return usvs


def identify_potential_multisyllable(usvs, threshold_ms=120):
    """Identify potential multi-syllable USVs based on duration."""
    print("\n" + "=" * 80)
    print(f"POTENTIAL MULTI-SYLLABLE USVs (duration > {threshold_ms}ms)")
    print("=" * 80)

    long_usvs = usvs[usvs['duration_ms'] > threshold_ms].copy()
    long_usvs = long_usvs.sort_values('duration_ms', ascending=False)

    print(f"\nFound {len(long_usvs)} USVs longer than {threshold_ms}ms")

    if 'split' in long_usvs.columns:
        print("\nBy split:")
        for split in ['train', 'val', 'test']:
            count = (long_usvs['split'] == split).sum()
            total = (usvs['split'] == split).sum()
            pct = count / total * 100 if total > 0 else 0
            print(f"  {split}: {count}/{total} ({pct:.1f}%)")

    print(f"\nTop 10 longest USVs:")
    print(long_usvs[['candidate_id', 'source_file', 'duration_ms', 'split']].head(10).to_string(index=False))

    return long_usvs


def analyze_recording_diversity(df):
    """Analyze diversity of recordings in each split."""
    print("\n" + "=" * 80)
    print("RECORDING DIVERSITY")
    print("=" * 80)

    usvs = df[df['label'] == 'USV'].copy()

    if 'split' not in usvs.columns or 'source_file' not in usvs.columns:
        print("Missing split or source_file information")
        return

    for split in ['train', 'val', 'test']:
        split_data = usvs[usvs['split'] == split]
        if len(split_data) == 0:
            continue

        recordings = split_data['source_file'].unique()
        samples_per_recording = split_data.groupby('source_file').size()

        print(f"\n{split.upper()} set:")
        print(f"  Total USVs: {len(split_data)}")
        print(f"  Unique recordings: {len(recordings)}")
        print(f"  USVs per recording: {samples_per_recording.mean():.1f} ± {samples_per_recording.std():.1f}")
        print(f"  Range: [{samples_per_recording.min()}, {samples_per_recording.max()}]")


def create_visualizations(usvs, output_dir='analysis'):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall duration distribution
    ax = axes[0, 0]
    ax.hist(usvs['duration_ms'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(usvs['duration_ms'].median(), color='red', linestyle='--',
               label=f'Median: {usvs["duration_ms"].median():.1f}ms')
    ax.axvline(100, color='orange', linestyle='--', alpha=0.5, label='100ms (potential multi-syllable)')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Count')
    ax.set_title('USV Duration Distribution (All Data)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Duration by split
    ax = axes[0, 1]
    if 'split' in usvs.columns:
        for split in ['train', 'val', 'test']:
            split_data = usvs[usvs['split'] == split]['duration_ms']
            if len(split_data) > 0:
                ax.hist(split_data, bins=30, alpha=0.5, label=f'{split} (n={len(split_data)})', edgecolor='black')
        ax.axvline(100, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('Count')
        ax.set_title('USV Duration by Split')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Box plot by split
    ax = axes[1, 0]
    if 'split' in usvs.columns:
        split_data = [usvs[usvs['split'] == s]['duration_ms'].values
                      for s in ['train', 'val', 'test']]
        ax.boxplot(split_data, tick_labels=['Train', 'Val', 'Test'], patch_artist=True)
        ax.axhline(100, color='red', linestyle='--', alpha=0.3, label='100ms threshold')
        ax.set_ylabel('Duration (ms)')
        ax.set_title('USV Duration Distribution by Split')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative distribution
    ax = axes[1, 1]
    sorted_durations = np.sort(usvs['duration_ms'].values)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    ax.plot(sorted_durations, cumulative, linewidth=2)
    ax.axvline(100, color='red', linestyle='--', alpha=0.5, label='100ms')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Duration Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add percentile annotations
    p50 = np.percentile(sorted_durations, 50)
    p90 = np.percentile(sorted_durations, 90)
    ax.text(p50, 0.5, f' 50th: {p50:.0f}ms', va='bottom')
    ax.text(p90, 0.9, f' 90th: {p90:.0f}ms', va='bottom')

    plt.tight_layout()

    output_path = output_dir / 'training_data_duration_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("TRAINING DATA COMPOSITION ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    print("Loading labels and split information...")
    df = load_labels_with_splits()
    print(f"Loaded {len(df)} total labels")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print()

    # Analyze duration
    usvs = analyze_duration_distribution(df)

    # Identify potential multi-syllable
    long_usvs = identify_potential_multisyllable(usvs, threshold_ms=120)

    # Analyze recording diversity
    analyze_recording_diversity(df)

    # Create visualizations
    create_visualizations(usvs)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    train_long = long_usvs[long_usvs['split'] == 'train']
    test_long = long_usvs[long_usvs['split'] == 'test']

    train_total = len(usvs[usvs['split'] == 'train'])
    test_total = len(usvs[usvs['split'] == 'test'])

    train_long_pct = len(train_long) / train_total * 100 if train_total > 0 else 0
    test_long_pct = len(test_long) / test_total * 100 if test_total > 0 else 0

    print(f"\nLong USVs (>120ms) in training: {len(train_long)}/{train_total} ({train_long_pct:.1f}%)")
    print(f"Long USVs (>120ms) in test: {len(test_long)}/{test_total} ({test_long_pct:.1f}%)")

    if abs(train_long_pct - test_long_pct) > 5:
        print("\nPOTENTIAL BIAS DETECTED:")
        print(f"   Test set has {'more' if test_long_pct > train_long_pct else 'fewer'} long-duration USVs than training")
        print("   This could explain poor performance on multi-syllable calls")

    median_dur = usvs['duration_ms'].median()
    mean_dur = usvs['duration_ms'].mean()

    print(f"\nTypical USV duration: {median_dur:.1f}ms (median), {mean_dur:.1f}ms (mean)")

    if median_dur < 80:
        print("-> Training data appears to be dominated by SHORT single-syllable USVs")
        print("-> Model may struggle with longer, complex multi-syllable patterns")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
