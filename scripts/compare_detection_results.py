"""Compare old vs new detection results to identify changes from continuity tuning.

Creates a comprehensive comparison report showing:
1. Overall statistics (before/after)
2. Which long USVs were split (and into how many pieces)
3. Recordings most affected by the changes
4. Specific examples to review in labeling tool
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_and_compare_candidates():
    """Load old and new candidates and create comparison."""
    print("=" * 80)
    print("DETECTION RESULTS COMPARISON: Old vs New (Tuned)")
    print("=" * 80)
    print()

    # Load candidates
    old_df = pd.read_csv('candidates.csv')
    new_df = pd.read_csv('candidates_v2.csv')

    print("Overall Statistics:")
    print(f"  Old candidates: {len(old_df)}")
    print(f"  New candidates: {len(new_df)}")
    print(f"  Change: {len(new_df) - len(old_df):+d} ({100*(len(new_df)-len(old_df))/len(old_df):+.1f}%)")
    print()

    # Duration statistics
    print("Duration Statistics:")
    print(f"  Old: mean={old_df['duration_ms'].mean():.1f}ms, median={old_df['duration_ms'].median():.1f}ms")
    print(f"  New: mean={new_df['duration_ms'].mean():.1f}ms, median={new_df['duration_ms'].median():.1f}ms")
    print()

    # Long USVs
    old_long = len(old_df[old_df['duration_ms'] > 120])
    new_long = len(new_df[new_df['duration_ms'] > 120])
    print("Long USVs (>120ms):")
    print(f"  Old: {old_long} ({100*old_long/len(old_df):.1f}%)")
    print(f"  New: {new_long} ({100*new_long/len(new_df):.1f}%)")
    print(f"  Reduction: {old_long - new_long} long USVs split")
    print()

    # Very long USVs
    old_very_long = len(old_df[old_df['duration_ms'] > 200])
    new_very_long = len(new_df[new_df['duration_ms'] > 200])
    print("Very Long USVs (>200ms):")
    print(f"  Old: {old_very_long}")
    print(f"  New: {new_very_long}")
    print()

    return old_df, new_df


def find_split_candidates(old_df, new_df):
    """Identify which old candidates were split into multiple new candidates."""
    print("=" * 80)
    print("SPLIT CANDIDATE ANALYSIS")
    print("=" * 80)
    print()

    # Focus on the 29 longest old candidates
    old_long = old_df.nsmallest(30, 'duration_ms', keep='all')
    old_long = old_long[old_long['duration_ms'] > 120].copy()
    old_long = old_long.sort_values('duration_ms', ascending=False)

    print(f"Analyzing {len(old_long)} old candidates > 120ms")
    print()

    splits = []

    for _, old_row in old_long.iterrows():
        source_file = old_row['source_file']
        old_start = old_row['start_ms']
        old_end = old_row['end_ms']
        old_duration = old_row['duration_ms']

        # Find new candidates that overlap with this old candidate
        # Allow 50ms tolerance on either side
        tolerance_ms = 50
        new_overlapping = new_df[
            (new_df['source_file'] == source_file) &
            (new_df['start_ms'] < old_end + tolerance_ms) &
            (new_df['end_ms'] > old_start - tolerance_ms)
        ].copy()

        new_overlapping = new_overlapping.sort_values('start_ms')

        if len(new_overlapping) > 1:
            # This candidate was split!
            splits.append({
                'source_file': source_file,
                'old_candidate_id': old_row['candidate_id'],
                'old_start_ms': old_start,
                'old_end_ms': old_end,
                'old_duration_ms': old_duration,
                'old_peak_freq_hz': old_row['peak_freq_hz'],
                'num_splits': len(new_overlapping),
                'new_candidate_ids': new_overlapping['candidate_id'].tolist(),
                'new_durations': new_overlapping['duration_ms'].tolist(),
                'new_peak_freqs': new_overlapping['peak_freq_hz'].tolist()
            })

    print(f"Found {len(splits)} candidates that were split")
    print()

    if splits:
        print("Top 10 Largest Splits:")
        print()

        splits_df = pd.DataFrame(splits)
        splits_df = splits_df.sort_values('old_duration_ms', ascending=False)

        for i, row in splits_df.head(10).iterrows():
            print(f"{i+1}. {row['source_file']}")
            print(f"   OLD: {row['old_candidate_id']}")
            print(f"        {row['old_start_ms']:.1f}-{row['old_end_ms']:.1f}ms "
                  f"(duration={row['old_duration_ms']:.1f}ms, peak={row['old_peak_freq_hz']:.0f}Hz)")
            print(f"   NEW: Split into {row['num_splits']} candidates:")
            for j, (cid, dur, freq) in enumerate(zip(row['new_candidate_ids'],
                                                       row['new_durations'],
                                                       row['new_peak_freqs']), 1):
                print(f"        {j}. {cid}: {dur:.1f}ms at {freq:.0f}Hz")
            print()

        # Save to CSV for easy review
        output_path = Path('analysis/split_candidates.csv')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        splits_df.to_csv(output_path, index=False)
        print(f"Split candidates saved to: {output_path}")
        print()

    return splits


def compare_by_recording(old_df, new_df):
    """Compare candidate counts by recording."""
    print("=" * 80)
    print("CHANGES BY RECORDING")
    print("=" * 80)
    print()

    old_counts = old_df.groupby('source_file').size().to_dict()
    new_counts = new_df.groupby('source_file').size().to_dict()

    all_files = set(old_counts.keys()) | set(new_counts.keys())

    changes = []
    for file in all_files:
        old_count = old_counts.get(file, 0)
        new_count = new_counts.get(file, 0)
        change = new_count - old_count
        pct_change = 100 * change / old_count if old_count > 0 else 0

        changes.append({
            'source_file': file,
            'old_count': old_count,
            'new_count': new_count,
            'change': change,
            'pct_change': pct_change
        })

    changes_df = pd.DataFrame(changes)
    changes_df = changes_df.sort_values('change', ascending=False)

    # Show recordings with biggest increases (likely had multi-syllable splits)
    print("Recordings with Most New Candidates (likely had multi-syllable splits):")
    print()
    top_increases = changes_df[changes_df['change'] > 0].head(10)
    for _, row in top_increases.iterrows():
        print(f"  {row['source_file']}: {row['old_count']} -> {row['new_count']} "
              f"({row['change']:+d}, {row['pct_change']:+.0f}%)")
    print()

    # Show recordings with decreases (unexpected - should investigate)
    decreases = changes_df[changes_df['change'] < 0]
    if len(decreases) > 0:
        print("Recordings with FEWER Candidates (unexpected - should review):")
        print()
        for _, row in decreases.iterrows():
            print(f"  {row['source_file']}: {row['old_count']} -> {row['new_count']} "
                  f"({row['change']:+d}, {row['pct_change']:+.0f}%)")
        print()

    return changes_df


def create_review_list(splits):
    """Create a prioritized list of candidates to review in labeling tool."""
    print("=" * 80)
    print("REVIEW CHECKLIST")
    print("=" * 80)
    print()

    if not splits:
        print("No splits detected - nothing to review!")
        return

    splits_df = pd.DataFrame(splits)

    print("Recommended samples to review in labeling tool:")
    print()

    # Review the biggest splits first
    print("1. BIGGEST SPLITS (verify multi-syllable -> single syllables):")
    print()
    for i, row in splits_df.nsmallest(5, 'old_duration_ms').iterrows():
        print(f"   Recording: {row['source_file']}")
        print(f"   Old candidate ID: {row['old_candidate_id']} ({row['old_duration_ms']:.0f}ms)")
        print(f"   New candidate IDs: {', '.join(row['new_candidate_ids'])} "
              f"({row['num_splits']} pieces)")
        print(f"   -> Open labeling tool and search for these IDs")
        print()

    # Create a CSV list for easy filtering in labeling tool
    review_list = []
    for _, row in splits_df.iterrows():
        for cid in row['new_candidate_ids']:
            review_list.append({
                'candidate_id': cid,
                'source_file': row['source_file'],
                'review_reason': f'Split from {row["old_duration_ms"]:.0f}ms candidate',
                'priority': 'high' if row['old_duration_ms'] > 200 else 'medium'
            })

    review_df = pd.DataFrame(review_list)
    output_path = Path('analysis/review_list.csv')
    review_df.to_csv(output_path, index=False)
    print(f"Full review list saved to: {output_path}")
    print(f"  Total candidates to review: {len(review_df)}")
    print()
    print("To review in labeling tool:")
    print("  1. Load candidates_v2.csv")
    print("  2. Filter by candidate IDs from review_list.csv")
    print("  3. Verify each split looks like a complete single USV (not partial)")
    print()


def create_visualization(old_df, new_df):
    """Create before/after visualization."""
    print("=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Duration histograms
    ax = axes[0, 0]
    ax.hist(old_df['duration_ms'], bins=50, alpha=0.6, label='Old', edgecolor='black')
    ax.hist(new_df['duration_ms'], bins=50, alpha=0.6, label='New (Tuned)', edgecolor='black')
    ax.axvline(120, color='red', linestyle='--', alpha=0.5, label='120ms threshold')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Duration Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoom on long USVs
    ax = axes[0, 1]
    old_long = old_df[old_df['duration_ms'] > 100]['duration_ms']
    new_long = new_df[new_df['duration_ms'] > 100]['duration_ms']
    ax.hist(old_long, bins=30, alpha=0.6, label=f'Old (n={len(old_long)})', edgecolor='black')
    ax.hist(new_long, bins=30, alpha=0.6, label=f'New (n={len(new_long)})', edgecolor='black')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Long USVs (>100ms) - Zoomed View')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative distribution
    ax = axes[1, 0]
    old_sorted = np.sort(old_df['duration_ms'].values)
    new_sorted = np.sort(new_df['duration_ms'].values)
    old_cumulative = np.arange(1, len(old_sorted) + 1) / len(old_sorted)
    new_cumulative = np.arange(1, len(new_sorted) + 1) / len(new_sorted)
    ax.plot(old_sorted, old_cumulative, linewidth=2, label='Old')
    ax.plot(new_sorted, new_cumulative, linewidth=2, label='New (Tuned)')
    ax.axvline(120, color='red', linestyle='--', alpha=0.5, label='120ms')
    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Duration Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot comparison
    ax = axes[1, 1]
    ax.boxplot([old_df['duration_ms'], new_df['duration_ms']],
               labels=['Old', 'New (Tuned)'],
               patch_artist=True)
    ax.axhline(120, color='red', linestyle='--', alpha=0.3, label='120ms')
    ax.set_ylabel('Duration (ms)')
    ax.set_title('Duration Distribution Box Plot')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = Path('analysis/detection_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    print()
    plt.close()


def main():
    # Load and compare
    old_df, new_df = load_and_compare_candidates()

    # Find split candidates
    splits = find_split_candidates(old_df, new_df)

    # Compare by recording
    changes_df = compare_by_recording(old_df, new_df)

    # Create review list
    create_review_list(splits)

    # Create visualization
    create_visualization(old_df, new_df)

    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print()
    print("Files created:")
    print("  - analysis/split_candidates.csv")
    print("  - analysis/review_list.csv")
    print("  - analysis/detection_comparison.png")
    print()
    print("Next steps:")
    print("  1. Review the visualization: analysis/detection_comparison.png")
    print("  2. Check split_candidates.csv to see which long USVs were split")
    print("  3. Use review_list.csv to prioritize manual review in labeling tool")
    print()


if __name__ == '__main__':
    main()
