"""Test different merge gap settings on known multi-syllable USV examples.

This script validates the proposed strategy for splitting multi-syllable USVs
by comparing detection results with different merge_gap_ms and
segment_continuity_enabled settings.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.config import DetectionConfig
from usv_spectrogram.detection.energy_detector import EnergyDetector


def test_config_on_file(wav_path: Path, config: DetectionConfig, config_name: str) -> dict:
    """Run detection with a specific config and return summary statistics."""
    detector = EnergyDetector(config)
    candidates = detector.detect(wav_path)

    if not candidates:
        return {
            'config': config_name,
            'count': 0,
            'mean_duration': 0,
            'median_duration': 0,
            'max_duration': 0,
            'min_duration': 0,
            'candidates': []
        }

    durations = [c.duration_ms for c in candidates]

    return {
        'config': config_name,
        'count': len(candidates),
        'mean_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'max_duration': np.max(durations),
        'min_duration': np.min(durations),
        'candidates': candidates
    }


def compare_configs_on_long_usvs():
    """Compare different merge gap settings on the 29 longest USVs."""
    print("=" * 80)
    print("MERGE GAP SENSITIVITY ANALYSIS")
    print("Testing on known multi-syllable USV examples")
    print("=" * 80)
    print()

    # Load the 29 longest USVs
    labels_df = pd.read_csv('labels.csv')
    candidates_df = pd.read_csv('candidates.csv')
    data = labels_df.merge(candidates_df, on='candidate_id', how='left')
    usvs = data[data['label'] == 'USV'].copy()

    # Get longest USVs (> 120ms)
    long_usvs = usvs[usvs['duration_ms'] > 120].copy()
    long_usvs = long_usvs.sort_values('duration_ms', ascending=False)

    print(f"Found {len(long_usvs)} USVs > 120ms to test on")
    print()

    # Get unique source files from long USVs
    test_files = long_usvs['source_file'].unique()
    print(f"Test files: {len(test_files)} unique recordings")

    # Define test configurations
    configs = [
        ('Current (merge=10ms, continuity=ON)', DetectionConfig(
            merge_gap_ms=10.0,
            segment_continuity_enabled=True
        )),
        ('Proposed A (merge=5ms, continuity=OFF)', DetectionConfig(
            merge_gap_ms=5.0,
            segment_continuity_enabled=False
        )),
        ('Proposed B (merge=2ms, continuity=OFF)', DetectionConfig(
            merge_gap_ms=2.0,
            segment_continuity_enabled=False
        )),
        ('Proposed C (merge=0ms, continuity=OFF)', DetectionConfig(
            merge_gap_ms=0.0,
            segment_continuity_enabled=False
        )),
        ('Continuity only (merge=0ms, continuity=ON)', DetectionConfig(
            merge_gap_ms=0.0,
            segment_continuity_enabled=True
        )),
    ]

    # Test on a sample of long-USV files (take first 5 to save time)
    wav_dir = Path('5970 USV')
    results_by_config = {name: [] for name, _ in configs}

    print("\nTesting configurations on sample files...")
    print()

    for i, source_file in enumerate(test_files[:5], 1):
        wav_path = wav_dir / source_file
        if not wav_path.exists():
            print(f"Warning: {wav_path.name} not found, skipping")
            continue

        print(f"[{i}/5] Testing {wav_path.name}...")

        for config_name, config in configs:
            result = test_config_on_file(wav_path, config, config_name)
            results_by_config[config_name].append(result)
            print(f"  {config_name}: {result['count']} candidates, "
                  f"max duration={result['max_duration']:.1f}ms")

    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    # Aggregate results
    summary_rows = []
    for config_name, _ in configs:
        results = results_by_config[config_name]
        if not results:
            continue

        total_candidates = sum(r['count'] for r in results)
        all_durations = []
        for r in results:
            all_durations.extend([c.duration_ms for c in r['candidates']])

        if all_durations:
            summary_rows.append({
                'Configuration': config_name,
                'Total Candidates': total_candidates,
                'Mean Duration (ms)': f"{np.mean(all_durations):.1f}",
                'Median Duration (ms)': f"{np.median(all_durations):.1f}",
                'Max Duration (ms)': f"{np.max(all_durations):.1f}",
                '% > 120ms': f"{100 * np.mean(np.array(all_durations) > 120):.1f}%",
                '% > 80ms': f"{100 * np.mean(np.array(all_durations) > 80):.1f}%",
            })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()

    # Save results
    output_path = Path('analysis/merge_gap_test_results.csv')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find config with lowest % > 120ms
    best_config = summary_df.loc[summary_df['% > 120ms'].str.rstrip('%').astype(float).idxmin()]

    print("Best configuration for eliminating multi-syllable USVs:")
    print(f"  {best_config['Configuration']}")
    print(f"  -> {best_config['% > 120ms']} of candidates > 120ms")
    print(f"  -> {best_config['Total Candidates']} total candidates (vs {summary_rows[0]['Total Candidates']} current)")
    print()

    # Check for over-splitting
    current_total = int(summary_rows[0]['Total Candidates'])
    best_total = int(best_config['Total Candidates'])
    increase_pct = 100 * (best_total - current_total) / current_total

    if increase_pct > 50:
        print("WARNING: Candidate count increased by >50% - may be over-splitting")
    elif increase_pct < 0:
        print("WARNING: Candidate count DECREASED - unexpected, check config")
    else:
        print(f"✓ Candidate count increase: {increase_pct:.1f}% (acceptable)")

    print()

    return summary_df


def analyze_specific_example():
    """Detailed analysis of a specific known multi-syllable example."""
    print("=" * 80)
    print("DETAILED EXAMPLE ANALYSIS")
    print("=" * 80)
    print()

    # Use the longest USV as example: 2024-09-30_11-20-56_0000033_00001033 (449ms)
    example_file = '2024-09-30_11-20-56_0000033.wav'
    example_time_ms = 1033  # Start time from candidate_id

    wav_path = Path('5970 USV') / example_file
    if not wav_path.exists():
        print(f"Example file not found: {wav_path}")
        return

    print(f"Example: {example_file}")
    print(f"Expected multi-syllable USV around {example_time_ms}ms")
    print()

    # Current config
    config_current = DetectionConfig()
    detector_current = EnergyDetector(config_current)
    candidates_current = detector_current.detect(wav_path)

    # Proposed config
    config_split = DetectionConfig(
        merge_gap_ms=5.0,
        segment_continuity_enabled=False
    )
    detector_split = EnergyDetector(config_split)
    candidates_split = detector_split.detect(wav_path)

    print(f"Current config: {len(candidates_current)} candidates")
    print(f"Split config:   {len(candidates_split)} candidates")
    print()

    # Find candidates near the example time
    tolerance_ms = 500

    print(f"Candidates near {example_time_ms}ms (±{tolerance_ms}ms):")
    print()

    print("CURRENT CONFIG:")
    near_current = [c for c in candidates_current
                   if abs(c.start_ms - example_time_ms) < tolerance_ms]
    for c in near_current:
        print(f"  {c.start_ms:.1f}-{c.end_ms:.1f}ms (duration={c.duration_ms:.1f}ms, "
              f"peak={c.peak_freq_hz:.0f}Hz)")

    print()
    print("SPLIT CONFIG:")
    near_split = [c for c in candidates_split
                 if abs(c.start_ms - example_time_ms) < tolerance_ms]
    for c in near_split:
        print(f"  {c.start_ms:.1f}-{c.end_ms:.1f}ms (duration={c.duration_ms:.1f}ms, "
              f"peak={c.peak_freq_hz:.0f}Hz)")

    print()

    if len(near_split) > len(near_current):
        print(f"✓ Split config detected {len(near_split)} segments vs {len(near_current)} in current config")
        print("  -> Multi-syllable call successfully split!")
    elif len(near_split) == len(near_current):
        print("⚠ Same number of segments detected - may need more aggressive splitting")
    else:
        print("✗ Fewer segments in split config - unexpected!")

    print()


def main():
    # Run comparison analysis
    summary_df = compare_configs_on_long_usvs()

    print()

    # Run detailed example
    analyze_specific_example()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
