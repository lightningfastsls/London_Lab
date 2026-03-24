"""Test continuity parameter tuning to distinguish energy dips from syllable gaps.

Goal: Find parameters where:
1. Single USV with energy dips -> 1 detection (keep whole USV)
2. Multi-syllable calls -> multiple detections (split syllables)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.detection.config import DetectionConfig
from usv_spectrogram.detection.energy_detector import EnergyDetector


# Test cases
TEST_CASES = [
    {
        'name': 'Single USV with dips',
        'description': 'Should stay as 1 detection (complete USV)',
        'file': '2024-09-30_11-18-17_0000001.wav',
        'time_ms': 777,
        'expected_count': 1,
        'tolerance_ms': 100
    },
    {
        'name': 'Multi-syllable call (446ms)',
        'description': 'Should split into 3-4 detections (separate syllables)',
        'file': '2024-09-30_11-20-56_0000033.wav',
        'time_ms': 1033,
        'expected_count': 4,  # Based on previous test showing 4 syllables
        'tolerance_ms': 500
    },
]


def test_config(config: DetectionConfig, test_case: dict, wav_dir: Path) -> dict:
    """Test a configuration on a specific test case."""
    wav_path = wav_dir / test_case['file']

    if not wav_path.exists():
        return {
            'error': f"File not found: {wav_path}",
            'count': 0,
            'candidates': []
        }

    detector = EnergyDetector(config)
    all_candidates = detector.detect(wav_path)

    # Find candidates near the expected time
    time_ms = test_case['time_ms']
    tolerance_ms = test_case['tolerance_ms']

    near_candidates = [
        c for c in all_candidates
        if abs(c.start_ms - time_ms) < tolerance_ms or
           (c.start_ms <= time_ms <= c.end_ms)
    ]

    return {
        'count': len(near_candidates),
        'candidates': near_candidates,
        'total_candidates': len(all_candidates)
    }


def score_config(results: list[dict], test_cases: list[dict]) -> dict:
    """Score a configuration based on how well it matches expected behavior."""
    score = 0
    max_score = len(test_cases)
    details = []

    for result, test_case in zip(results, test_cases):
        expected = test_case['expected_count']
        actual = result['count']

        # Perfect match gets 1 point
        if actual == expected:
            points = 1.0
            status = 'PERFECT'
        # Off by 1 gets 0.5 points
        elif abs(actual - expected) == 1:
            points = 0.5
            status = 'CLOSE'
        # Wrong direction gets 0 points
        else:
            points = 0.0
            if (actual > expected and 'Single' in test_case['name']) or \
               (actual < expected and 'Multi-syllable' in test_case['name']):
                status = 'WRONG DIRECTION'
            else:
                status = 'MISS'

        score += points
        details.append({
            'test_case': test_case['name'],
            'expected': expected,
            'actual': actual,
            'points': points,
            'status': status
        })

    return {
        'score': score,
        'max_score': max_score,
        'percentage': 100 * score / max_score if max_score > 0 else 0,
        'details': details
    }


def test_parameter_grid():
    """Test grid of continuity parameters."""
    print("=" * 80)
    print("CONTINUITY PARAMETER TUNING")
    print("Testing configurations to distinguish energy dips from syllable gaps")
    print("=" * 80)
    print()

    wav_dir = Path('5970 USV')

    # Display test cases
    print("Test Cases:")
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"  {i}. {tc['name']}")
        print(f"     File: {tc['file']}")
        print(f"     Time: {tc['time_ms']}ms")
        print(f"     Expected: {tc['expected_count']} detection(s)")
        print(f"     Goal: {tc['description']}")
        print()

    # Parameter grid
    param_grid = {
        'max_gap_ms': [5.0, 8.0, 10.0, 15.0, 20.0],
        'freq_tolerance_hz': [1500.0, 2000.0, 3000.0],
        'energy_tolerance_db': [10.0, 15.0, 20.0],
        'merge_gap_ms': [3.0, 5.0, 10.0]
    }

    print(f"Parameter Grid:")
    print(f"  segment_continuity_max_gap_ms: {param_grid['max_gap_ms']}")
    print(f"  segment_continuity_freq_tolerance_hz: {param_grid['freq_tolerance_hz']}")
    print(f"  segment_continuity_energy_tolerance_db: {param_grid['energy_tolerance_db']}")
    print(f"  merge_gap_ms: {param_grid['merge_gap_ms']}")

    total_configs = (len(param_grid['max_gap_ms']) *
                    len(param_grid['freq_tolerance_hz']) *
                    len(param_grid['energy_tolerance_db']) *
                    len(param_grid['merge_gap_ms']))
    print(f"\nTotal configurations to test: {total_configs}")
    print()

    # Test all combinations
    results_list = []

    for i, (max_gap, freq_tol, energy_tol, merge_gap) in enumerate(
        product(param_grid['max_gap_ms'],
                param_grid['freq_tolerance_hz'],
                param_grid['energy_tolerance_db'],
                param_grid['merge_gap_ms']),
        1
    ):
        if i % 10 == 0:
            print(f"Progress: {i}/{total_configs} configurations tested...")

        config = DetectionConfig(
            segment_continuity_enabled=True,
            segment_continuity_max_gap_ms=max_gap,
            segment_continuity_freq_tolerance_hz=freq_tol,
            segment_continuity_energy_tolerance_db=energy_tol,
            merge_gap_ms=merge_gap
        )

        # Test on both cases
        test_results = [test_config(config, tc, wav_dir) for tc in TEST_CASES]

        # Score the configuration
        score_info = score_config(test_results, TEST_CASES)

        results_list.append({
            'max_gap_ms': max_gap,
            'freq_tolerance_hz': freq_tol,
            'energy_tolerance_db': energy_tol,
            'merge_gap_ms': merge_gap,
            'score': score_info['score'],
            'percentage': score_info['percentage'],
            'single_usv_count': test_results[0]['count'],
            'multi_syllable_count': test_results[1]['count'],
            'details': score_info['details']
        })

    print(f"Progress: {total_configs}/{total_configs} configurations tested!")
    print()

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Sort by score
    results_df = results_df.sort_values('score', ascending=False)

    # Save all results
    output_path = Path('analysis/continuity_tuning_results.csv')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_path, index=False)
    print(f"Full results saved to: {output_path}")
    print()

    # Display top configurations
    print("=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    print()

    top_configs = results_df.head(10)

    for i, row in top_configs.iterrows():
        print(f"Rank #{len(top_configs) - list(top_configs.index).index(i)}:")
        print(f"  Score: {row['score']:.1f}/2.0 ({row['percentage']:.0f}%)")
        print(f"  Parameters:")
        print(f"    max_gap_ms = {row['max_gap_ms']:.1f}")
        print(f"    freq_tolerance_hz = {row['freq_tolerance_hz']:.0f}")
        print(f"    energy_tolerance_db = {row['energy_tolerance_db']:.0f}")
        print(f"    merge_gap_ms = {row['merge_gap_ms']:.1f}")
        print(f"  Results:")
        print(f"    Single USV with dips: {row['single_usv_count']} detection(s) (want 1)")
        print(f"    Multi-syllable call: {row['multi_syllable_count']} detection(s) (want 4)")
        print()

    # Find best perfect score
    perfect_scores = results_df[results_df['score'] == 2.0]

    if len(perfect_scores) > 0:
        print("=" * 80)
        print("PERFECT CONFIGURATIONS (Score 2.0/2.0)")
        print("=" * 80)
        print()
        print(f"Found {len(perfect_scores)} configuration(s) with perfect score!")
        print()

        # Get the most conservative (smallest gaps) among perfect scores
        best = perfect_scores.nsmallest(1, ['max_gap_ms', 'merge_gap_ms']).iloc[0]

        print("RECOMMENDED CONFIGURATION (most conservative among perfect scores):")
        print(f"  max_gap_ms = {best['max_gap_ms']:.1f}")
        print(f"  freq_tolerance_hz = {best['freq_tolerance_hz']:.0f}")
        print(f"  energy_tolerance_db = {best['energy_tolerance_db']:.0f}")
        print(f"  merge_gap_ms = {best['merge_gap_ms']:.1f}")
        print()
    else:
        print("=" * 80)
        print("NO PERFECT CONFIGURATIONS FOUND")
        print("=" * 80)
        print()
        print("Best configuration:")
        best = results_df.iloc[0]
        print(f"  Score: {best['score']:.1f}/2.0 ({best['percentage']:.0f}%)")
        print(f"  max_gap_ms = {best['max_gap_ms']:.1f}")
        print(f"  freq_tolerance_hz = {best['freq_tolerance_hz']:.0f}")
        print(f"  energy_tolerance_db = {best['energy_tolerance_db']:.0f}")
        print(f"  merge_gap_ms = {best['merge_gap_ms']:.1f}")
        print()
        print("Note: May need manual review or additional test cases")
        print()

    return results_df


def detailed_comparison():
    """Detailed comparison of current vs recommended configuration."""
    print("=" * 80)
    print("DETAILED COMPARISON: CURRENT VS TUNED")
    print("=" * 80)
    print()

    wav_dir = Path('5970 USV')

    # Current config
    config_current = DetectionConfig()

    # Tuned config (using hypothesis from analysis)
    config_tuned = DetectionConfig(
        segment_continuity_enabled=True,
        segment_continuity_max_gap_ms=8.0,
        segment_continuity_freq_tolerance_hz=2000.0,
        segment_continuity_energy_tolerance_db=15.0,
        merge_gap_ms=3.0
    )

    configs = [
        ('CURRENT', config_current),
        ('TUNED', config_tuned)
    ]

    for test_case in TEST_CASES:
        print(f"Test Case: {test_case['name']}")
        print(f"File: {test_case['file']}, Time: {test_case['time_ms']}ms")
        print(f"Expected: {test_case['expected_count']} detection(s)")
        print()

        for config_name, config in configs:
            result = test_config(config, test_case, wav_dir)

            print(f"{config_name} Configuration:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Detections found: {result['count']}")
                for j, c in enumerate(result['candidates'], 1):
                    print(f"    {j}. {c.start_ms:.1f}-{c.end_ms:.1f}ms "
                          f"(duration={c.duration_ms:.1f}ms, peak={c.peak_freq_hz:.0f}Hz)")
            print()

        print("-" * 80)
        print()


def main():
    # Run detailed comparison first (quick check)
    detailed_comparison()

    # Run full parameter grid search
    results_df = test_parameter_grid()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
