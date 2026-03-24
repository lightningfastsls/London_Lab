"""
Analyze model predictions to understand test set performance.

Usage:
  1. First generate predictions on test set:
     python scripts/evaluate_model.py --model models/best_model.pt --test-csv splits/test.csv --save-predictions test_predictions.csv

  2. Then run this script:
     python scripts/analyze_predictions.py --predictions test_predictions.csv --test-csv splits/test.csv
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np


def analyze_confidence_by_correctness(df):
    """Analyze model confidence for correct vs incorrect predictions."""
    print("=" * 80)
    print("MODEL CONFIDENCE ANALYSIS")
    print("=" * 80)

    # Separate correct and incorrect
    correct = df[df['predicted_label'] == df['true_label']]
    incorrect = df[df['predicted_label'] != df['true_label']]

    print(f"\nCorrectly classified: {len(correct)} / {len(df)} ({len(correct)/len(df)*100:.1f}%)")
    print(f"Incorrectly classified: {len(incorrect)} / {len(df)} ({len(incorrect)/len(df)*100:.1f}%)")

    # Confidence statistics
    print("\nConfidence (probability) for CORRECT predictions:")
    print(f"  Mean: {correct['confidence'].mean():.3f}")
    print(f"  Median: {correct['confidence'].median():.3f}")
    print(f"  Std: {correct['confidence'].std():.3f}")
    print(f"  Min: {correct['confidence'].min():.3f}")
    print(f"  Max: {correct['confidence'].max():.3f}")

    if len(incorrect) > 0:
        print("\nConfidence (probability) for INCORRECT predictions:")
        print(f"  Mean: {incorrect['confidence'].mean():.3f}")
        print(f"  Median: {incorrect['confidence'].median():.3f}")
        print(f"  Std: {incorrect['confidence'].std():.3f}")
        print(f"  Min: {incorrect['confidence'].min():.3f}")
        print(f"  Max: {incorrect['confidence'].max():.3f}")


def analyze_errors_by_class(df):
    """Analyze which class has more errors."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS BY CLASS")
    print("=" * 80)

    # True Positives, False Positives, True Negatives, False Negatives
    tp = len(df[(df['true_label'] == 'USV') & (df['predicted_label'] == 'USV')])
    fp = len(df[(df['true_label'] == 'Not USV') & (df['predicted_label'] == 'USV')])
    tn = len(df[(df['true_label'] == 'Not USV') & (df['predicted_label'] == 'Not USV')])
    fn = len(df[(df['true_label'] == 'USV') & (df['predicted_label'] == 'Not USV')])

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted USV    Predicted Not USV")
    print(f"True USV              {tp:>4}              {fn:>4}")
    print(f"True Not USV          {fp:>4}              {tn:>4}")

    # Metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}  (of predicted USV, how many are correct)")
    print(f"  Recall:    {recall:.3f}  (of true USV, how many detected)")
    print(f"  F1 Score:  {f1:.3f}")

    # Error breakdown
    print(f"\nError Breakdown:")
    print(f"  False Negatives (missed USVs): {fn} / {tp + fn} = {fn/(tp+fn)*100 if (tp+fn)>0 else 0:.1f}%")
    print(f"  False Positives (false alarms): {fp} / {tn + fp} = {fp/(tn+fp)*100 if (tn+fp)>0 else 0:.1f}%")

    if fn > fp:
        print(f"\n  [INSIGHT] Model is CONSERVATIVE - misses more USVs than false alarms")
        print(f"            This explains low recall ({recall:.1%})")
    elif fp > fn:
        print(f"\n  [INSIGHT] Model is AGGRESSIVE - more false alarms than missed USVs")
    else:
        print(f"\n  [INSIGHT] Errors are balanced between both classes")


def analyze_errors_by_recording(df):
    """Analyze which recordings have the most errors."""
    print("\n" + "=" * 80)
    print("ERRORS BY RECORDING")
    print("=" * 80)

    recording_stats = []
    for recording in sorted(df['source_file'].unique()):
        rec_df = df[df['source_file'] == recording]

        correct = (rec_df['predicted_label'] == rec_df['true_label']).sum()
        total = len(rec_df)
        accuracy = correct / total if total > 0 else 0

        recording_stats.append({
            'recording': recording,
            'total': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': accuracy,
        })

    stats_df = pd.DataFrame(recording_stats).sort_values('accuracy')

    print("\nWorst performing recordings:")
    print("-" * 80)
    print(f"{'Recording':<45} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 80)

    for _, row in stats_df.head(10).iterrows():
        print(f"{row['recording']:<45} {row['correct']:>8} {row['total']:>8} {row['accuracy']:>9.1%}")

    print("\nBest performing recordings:")
    print("-" * 80)
    for _, row in stats_df.tail(5).iterrows():
        print(f"{row['recording']:<45} {row['correct']:>8} {row['total']:>8} {row['accuracy']:>9.1%}")


def analyze_low_confidence_errors(df, threshold=0.6):
    """Find samples where model made wrong predictions with high confidence."""
    print("\n" + "=" * 80)
    print(f"HIGH CONFIDENCE ERRORS (confidence > {threshold})")
    print("=" * 80)

    # Incorrect predictions with high confidence
    high_conf_errors = df[
        (df['predicted_label'] != df['true_label']) &
        (df['confidence'] > threshold)
    ].sort_values('confidence', ascending=False)

    if len(high_conf_errors) == 0:
        print(f"\n[GOOD] No high-confidence errors found (threshold={threshold})")
        return

    print(f"\nFound {len(high_conf_errors)} high-confidence errors:")
    print("-" * 80)
    print(f"{'Candidate ID':<50} {'True':>10} {'Pred':>10} {'Conf':>8}")
    print("-" * 80)

    for _, row in high_conf_errors.head(20).iterrows():
        print(f"{row['candidate_id']:<50} {row['true_label']:>10} {row['predicted_label']:>10} {row['confidence']:>8.3f}")

    print(f"\nThese are concerning - model is very wrong with high confidence")
    print(f"Review these samples visually to understand what the model is learning")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model predictions to understand test set performance"
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV (from evaluate_model.py --save-predictions)'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default='splits/test.csv',
        help='Path to test split CSV'
    )
    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")

    try:
        pred_df = pd.read_csv(args.predictions)
    except FileNotFoundError:
        print(f"\n[ERROR] Predictions file not found: {args.predictions}")
        print("\nTo generate predictions, run:")
        print(f"  python scripts/evaluate_model.py --model models/best_model.pt --test-csv {args.test_csv} --save-predictions {args.predictions}")
        sys.exit(1)

    # Check required columns
    required_cols = ['candidate_id', 'source_file', 'true_label', 'predicted_label', 'confidence']
    missing_cols = [col for col in required_cols if col not in pred_df.columns]

    if missing_cols:
        print(f"\n[ERROR] Missing columns in predictions file: {missing_cols}")
        print(f"Found columns: {list(pred_df.columns)}")
        sys.exit(1)

    print(f"Loaded {len(pred_df)} predictions\n")

    # Run analyses
    analyze_confidence_by_correctness(pred_df)
    analyze_errors_by_class(pred_df)
    analyze_errors_by_recording(pred_df)
    analyze_low_confidence_errors(pred_df, threshold=0.6)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nRECOMMENDATIONS:")
    print("1. Review high-confidence error samples visually")
    print("2. Check if worst-performing recordings have unique characteristics")
    print("3. Consider per-recording analysis during training (recording as feature)")
    print("4. If false negatives >> false positives: lower classification threshold")
    print("5. If overfitting: add regularization, data augmentation, or early stopping")


if __name__ == '__main__':
    main()
