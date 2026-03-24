"""Threshold sweep analysis for CNN classifier.

Evaluates model performance at different classification thresholds to find
optimal threshold for maximizing F1 score.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.evaluate import load_model_checkpoint, evaluate_model
from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
from torch.utils.data import DataLoader


def compute_metrics_at_threshold(y_true, y_proba, threshold):
    """Compute classification metrics at given threshold.

    Args:
        y_true: True labels (numpy array)
        y_proba: Predicted probabilities (numpy array)
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    accuracy = (y_pred == y_true).mean()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def run_threshold_sweep(
    model_path,
    data_csv,
    thresholds=None,
    batch_size=16,
    device=None,
    output_dir=None
):
    """Run threshold sweep analysis.

    Args:
        model_path: Path to trained model checkpoint
        data_csv: Path to data split CSV (test.csv or val.csv)
        thresholds: List of thresholds to test
        batch_size: Batch size for evaluation
        device: Device to run on
        output_dir: Directory to save results

    Returns:
        DataFrame with results for all thresholds
    """
    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    if output_dir is None:
        output_dir = Path('analysis')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("THRESHOLD SWEEP ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Data: {data_csv}")
    print(f"Device: {device}")
    print(f"Thresholds: {thresholds}")
    print()

    # Load model
    print("Loading model...")
    model, checkpoint = load_model_checkpoint(
        Path(model_path),
        USVClassifierCNN,
        device=device
    )
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print()

    # Create data loader
    print("Loading dataset...")
    dataset = USVDataset(data_csv)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=pad_collate_fn
    )
    print(f"Dataset: {len(dataset)} samples")
    print(f"Class distribution: {dataset.class_counts}")
    print()

    # Run evaluation once to get probabilities
    print("Running model evaluation...")
    metrics_default, y_pred_default, y_proba, y_true = evaluate_model(
        model,
        data_loader,
        device=device,
        verbose=False
    )
    print(f"Evaluation complete")
    print(f"Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"Probability mean: {y_proba.mean():.4f}")
    print(f"Probability std: {y_proba.std():.4f}")
    print()

    # Compute metrics at each threshold
    print("Computing metrics at each threshold...")
    results = []
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_proba, threshold)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Find optimal thresholds
    best_f1_idx = results_df['f1'].idxmax()
    best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
    best_f1_score = results_df.loc[best_f1_idx, 'f1']

    # Find threshold for recall >= 0.85
    high_recall_rows = results_df[results_df['recall'] >= 0.85]
    if len(high_recall_rows) > 0:
        # Pick the one with highest precision
        high_recall_idx = high_recall_rows['precision'].idxmax()
        high_recall_threshold = results_df.loc[high_recall_idx, 'threshold']
        high_recall_precision = results_df.loc[high_recall_idx, 'precision']
        high_recall_recall = results_df.loc[high_recall_idx, 'recall']
        high_recall_f1 = results_df.loc[high_recall_idx, 'f1']
    else:
        high_recall_threshold = None
        high_recall_precision = None
        high_recall_recall = None
        high_recall_f1 = None

    # Print results table
    print()
    print("=" * 80)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("=" * 80)
    print()
    print(f"OPTIMAL THRESHOLD (max F1): {best_f1_threshold:.2f}")
    print(f"  F1 Score: {best_f1_score:.4f}")
    print(f"  Accuracy: {results_df.loc[best_f1_idx, 'accuracy']:.4f}")
    print(f"  Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
    print(f"  Recall: {results_df.loc[best_f1_idx, 'recall']:.4f}")
    print()

    if high_recall_threshold is not None:
        print(f"HIGH RECALL THRESHOLD (recall >= 0.85): {high_recall_threshold:.2f}")
        print(f"  Recall: {high_recall_recall:.4f}")
        print(f"  Precision: {high_recall_precision:.4f}")
        print(f"  F1 Score: {high_recall_f1:.4f}")
        print()
    else:
        print("No threshold achieved recall >= 0.85")
        print()

    # Save results to CSV
    csv_path = output_dir / 'threshold_sweep_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All metrics vs threshold
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(results_df['threshold'], results_df['precision'], 's-', label='Precision', linewidth=2)
    ax.plot(results_df['threshold'], results_df['recall'], '^-', label='Recall', linewidth=2)
    ax.plot(results_df['threshold'], results_df['f1'], 'd-', label='F1', linewidth=2)
    ax.axvline(best_f1_threshold, color='red', linestyle='--', alpha=0.5, label=f'Optimal ({best_f1_threshold:.2f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 2: F1 Score vs Threshold
    ax = axes[0, 1]
    ax.plot(results_df['threshold'], results_df['f1'], 'o-', color='green', linewidth=2)
    ax.axvline(best_f1_threshold, color='red', linestyle='--', alpha=0.5, label=f'Max F1 at {best_f1_threshold:.2f}')
    ax.axhline(best_f1_score, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 3: Precision-Recall tradeoff
    ax = axes[1, 0]
    ax.plot(results_df['recall'], results_df['precision'], 'o-', linewidth=2)
    for i, row in results_df.iterrows():
        ax.annotate(f"{row['threshold']:.2f}",
                   (row['recall'], row['precision']),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    # Plot 4: Confusion matrix counts
    ax = axes[1, 1]
    best_row = results_df.loc[best_f1_idx]
    ax.bar(['TP', 'FP', 'FN', 'TN'],
           [best_row['tp'], best_row['fp'], best_row['fn'], best_row['tn']],
           color=['green', 'orange', 'red', 'blue'],
           alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Confusion Matrix at Optimal Threshold ({best_f1_threshold:.2f})')
    ax.grid(True, alpha=0.3, axis='y')

    # Add counts on bars
    for i, (label, count) in enumerate(zip(['TP', 'FP', 'FN', 'TN'],
                                           [best_row['tp'], best_row['fp'], best_row['fn'], best_row['tn']])):
        ax.text(i, count, str(int(count)), ha='center', va='bottom')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'threshold_sweep_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()

    print()
    print("=" * 80)

    return results_df, best_f1_threshold, best_f1_score


def main():
    parser = argparse.ArgumentParser(description='Threshold sweep analysis')
    parser.add_argument('--model', type=str, default='models/clean_test/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='splits/test.csv',
                       help='Path to data CSV (test.csv or val.csv)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Output directory for results')

    args = parser.parse_args()

    run_threshold_sweep(
        model_path=args.model,
        data_csv=args.data,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
