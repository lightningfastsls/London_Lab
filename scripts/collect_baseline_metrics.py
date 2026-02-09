"""Collect baseline metrics for current model before implementing jittering.

This script runs evaluation on the current production model and saves
structured metrics to JSON for later comparison after retraining.

Usage:
    python scripts/collect_baseline_metrics.py --model models/production/best_model.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models import USVClassifierCNN, USVDataset
from usv_spectrogram.models.data_loader import pad_collate_fn
from usv_spectrogram.models.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    load_model_checkpoint
)
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    parser = argparse.ArgumentParser(
        description='Collect baseline metrics from current model'
    )

    # Model
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )

    # Data
    parser.add_argument(
        '--test-csv',
        type=Path,
        default=Path('spectrograms_training/test.csv'),
        help='Path to test split CSV (default: spectrograms_training/test.csv)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/baseline_metrics'),
        help='Directory to save metrics (default: analysis/baseline_metrics)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu, default: auto)'
    )

    args = parser.parse_args()

    # Verify files exist
    if not args.model.exists():
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)
    if not args.test_csv.exists():
        print(f"Error: Test CSV not found: {args.test_csv}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.model}")
    model, checkpoint = load_model_checkpoint(args.model, USVClassifierCNN, device)
    model.eval()

    # Load test dataset
    print(f"Loading test dataset from: {args.test_csv}")
    test_dataset = USVDataset(
        csv_path=args.test_csv,
        normalize_mode='per_image'  # Match training default
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0
    )

    print(f"Test set size: {len(test_dataset)} samples")

    # Evaluate model
    print("\nRunning evaluation...")
    metrics, y_pred, y_proba, y_true = evaluate_model(model, test_loader, device)

    # Compute AUC scores
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    # Print metrics
    print("\n" + "="*60)
    print("BASELINE METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print("="*60)

    # Save structured metrics to JSON
    baseline_data = {
        "model_path": str(args.model),
        "evaluation_date": datetime.now().isoformat(),
        "test_set": str(args.test_csv),
        "test_set_size": len(test_dataset),
        "metrics": {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1_score": float(metrics['f1']),
            "specificity": float(metrics['specificity']),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr)
        },
        "confusion_matrix": {
            "true_positives": int(metrics['true_positives']),
            "false_positives": int(metrics['false_positives']),
            "true_negatives": int(metrics['true_negatives']),
            "false_negatives": int(metrics['false_negatives'])
        },
        "notes": "Baseline before constrained jittering implementation (Phase 0)"
    }

    metrics_file = args.output_dir / "baseline_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\nMetrics saved to: {metrics_file}")

    # Generate and save plots
    print("\nGenerating evaluation plots...")

    # Confusion matrix
    cm_path = args.output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true,
        y_pred,
        output_path=cm_path
    )
    print(f"  Confusion matrix: {cm_path}")

    # ROC curve
    roc_path = args.output_dir / "roc_curve.png"
    plot_roc_curve(
        y_true,
        y_proba,
        output_path=roc_path
    )
    print(f"  ROC curve: {roc_path}")

    # Precision-Recall curve
    pr_path = args.output_dir / "precision_recall_curve.png"
    plot_precision_recall_curve(
        y_true,
        y_proba,
        output_path=pr_path
    )
    print(f"  PR curve: {pr_path}")

    # Save predictions for error analysis
    predictions_file = args.output_dir / "predictions.csv"
    import pandas as pd
    import numpy as np

    predictions_df = pd.DataFrame({
        'true_label': np.array(y_true),
        'predicted_score': np.array(y_proba),
        'predicted_label': np.array(y_pred)
    })
    predictions_df.to_csv(predictions_file, index=False)
    print(f"  Predictions: {predictions_file}")

    print("\nBaseline collection complete!")
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
