"""Evaluate trained CNN model on test set.

Usage:
    python scripts/evaluate_model.py --model models/production/best_model.pt --test-csv splits/test.csv
"""

import argparse
import sys
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


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained CNN model on test set'
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
        default=Path('splits/test.csv'),
        help='Path to test split CSV'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )
    parser.add_argument(
        '--normalize-mode',
        type=str,
        default='per_image',
        choices=['per_image', 'global'],
        help='Normalization mode (default: per_image)'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to save plots (default: same as model dir)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save evaluation plots'
    )
    parser.add_argument(
        '--save-predictions',
        type=str,
        default=None,
        help='Save predictions to CSV file (for error analysis)'
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

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.model.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model, checkpoint = load_model_checkpoint(
        args.model,
        USVClassifierCNN,
        device=args.device
    )
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    print()

    # Load test data
    print("Loading test data...")
    test_dataset = USVDataset(args.test_csv, normalize_mode=args.normalize_mode)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate_fn
    )
    print(f"Test samples: {len(test_dataset)}")
    print(f"Class distribution: {test_dataset.class_counts}")
    print()

    # Evaluate
    print("Evaluating model...")
    metrics, y_pred, y_proba, y_true = evaluate_model(
        model,
        test_loader,
        device=args.device,
        verbose=True
    )
    print()

    # Save plots
    if args.save_plots:
        print("Generating plots...")

        # Confusion matrix
        cm_path = args.output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(y_true, y_pred, output_path=cm_path)

        # ROC curve
        roc_path = args.output_dir / 'roc_curve.png'
        plot_roc_curve(y_true, y_proba, output_path=roc_path)

        # Precision-recall curve
        pr_path = args.output_dir / 'precision_recall_curve.png'
        plot_precision_recall_curve(y_true, y_proba, output_path=pr_path)

        print()

    # Save metrics
    import json
    metrics_path = args.output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Save predictions if requested
    if args.save_predictions:
        import pandas as pd
        # Create inverse label map
        label_map_inv = {v: k for k, v in test_dataset.label_map.items()}

        predictions_df = pd.DataFrame({
            'candidate_id': [test_dataset.df.iloc[i]['candidate_id'] for i in range(len(test_dataset))],
            'source_file': [test_dataset.df.iloc[i]['source_file'] for i in range(len(test_dataset))],
            'true_label': [label_map_inv[y] for y in y_true],
            'predicted_label': [label_map_inv[y] for y in y_pred],
            'confidence': y_proba,
        })
        pred_path = Path(args.save_predictions)
        predictions_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to: {pred_path}")

    print()
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
