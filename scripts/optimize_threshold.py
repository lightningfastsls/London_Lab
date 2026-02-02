"""Find optimal classification threshold for the retrained CNN.

Tests multiple thresholds and finds:
1. Best F1 score (balanced precision/recall)
2. High-recall threshold (achieves target recall with best precision)

CRITICAL: Matches data_loader.py preprocessing exactly (per-image normalization).

Usage:
    python scripts/optimize_threshold.py \\
        --model models/full_retrained_cnn/best_model.pt \\
        --test-csv data/full_training_dataset/test.csv \\
        --output-dir analysis/threshold_optimization \\
        --target-recall 0.90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN


def load_model(model_path: Path, device: str = 'auto') -> tuple:
    """Load trained model.

    Returns:
        model, device
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Loading model from {model_path}...")
    print(f"Device: {device}")

    model = USVClassifierCNN()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device


def load_and_predict(model: torch.nn.Module, device: torch.device, spec_path: Path) -> float:
    """Load spectrogram and predict probability.

    CRITICAL: Matches data_loader.py preprocessing exactly (lines 76-87):
    - Convert to grayscale ('L' mode)
    - Per-image normalize to [0, 1]
    """
    # Load image and convert to grayscale (match data_loader.py line 76)
    img = Image.open(spec_path).convert('L')
    spectrogram = np.array(img, dtype=np.float32)

    # Per-image normalization (match data_loader.py lines 82-87)
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    if spec_max > spec_min:
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        spectrogram = np.zeros_like(spectrogram)

    # Convert to tensor (1, 1, H, W)
    x = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        prob = model.predict_proba(x).item()

    return prob


def get_predictions(
    model: torch.nn.Module,
    device: torch.device,
    test_csv: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions for all test samples.

    Returns:
        probabilities, labels (both as numpy arrays)
    """
    df = pd.read_csv(test_csv)

    # Ensure 'spectrogram_path' column exists
    if 'spectrogram_path' not in df.columns:
        raise ValueError("Test CSV missing 'spectrogram_path' column")

    # Resolve paths (may be relative or just filenames)
    repo_root = Path(__file__).parent.parent

    probabilities = []
    labels = []

    print(f"Predicting on {len(df)} test samples...")

    for idx, row in df.iterrows():
        spec_path_str = row['spectrogram_path']

        # Try to resolve path
        spec_path = Path(spec_path_str)
        if not spec_path.is_absolute():
            # Try relative to repo root
            spec_path = repo_root / spec_path_str

        if not spec_path.exists():
            print(f"  Warning: Spectrogram not found: {spec_path}")
            continue

        # Predict
        try:
            prob = load_and_predict(model, device, spec_path)
            probabilities.append(prob)

            # Parse label
            label = 1 if row['label'] == 'USV' else 0
            labels.append(label)

        except Exception as e:
            print(f"  Error predicting {spec_path}: {e}")
            continue

    return np.array(probabilities), np.array(labels)


def compute_metrics_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Compute classification metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / len(labels)

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def optimize_threshold(
    model_path: Path,
    test_csv: Path,
    output_dir: Path,
    target_recall: float = 0.90,
    device: str = 'auto',
) -> None:
    """Find optimal classification threshold."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, device = load_model(model_path, device)

    # Get predictions
    probs, labels = get_predictions(model, device, test_csv)

    print(f"\nPredictions obtained:")
    print(f"  Total samples: {len(probs)}")
    print(f"  USV: {labels.sum()}")
    print(f"  Not USV: {(1 - labels).sum()}")
    print(f"  Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

    # Test multiple thresholds
    print(f"\nTesting thresholds from 0.05 to 0.95...")
    thresholds = np.arange(0.05, 0.96, 0.05)
    results = []

    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(probs, labels, thresh)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Find optimal thresholds
    best_f1_idx = results_df['f1'].idxmax()
    best_f1_thresh = results_df.loc[best_f1_idx, 'threshold']

    # Find threshold that achieves target recall with best precision
    high_recall = results_df[results_df['recall'] >= target_recall]
    if len(high_recall) > 0:
        best_recall_idx = high_recall['precision'].idxmax()
        best_recall_thresh = high_recall.loc[best_recall_idx, 'threshold']
    else:
        # No threshold achieves target recall
        best_recall_idx = results_df['recall'].idxmax()
        best_recall_thresh = results_df.loc[best_recall_idx, 'threshold']
        print(f"  Warning: No threshold achieves {target_recall:.0%} recall")
        print(f"  Using threshold with max recall: {results_df.loc[best_recall_idx, 'recall']:.3f}")

    # Plot results
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Metrics vs Threshold
    ax = axes[0]
    ax.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision', linewidth=2)
    ax.plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall', linewidth=2)
    ax.plot(results_df['threshold'], results_df['f1'], 'r-', label='F1', linewidth=2)
    ax.axvline(x=best_f1_thresh, color='r', linestyle='--', alpha=0.5, label=f'Best F1 ({best_f1_thresh:.2f})')
    ax.axvline(x=best_recall_thresh, color='g', linestyle='--', alpha=0.5, label=f'High Recall ({best_recall_thresh:.2f})')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 2: Precision-Recall Curve
    ax = axes[1]
    ax.plot(results_df['recall'], results_df['precision'], 'b-', linewidth=2)
    ax.scatter(
        [results_df.loc[best_f1_idx, 'recall']],
        [results_df.loc[best_f1_idx, 'precision']],
        color='red', s=150, zorder=5, marker='*',
        label=f'Best F1 (thresh={best_f1_thresh:.2f})'
    )
    if best_recall_idx != best_f1_idx:
        ax.scatter(
            [results_df.loc[best_recall_idx, 'recall']],
            [results_df.loc[best_recall_idx, 'precision']],
            color='green', s=150, zorder=5, marker='o',
            label=f'High Recall (thresh={best_recall_thresh:.2f})'
        )
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = output_dir / 'threshold_optimization.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {plot_path}")

    # Save detailed results
    results_csv = output_dir / 'threshold_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"  Saved results: {results_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest F1 threshold: {best_f1_thresh:.2f}")
    print(f"  Precision: {results_df.loc[best_f1_idx, 'precision']:.3f}")
    print(f"  Recall: {results_df.loc[best_f1_idx, 'recall']:.3f}")
    print(f"  F1: {results_df.loc[best_f1_idx, 'f1']:.3f}")
    print(f"  Accuracy: {results_df.loc[best_f1_idx, 'accuracy']:.3f}")

    print(f"\nHigh-recall threshold (target {target_recall:.0%}): {best_recall_thresh:.2f}")
    high_recall_row = results_df.loc[best_recall_idx]
    print(f"  Precision: {high_recall_row['precision']:.3f}")
    print(f"  Recall: {high_recall_row['recall']:.3f}")
    print(f"  F1: {high_recall_row['f1']:.3f}")
    print(f"  Accuracy: {high_recall_row['accuracy']:.3f}")

    # Save recommendation
    rec_path = output_dir / 'recommended_threshold.txt'
    with open(rec_path, 'w') as f:
        f.write("THRESHOLD OPTIMIZATION RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Best F1 threshold: {best_f1_thresh:.2f}\n")
        f.write(f"  Precision: {results_df.loc[best_f1_idx, 'precision']:.3f}\n")
        f.write(f"  Recall: {results_df.loc[best_f1_idx, 'recall']:.3f}\n")
        f.write(f"  F1: {results_df.loc[best_f1_idx, 'f1']:.3f}\n")
        f.write(f"  Accuracy: {results_df.loc[best_f1_idx, 'accuracy']:.3f}\n\n")

        f.write(f"High-recall threshold: {best_recall_thresh:.2f}\n")
        f.write(f"  Precision: {high_recall_row['precision']:.3f}\n")
        f.write(f"  Recall: {high_recall_row['recall']:.3f}\n")
        f.write(f"  F1: {high_recall_row['f1']:.3f}\n")
        f.write(f"  Accuracy: {high_recall_row['accuracy']:.3f}\n\n")

        f.write("RECOMMENDATION:\n")
        f.write(f"  For batch detection: Use {best_recall_thresh:.2f}\n")
        f.write(f"  (Prioritizes catching all USVs over avoiding false positives)\n\n")
        f.write(f"  For balanced performance: Use {best_f1_thresh:.2f}\n")
        f.write(f"  (Balances precision and recall)\n")

    print(f"\nRecommendation saved to: {rec_path}")
    print(f"\nUse the recommended threshold in:")
    print(f"  - USVClassifierCNN.optimal_threshold attribute")
    print(f"  - Batch detection scripts")
    print(f"  - PyQt6 app default threshold")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize classification threshold for CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (e.g., models/full_retrained_cnn/best_model.pt)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="Path to test CSV (e.g., data/full_training_dataset/test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Target recall for high-recall threshold (default: 0.90)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Device to use (default: auto)",
    )

    args = parser.parse_args()

    optimize_threshold(
        model_path=args.model,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        target_recall=args.target_recall,
        device=args.device,
    )


if __name__ == "__main__":
    main()
