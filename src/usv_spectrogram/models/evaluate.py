"""Evaluation tools for USV classifier."""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: str = None,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run on (cuda or cpu)
        verbose: Whether to print results

    Returns:
        Dictionary of metrics (loss, accuracy, precision, recall, f1)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in data_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass — squeeze only dim=1, preserve batch dim
            outputs = model(spectrograms).squeeze(dim=1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * spectrograms.size(0)

            # Get probabilities and predictions
            proba = torch.sigmoid(outputs)
            predictions = (proba >= 0.5).float()

            all_probabilities.extend(proba.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    # Compute metrics
    avg_loss = total_loss / len(all_labels)
    accuracy = (all_predictions == all_labels).mean()

    # Precision, Recall, F1
    tp = ((all_predictions == 1) & (all_labels == 1)).sum()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum()
    fn = ((all_predictions == 0) & (all_labels == 1)).sum()
    tn = ((all_predictions == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }

    if verbose:
        print("=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print()
        print("Confusion Matrix:")
        print(f"  TP: {tp:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TN: {tn:4d}")
        print("=" * 80)

    return metrics, all_predictions, all_probabilities, all_labels


def plot_training_history(
    history_path: Path,
    output_path: Path = None,
    show: bool = False
):
    """Plot training history curves.

    Args:
        history_path: Path to training_history.json
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision, Recall, F1
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].plot(history['val_f1'], label='F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning Rate
    axes[1, 1].plot(history['learning_rates'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = None,
    show: bool = False
):
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ['Not USV', 'USV']
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        xlabel='Predicted Label',
        ylabel='True Label',
        title='Confusion Matrix'
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path = None,
    show: bool = False
):
    """Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path = None,
    show: bool = False
):
    """Plot precision-recall curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_path: Path to save plot (optional)
        show: Whether to display plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved precision-recall curve to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve_annotated(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path = None,
    thresholds_to_mark: list = None,
    show: bool = False
):
    """Plot ROC curve with threshold annotations at key operating points.

    Marks specific thresholds on the curve so you can see the TPR/FPR
    trade-off at each decision value. Also marks Youden's J optimal point.

    Args:
        y_true: True labels (0 or 1)
        y_proba: Predicted probabilities
        output_path: Path to save plot (optional)
        thresholds_to_mark: List of thresholds to annotate (default: common DV values)
        show: Whether to display plot
    """
    if thresholds_to_mark is None:
        thresholds_to_mark = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic: optimal point maximizing (TPR - FPR)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_j_idx] if best_j_idx < len(thresholds) else 0.5

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5, label='Random')

    # Mark Youden's J optimal point
    ax.plot(fpr[best_j_idx], tpr[best_j_idx], 'r*', markersize=15,
            label=f"Youden's J optimal (t={best_threshold:.3f})")

    # Mark requested thresholds
    for t in thresholds_to_mark:
        # Find closest threshold in the ROC curve
        idx = np.argmin(np.abs(thresholds - t)) if len(thresholds) > 0 else 0
        if idx < len(fpr):
            ax.plot(fpr[idx], tpr[idx], 'ko', markersize=6)
            ax.annotate(
                f't={t:.2f}\nTPR={tpr[idx]:.2f}\nFPR={fpr[idx]:.2f}',
                xy=(fpr[idx], tpr[idx]),
                xytext=(12, -12), textcoords='offset points',
                fontsize=7, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve with Decision Value Annotations', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved annotated ROC curve to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return best_threshold, roc_auc


def plot_pr_curve_annotated(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path = None,
    thresholds_to_mark: list = None,
    show: bool = False
):
    """Plot precision-recall curve with threshold annotations.

    Args:
        y_true: True labels (0 or 1)
        y_proba: Predicted probabilities
        output_path: Path to save plot (optional)
        thresholds_to_mark: List of thresholds to annotate
        show: Whether to display plot
    """
    if thresholds_to_mark is None:
        thresholds_to_mark = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    # F1-optimal point
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')

    # Mark F1-optimal point
    ax.plot(recall[best_f1_idx], precision[best_f1_idx], 'r*', markersize=15,
            label=f'Best F1 (t={best_threshold:.3f}, F1={f1_scores[best_f1_idx]:.3f})')

    # Mark requested thresholds
    for t in thresholds_to_mark:
        idx = np.argmin(np.abs(thresholds - t)) if len(thresholds) > 0 else 0
        if idx < len(recall):
            ax.plot(recall[idx], precision[idx], 'ko', markersize=6)
            ax.annotate(
                f't={t:.2f}\nP={precision[idx]:.2f}\nR={recall[idx]:.2f}',
                xy=(recall[idx], precision[idx]),
                xytext=(12, -12), textcoords='offset points',
                fontsize=7, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve with Decision Value Annotations', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved annotated PR curve to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return best_threshold, pr_auc


def generate_threshold_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path = None,
    thresholds: list = None,
    verbose: bool = True
):
    """Generate a table of metrics at various decision thresholds.

    This is the key output for choosing a decision value (DV): it shows
    exactly how many candidates would pass at each threshold and the
    corresponding precision/recall trade-off.

    Args:
        y_true: True labels (0 or 1)
        y_proba: Predicted probabilities
        output_path: Path to save CSV (optional)
        thresholds: List of thresholds to evaluate (default: 14 common values)
        verbose: Whether to print the table

    Returns:
        List of dicts with metrics at each threshold
    """
    import csv as csv_module

    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Also find Youden's J optimal
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr_curve - fpr_curve
    best_j_idx = np.argmax(j_scores)
    youden_threshold = roc_thresholds[best_j_idx] if best_j_idx < len(roc_thresholds) else 0.5

    # Add Youden's threshold if not already close to an existing one
    if not any(abs(t - youden_threshold) < 0.01 for t in thresholds):
        thresholds.append(round(youden_threshold, 4))
        thresholds.sort()

    n_total = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_total - n_positive

    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())

        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        n_predicted_pos = tp + fp

        is_youden = abs(t - youden_threshold) < 0.01

        rows.append({
            'threshold': t,
            'TPR_recall': recall_val,
            'FPR': fpr_val,
            'precision': precision_val,
            'specificity': specificity_val,
            'F1': f1_val,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'n_predicted_positive': n_predicted_pos,
            'pct_passing': n_predicted_pos / n_total * 100 if n_total > 0 else 0,
            'is_youden_optimal': is_youden,
        })

    if verbose:
        print(f"\nThreshold Analysis (n={n_total}, positives={n_positive}, negatives={n_negative})")
        print("=" * 110)
        print(f"{'Thresh':>7} {'TPR':>6} {'FPR':>6} {'Prec':>6} {'Spec':>6} {'F1':>6} "
              f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Pass':>5} {'%Pass':>6} {'Note':>8}")
        print("-" * 110)
        for r in rows:
            note = "YOUDEN" if r['is_youden_optimal'] else ""
            print(f"{r['threshold']:>7.3f} {r['TPR_recall']:>6.3f} {r['FPR']:>6.3f} "
                  f"{r['precision']:>6.3f} {r['specificity']:>6.3f} {r['F1']:>6.3f} "
                  f"{r['TP']:>5d} {r['FP']:>5d} {r['FN']:>5d} {r['TN']:>5d} "
                  f"{r['n_predicted_positive']:>5d} {r['pct_passing']:>5.1f}% {note:>8}")
        print("=" * 110)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv_module.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved threshold table to: {output_path}")

    return rows


def load_model_checkpoint(checkpoint_path: Path, model_class, device: str = None):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to instantiate
        device: Device to load on

    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Load checkpoint (weights_only=False for our own trusted checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model with architecture from checkpoint (if available)
    model_kwargs = {}
    if 'num_filters' in checkpoint:
        model_kwargs['num_filters'] = checkpoint['num_filters']
    if 'dense_units' in checkpoint:
        model_kwargs['dense_units'] = checkpoint['dense_units']
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint
