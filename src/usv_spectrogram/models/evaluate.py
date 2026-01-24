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

            # Forward pass
            outputs = model(spectrograms).squeeze()
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

    # Create model and load weights
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint
