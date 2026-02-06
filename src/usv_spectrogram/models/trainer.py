"""Training loop for USV classifier."""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TrainingHistory:
    """Container for training history metrics."""
    train_loss: list
    train_acc: list
    val_loss: list
    val_acc: list
    val_precision: list
    val_recall: list
    val_f1: list
    learning_rates: list

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1,
            'learning_rates': self.learning_rates,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingHistory':
        """Create from dictionary."""
        return cls(**d)


class Trainer:
    """Training loop with early stopping and checkpointing.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        class_weights: Optional class weights for imbalanced data
        learning_rate: Initial learning rate
        patience: Early stopping patience (epochs)
        min_delta: Minimum improvement for early stopping
        lr_scheduler_patience: ReduceLROnPlateau patience
        lr_scheduler_factor: Learning rate reduction factor
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on (cuda or cpu)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        class_weights: Optional[Dict[int, float]] = None,
        learning_rate: float = 0.001,
        patience: int = 15,
        min_delta: float = 0.001,
        lr_scheduler_patience: int = 5,
        lr_scheduler_factor: float = 0.5,
        checkpoint_dir: Path = Path('checkpoints'),
        device: str = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Loss function: BCEWithLogitsLoss (more stable than BCELoss + Sigmoid)
        if class_weights is not None:
            # Compute pos_weight for class imbalance
            # pos_weight = weight_positive / weight_negative
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            print(f"Using class-weighted loss: pos_weight={pos_weight.item():.3f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience
        )

        # Early stopping state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self) -> tuple:
        """Train for one epoch.

        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for spectrograms, labels in self.train_loader:
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            # Forward pass (model outputs logits)
            self.optimizer.zero_grad()
            outputs = self.model(spectrograms).squeeze()

            # Compute loss (BCEWithLogitsLoss applies sigmoid internally)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * spectrograms.size(0)

            # Compute accuracy (apply sigmoid to get probabilities)
            with torch.no_grad():
                proba = torch.sigmoid(outputs)
                predictions = (proba >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self) -> tuple:
        """Validate on validation set.

        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
            precision: Validation precision
            recall: Validation recall
            f1: Validation F1 score
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for spectrograms, labels in self.val_loader:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)

                # Forward pass (outputs logits)
                outputs = self.model(spectrograms).squeeze()

                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * spectrograms.size(0)

                # Apply sigmoid to get probabilities, then threshold
                proba = torch.sigmoid(outputs)
                predictions = (proba >= 0.5).float()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        avg_loss = total_loss / len(all_labels)
        accuracy = (all_predictions == all_labels).mean()

        # Precision, Recall, F1
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return avg_loss, accuracy, precision, recall, f1

    def save_checkpoint(self, filepath: Path, epoch: int, metrics: dict):
        """Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint

    def train(self, num_epochs: int, verbose: bool = True) -> TrainingHistory:
        """Train the model.

        Args:
            num_epochs: Maximum number of epochs
            verbose: Whether to print progress

        Returns:
            TrainingHistory object with metrics
        """
        history = TrainingHistory(
            train_loss=[],
            train_acc=[],
            val_loss=[],
            val_acc=[],
            val_precision=[],
            val_recall=[],
            val_f1=[],
            learning_rates=[]
        )

        if verbose:
            print(f"Training on {self.device}")
            print(f"Training samples: {len(self.train_loader.dataset)}")
            print(f"Validation samples: {len(self.val_loader.dataset)}")
            print("-" * 80)

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            history.train_loss.append(train_loss)
            history.train_acc.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)
            history.val_precision.append(val_prec)
            history.val_recall.append(val_rec)
            history.val_f1.append(val_f1)
            history.learning_rates.append(current_lr)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                      f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
                print(f"  LR: {current_lr:.6f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping and checkpointing
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                best_path = self.checkpoint_dir / 'best_model.pt'
                self.save_checkpoint(
                    best_path,
                    epoch,
                    {
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                    }
                )
                if verbose:
                    print(f"  > Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Check early stopping
            if self.epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best val_loss: {self.best_val_loss:.4f}")
                break

            if verbose:
                print()

        # Save final model
        final_path = self.checkpoint_dir / 'final_model.pt'
        self.save_checkpoint(
            final_path,
            epoch,
            {
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
            }
        )

        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history.to_dict(), f, indent=2)

        # Auto-generate training curves
        try:
            from .evaluate import plot_training_history

            plot_path = self.checkpoint_dir / 'training_curves.png'
            plot_training_history(
                history_path=history_path,
                output_path=plot_path,
                show=False
            )
            if verbose:
                print(f"Training curves saved to: {plot_path}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not generate training curves: {e}")

        if verbose:
            print(f"\nTraining complete!")
            print(f"Final model saved to: {final_path}")
            print(f"Best model saved to: {self.checkpoint_dir / 'best_model.pt'}")
            print(f"Training history saved to: {history_path}")

        return history
