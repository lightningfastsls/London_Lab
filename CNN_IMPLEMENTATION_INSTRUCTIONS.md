# Task: Implement CNN for USV Detection

## Overview

Build a CNN binary classifier to detect USVs in spectrogram images. This is Stage 1 of the USV detection pipeline—detecting whether a spectrogram contains a USV or not.

**Reference documents:**
- `usv_signal_processing_reference.md` — Technical background
- `USV_DETECTION_IMPLEMENTATION_PLAN.md` — Pipeline architecture

---

## Current State

- ~420 labeled USVs
- ~62 labeled noise examples (need more—see class imbalance section)
- ~8 uncertain (exclude from training)
- Spectrograms already extracted as images
- Labels in CSV format

---

## Project Structure

```
src/usv_spectrogram/
├── models/
│   ├── __init__.py
│   ├── cnn_classifier.py      # CNN architecture definition
│   ├── data_loader.py         # Dataset and data loading
│   ├── trainer.py             # Training loop
│   ├── evaluate.py            # Evaluation metrics
│   └── config.py              # Hyperparameters
scripts/
├── train_cnn.py               # CLI for training
├── evaluate_model.py          # CLI for evaluation
└── predict.py                 # CLI for inference on new data
```

---

## Implementation: Data Loading

### `src/usv_spectrogram/models/data_loader.py`

```python
"""
Data loading utilities for USV classification.

Key requirements:
- Load spectrograms from PNG files
- Handle variable-size spectrograms (will use GlobalAveragePooling in model)
- Apply per-spectrogram normalization
- Handle class imbalance via sampling or class weights
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd


class USVDataset(Dataset):
    """
    Dataset for USV spectrogram classification.
    
    Loads spectrograms and labels from a CSV file.
    CSV format: candidate_id, spectrogram_path, label (usv/noise)
    """
    
    def __init__(
        self, 
        csv_path: Path,
        spectrogram_dir: Path,
        transform=None,
        normalize: str = 'per_sample'  # 'per_sample', 'fixed', or 'none'
    ):
        """
        Args:
            csv_path: Path to CSV with columns [candidate_id, spectrogram_path, label]
            spectrogram_dir: Base directory for spectrogram files
            transform: Optional augmentation transforms
            normalize: Normalization strategy
        """
        self.df = pd.read_csv(csv_path)
        self.spectrogram_dir = Path(spectrogram_dir)
        self.transform = transform
        self.normalize = normalize
        
        # Filter out uncertain labels
        self.df = self.df[self.df['label'].isin(['usv', 'noise'])]
        
        # Convert labels to binary
        self.label_map = {'noise': 0, 'usv': 1}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load spectrogram
        spec_path = self.spectrogram_dir / row['spectrogram_path']
        spectrogram = np.array(Image.open(spec_path).convert('L'))  # Grayscale
        spectrogram = spectrogram.astype(np.float32)
        
        # Normalize
        if self.normalize == 'per_sample':
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        elif self.normalize == 'fixed':
            spectrogram = spectrogram / 255.0  # Assume 8-bit PNG
        
        # Apply transforms (augmentation)
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # Convert to tensor: (1, H, W) for single channel
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)
        
        # Get label
        label = self.label_map[row['label']]
        label = torch.tensor(label, dtype=torch.float32)
        
        return spectrogram, label
    
    def get_class_weights(self):
        """
        Compute class weights inversely proportional to class frequency.
        
        Returns dict suitable for loss function weighting.
        """
        counts = self.df['label'].value_counts()
        total = len(self.df)
        weights = {
            0: total / (2 * counts.get('noise', 1)),
            1: total / (2 * counts.get('usv', 1))
        }
        return weights


def create_data_loaders(
    train_csv: Path,
    val_csv: Path,
    spectrogram_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Returns (train_loader, val_loader)
    """
    train_dataset = USVDataset(train_csv, spectrogram_dir)
    val_dataset = USVDataset(val_csv, spectrogram_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

---

## Implementation: CNN Architecture

### `src/usv_spectrogram/models/cnn_classifier.py`

```python
"""
CNN architecture for USV binary classification.

Design decisions:
- GlobalAveragePooling to handle variable-size spectrograms
- Dropout for regularization (small dataset)
- Batch normalization for training stability
- Simple architecture to start—can increase complexity if underfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class USVClassifierCNN(nn.Module):
    """
    CNN for binary USV classification.
    
    Architecture:
        Conv blocks (3x) → GlobalAveragePool → Dense → Sigmoid
    
    Handles variable-size input via GlobalAveragePooling.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_filters: list[int] = [32, 64, 128],
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        prev_channels = in_channels
        for num_filter in num_filters:
            block = self._make_conv_block(prev_channels, num_filter)
            self.conv_blocks.append(block)
            prev_channels = num_filter
        
        # Global average pooling (handles variable input size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a conv block: Conv → BatchNorm → ReLU → MaxPool"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width)
               Height and width can vary between batches.
        
        Returns:
            Tensor of shape (batch, 1) with values in [0, 1]
        """
        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Global pooling → (batch, num_filters, 1, 1)
        x = self.global_pool(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of USV class."""
        return self.forward(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions."""
        proba = self.forward(x)
        return (proba >= threshold).float()


class USVClassifierCNNLarge(nn.Module):
    """
    Larger CNN for when you have more data (5000+ examples).
    
    More layers, more filters, but same basic structure.
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
```

---

## Implementation: Training Loop

### `src/usv_spectrogram/models/trainer.py`

```python
"""
Training loop for USV classifier.

Features:
- Class weighting for imbalanced data
- Early stopping to prevent overfitting
- Learning rate scheduling
- Checkpointing best model
- Logging training/validation metrics
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
from datetime import datetime


class Trainer:
    """
    Trainer for USV classifier.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        class_weights: dict = None,
        learning_rate: float = 1e-3,
        device: str = 'auto',
        checkpoint_dir: Path = Path('checkpoints')
    ):
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function with class weights
        if class_weights:
            # For binary classification with imbalance, use pos_weight
            # pos_weight = weight of positive class (USV) relative to negative (noise)
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            # Note: If using BCEWithLogitsLoss, remove Sigmoid from model's last layer
            # Or use BCELoss with Sigmoid in model (current setup)
            self.criterion = nn.BCELoss(reduction='none')
            self.class_weights = class_weights
        else:
            self.criterion = nn.BCELoss()
            self.class_weights = None
        
        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for spectrograms, labels in self.train_loader:
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectrograms).squeeze()
            
            # Compute loss with class weights
            if self.class_weights:
                weights = torch.where(
                    labels == 1,
                    torch.tensor(self.class_weights[1], device=self.device),
                    torch.tensor(self.class_weights[0], device=self.device)
                )
                loss = (self.criterion(outputs, labels) * weights).mean()
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * spectrograms.size(0)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> tuple[float, float, float, float]:
        """Validate model. Returns (loss, accuracy, precision, recall)."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for spectrograms, labels in self.val_loader:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(spectrograms).squeeze()
                loss = nn.BCELoss()(outputs, labels)
                
                total_loss += loss.item() * spectrograms.size(0)
                predictions = (outputs >= 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / len(all_labels)
        accuracy = (all_predictions == all_labels).mean()
        
        # Precision and recall for USV class
        true_positives = ((all_predictions == 1) & (all_labels == 1)).sum()
        predicted_positives = (all_predictions == 1).sum()
        actual_positives = (all_labels == 1).sum()
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        
        return avg_loss, accuracy, precision, recall
    
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Maximum epochs to train
            early_stopping_patience: Stop if val acc doesn't improve for this many epochs
        
        Returns:
            Training history dict
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_precision, val_recall = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
            
            # Checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pt')
                patience_counter = 0
                print(f"  ✓ New best model saved (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model and history
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, path)
    
    def save_history(self):
        """Save training history to JSON."""
        path = self.checkpoint_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']


# Need numpy for metrics
import numpy as np
```

---

## Implementation: Evaluation

### `src/usv_spectrogram/models/evaluate.py`

```python
"""
Evaluation utilities for USV classifier.

Provides detailed metrics including:
- Accuracy, precision, recall, F1
- Confusion matrix
- Per-population breakdown (lab vs wild mice)
- Confidence distribution analysis
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_model(
    model,
    data_loader,
    device: str = 'auto',
    threshold: float = 0.5
) -> dict:
    """
    Comprehensive model evaluation.
    
    Returns dict with all metrics.
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for spectrograms, labels in data_loader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms).squeeze()
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_predictions = (all_probs >= threshold).astype(int)
    
    # Compute metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
        'threshold': threshold,
        'num_samples': len(all_labels),
        'num_positive': int(all_labels.sum()),
        'num_negative': int(len(all_labels) - all_labels.sum())
    }
    
    return results


def plot_training_history(history: dict, save_path: Path = None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Training')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Training')
    axes[0, 1].legend()
    
    # Precision and Recall
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Precision and Recall')
    axes[1, 0].legend()
    
    # Learning rate
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_confusion_matrix(cm, save_path: Path = None):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Noise', 'USV'])
    ax.set_yticklabels(['Noise', 'USV'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Add values
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=20)
    
    plt.colorbar(im)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def evaluate_by_population(
    model,
    data_loader,
    population_labels: list,  # List parallel to data_loader samples: 'lab' or 'wild'
    device: str = 'auto'
) -> dict:
    """
    Evaluate model separately for each population.
    
    CRITICAL: This catches generalization failures between lab and wild mice.
    See usv_signal_processing_reference.md Section 3.5 and 4.7.
    """
    # ... implementation similar to evaluate_model but stratified by population
    pass
```

---

## Implementation: Training Script

### `scripts/train_cnn.py`

```python
"""
CLI script to train USV classifier.

Usage:
    python scripts/train_cnn.py --train-csv data/splits/train.csv \
                                --val-csv data/splits/val.csv \
                                --spectrogram-dir data/candidates/spectrograms \
                                --output-dir models/experiment_001
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.data_loader import USVDataset, create_data_loaders
from usv_spectrogram.models.trainer import Trainer
from usv_spectrogram.models.evaluate import plot_training_history


def main():
    parser = argparse.ArgumentParser(description='Train USV classifier')
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--spectrogram-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('models/default'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--use-class-weights', action='store_true')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        args.train_csv,
        args.val_csv,
        args.spectrogram_dir,
        batch_size=args.batch_size
    )
    
    # Get class weights if needed
    class_weights = None
    if args.use_class_weights:
        train_dataset = USVDataset(args.train_csv, args.spectrogram_dir)
        class_weights = train_dataset.get_class_weights()
        print(f"Using class weights: {class_weights}")
    
    # Create model
    print("Creating model...")
    model = USVClassifierCNN()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.output_dir
    )
    
    # Train
    print("Starting training...")
    history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping
    )
    
    # Plot results
    plot_training_history(history, save_path=args.output_dir / 'training_curves.png')
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
```

---

## Configuration Defaults

### `src/usv_spectrogram/models/config.py`

```python
"""
Default configuration for USV classifier training.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    # Data
    batch_size: int = 32
    num_workers: int = 4
    normalize: str = 'per_sample'
    
    # Model
    model_type: str = 'small'  # 'small' or 'large'
    num_filters: list = None  # Default: [32, 64, 128]
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    
    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 50
    early_stopping_patience: int = 10
    use_class_weights: bool = True
    
    # Augmentation (future)
    use_augmentation: bool = False
    
    def __post_init__(self):
        if self.num_filters is None:
            self.num_filters = [32, 64, 128]
```

---

## Before Training Checklist

Before running training, verify:

- [ ] Train/val/test splits created (by recording, not by candidate)
- [ ] Class balance checked (see class imbalance section below)
- [ ] All spectrogram files exist
- [ ] No recording in multiple splits
- [ ] Both populations (lab/wild) represented in val and test sets

---

## Class Imbalance: Must Fix First

Current state: ~420 USV, ~62 noise (7:1 ratio)

**Required action before training:**

1. Add noise examples by extracting random non-candidate segments:

```python
# Script to add noise examples
def extract_random_noise_segments(
    wav_files: list[Path],
    candidates_csv: Path,  # To know which times to AVOID
    output_dir: Path,
    num_samples: int = 300,
    segment_duration_ms: float = 150
):
    """
    Extract random spectrogram segments from regions NOT flagged as candidates.
    These are guaranteed (or near-guaranteed) to be noise.
    """
    # Load candidate times to avoid
    # For each wav file:
    #   - Pick random times that don't overlap with any candidate
    #   - Extract spectrogram
    #   - Save to output_dir
    # Create CSV with labels = 'noise'
    pass
```

2. Quickly label them (most will be obvious noise)

3. Target: At least 150-200 noise examples (1:3 ratio minimum)

---

## Expected Results

With ~500 examples and proper class balancing:

| Metric | Reasonable Target | Red Flag |
|--------|------------------|----------|
| Val Accuracy | 80-90% | <70% or >95% (overfitting) |
| Val Precision | >80% | <60% |
| Val Recall | >85% | <70% (missing USVs) |
| Train-Val Gap | <10% | >20% (overfitting) |

If results are worse than these, likely causes:
1. Class imbalance not addressed
2. Data leakage (recording in multiple splits)
3. Normalization issues between train/val

---

## Next Steps After Training

1. Evaluate on held-out test set (only once!)
2. Evaluate separately on lab vs. wild mice
3. If performance is acceptable, use model to detect USVs in unlabeled recordings
4. If not acceptable, diagnose and iterate:
   - More data?
   - Augmentation?
   - Larger model?
   - Check for data issues?
