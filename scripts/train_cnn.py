"""Train CNN classifier for USV detection.

Usage:
    python scripts/train_cnn.py --train-csv splits/train.csv --val-csv splits/val.csv
    python scripts/train_cnn.py --train-csv splits/train.csv --val-csv splits/val.csv --num-epochs 50 --batch-size 32
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.models import (
    TrainingConfig,
    USVClassifierCNN,
    create_data_loaders,
    Trainer
)


# Model size configurations
# Guidelines for model selection based on dataset size:
# - small (~101K params): 2K-10K samples - good starting point
# - mid (~207K params): 10K-15K samples - wider deep layers, small head, good inference/capacity tradeoff
# - medium (~400K params): 15K-20K samples - use when mid model shows underfitting
# - large (~1.6M params): 20K-30K+ samples - only with large datasets and strong regularization
#
# Scaling decision criteria:
# - Train loss AND val loss both high, plateau together → Underfitting → Scale up
# - Train loss low, val loss much higher and diverging → Overfitting → More data/regularization
# - Both losses low and close together → Good fit → Current size appropriate
MODEL_CONFIGS = {
    "small": {
        "filters": [32, 64, 128],
        "dense_units": 64,
        "params_approx": "~101K",
        "recommended_samples": "2K-10K"
    },
    "mid": {
        "filters": [32, 96, 192],
        "dense_units": 64,
        "params_approx": "~207K",
        "recommended_samples": "10K-15K"
    },
    "medium": {
        "filters": [64, 128, 256],
        "dense_units": 128,
        "params_approx": "~400K",
        "recommended_samples": "15K-20K"
    },
    "large": {
        "filters": [128, 256, 512],
        "dense_units": 256,
        "params_approx": "~1.6M",
        "recommended_samples": "20K-30K+"
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN classifier for USV detection'
    )

    # Data paths
    parser.add_argument(
        '--train-csv',
        type=Path,
        default=Path('splits/train.csv'),
        help='Path to training split CSV'
    )
    parser.add_argument(
        '--val-csv',
        type=Path,
        default=Path('splits/val.csv'),
        help='Path to validation split CSV'
    )

    # Training config
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (L2 regularization) coefficient (default: 1e-4)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Maximum number of epochs (default: 100)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience (default: 15)'
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.5,
        help='Dropout rate in classifier head (default: 0.5)'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='small',
        choices=['small', 'mid', 'medium', 'large'],
        help='Model size configuration: small (~101K params, 2K-10K samples), '
             'medium (~400K params, 10K-20K samples), large (~1.6M params, 20K-30K+ samples). '
             'Default: small'
    )
    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        default=False,
        help='Use class weights for imbalanced data'
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
        default=Path('checkpoints'),
        help='Directory to save checkpoints (default: checkpoints/)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda or cpu, default: auto)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Verify input files exist
    if not args.train_csv.exists():
        print(f"Error: Training CSV not found: {args.train_csv}")
        sys.exit(1)
    if not args.val_csv.exists():
        print(f"Error: Validation CSV not found: {args.val_csv}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_config = MODEL_CONFIGS[args.model_size]

    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Max epochs: {args.num_epochs}")
    print(f"Patience: {args.patience}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Model size: {args.model_size} ({model_config['params_approx']} params, "
          f"recommended for {model_config['recommended_samples']} samples)")
    print(f"Use class weights: {args.use_class_weights}")
    print(f"Normalize mode: {args.normalize_mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    print()

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, class_weights = create_data_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=2,  # Parallel data loading (0 for Windows, 2+ for Linux/WSL)
        normalize_mode=args.normalize_mode,
        use_class_weights=args.use_class_weights
    )
    print()

    # Create model
    print("Creating model...")
    model = USVClassifierCNN(
        num_filters=model_config['filters'],
        dense_units=model_config['dense_units'],
        dropout_rate=args.dropout_rate
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: USVClassifierCNN ({args.model_size})")
    print(f"Architecture: {model_config['filters']} filters, {model_config['dense_units']} dense units")
    print(f"Total parameters: {num_params:,}")
    print(f"Recommended dataset size: {model_config['recommended_samples']} samples")
    print()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights if args.use_class_weights else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.output_dir,
        device=args.device
    )

    # Train
    print("Starting training...")
    print()
    history = trainer.train(num_epochs=args.num_epochs, verbose=True)

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {min(history.val_loss):.4f}")
    print(f"Best validation accuracy: {max(history.val_acc):.4f}")
    print(f"Best validation F1: {max(history.val_f1):.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
