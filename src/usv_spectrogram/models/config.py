"""Training configuration for USV classifier."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training USV classifier.

    Args:
        batch_size: Number of samples per batch
        learning_rate: Initial learning rate for optimizer
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement for early stopping
        use_class_weights: Whether to use class weighting for imbalanced data
        dropout_rate: Dropout probability in classifier head
        lr_scheduler_patience: ReduceLROnPlateau patience (epochs)
        lr_scheduler_factor: Learning rate reduction factor
        checkpoint_dir: Directory to save model checkpoints
        seed: Random seed for reproducibility
    """

    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 15
    min_delta: float = 0.001
    use_class_weights: bool = True
    dropout_rate: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    checkpoint_dir: Path = Path('checkpoints')
    seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")

        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

        if self.lr_scheduler_factor <= 0 or self.lr_scheduler_factor >= 1:
            raise ValueError(
                f"lr_scheduler_factor must be in (0, 1), got {self.lr_scheduler_factor}"
            )

        object.__setattr__(self, 'checkpoint_dir', Path(self.checkpoint_dir))
