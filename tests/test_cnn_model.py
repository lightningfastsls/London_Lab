"""Tests for CNN model modules (classifier, data_loader, config, trainer, evaluate)."""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN, USVClassifierCNNLarge
from usv_spectrogram.models.data_loader import (
    USVDataset,
    pad_collate_fn,
    create_data_loaders
)
from usv_spectrogram.models.config import TrainingConfig
from usv_spectrogram.models.trainer import Trainer, TrainingHistory
from usv_spectrogram.models.evaluate import evaluate_model


# =============================================================================
# CNN Classifier Tests
# =============================================================================

def test_cnn_classifier_instantiation_default():
    """Model instantiation with default config."""
    model = USVClassifierCNN()

    assert model.num_filters == [32, 64, 128]
    assert model.dense_units == 64
    assert model.dropout_rate == 0.5
    assert model.optimal_threshold == 0.05
    assert isinstance(model, torch.nn.Module)


def test_cnn_classifier_instantiation_custom():
    """Model instantiation with custom parameters."""
    model = USVClassifierCNN(
        num_filters=[16, 32],
        dense_units=32,
        dropout_rate=0.3,
        optimal_threshold=0.1
    )

    assert model.num_filters == [16, 32]
    assert model.dense_units == 32
    assert model.dropout_rate == 0.3
    assert model.optimal_threshold == 0.1


def test_cnn_classifier_forward_pass_shape():
    """Forward pass produces correct output shape (batch_size, 1)."""
    model = USVClassifierCNN()

    # Test with single sample
    x = torch.randn(1, 1, 64, 128)
    output = model(x)
    assert output.shape == (1, 1)

    # Test with batch
    x = torch.randn(8, 1, 64, 128)
    output = model(x)
    assert output.shape == (8, 1)


def test_cnn_classifier_forward_pass_variable_sizes():
    """Forward pass handles variable input sizes (due to global pooling)."""
    model = USVClassifierCNN()

    # Different input sizes
    x1 = torch.randn(4, 1, 32, 64)
    x2 = torch.randn(4, 1, 128, 256)
    x3 = torch.randn(4, 1, 100, 200)

    # All should produce (batch_size, 1) output
    assert model(x1).shape == (4, 1)
    assert model(x2).shape == (4, 1)
    assert model(x3).shape == (4, 1)


def test_cnn_classifier_predict_proba():
    """predict_proba outputs valid probabilities in [0, 1]."""
    model = USVClassifierCNN()
    model.eval()

    x = torch.randn(8, 1, 64, 128)
    proba = model.predict_proba(x)

    assert proba.shape == (8, 1)
    assert torch.all(proba >= 0.0)
    assert torch.all(proba <= 1.0)


def test_cnn_classifier_predict():
    """predict returns binary predictions based on threshold."""
    model = USVClassifierCNN(optimal_threshold=0.5)
    model.eval()

    x = torch.randn(8, 1, 64, 128)
    predictions = model.predict(x)

    assert predictions.shape == (8, 1)
    assert torch.all((predictions == 0.0) | (predictions == 1.0))


def test_cnn_classifier_predict_custom_threshold():
    """predict respects custom threshold parameter."""
    model = USVClassifierCNN(optimal_threshold=0.5)
    model.eval()

    # Create controlled input that produces known probability
    x = torch.randn(1, 1, 64, 128)

    # Test with different thresholds
    pred_low = model.predict(x, threshold=0.01)
    pred_high = model.predict(x, threshold=0.99)

    # With very low threshold, more likely to predict 1
    # With very high threshold, more likely to predict 0
    assert pred_low.shape == (1, 1)
    assert pred_high.shape == (1, 1)


def test_cnn_classifier_parameter_count():
    """Model parameter count is reasonable (~90K for default config)."""
    model = USVClassifierCNN()

    total_params = sum(p.numel() for p in model.parameters())

    # Should be around 90K parameters for default config
    # Allow range of 50K-150K to account for variations
    assert 50_000 <= total_params <= 150_000


def test_cnn_classifier_large_instantiation():
    """Large CNN model instantiation."""
    model = USVClassifierCNNLarge()

    assert model.dropout_rate == 0.5
    assert model.optimal_threshold == 0.40

    # Should have significantly more parameters than small model
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 150_000  # Larger than small model


def test_cnn_classifier_large_forward_pass():
    """Large CNN forward pass works correctly."""
    model = USVClassifierCNNLarge()

    x = torch.randn(4, 1, 64, 128)
    output = model(x)

    assert output.shape == (4, 1)


# =============================================================================
# Data Loader Tests
# =============================================================================

@pytest.fixture
def mock_csv_with_images(tmp_path):
    """Create mock CSV file and corresponding PNG images for testing."""
    # Create image directory
    img_dir = tmp_path / "spectrograms"
    img_dir.mkdir()

    # Create mock spectrogram images
    data = []
    for i in range(10):
        # Create grayscale image
        img = Image.new('L', (128, 64), color=128)
        img_path = img_dir / f"spec_{i}.png"
        img.save(img_path)

        # Alternate labels
        label = 'USV' if i % 2 == 0 else 'Not USV'

        data.append({
            'candidate_id': f'cand_{i}',
            'spectrogram_path': str(img_path),
            'label': label
        })

    # Add one 'Uncertain' sample (should be filtered)
    img = Image.new('L', (128, 64), color=128)
    img_path = img_dir / "spec_uncertain.png"
    img.save(img_path)
    data.append({
        'candidate_id': 'cand_uncertain',
        'spectrogram_path': str(img_path),
        'label': 'Uncertain'
    })

    # Create CSV
    df = pd.DataFrame(data)
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_usv_dataset_creation(mock_csv_with_images):
    """USVDataset creation from CSV."""
    dataset = USVDataset(mock_csv_with_images)

    # Should filter out 'Uncertain' label
    assert len(dataset) == 10
    assert 'USV' in dataset.class_counts
    assert 'Not USV' in dataset.class_counts


def test_usv_dataset_getitem(mock_csv_with_images):
    """USVDataset __getitem__ returns correct format."""
    dataset = USVDataset(mock_csv_with_images)

    spectrogram, label = dataset[0]

    # Check shapes and types
    assert isinstance(spectrogram, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert spectrogram.shape[0] == 1  # Single channel
    assert spectrogram.ndim == 3  # (1, H, W)
    assert label.shape == ()  # Scalar

    # Check value ranges
    assert torch.all(spectrogram >= 0.0)
    assert torch.all(spectrogram <= 1.0)
    assert label.item() in [0.0, 1.0]


def test_usv_dataset_label_mapping(mock_csv_with_images):
    """USVDataset maps labels correctly."""
    dataset = USVDataset(mock_csv_with_images)

    # Check first sample (should be 'USV' = 1)
    _, label0 = dataset[0]
    assert label0.item() == 1.0

    # Check second sample (should be 'Not USV' = 0)
    _, label1 = dataset[1]
    assert label1.item() == 0.0


def test_usv_dataset_normalization_modes(mock_csv_with_images):
    """USVDataset handles different normalization modes."""
    # Per-image normalization
    dataset_per_image = USVDataset(mock_csv_with_images, normalize_mode='per_image')
    spec, _ = dataset_per_image[0]
    assert torch.all(spec >= 0.0) and torch.all(spec <= 1.0)

    # Global normalization
    dataset_global = USVDataset(mock_csv_with_images, normalize_mode='global')
    spec, _ = dataset_global[0]
    assert torch.all(spec >= 0.0) and torch.all(spec <= 1.0)


def test_usv_dataset_get_class_weights(mock_csv_with_images):
    """USVDataset computes class weights correctly."""
    dataset = USVDataset(mock_csv_with_images)
    weights = dataset.get_class_weights()

    # Should have weights for both classes
    assert 0 in weights
    assert 1 in weights
    assert weights[0] > 0
    assert weights[1] > 0


def test_pad_collate_fn():
    """pad_collate_fn pads batch to maximum dimensions."""
    # Create spectrograms of different sizes
    spec1 = torch.randn(1, 32, 64)
    spec2 = torch.randn(1, 64, 128)
    spec3 = torch.randn(1, 48, 96)

    label1 = torch.tensor(0.0)
    label2 = torch.tensor(1.0)
    label3 = torch.tensor(0.0)

    batch = [(spec1, label1), (spec2, label2), (spec3, label3)]

    spectrograms, labels = pad_collate_fn(batch)

    # All should be padded to max dimensions (64, 128)
    assert spectrograms.shape == (3, 1, 64, 128)
    assert labels.shape == (3,)
    assert torch.equal(labels, torch.tensor([0.0, 1.0, 0.0]))


def test_create_data_loaders(tmp_path):
    """create_data_loaders creates train and val loaders."""
    # Create minimal CSV files
    img_dir = tmp_path / "spectrograms"
    img_dir.mkdir()

    def create_csv(name, num_samples):
        data = []
        for i in range(num_samples):
            img = Image.new('L', (128, 64), color=128)
            img_path = img_dir / f"{name}_spec_{i}.png"
            img.save(img_path)

            data.append({
                'candidate_id': f'{name}_{i}',
                'spectrogram_path': str(img_path),
                'label': 'USV' if i % 2 == 0 else 'Not USV'
            })

        df = pd.DataFrame(data)
        csv_path = tmp_path / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    train_csv = create_csv('train', 8)
    val_csv = create_csv('val', 4)

    train_loader, val_loader, class_weights = create_data_loaders(
        train_csv,
        val_csv,
        batch_size=4,
        num_workers=0
    )

    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 4
    assert class_weights is not None


# =============================================================================
# Config Tests
# =============================================================================

def test_training_config_default_values():
    """TrainingConfig has sensible default values."""
    config = TrainingConfig()

    assert config.batch_size == 16
    assert config.learning_rate == 0.001
    assert config.num_epochs == 100
    assert config.patience == 15
    assert config.min_delta == 0.001
    assert config.use_class_weights is True
    assert config.dropout_rate == 0.5
    assert config.lr_scheduler_patience == 5
    assert config.lr_scheduler_factor == 0.5
    assert config.checkpoint_dir == Path('checkpoints')
    assert config.seed == 42


def test_training_config_custom_values():
    """TrainingConfig accepts custom values."""
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.0001,
        num_epochs=50,
        patience=10
    )

    assert config.batch_size == 32
    assert config.learning_rate == 0.0001
    assert config.num_epochs == 50
    assert config.patience == 10


def test_training_config_validation_batch_size():
    """TrainingConfig validates batch_size must be positive."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(batch_size=-1)


def test_training_config_validation_learning_rate():
    """TrainingConfig validates learning_rate must be positive."""
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainingConfig(learning_rate=0)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TrainingConfig(learning_rate=-0.001)


def test_training_config_validation_num_epochs():
    """TrainingConfig validates num_epochs must be positive."""
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        TrainingConfig(num_epochs=0)


def test_training_config_validation_patience():
    """TrainingConfig validates patience must be positive."""
    with pytest.raises(ValueError, match="patience must be positive"):
        TrainingConfig(patience=0)


def test_training_config_validation_dropout_rate():
    """TrainingConfig validates dropout_rate in [0, 1)."""
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TrainingConfig(dropout_rate=-0.1)

    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TrainingConfig(dropout_rate=1.0)

    # Valid edge case
    config = TrainingConfig(dropout_rate=0.0)
    assert config.dropout_rate == 0.0


def test_training_config_validation_lr_scheduler_factor():
    """TrainingConfig validates lr_scheduler_factor in (0, 1)."""
    with pytest.raises(ValueError, match="lr_scheduler_factor must be in"):
        TrainingConfig(lr_scheduler_factor=0.0)

    with pytest.raises(ValueError, match="lr_scheduler_factor must be in"):
        TrainingConfig(lr_scheduler_factor=1.0)

    with pytest.raises(ValueError, match="lr_scheduler_factor must be in"):
        TrainingConfig(lr_scheduler_factor=-0.1)


def test_training_config_frozen():
    """TrainingConfig is frozen (immutable)."""
    config = TrainingConfig()

    with pytest.raises(Exception):  # FrozenInstanceError
        config.batch_size = 32


# =============================================================================
# Trainer Tests (Basic Smoke Tests)
# =============================================================================

@pytest.fixture
def dummy_data_loaders():
    """Create minimal data loaders with random tensors for testing."""
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            spec = torch.randn(1, 32, 64)
            label = torch.tensor(float(idx % 2))
            return spec, label

    from torch.utils.data import DataLoader

    train_dataset = DummyDataset(16)
    val_dataset = DummyDataset(8)

    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=pad_collate_fn)

    return train_loader, val_loader


def test_trainer_instantiation(dummy_data_loaders, tmp_path):
    """Trainer instantiation works correctly."""
    train_loader, val_loader = dummy_data_loaders
    model = USVClassifierCNN()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "checkpoints",
        device='cpu'
    )

    assert trainer.model is not None
    assert trainer.device.type == 'cpu'
    assert trainer.checkpoint_dir.exists()


def test_trainer_with_class_weights(dummy_data_loaders, tmp_path):
    """Trainer handles class weights correctly."""
    train_loader, val_loader = dummy_data_loaders
    model = USVClassifierCNN()

    class_weights = {0: 1.0, 1: 2.0}

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        checkpoint_dir=tmp_path / "checkpoints",
        device='cpu'
    )

    assert trainer.criterion is not None


def test_trainer_single_training_step(dummy_data_loaders, tmp_path):
    """Single training step doesn't crash."""
    train_loader, val_loader = dummy_data_loaders
    model = USVClassifierCNN()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "checkpoints",
        device='cpu'
    )

    # Run single training epoch
    train_loss, train_acc = trainer.train_epoch()

    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    assert train_loss >= 0.0
    assert 0.0 <= train_acc <= 1.0


def test_trainer_single_validation_step(dummy_data_loaders, tmp_path):
    """Single validation step doesn't crash."""
    train_loader, val_loader = dummy_data_loaders
    model = USVClassifierCNN()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "checkpoints",
        device='cpu'
    )

    # Run validation
    val_loss, val_acc, val_prec, val_rec, val_f1 = trainer.validate()

    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert val_loss >= 0.0
    assert 0.0 <= val_acc <= 1.0


def test_trainer_checkpoint_save_load(dummy_data_loaders, tmp_path):
    """Trainer can save and load checkpoints."""
    train_loader, val_loader = dummy_data_loaders
    model = USVClassifierCNN()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "checkpoints",
        device='cpu'
    )

    # Save checkpoint
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    metrics = {'val_loss': 0.5, 'val_acc': 0.8}
    trainer.save_checkpoint(checkpoint_path, epoch=5, metrics=metrics)

    assert checkpoint_path.exists()

    # Load checkpoint
    checkpoint = trainer.load_checkpoint(checkpoint_path)

    assert checkpoint['epoch'] == 5
    assert checkpoint['metrics']['val_loss'] == 0.5


def test_trainer_full_training_loop(dummy_data_loaders, tmp_path):
    """Full training loop for 2 epochs doesn't crash."""
    train_loader, val_loader = dummy_data_loaders
    model = USVClassifierCNN()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "checkpoints",
        device='cpu',
        patience=10
    )

    # Train for just 2 epochs
    history = trainer.train(num_epochs=2, verbose=False)

    assert isinstance(history, TrainingHistory)
    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2

    # Check that checkpoints were created
    assert (tmp_path / "checkpoints" / "final_model.pt").exists()
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()


def test_training_history_to_dict():
    """TrainingHistory converts to dict correctly."""
    history = TrainingHistory(
        train_loss=[0.5, 0.4],
        train_acc=[0.7, 0.8],
        val_loss=[0.6, 0.5],
        val_acc=[0.65, 0.75],
        val_precision=[0.7, 0.8],
        val_recall=[0.6, 0.7],
        val_f1=[0.65, 0.75],
        learning_rates=[0.001, 0.001]
    )

    d = history.to_dict()

    assert 'train_loss' in d
    assert 'val_f1' in d
    assert d['train_loss'] == [0.5, 0.4]


def test_training_history_from_dict():
    """TrainingHistory creates from dict correctly."""
    d = {
        'train_loss': [0.5, 0.4],
        'train_acc': [0.7, 0.8],
        'val_loss': [0.6, 0.5],
        'val_acc': [0.65, 0.75],
        'val_precision': [0.7, 0.8],
        'val_recall': [0.6, 0.7],
        'val_f1': [0.65, 0.75],
        'learning_rates': [0.001, 0.001]
    }

    history = TrainingHistory.from_dict(d)

    assert history.train_loss == [0.5, 0.4]
    assert history.val_f1 == [0.65, 0.75]


# =============================================================================
# Evaluate Tests
# =============================================================================

def test_evaluate_model_with_known_inputs(dummy_data_loaders):
    """evaluate_model computes metrics correctly with known inputs."""
    _, val_loader = dummy_data_loaders
    model = USVClassifierCNN()
    model.eval()

    metrics, predictions, probabilities, labels = evaluate_model(
        model,
        val_loader,
        device='cpu',
        verbose=False
    )

    # Check that metrics are returned
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'specificity' in metrics

    # Check that arrays have correct length
    assert len(predictions) == 8
    assert len(probabilities) == 8
    assert len(labels) == 8


def test_evaluate_model_metric_ranges(dummy_data_loaders):
    """evaluate_model produces metrics in valid ranges."""
    _, val_loader = dummy_data_loaders
    model = USVClassifierCNN()
    model.eval()

    metrics, _, _, _ = evaluate_model(
        model,
        val_loader,
        device='cpu',
        verbose=False
    )

    # All metrics should be in [0, 1]
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['f1'] <= 1.0
    assert 0.0 <= metrics['specificity'] <= 1.0


def test_evaluate_model_confusion_matrix_counts(dummy_data_loaders):
    """evaluate_model computes confusion matrix counts."""
    _, val_loader = dummy_data_loaders
    model = USVClassifierCNN()
    model.eval()

    metrics, _, _, _ = evaluate_model(
        model,
        val_loader,
        device='cpu',
        verbose=False
    )

    # Check confusion matrix counts sum to total samples
    tp = metrics['true_positives']
    fp = metrics['false_positives']
    tn = metrics['true_negatives']
    fn = metrics['false_negatives']

    assert tp + fp + tn + fn == 8  # Total samples in val set
