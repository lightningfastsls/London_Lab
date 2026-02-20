# CNN Classifier

**Phase:** 5 (CNN Classifier)
**ADRs:** ADR-005 (Class Weighting), ADR-006 (CNN Architecture), ADR-008 (Negative Samples), ADR-009 (Model Artifacts)
**Tests:** `tests/test_cnn_model.py` — 37 tests

## Purpose

The CNN classifier is the precision stage of a two-stage USV detection pipeline. Stage 1 (EnergyDetector) generates high-recall candidates; Stage 2 (this CNN) scores them as USV vs. noise. The model is small (~101K parameters), designed to train on limited labeled data (1K-10K samples), and outputs per-window probability predictions that feed into hysteresis-based detection logic.

## Architecture

### `USVClassifierCNN` (Small — Default)

```
Input: (batch, 1, H, W) — grayscale spectrogram patch

Feature Extractor:
  Block 1: Conv2d(1->32, 3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2)
  Block 2: Conv2d(32->64, 3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2)
  Block 3: Conv2d(64->128, 3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2)

Global Average Pooling: AdaptiveAvgPool2d((1, 1))

Classifier Head:
  Flatten -> Linear(128->64) -> ReLU -> Dropout(0.5) -> Linear(64->1)

Output: (batch, 1) — logits (no sigmoid; use BCEWithLogitsLoss)
```

- **Parameters:** ~101K total
- **Variable input sizes:** Global average pooling handles any (H, W)
- **Optimal threshold:** 0.05 (calibrated from full retraining evaluation)

### `USVClassifierCNNLarge` (Alternative)

- 5 conv blocks: [32, 64, 128, 256, 512]
- Larger classifier head: Linear(512->128->1)
- Not recommended for datasets < 5,000 samples (overfitting risk)

## Public Interface

### Model Class

```python
class USVClassifierCNN(nn.Module):
    def __init__(
        self,
        num_filters: List[int] = None,       # Default: [32, 64, 128]
        dense_units: int = 64,
        dropout_rate: float = 0.5,
        optimal_threshold: float = 0.05,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (batch, 1). For training with BCEWithLogitsLoss."""

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities [0, 1]. For inference."""

    def predict(self, x: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """Returns binary predictions {0, 1}. Default threshold: self.optimal_threshold."""
```

### Data Loading

```python
class USVDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,               # CSV with: candidate_id, spectrogram_path, label
        normalize_mode: str = 'per_image',   # 'per_image' or 'global'
    ): ...

def pad_collate_fn(batch):
    """Custom collate: pads variable-size spectrograms to max dims in batch."""

def create_data_loaders(train_csv, val_csv, batch_size, ...):
    """Create train/val DataLoaders with pad_collate_fn."""
```

### Training

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        class_weights: Optional[Dict[int, float]] = None,
        learning_rate: float = 0.001,
        patience: int = 15,
        checkpoint_dir: Path = Path('checkpoints'),
        device: str = None,
    ): ...

    def train(self, num_epochs: int, verbose: bool = True) -> TrainingHistory:
        """Full training loop with early stopping and checkpointing."""

    def train_epoch(self) -> tuple[float, float]:
        """Single training epoch. Returns (loss, accuracy)."""

    def validate(self) -> tuple[float, float, float, float, float]:
        """Validation. Returns (loss, accuracy, precision, recall, f1)."""
```

### Configuration

```python
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 15          # Early stopping patience
    min_delta: float = 0.001
    use_class_weights: bool = True
    dropout_rate: float = 0.5
    checkpoint_dir: Path = Path('checkpoints')
    seed: int = 42
```

## Data Model

### Input Format

- **Format:** Grayscale spectrogram patches (PNG, mode 'L')
- **Shape:** Variable (H, W) — typically 32-256 height x 50-200 width
- **Normalization:** Per-image min-max scaling to [0.0, 1.0]
- **Tensor shape:** (batch, 1, H, W)

### Output

- **Training:** Logits (raw) for `BCEWithLogitsLoss`
- **Inference:** Probabilities [0.0, 1.0] via `predict_proba()`
- **Binary:** {0, 1} via `predict()` at configurable threshold

### Labels

- `1.0` = USV, `0.0` = Not USV
- CSV columns: `candidate_id`, `spectrogram_path`, `label` (string "USV" or "Not USV")

## Model Artifacts (ADR-009)

Checkpoints saved as PyTorch `.pt` files:

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': optimizer_dict,
    'scheduler_state_dict': scheduler_dict,
    'metrics': {'val_loss': float, 'val_acc': float, 'val_f1': float}
}
```

| Location | Purpose |
|----------|---------|
| `checkpoints/best_model.pt` | Lowest validation loss (auto-selected) |
| `checkpoints/final_model.pt` | Last epoch before early stopping |
| `checkpoints/training_history.json` | Loss/metrics curves |
| `models/full_retrained_cnn/best_model.pt` | Production model |

**Loading a trained model:**

```python
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
model = USVClassifierCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Usage Examples

### Training

```bash
# Small model (recommended for 2K-10K samples)
python scripts/train_cnn.py \
    --train-csv splits/train.csv \
    --val-csv splits/val.csv \
    --model-size small \
    --num-epochs 100 \
    --use-class-weights

# Medium model (10K-20K samples)
python scripts/train_cnn.py --model-size medium
```

### Inference (standalone)

```python
model = USVClassifierCNN()
checkpoint = torch.load('best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Single patch
patch = torch.randn(1, 1, 64, 128)  # (batch=1, channels=1, H, W)
prob = model.predict_proba(patch)     # tensor([[0.87]])
label = model.predict(patch)          # tensor([[1.]])  (threshold=0.05)
```

### Inference in Detection App (Sliding Window)

The detection app uses `SlidingInference` for full-spectrogram scanning:

```python
from usv_spectrogram.app.core.sliding_inference import SlidingInference

engine = SlidingInference(
    model_path="models/full_retrained_cnn/best_model.pt",
    window_width_px=100,     # ~43ms window
    hop_px=10,               # 10-pixel stride
    batch_size=32,
)
result = engine.run(spectrogram_db, times, frequencies)
# result.probabilities: (n_windows,) array of [0, 1] scores
# result.times: (n_windows,) array of time values in seconds
```

Probabilities then feed into `HysteresisDetector` for final detection boundaries.

## Key Decisions

1. **3 conv blocks, ~101K params** (ADR-006) — balances expressiveness against overfitting on small datasets. Scaling configs (medium, large) available when data grows.
2. **Class weighting 3.0x for USV class** (ADR-005) — biases toward recall; false positives are cheaper than missed USVs in a research context.
3. **Three-source negative sampling** (ADR-008) — trains on random chunks, inter-USV gaps, and low-energy regions to prevent classifier blindness to "no USV" cases.
4. **Global average pooling** — handles variable input sizes without fixed dimensions, enabling flexible spectrogram patch extraction.
5. **Logit output + BCEWithLogitsLoss** — numerically more stable than sigmoid + BCELoss.
6. **Optimal threshold 0.05** (ADR-003) — calibrated from full evaluation sweep. Very low because the model is biased toward recall.

## Integration Points

### Consumes
- Spectrogram patches (PNG) from `SpectrogramExtractor`
- Labels from `LabelStorage` / labeling tools
- Split CSVs from `dataset/splits.py`

### Consumed By
- `app/core/sliding_inference.py` — sliding window inference in detection app
- `app/core/detection_logic.py` — hysteresis detection converts probabilities to boundaries
- `scripts/train_cnn.py` — training CLI
- `scripts/evaluate_model.py` — evaluation CLI
