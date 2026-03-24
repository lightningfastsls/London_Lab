# CNN Classifier Quick Start Guide

## Overview

Binary CNN classifier for detecting USVs in spectrogram images. Trained on 1,047 labeled spectrograms from the USV detection pipeline.

---

## Installation

First, install PyTorch and ML dependencies:

```powershell
.\.venv\Scripts\pip.exe install torch torchvision pandas pillow scikit-learn
```

---

## Dataset

The dataset is already prepared and split:
- **Train**: 740 samples (433 USV / 307 Not USV)
- **Val**: 178 samples (108 USV / 70 Not USV)
- **Test**: 129 samples (68 USV / 61 Not USV)

Files:
- `splits/train.csv` - Training split
- `splits/val.csv` - Validation split
- `splits/test.csv` - Test split (evaluate ONLY ONCE after final training)

---

## Training

### Basic Training (Quick Test)

Run a 10-epoch test to verify everything works:

```powershell
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv splits/train.csv `
    --val-csv splits/val.csv `
    --num-epochs 10 `
    --output-dir models/test_run
```

### Full Training (Production)

Train with early stopping and class weighting:

```powershell
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv splits/train.csv `
    --val-csv splits/val.csv `
    --num-epochs 100 `
    --patience 15 `
    --use-class-weights `
    --output-dir models/production
```

Training will stop early if validation loss doesn't improve for 15 epochs. The best model (lowest validation loss) is saved to `models/production/best_model.pt`.

### Training Options

```powershell
--batch-size 16              # Batch size (default: 16)
--learning-rate 0.001        # Initial LR (default: 0.001)
--num-epochs 100             # Max epochs (default: 100)
--patience 15                # Early stopping patience (default: 15)
--dropout-rate 0.5           # Dropout in classifier head (default: 0.5)
--use-class-weights          # Enable class weighting for imbalance
--normalize-mode per_image   # Normalization: per_image or global
--output-dir checkpoints/    # Where to save models
--device cuda                # Force device (default: auto)
--seed 42                    # Random seed (default: 42)
```

### Outputs

After training, you'll find:
- `best_model.pt` - Best model checkpoint (lowest val loss)
- `final_model.pt` - Final model checkpoint (last epoch)
- `training_history.json` - Training metrics per epoch

---

## Evaluation

Evaluate the trained model on the test set:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py `
    --model models/production/best_model.pt `
    --test-csv splits/test.csv `
    --output-dir models/production
```

**IMPORTANT**: Only evaluate on the test set ONCE after final training to avoid overfitting to test data.

### Outputs

- `test_metrics.json` - Accuracy, precision, recall, F1, confusion matrix
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve with AUC
- `precision_recall_curve.png` - PR curve with AUC

---

## Inference

### Single Image Prediction

```powershell
.\.venv\Scripts\python.exe scripts/predict.py `
    --model models/production/best_model.pt `
    --image path/to/spectrogram.png
```

Output:
```
Probability: 0.9823
Prediction: USV
```

### Batch Prediction

Predict on all candidates from a CSV:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py `
    --model models/production/best_model.pt `
    --csv candidates.csv `
    --output predictions.csv
```

The CSV must have a `spectrogram_path` column. Output CSV will have:
- `probability` - USV probability (0 to 1)
- `prediction` - Binary prediction (0 or 1)
- `predicted_label` - Label string ('USV' or 'Not USV')

---

## Expected Performance

With ~1,047 samples, expect:

| Metric | Target | Acceptable | Red Flag |
|--------|--------|------------|----------|
| Val Accuracy | 85-90% | 80-85% | <75% or >95% |
| Val Precision | >85% | >75% | <70% |
| Val Recall | >85% | >75% | <70% |
| Test Accuracy | 80-85% | 75-80% | <70% |
| Train-Val Gap | <5% | <10% | >15% (overfitting) |

**Red Flag Diagnosis:**
- **<75% accuracy** → Check data quality, verify label correctness
- **>95% accuracy** → Data leakage! Verify recordings not in multiple splits
- **>15% train-val gap** → Overfitting, increase dropout or reduce model size
- **Low recall (<70%)** → Class weighting not working, check pos_weight

---

## Model Architecture

**USVClassifierCNN (Small)**
- **Use for**: 1,000-5,000 samples
- **Filters**: [32, 64, 128]
- **Parameters**: ~90K
- **Architecture**:
  ```
  Conv(1→32) + BN + ReLU + MaxPool
  Conv(32→64) + BN + ReLU + MaxPool
  Conv(64→128) + BN + ReLU + MaxPool
  GlobalAvgPool
  Linear(128→64) + ReLU + Dropout(0.5)
  Linear(64→1)  # Logits
  ```

**USVClassifierCNNLarge**
- **Use for**: 5,000+ samples (will overfit on small datasets!)
- **Filters**: [32, 64, 128, 256, 512]
- **Parameters**: ~500K

---

## Troubleshooting

### Training crashes with OOM
- Reduce batch size: `--batch-size 8`
- Or use CPU: `--device cpu` (slower)

### Validation accuracy stuck at ~50%
- Model is predicting random / all same class
- Check class weights: should see `pos_weight` printed at start
- Try without class weights first to debug

### Loss is NaN
- Learning rate too high, reduce: `--learning-rate 0.0001`
- Check for corrupted spectrograms in dataset

### Model not learning (accuracy ~55-60%)
- Increase epochs: `--num-epochs 200`
- Reduce learning rate: `--learning-rate 0.0005`
- Try different normalization: `--normalize-mode global`

---

## Integration with Detection Pipeline

After training, integrate the classifier into the detection pipeline:

1. **Stage 1**: Energy detector generates candidates (existing `run_detection.py`)
2. **Stage 2**: CNN classifier filters candidates → This model!
3. **Stage 3**: Human review of high-confidence detections

Example workflow:
```powershell
# Step 1: Run energy detector
python scripts/run_detection.py --wav-dir "5970 USV" --output candidates_raw.csv

# Step 2: Extract spectrograms
python scripts/extract_spectrograms.py --candidates candidates_raw.csv --output-dir spectrograms_raw/

# Step 3: Run CNN classifier
python scripts/predict.py --model models/production/best_model.pt --csv candidates_raw.csv --output candidates_filtered.csv

# Step 4: Keep high-confidence USVs (e.g., probability > 0.8)
# Filter candidates_filtered.csv where probability > 0.8
```

---

## Files Reference

**Source Code:**
- `src/usv_spectrogram/models/config.py` - TrainingConfig dataclass
- `src/usv_spectrogram/models/data_loader.py` - Dataset and data loaders
- `src/usv_spectrogram/models/cnn_classifier.py` - CNN model architecture
- `src/usv_spectrogram/models/trainer.py` - Training loop with early stopping
- `src/usv_spectrogram/models/evaluate.py` - Evaluation metrics and plots

**Scripts:**
- `scripts/train_cnn.py` - Training CLI
- `scripts/evaluate_model.py` - Evaluation CLI
- `scripts/predict.py` - Inference CLI

**Data:**
- `splits/train.csv` - Training split (740 samples)
- `splits/val.csv` - Validation split (178 samples)
- `splits/test.csv` - Test split (129 samples)

---

## Next Steps

1. **Install dependencies** (see Installation section)
2. **Run quick test** (10 epochs) to verify setup
3. **Full training** (50-100 epochs with early stopping)
4. **Evaluate on test set** (ONLY ONCE)
5. **Document performance** in `IMPLEMENTATION_PROGRESS.md`
6. **Integrate into pipeline** for Stage 2 filtering
