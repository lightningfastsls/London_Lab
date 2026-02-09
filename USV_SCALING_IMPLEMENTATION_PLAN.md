# USV Detection Pipeline - Scaling to 30K Labels

## Implementation Plan for Claude Code

**Date**: February 5, 2026  
**Context**: Scaling USV labeling from ~2K to 30K samples to improve CNN performance and confidence calibration.

---

## Background Context

### Key Concepts Informing This Plan

1. **Positional Bias Problem**: CNNs trained on centered USVs learn a shortcut - they look for energy in the middle, not USV features. Solution: constrained jittering to create training samples with USVs at different positions while ensuring sufficient USV content remains.

2. **Model Scaling**: Match model capacity to data size. Current 101K parameter model is appropriate for ~2K samples. Scale filter counts (64→128→256) as data approaches 15-30K samples. Monitor training curves to decide when to scale.

3. **Weight Decay**: Penalizes large weights, encouraging smoother decision boundaries. Helps prevent overfitting as model size increases. Starting value: 1e-4.

4. **Label Quality Over Quantity**: Arbitrary windowing can create samples labeled "USV" that contain no USV. Constrained jittering solves this by ensuring minimum USV overlap in each generated sample.

### Current State

- CNN binary classifier: 89.7% precision, 93.8% recall, F1 91.7%
- Optimal threshold: 0.05 (app uses 0.04 high / 0.03 low due to probability compression)
- Training data: ~1,930 samples
- PyQt6 detection app: built and working
- Model location: `models/full_retrained_cnn/best_model.pt`

### Project Structure Reference

```
mickey_london_lab/
├── src/usv_spectrogram/
│   ├── app/                    # PyQt6 detection app (main labeling tool)
│   ├── models/
│   │   ├── cnn_classifier.py
│   │   ├── data_loader.py
│   │   └── trainer.py
│   ├── detection/
│   │   └── energy_detector.py
│   └── labeling/
│       └── labelling_app/      # Older labeling app with boundary adjustment
├── scripts/
│   ├── train_cnn.py
│   ├── evaluate_experiment.py
│   ├── optimize_threshold.py
│   └── generate_comprehensive_negatives.py
└── models/
    └── full_retrained_cnn/
        └── best_model.pt
```

---

## Phase 0: Verification and Baseline

**Goal**: Confirm existing functionality works before adding new features.

### Tasks

#### 0.1 Verify App Save Functions

Test the following in `src/usv_spectrogram/app/`:

1. **"Save Current View"** - Confirm it only saves detections currently visible on screen, not all detections in the file
2. **"Save All Detections"** - Confirm JSON, PNG, and CSV all generate correctly
3. **Duplicate detection prevention** - Save the same region twice (e.g., at different threshold settings), confirm no duplicate entries are created in the output files

Document any bugs found in a `BUGS.md` file.

#### 0.2 Establish Baseline Metrics

```powershell
python scripts/evaluate_experiment.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir analysis/baseline_evaluation
```

Save results as `analysis/baseline_metrics.json` containing:
- Precision, recall, F1
- Probability distribution statistics (mean, std, min, max for USV and non-USV classes)
- Threshold used
- Date of evaluation

**Output**: Verified working app, documented baseline performance.

---

## Phase 1: Add Boundary Adjustment to Detection App

**Goal**: Enable manual refinement of detection boundaries within the main PyQt6 app.

### Location

Modify files in `src/usv_spectrogram/app/`

### UI Changes

1. **Enter edit mode**: When user clicks on a detection (green shaded region in probability view or boundary line in spectrogram view), enter "edit mode" for that detection

2. **Draggable handles**: In edit mode, show draggable handles at start and end boundaries

3. **Bidirectional adjustment**: Allow dragging either boundary left OR right (not limited to one direction)

4. **Real-time feedback**: Show detection duration updating in real-time as boundaries move

5. **Confirm/cancel**: 
   - Press Enter or click elsewhere to confirm adjustment
   - Press Escape to cancel and revert to original boundaries

6. **Keyboard shortcut**: Add 'E' key to enter edit mode for currently selected/hovered detection

### Implementation Details

- Store original boundaries so user can reset if needed
- Update both spectrogram overlay (green/cyan lines) AND probability view (shaded region) simultaneously
- When saving, use the adjusted boundaries
- Add `adjusted: true` flag to detection metadata when boundaries have been manually modified
- Store both original and adjusted boundaries in metadata for traceability

### Why This Matters

Precise boundaries are needed for constrained jittering in Phase 3. Even though energy detector is generally accurate, the ability to correct occasional conservative detections improves label quality.

---

## Phase 2: Progressive Labeling Workflow

**Goal**: Implement threshold-based progressive labeling strategy for efficiency at scale.

### Location

Modify `src/usv_spectrogram/app/main_window.py` and related UI files.

### 2.1 Threshold Presets

Add preset buttons to the UI:

| Preset Name | High Threshold | Low Threshold | Use Case |
|-------------|----------------|---------------|----------|
| High Confidence | 0.10 | 0.08 | First pass - obvious USVs |
| Medium | 0.06 | 0.04 | Second pass - clear USVs |
| Low | 0.04 | 0.03 | Third pass - faint USVs |

Implementation:
- Add three buttons above or beside the threshold sliders
- Clicking a preset sets both sliders to those values
- User can still fine-tune with sliders after selecting preset
- Store presets in a config file (`app_config.json`) so they can be adjusted as model improves

### 2.2 Labeling Session Tracking

For each saved detection, store additional metadata:

```json
{
  "threshold_preset": "high_confidence",
  "threshold_high": 0.10,
  "threshold_low": 0.08,
  "session_timestamp": "2026-02-05T14:30:00",
  "session_id": "uuid-here"
}
```

This enables later analysis of detection confidence tiers.

### 2.3 Visual Indication of Saved Detections

Different visual styles for detection states:

| State | Visual Style |
|-------|--------------|
| Unsaved (current session) | Solid green shading (current behavior) |
| Saved (current session) | Solid blue shading or green with checkmark |
| Saved (previous session) | Hatched pattern or gray shading |

Load saved detection locations from tracking file on WAV open, overlay them on the probability view.

### Why This Matters

At 30K labels, workflow efficiency is critical. Visual tracking prevents re-reviewing already-labeled regions.

---

## Phase 3: Constrained Jittering for Training Data Generation

**Goal**: Generate positionally-diverse training samples while ensuring label validity.

### Location

Create new script: `scripts/generate_jittered_training_data.py`

### The Algorithm

```python
def generate_jittered_samples(detection, config):
    """
    Generate multiple training samples from one detection with USV at different positions.
    
    Parameters:
    - detection: dict with start_ms, end_ms (the USV boundaries)
    - config: dict with window_ms, min_overlap_fraction, n_samples, context_padding_ms
    
    Returns:
    - List of sample specs, each with extraction window boundaries and metadata
    """
    
    # 1. Get USV boundaries
    usv_start = detection['start_ms']
    usv_end = detection['end_ms']
    usv_duration = usv_end - usv_start
    usv_center = (usv_start + usv_end) / 2
    
    # 2. Configuration
    window_ms = config['window_ms']  # e.g., 40ms
    min_overlap_fraction = config['min_overlap_fraction']  # e.g., 0.5
    n_samples = config['n_samples']  # e.g., 5
    context_padding_ms = config['context_padding_ms']  # e.g., 20ms
    
    # 3. Calculate minimum required overlap in ms
    min_overlap_ms = usv_duration * min_overlap_fraction
    
    # 4. Calculate valid jitter range
    # When window is centered on USV center, USV is centered in window
    # Max jitter = how far we can shift before overlap drops below minimum
    
    if usv_duration >= window_ms:
        # USV longer than window - no jitter possible while maintaining overlap
        # Just use centered extraction
        max_jitter = 0
    else:
        # max_jitter = (window_ms / 2) - min_overlap_ms
        # This ensures at least min_overlap_ms of USV remains in window
        max_jitter = (window_ms / 2) - min_overlap_ms
    
    # 5. Generate jitter values
    if max_jitter <= 0:
        jitter_values = [0]  # Only centered sample
    else:
        jitter_values = np.linspace(-max_jitter, max_jitter, n_samples)
    
    # 6. Generate sample specifications
    samples = []
    for i, jitter in enumerate(jitter_values):
        window_center = usv_center + jitter
        window_start = window_center - (window_ms / 2) - context_padding_ms
        window_end = window_center + (window_ms / 2) + context_padding_ms
        
        samples.append({
            'window_start_ms': window_start,
            'window_end_ms': window_end,
            'jitter_ms': jitter,
            'jitter_index': i,
            'original_detection_id': detection['id'],
            'usv_position_in_window': 'left' if jitter > 0 else 'right' if jitter < 0 else 'center'
        })
    
    return samples
```

### Edge Cases to Handle

1. **USV longer than window**: Use centered extraction only (no jitter)
2. **USV very short** (e.g., 5ms): Large jitter range is fine - short USVs can appear anywhere
3. **Detection near recording start/end**: Clamp extraction window to valid audio range
4. **Negative window boundaries**: Shift window to start at 0 if window_start < 0

### Output Format

Directory structure:
```
data/jittered_training_data/
├── spectrograms/
│   ├── {detection_id}_jitter0.png
│   ├── {detection_id}_jitter1.png
│   └── ...
├── metadata.json        # Full metadata for all samples
└── samples_summary.csv  # Quick reference table
```

Metadata per sample:
```json
{
  "sample_id": "det001_jitter2",
  "original_detection_id": "det001",
  "source_wav": "recording_001.wav",
  "jitter_ms": 5.0,
  "window_start_ms": 1250.0,
  "window_end_ms": 1310.0,
  "usv_start_in_window_ms": 25.0,
  "usv_end_in_window_ms": 35.0,
  "label": "USV"
}
```

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window-ms` | 40 | Extraction window size (excluding padding) |
| `--context-padding-ms` | 20 | Padding added to each side |
| `--min-overlap-fraction` | 0.5 | Minimum fraction of USV that must be in window |
| `--n-samples` | 5 | Number of jittered samples per detection |
| `--input-dir` | - | Directory with saved detection JSONs |
| `--output-dir` | - | Output directory for generated samples |

### Command

```powershell
python scripts/generate_jittered_training_data.py `
    --input-dir data/labeled_detections `
    --output-dir data/jittered_training_data `
    --window-ms 40 `
    --context-padding-ms 20 `
    --min-overlap-fraction 0.5 `
    --n-samples 5
```

---

## Phase 4: Training Infrastructure Improvements

### 4A: Training Curves and Monitoring

**Location**: Modify `scripts/train_cnn.py`

#### Save Per-Epoch Metrics

After each epoch, append to a metrics list:
```json
{
  "epoch": 1,
  "train_loss": 0.452,
  "val_loss": 0.489,
  "train_acc": 0.823,
  "val_acc": 0.801,
  "learning_rate": 0.001,
  "timestamp": "2026-02-05T14:30:00"
}
```

Save as `{output_dir}/training_metrics.json` at end of training.

#### Generate Training Curve Plot

At end of training, generate matplotlib plot:
- Subplot 1: Train loss vs Val loss over epochs
- Subplot 2: Train accuracy vs Val accuracy over epochs
- Mark early stopping point if triggered
- Save as `{output_dir}/training_curves.png`

#### Create Standalone Plotting Script

Create `scripts/plot_training_curves.py`:

```powershell
python scripts/plot_training_curves.py `
    --metrics-file models/experiment_001/training_metrics.json `
    --output-file models/experiment_001/training_curves.png
```

### 4B: Weight Decay Integration

**Location**: Modify `scripts/train_cnn.py` and `src/usv_spectrogram/models/trainer.py`

Add command-line parameter:
```
--weight-decay FLOAT    Weight decay (L2 regularization) coefficient. Default: 1e-4
```

Apply to optimizer:
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay  # Add this
)
```

Log weight decay value in training metadata/config saved with model.

**Starting value**: 1e-4. Increase to 1e-3 if overfitting persists after adding more data.

### 4C: Model Scaling Preparation

**Location**: Modify `src/usv_spectrogram/models/cnn_classifier.py`

Add model size configurations:

```python
MODEL_CONFIGS = {
    "small": {
        "filters": [32, 64, 128],
        "dense_units": 64,
        "description": "~101K params, use for 2K-10K samples"
    },
    "medium": {
        "filters": [64, 128, 256],
        "dense_units": 128,
        "description": "~400K params, use for 10K-20K samples"
    },
    "large": {
        "filters": [128, 256, 512],
        "dense_units": 256,
        "description": "~1.6M params, use for 20K+ samples"
    }
}
```

Add `--model-size` parameter to training script:
```
--model-size {small,medium,large}    Model size configuration. Default: small
```

#### Scaling Decision Guidelines

Document these in the training script or README:

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Train loss AND val loss both high, plateau together | Underfitting | Scale up model size |
| Train loss low, val loss much higher and diverging | Overfitting | Add regularization (weight decay, dropout) or more data |
| Both losses low and close together | Good fit | Current size is appropriate |
| Adding data improves val loss | Model can learn more | Keep adding data, consider scaling up |
| Adding data doesn't improve val loss | Possible saturation | Check data quality, try scaling up |

#### Recommended Progression

| Label Count | Recommended Model Size |
|-------------|------------------------|
| 2K - 5K | small |
| 5K - 10K | small, monitor for underfitting |
| 10K - 15K | small or medium |
| 15K - 25K | medium |
| 25K - 30K+ | medium or large |

### 4D: Threshold Re-optimization Reminder

**IMPORTANT**: After each major retraining, re-run threshold optimization:

```powershell
python scripts/optimize_threshold.py `
    --model models/NEW_MODEL/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --spec-dir data/full_training_dataset/spectrograms `
    --output-dir analysis/threshold_optimization_NEW
```

Probability distributions WILL change with retraining. Old thresholds (0.04/0.03) may not be optimal for new model.

---

## Phase 5: Negative Sample Generation (Refined)

**Goal**: Scale negative samples proportionally with positive samples.

### Location

Modify `scripts/generate_comprehensive_negatives.py`

### Scaling Strategy

Maintain approximate ratio as positives scale:

| Positive Samples | Negative Samples | Total |
|------------------|------------------|-------|
| 2K | 1K | 3K |
| 10K | 5K | 15K |
| 20K | 10K | 30K |
| 30K | 15K | 45K |

### Negative Sample Types

Keep all three types from original approach:

1. **Random position** (50% of negatives): Arbitrary audio segments
2. **Inter-USV gaps** (30% of negatives): Regions between detected USVs - hard negatives with similar noise characteristics
3. **Low-energy regions** (20% of negatives): Ensures model doesn't learn "high energy = USV"

### New Addition: Hard Negative Mining

After initial labeling pass, identify CNN false positives:

1. Run sliding window inference on unlabeled recordings
2. Find regions where CNN probability > 0.3 but no detection was saved by human
3. Surface these for manual review
4. Confirmed false positives become hard negatives for next training round

Create `scripts/mine_hard_negatives.py`:
```powershell
python scripts/mine_hard_negatives.py `
    --model models/full_retrained_cnn/best_model.pt `
    --wav-dir "unlabeled_recordings" `
    --labeled-detections-dir data/labeled_detections `
    --output-dir data/hard_negative_candidates `
    --probability-threshold 0.3
```

---

## Phase 6: Quality Control Infrastructure

### 6A: Label Review Tool

**Location**: Create `scripts/review_labels.py`

Simple CLI tool for reviewing saved detections:

1. Load saved detections from a directory
2. Display each detection spectrogram (using matplotlib or simple image viewer)
3. User input:
   - `y` or Enter: Confirm label
   - `n`: Reject label (flag for removal)
   - `s`: Skip (uncertain)
   - `q`: Quit review session
4. Save review results to `review_results.json`
5. Summary at end: X confirmed, Y rejected, Z skipped

```powershell
python scripts/review_labels.py `
    --input-dir data/jittered_training_data/spectrograms `
    --output-file data/jittered_training_data/review_results.json
```

### 6B: Cross-Validation Sanity Check

Before major retraining:

1. Split data by recording (already implemented)
2. Compare train vs validation performance
3. If val performance is >10% worse than train, investigate:
   - Are some recordings much harder than others?
   - Are there labeling inconsistencies?

### 6C: Outlier Detection

After training, identify potential label errors:

Create `scripts/find_label_outliers.py`:

1. Load trained model
2. Run inference on training set
3. Find samples where |prediction - label| > 0.7
   - Label = 1 (USV) but model says < 0.3
   - Label = 0 (not USV) but model says > 0.7
4. Export these for manual review

```powershell
python scripts/find_label_outliers.py `
    --model models/full_retrained_cnn/best_model.pt `
    --data-csv data/full_training_dataset/train.csv `
    --spec-dir data/full_training_dataset/spectrograms `
    --output-dir analysis/label_outliers `
    --threshold 0.7
```

---

## Execution Order

### Recommended Sequence

1. **Phase 0** - Verification (do first, confirms foundation)
2. **Phase 1** - Boundary adjustment (improves label precision)
3. **Phase 4A/4B** - Training curves and weight decay (needed before any retraining)
4. **Phase 3** - Constrained jittering (addresses positional bias)
5. **Phase 2** - Progressive workflow (improves labeling efficiency)
6. **Phase 4C** - Model scaling (when approaching 10K+ samples)
7. **Phase 5** - Negative scaling (as positives grow)
8. **Phase 6** - Quality control (ongoing)

### Milestone Checkpoints

| Milestone | Samples | Actions |
|-----------|---------|---------|
| 5K labels | 5K | Retrain with weight decay, evaluate improvement |
| 10K labels | 10K | Consider medium model size, run outlier detection |
| 20K labels | 20K | Switch to medium model, full QC pass |
| 30K labels | 30K | Evaluate large model, comprehensive evaluation |

---

## Key Parameters Reference

| Parameter | Value | Context |
|-----------|-------|---------|
| Weight decay | 1e-4 | Starting point, increase if overfitting |
| Window size | 40ms | Training sample extraction |
| Context padding | 20ms | Added to each side of detection |
| Min USV overlap | 50% | For constrained jittering validity |
| Threshold presets | 0.10/0.08, 0.06/0.04, 0.04/0.03 | High/medium/low confidence |
| Model scaling | small→medium at ~10K | Monitor training curves to decide |

---

## Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `scripts/generate_jittered_training_data.py` | 3 | Positional augmentation |
| `scripts/plot_training_curves.py` | 4A | Visualize training progress |
| `scripts/mine_hard_negatives.py` | 5 | Find CNN false positives |
| `scripts/review_labels.py` | 6A | Manual QC tool |
| `scripts/find_label_outliers.py` | 6C | Find potential label errors |
| `app_config.json` | 2 | Store threshold presets |

---

## Notes for Claude Code

- User works in PyCharm on Windows
- Use PowerShell command syntax (backtick for line continuation)
- Prioritize learning and correctness over speed
- Ask for clarification on any ambiguous requirements
- Test changes incrementally - don't batch too many modifications
- The main labeling app is at `src/usv_spectrogram/app/`
- User prefers to understand concepts before implementation
