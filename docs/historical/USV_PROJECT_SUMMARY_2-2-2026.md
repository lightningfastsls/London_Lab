# USV Detection Pipeline - Project Summary

## Project Overview

**Goal:** Build a pipeline to detect and classify mouse ultrasonic vocalizations (USVs) from 300 kHz audio recordings, comparing wild vs. lab mice courtship behavior. Two mice per cage means overlapping calls.

**Current Status:** Detection pipeline complete and working. Ready for batch detection and clustering exploration.

---

## What Was Built

### 1. Signal Processing Foundation
- Spectrograms: 2D time-frequency representations using STFT
- Parameters: n_fft=512, hop_length=128, freq_range=25-110 kHz
- Dynamic normalization: vmin/vmax based on per-recording statistics
- Colormap: magma (dark background, bright USVs)

### 2. CNN Binary Classifier (USV vs. Not USV)

**Architecture:**
- 3 conv blocks (32→64→128 filters) with BatchNorm and MaxPool
- GlobalAveragePooling (handles variable-size spectrograms)
- Dense(64) → Dropout(0.5) → Dense(1) → Sigmoid
- ~101,000 parameters

**Final Model Performance:**
| Metric | Value |
|--------|-------|
| Precision | 89.7% |
| Recall | 93.8% |
| F1 Score | 91.7% |
| Optimal Threshold | 0.05 |

**Critical Discovery:** Original CNN was trained only on energy-detector candidates, so it thought everything was a USV (0.997 probability on random audio). Fixed by retraining with comprehensive negative samples.

### 3. Training Data

**Final Dataset (~1,930 samples):**
- Original USVs: ~460
- Original Not USV (energy-detector false positives): ~470
- Random position negatives: 500
- Inter-USV gap negatives: 300
- Low-energy region negatives: 200

**Key Principles:**
- Split by recording (not by candidate) to prevent data leakage
- Class weights (3.0× for USV) to protect recall
- Three types of negatives for robust "no USV" learning

---

## Key Technical Lessons Learned

### Probability Compression
The model outputs probabilities in compressed range (0.00-0.74) due to class weighting. This is fine—what matters is separation between classes, not absolute values. USV mean (0.742) vs non-USV mean (0.057) = 0.68 gap = excellent separation.

### Recording-Level Variance
Different recordings have different characteristics (noise floor, microphone, etc.). Per-spectrogram normalization helps but doesn't fully eliminate this. Test on diverse recordings.

### Training Data Bias
A model only learns what it sees. Original CNN never saw random audio as negative, so it couldn't recognize "no USV." Adding diverse negatives fixed this completely.

---

## Project Structure

```
mickey_london_lab/
├── src/usv_spectrogram/
│   ├── models/
│   │   ├── cnn_classifier.py      # CNN architecture
│   │   ├── data_loader.py         # Dataset and DataLoader
│   │   └── trainer.py             # Training loop
│   ├── detection/
│   │   └── energy_detector.py     # Initial candidate detection
│   └── clustering/                 # (To be implemented)
│       ├── feature_extractor.py
│       ├── visualize.py
│       └── cluster.py
├── scripts/
│   ├── train_cnn.py
│   ├── evaluate_experiment.py
│   ├── generate_comprehensive_negatives.py
│   ├── optimize_threshold.py
│   └── batch_detect_for_clustering.py
├── models/
│   └── full_retrained_cnn/
│       └── best_model.pt          # Production model (threshold=0.05)
├── data/
│   ├── full_training_dataset/     # Final training data
│   └── comprehensive_negatives/   # Generated negative samples
└── analysis/
    └── threshold_optimization/    # Threshold analysis results
```

---

## Pending Work

### 1. Unsupervised USV Clustering
**Status:** Plan created, not yet implemented

**Approach:**
1. Use CNN as feature extractor (remove classification head)
2. Extract features from all detected USVs
3. Visualize with t-SNE/UMAP
4. Cluster with k-means and HDBSCAN
5. Compare cluster distributions between recordings/populations

**Purpose:** Discover natural USV types without predefined categories. May reveal wild vs. lab mouse differences.

**Document:** `USV_CLUSTERING_EXPLORATION_PLAN.md`

### 2. Batch Detection for Full Recordings
**Status:** Now viable with retrained model

**Command:**
```powershell
python scripts/batch_detect_for_clustering.py --threshold 0.05
```

---

## Important Files to Reference

| File | Purpose |
|------|---------|
| `usv_signal_processing_reference.md` | Technical subtleties (20+ items) |
| `USV_DETECTION_IMPLEMENTATION_PLAN.md` | Original 5-stage pipeline plan |
| `CNN_IMPLEMENTATION_INSTRUCTIONS.md` | CNN architecture and training details |
| `USV_DETECTION_APP_IMPLEMENTATION.md` | PyQt6 app implementation plan |
| `USV_CLUSTERING_EXPLORATION_PLAN.md` | Unsupervised clustering plan |
| `FULL_CNN_RETRAINING_PLAN.md` | How the final model was trained |

---

## Commands Cheat Sheet

```powershell
# Train CNN
python scripts/train_cnn.py `
    --train-csv data/full_training_dataset/train.csv `
    --val-csv data/full_training_dataset/val.csv `
    --spectrogram-dir data/full_training_dataset/spectrograms `
    --output-dir models/new_model `
    --num-epochs 50 --patience 15 --use-class-weights

# Evaluate model
python scripts/evaluate_experiment.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir analysis/evaluation

# Optimize threshold
python scripts/optimize_threshold.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --spec-dir data/full_training_dataset/spectrograms `
    --output-dir analysis/threshold_optimization

# Generate negative samples
python scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/comprehensive_negatives `
    --n-random 500 --n-inter-usv 300 --n-low-energy 200

# Run app (when implemented)
python scripts/run_app.py models/full_retrained_cnn/best_model.pt
```

---

## User Preferences (Shachar)

- Works in PyCharm
- Prefers learning over speed, then performance optimization
- Uses Claude Code and codex for implementation
- Has access to compute cluster if needed
- Can scale to 50000+ labels if necessary
- Primarily working with wild mice (two types), not lab mice

---

## Scientific Context

**Research Question:** How do wild mouse USV patterns compare to lab mice during courtship?

**Key Insight from User:** USVs might not be meaningful as isolated units—they could be like phonemes, meaningful only in sequence. This suggests future work on sequence models (RNN/Transformer) to analyze "conversations," not just individual calls.

**Current Approach:** Binary detection first (find USVs), then clustering (discover types), potentially sequence analysis later.

---

## Starting a New Chat

Copy this to start a new conversation with full context:

> "I'm working on a USV (Ultrasonic Vocalization) detection pipeline for mouse courtship behavior research. Here's where I am:
> 
> **Completed:**
> - CNN binary classifier (USV vs. Not USV) trained and working
> - 89.7% precision, 93.8% recall, optimal threshold 0.05
> - Fixed training data bias by adding diverse negative samples
> - Model can now do batch detection on full recordings
> - PyQt6 visualization app (plan exists in USV_DETECTION_APP_IMPLEMENTATION.md)
> 
> **Pending:**
> - Unsupervised clustering to discover USV types (plan exists in USV_CLUSTERING_EXPLORATION_PLAN.md)
> 
> **Key technical details:**
> - 300 kHz sample rate, 25-110 kHz frequency range
> - Spectrograms: n_fft=512, hop_length=128, magma colormap
> - CNN: 3 conv blocks + GlobalAveragePooling + Dense
> - Production model: models/full_retrained_cnn/best_model.pt
> 
> I want to work on [SPECIFY NEXT TASK]."

---

## Version History

- **Initial:** Energy detector + manual labeling + CNN classifier
- **Problem:** CNN only trained on energy-detector candidates → 99.7% false positive on random audio
- **Solution:** Retrained with 1000 diverse negatives (random, inter-USV gaps, low-energy)
- **Final:** Production model with 0.05 threshold, ready for batch detection
