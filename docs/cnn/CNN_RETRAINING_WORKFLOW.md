# Full CNN Retraining Workflow

## Overview

This workflow addresses the false positive issue where random chunks get 0.997 probability (should be <0.20) while maintaining USV recall >0.85.

**Strategy:** Add 1000 diverse negatives with 3.0x class weight boost

---

## Execution Steps

### Phase 1: Generate Comprehensive Negatives (~5-10 min)

```powershell
.\.venv\Scripts\python.exe scripts/generate_comprehensive_negatives.py `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir data/comprehensive_negatives `
    --n-random 500 `
    --n-inter-usv 300 `
    --n-low-energy 200 `
    --seed 42
```

**Expected Output:**
- `data/comprehensive_negatives/` - 1000 spectrogram PNGs
- `comprehensive_negatives_metadata.csv` - CSV with sample_type field
- Validation output showing dimensions (height=256, width 100-800px)

**Verify:**
- All 1000 samples generated
- Width range reasonable (median ~512px)
- No dimension warnings
- All labels are "Not USV"

---

### Phase 2: Create Full Training Dataset (~2-3 min)

```powershell
.\.venv\Scripts\python.exe scripts/create_full_training_dataset.py `
    --original-train spectrograms_training/train.csv `
    --original-val spectrograms_training/val.csv `
    --original-test spectrograms_training/test.csv `
    --negatives-csv data/comprehensive_negatives/comprehensive_negatives_metadata.csv `
    --negatives-dir data/comprehensive_negatives `
    --output-dir data/full_training_dataset `
    --seed 42
```

**Expected Output:**
- `data/full_training_dataset/train.csv` - 1852 samples (376 USV, 1476 Not USV)
- `data/full_training_dataset/val.csv` - ~280 samples (unchanged)
- `data/full_training_dataset/test.csv` - ~290 samples (unchanged)
- `data/full_training_dataset/spectrograms/` - All PNGs in one directory
- `data/full_training_dataset/class_weights.csv` - pos_weight ≈ 11.8

**Verify:**
- Train dataset: 20.3% USV, 79.7% Not USV
- pos_weight ≈ 11.8 (3.0x boost applied)
- Val/test unchanged

---

### Phase 3: Train Model (~15-30 min CPU, faster GPU)

```powershell
.\.venv\Scripts\python.exe scripts/train_cnn.py `
    --train-csv data/full_training_dataset/train.csv `
    --val-csv data/full_training_dataset/val.csv `
    --batch-size 32 `
    --num-epochs 50 `
    --patience 15 `
    --use-class-weights `
    --output-dir models/full_retrained_cnn
```

**Monitor:**
- Train loss decreasing steadily
- Val loss decreasing (may plateau around epoch 20-35)
- Val F1 improving
- Train-val gap <15%

**Expected Output:**
- `models/full_retrained_cnn/best_model.pt` - Best model by val F1
- `models/full_retrained_cnn/training_history.json` - Loss curves
- Early stopping likely around epoch 25-35

---

### Phase 4: Evaluate Model (~5-10 min)

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_experiment.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --wav-dir "5970 USV" `
    --labels-csv labels.csv `
    --output-dir analysis/full_retrained_evaluation
```

**Expected Output:**
- Three-panel histogram: USV samples, Not USV samples, Random chunks
- `experiment_metrics.json` - Mean probabilities
- `verdict.txt` - SUCCESS/PARTIAL/FAILED

**Success Criteria:**
- ✓ Random chunks: <0.20 (vs 0.997 baseline)
- ✓ USV samples: >0.85 (vs 0.624 experiment)
- ✓ Not USV samples: <0.30

---

### Phase 5: Optimize Threshold (~2-3 min)

```powershell
.\.venv\Scripts\python.exe scripts/optimize_threshold.py `
    --model models/full_retrained_cnn/best_model.pt `
    --test-csv data/full_training_dataset/test.csv `
    --output-dir analysis/threshold_optimization `
    --target-recall 0.90
```

**Expected Output:**
- `threshold_optimization.png` - Two-panel plot (metrics vs threshold, PR curve)
- `threshold_results.csv` - Full metrics table
- `recommended_threshold.txt` - Summary with recommendations

**Use Results:**
- Update `USVClassifierCNN.optimal_threshold` with recommended value
- Use in batch detection scripts
- Use in PyQt6 app default threshold

---

## Troubleshooting

### If USV recall is still low (<0.80):

1. **Increase class weight boost:**
   ```powershell
   # Rerun Phase 2 with higher boost
   python scripts/create_full_training_dataset.py ... --usv-boost 4.0
   # Then retrain (Phase 3)
   ```

2. **Use threshold optimization:**
   - Find high-recall threshold from Phase 5
   - May sacrifice some precision for better recall

### If random chunk probability is still high (>0.30):

1. **Add more random negatives:**
   ```powershell
   # Generate additional 500 random negatives
   python scripts/generate_comprehensive_negatives.py ... --n-random 1000
   ```

2. **Check spectrogram consistency:**
   - Verify render_mode="training" used
   - Compare training spectrograms to evaluation spectrograms visually

### If training is unstable:

1. **Reduce learning rate:**
   ```powershell
   python scripts/train_cnn.py ... --learning-rate 0.0005
   ```

2. **Increase batch size (if GPU memory allows):**
   ```powershell
   python scripts/train_cnn.py ... --batch-size 64
   ```

---

## Expected Timeline

| Phase | Time | Description |
|-------|------|-------------|
| 1 | 5-10 min | Generate 1000 negatives |
| 2 | 2-3 min | Create unified dataset |
| 3 | 15-30 min | Train model (CPU) |
| 4 | 5-10 min | Evaluate performance |
| 5 | 2-3 min | Optimize threshold |
| **Total** | **30-55 min** | Full pipeline |

---

## Files Created

```
scripts/
├── generate_comprehensive_negatives.py    (~700 lines)
├── create_full_training_dataset.py        (~400 lines)
└── optimize_threshold.py                  (~400 lines)

data/
├── comprehensive_negatives/
│   ├── neg_random_00000.png ... neg_random_00499.png
│   ├── neg_inter_usv_gap_00000.png ... neg_inter_usv_gap_00299.png
│   ├── neg_low_energy_00000.png ... neg_low_energy_00199.png
│   └── comprehensive_negatives_metadata.csv
└── full_training_dataset/
    ├── spectrograms/                      (~1900 PNGs)
    ├── train.csv                          (1852 samples)
    ├── val.csv                            (~280 samples)
    ├── test.csv                           (~290 samples)
    └── class_weights.csv                  (pos_weight ≈ 11.8)

models/
└── full_retrained_cnn/
    ├── best_model.pt
    └── training_history.json

analysis/
├── full_retrained_evaluation/
│   ├── evaluation_results.png
│   ├── experiment_metrics.json
│   └── verdict.txt
└── threshold_optimization/
    ├── threshold_optimization.png
    ├── threshold_results.csv
    └── recommended_threshold.txt
```

---

## Critical Consistency Checklist

Before running, verify:

- [x] SpectrogramExtractor used (NOT scipy.signal.spectrogram)
- [x] render_mode="training" (NOT "review")
- [x] ExtractionConfig: sr=300000, n_fft=512, hop=128
- [x] Frequency range: 20-120kHz
- [x] Colormap: magma
- [x] Dynamic range: MAD
- [x] Labels: "Not USV" (NOT "noise")
- [x] 3.0x class weight boost applied
- [x] Per-image normalization in threshold optimization

---

## Next Steps After Success

1. **Update model in production:**
   - Replace `checkpoints/best_model.pt` with `models/full_retrained_cnn/best_model.pt`
   - Update `USVClassifierCNN.optimal_threshold` with recommended value

2. **Update batch detection scripts:**
   - Use optimized threshold from Phase 5
   - Test on new data to verify false positive rate

3. **Update PyQt6 app:**
   - Use new model path
   - Update default threshold settings

4. **Document performance:**
   - Add results to IMPLEMENTATION_PROGRESS.md
   - Compare before/after metrics
   - Note any threshold adjustments needed

---

## Reference Documents

- `FULL_CNN_RETRAINING_PLAN.md` - Original plan (NOTE: Uses scipy, we use SpectrogramExtractor)
- `CNN_RETRAINING_EXPERIMENT_PLAN.md` - Session 18 experiment plan
- `IMPLEMENTATION_PROGRESS.md` - Session 19 entry
- `usv_signal_processing_reference.md` - STFT parameter reference
