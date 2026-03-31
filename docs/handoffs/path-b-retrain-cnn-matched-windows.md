# Handoff: Path B — Retrain CNN with Matched Training/Inference Windows

**Date:** 2026-03-25
**From:** Session working on ROC curve pipeline (Path A)
**Priority:** Medium (Path A provides immediate value; this is the proper long-term fix)

## Problem

The production CNN (`models/production/best_model.pt`, trained 2026-02-02) has a **training/inference window mismatch**:

| Stage | Window size | Image width |
|-------|------------|-------------|
| **Training positives** | 105–135ms (USV + 50ms×2 context) | 210–270px |
| **Training negatives** (comprehensive) | 40ms (clamped) | 128px |
| **Inference** (`SlidingInference`) | 42.7ms (100 STFT columns) | 100px → padded to 512px |

This causes **probability compression** — the model never outputs above ~0.58 on real USVs because it never saw a USV in only 100px during training. The app compensates with absurdly low hysteresis thresholds (0.04/0.03).

**Full evidence:** `docs/reports/cnn-training-window-provenance.md`

## Goal

Retrain the CNN so training images match inference window size. This should:
- Eliminate probability compression → probabilities span full 0–1 range
- Allow a normal decision threshold (~0.5) instead of 0.04
- Improve discrimination (fewer false positives at any given recall)

## Three Options (pick one)

### Option A: Train on 42.7ms windows to match current SlidingInference (Recommended)
- Modify `DatasetAssembler` config: `jitter_window_ms=42.7`, `jitter_context_padding_ms=0.0`
- Or: set context so total = 42.7ms (e.g., window=30ms + 6.35ms padding each side)
- Training images will be 100px wide — exactly what `SlidingInference` feeds
- **No changes to inference code needed**

### Option B: Change SlidingInference to match training
- Keep assembler at 40ms + 20ms padding = 80ms (160px)
- Update `SlidingInference.window_width_px` from 100 to ~188 (80ms / 0.4267ms per column)
- Requires changes to `src/usv_spectrogram/app/core/sliding_inference.py`
- Wider window = fewer inference steps = faster, but less temporal resolution

### Option C: Compromise — assembler's 80ms + update SlidingInference
- Use existing `DatasetAssembler` config as-is (40ms + 20ms×2 = 80ms)
- Update `SlidingInference.window_width_px` to 188
- Least code change for training pipeline, moderate change to inference

## Key Files

| File | Role |
|------|------|
| `src/usv_spectrogram/dataset/assembler.py` | `DatasetAssembler` — generates training spectrograms with jitter |
| `scripts/run_training_cycle.py` | End-to-end: assemble → split → train → evaluate |
| `scripts/train_cnn.py` | CNN training script |
| `src/usv_spectrogram/models/cnn_classifier.py` | `USVClassifierCNN` architecture (3 conv blocks, GlobalAvgPool) |
| `src/usv_spectrogram/models/data_loader.py` | `USVDataset` + `pad_collate_fn` |
| `src/usv_spectrogram/app/core/sliding_inference.py` | `SlidingInference` — inference-time window extraction |
| `src/usv_spectrogram/detection/spectrogram_extractor.py` | Generates training-mode PNGs |
| `docs/reports/cnn-training-window-provenance.md` | Full investigation of current mismatch |

## Existing Infrastructure

The `DatasetAssembler` + `run_training_cycle.py` pipeline already exists but has never produced a model that replaced production. The pipeline:

1. `DatasetAssembler.assemble()` — extracts spectrograms from labeled candidates with jitter
2. `scripts/split_dataset.py` — stratified train/val/test split
3. `scripts/train_cnn.py` — trains CNN with class weights, early stopping
4. `scripts/generate_roc_curve.py` — evaluates with ROC/PR curves (built in current session, not yet committed)

## Labeled Data

- `splits/{train,val,test}.csv` — labeled candidates (candidate_id, final_label, source_file)
- Labels: USV, Not USV (some CSVs have `noise` which must be remapped to `Not USV`)
- `spectrograms_training/` — existing PNGs (variable width, from original extraction)
- Source WAVs: `5970 USV/` directory (300 kHz sample rate)

## Validation Criteria

After retraining:
1. **Probability range:** True USVs should score >0.7, noise should score <0.3 (no more compression)
2. **ROC AUC:** Should remain ≥0.95 (current: 0.97 on test split)
3. **Threshold table:** Youden's J optimal should be near 0.5 (currently 0.437 but on compressed probabilities)
4. **App integration:** Replace `models/production/best_model.pt`, verify `SlidingInference` works with new model
5. Run `scripts/generate_roc_curve.py --split test` to produce new curves for comparison

## Constraints

- Sample rate is always 300,000 Hz — specify `sr=300000` explicitly everywhere
- `pad_collate_fn` handles variable-width images in batches via zero-padding
- The CNN architecture (`USVClassifierCNN`) uses `AdaptiveAvgPool2d((1,1))` so it accepts any input size
- `optimal_threshold` in the model checkpoint should be updated to the new Youden's J value
- **Do NOT modify test expectations to make tests pass** — fix code instead
