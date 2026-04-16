# Handoff: Transformer Training — Deploy to GPU Rig

**Date:** 2026-04-14
**Status:** Data prepared, code synced, stuck on file transfer to GPU rig

## What was done this session

### 1. Fixed 3 training pipeline bugs
**File:** `usv_language/training/train_transformer.py`
- **Bug #1:** `load_bout_spectrograms()` glob changed from `*_bout*.npy` to `*.npy` + dual naming for rec_id (line ~206)
- **Bug #2:** `build_optimizer()` now uses module-type check instead of name-based for LayerNorm weight decay (line ~299)
- **Bug #3:** `masked_mse_loss()` returns `(predictions * 0).sum()` instead of detached `torch.tensor(0.0)` for all-padding batches (line ~167)

All 341 tests pass.

### 2. Added `--detection-csv` to prepare_data.py
**File:** `usv_language/data/prepare_data.py`
- Added `--detection-csv` CLI option as alternative to `--detection-dir`
- Routes to existing `BoutExtractor.extract_from_csv()` method
- Needed because batch detection results are flat JSONs, not subdirectories

### 3. Generated training CSV from batch detections
**File:** `results/batch_5970/detections_for_training.csv`
- Converted 6,400 batch detection JSONs → CSV with columns `wav_file, start_time_s, end_time_s`
- 8,036 detections from 1,414 recordings

### 4. Ran data preparation successfully
**Command used:**
```bash
.venv/bin/python -m usv_language.data.prepare_data \
    --detection-csv results/batch_5970/detections_for_training.csv \
    --wav-dir 5970/ \
    --output-dir usv_language/prepared_data \
    -v
```

**Output:** `usv_language/prepared_data/` (3.0 GB)
- 1,658 bout spectrograms from 1,414 recordings
- train: 1,328 / val: 169 / test: 161
- `normalization_stats.npz` computed from training set
- `pipeline_config.json` with all parameters

### 5. Fixed lazy import in `__init__.py`
**File:** `usv_language/data/__init__.py`
- Made `BoutSpectrogramConfig` and `compute_bout_spectrogram` lazy-loaded via `__getattr__`
- This avoids pulling in `soundfile`, `scipy`, `src.usv_spectrogram._stft_core` when only the training path is needed
- Without this fix, `import usv_language.data` fails on the GPU rig because those packages aren't installed there

## Current state — where you're stuck

The user is trying to deploy `usv_language/` to a GPU rig for training. The rig details:
- **SSH:** `ssh shachar@100.113.224.57` (Tailscale IP)
- **Port forward variant:** `ssh -L 9090:localhost:9090 shachar@100.113.224.57`
- **Target path:** `/opt/mickey_london_lab/`
- **OS:** Debian/Ubuntu with Python 3.12
- **Venv:** `/opt/mickey_london_lab/.venv/` (already created and activated)
- **Installed:** torch, numpy, h5py (already installed in venv)

### What's been transferred
The initial `rsync` of `usv_language/` to the rig completed (the pip install and module import worked), BUT the `__init__.py` lazy-import fix (step 5 above) hasn't been synced yet. The user was having trouble with `scp` — the command kept getting split across lines in their terminal.

### What needs to happen

1. **Sync the fixed `__init__.py`** from local to rig:
   ```bash
   scp /home/shachar/projects/mickey_london_lab/usv_language/data/__init__.py shachar@100.113.224.57:/opt/mickey_london_lab/usv_language/data/__init__.py
   ```
   (Must be run from local machine, all on ONE line)

2. **Run sanity check on rig** (5 epochs):
   ```bash
   ssh shachar@100.113.224.57
   cd /opt/mickey_london_lab
   source .venv/bin/activate
   python -m usv_language.training.train_transformer \
       --data-dir usv_language/prepared_data \
       --output-dir usv_language/checkpoints \
       --epochs 5
   ```

3. **If sanity check passes, full training** (200 epochs):
   ```bash
   python -m usv_language.training.train_transformer \
       --data-dir usv_language/prepared_data \
       --output-dir usv_language/checkpoints \
       --epochs 200
   ```

### Model details
- **Architecture:** Autoregressive spectrogram transformer, pre-norm, 8 layers
- **Parameters:** 25.7M (512 d_model, 2048 d_ffn, 8 heads, 170 freq bins)
- **Task:** Next-column prediction on bout spectrograms
- **Loss:** Masked MSE (handles variable-length sequences via padding mask)
- **Optimizer:** AdamW with parameter groups (no weight decay on LayerNorm/bias/embedding)
- **Training script auto-detects CUDA** — no flags needed for GPU

### After training completes
Copy checkpoints back to local:
```bash
scp -r shachar@100.113.224.57:/opt/mickey_london_lab/usv_language/checkpoints /home/shachar/projects/mickey_london_lab/usv_language/
```
