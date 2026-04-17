# HANDOFF: USV Convolutional Autoencoder Implementation

## Context
We completed a deep-read of AMVOC (Stoumpou et al. 2022) — a convolutional autoencoder pipeline for mouse USV analysis. The full technical extraction is in `AMVOC_deep_read_extraction.md` in arscontexta (or the project docs). This handoff is for building our own autoencoder informed by their design, adapted for wild-derived mouse USVs.

## What We Have
- ~7,575 detected USVs from wild-derived mice (CNN detector, ~207K params, ~90% precision, ~99% recall)
- Spectrograms already computed as part of the detection pipeline
- The USVs exist as detected segments with start/end times

## What We Need to Build
A PyTorch autoencoder training + feature extraction pipeline with these stages:
1. Extract and preprocess individual USV spectrograms from detected segments
2. Train a convolutional autoencoder (AMVOC-style, with improvements)
3. Extract bottleneck features
4. Run dimensionality reduction (PCA → UMAP)
5. Cluster in latent space
6. Visualize and compare with Didi's vectorization approach

## Discovery Tasks (Do These First)

### Task 1: Find Current Spectrogram Format
- Find where detected USV segments are stored (CSV? numpy arrays? what columns?)
- Find the spectrogram computation code — what FFT parameters are we using?
- Find what frequency range and resolution our spectrograms have
- Report: sample rate, window size, hop size, frequency range, frequency resolution
- Show me 3 example USV spectrograms with their shapes

### Task 2: Find Didi's Vectorization Code
- Search the codebase for any peak-frequency extraction, frequency contour, or vectorization code
- Find how Didi's approach represents each USV (peak frequency per column? fixed-length vector?)
- Report what you find — we need this for later comparison

### Task 3: Assess Data Scale
- Count total detected USVs available for training
- Show the distribution of USV durations (in ms or time frames)
- Determine what time_limit (in frames) would cover the mean + 1 SD of durations
- Check if 64 frames (128 ms at 2ms/frame) is appropriate for our data, or if we need a different size

## Implementation Plan

### Module 1: Data Preparation (`usv_autoencoder/data_prep.py`)

Based on AMVOC, implement:

```python
def prepare_usv_spectrograms(
    detected_segments,      # list of (start_time, end_time) or similar
    full_spectrogram,       # the recording's spectrogram matrix
    freq_bins,              # frequency axis
    f_low=30000,            # Hz — adjust if our range differs
    f_high=110000,          # Hz — adjust if our range differs
    time_limit=64,          # frames — verify with Task 3
    normalize='max'         # 'max' for AMVOC-style, could also try 'minmax'
) -> np.ndarray:
    """
    Extract, crop/pad, and normalize individual USV spectrograms.
    Returns array of shape (N, 1, time_limit, freq_bins_in_range).
    """
```

Key decisions from AMVOC:
- **Normalization:** Per-spectrogram max normalization (divide by max value → [0,1])
- **Short USVs:** Zero-pad symmetrically (center the USV)
- **Long USVs:** Center-crop (keep middle portion)
- **Channel dim:** Add channel dimension for PyTorch Conv2d: `(N, 1, T, F)`

### Module 2: Autoencoder Architecture (`usv_autoencoder/model.py`)

Build two variants:

**Variant A: AMVOC-faithful (baseline)**
- Exact AMVOC architecture: 3 conv layers (64→32→8), MaxPool2d(2,2), 3×3 kernels, same padding
- BCE loss with sigmoid output
- This is our comparison baseline

**Variant B: Improved (our primary)**
- Same overall structure but add:
  - `nn.BatchNorm2d` after each conv layer (before ReLU)
  - Optional dropout on bottleneck (e.g., 0.1)
  - Configurable bottleneck filter count (try 4, 8, 16)
- Same decoder structure (transposed convolutions)
- BCE loss

**Variant C: VAE (future — just stub the interface)**
- Same encoder but output mu and logvar instead of deterministic bottleneck
- KL divergence term added to loss
- Don't implement fully now, just define the class interface so we can swap in later

### Module 3: Training (`usv_autoencoder/train.py`)

```python
def train_autoencoder(
    model,
    train_loader,
    val_loader,           # AMVOC didn't use validation — we should
    n_epochs=20,          # higher than AMVOC's 2, but use early stopping
    lr=0.001,
    patience=5,           # early stopping patience
    device='cuda'
) -> dict:
    """
    Train with BCE loss, Adam optimizer.
    Return training history (loss curves) and best model state_dict.
    """
```

Improvements over AMVOC:
- **80/10/10 split** (train/val/test) instead of 80/20 with no validation
- **Early stopping** based on validation loss with patience=5
- **Save best model** (lowest val loss), not just final epoch
- **Log training curves** for later analysis

### Module 4: Feature Extraction (`usv_autoencoder/features.py`)

```python
def extract_features(
    model,
    data_loader,
    device='cuda'
) -> np.ndarray:
    """
    Pass all USVs through encoder, flatten bottleneck.
    Returns (N, bottleneck_dim) array.
    """

def postprocess_features(
    features,                    # (N, bottleneck_dim)
    variance_threshold_factor=1.2,  # AMVOC default
    pca_variance_ratio=0.95,        # AMVOC default
    return_intermediates=False       # for debugging
) -> np.ndarray:
    """
    AMVOC pipeline: VarianceThreshold → StandardScaler → PCA (95% variance).
    """
```

### Module 5: Visualization & Clustering (`usv_autoencoder/cluster.py`)

```python
def reduce_and_cluster(
    features,                # post-processed features from Module 4
    n_clusters=6,            # AMVOC default, but we should try range
    cluster_method='kmeans', # also try 'gmm', 'hdbscan'
    viz_method='umap',       # AMVOC used t-SNE; we prefer UMAP
    umap_n_neighbors=15,
    umap_min_dist=0.1
) -> tuple:
    """
    Returns (cluster_labels, 2d_embedding, cluster_metrics).
    """
```

### Module 6: Comparison Script (`usv_autoencoder/compare_representations.py`)

Compare three representation methods on the same USV set:
1. **AMVOC-style autoencoder features** (from Module 4)
2. **Didi's vectorization** (peak frequency per column — from Task 2)
3. **Raw flattened spectrograms** (baseline — just flatten the 64×160 input)

For each, run the same clustering pipeline and compute:
- Silhouette score
- Calinski-Harabasz score
- Visual inspection of UMAP embeddings side by side

## File Structure
```
usv_autoencoder/
├── __init__.py
├── data_prep.py          # Module 1
├── model.py              # Module 2 (all variants)
├── train.py              # Module 3
├── features.py           # Module 4
├── cluster.py            # Module 5
├── compare_representations.py  # Module 6
├── config.yaml           # All hyperparameters in one place
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_train_autoencoder.ipynb
    └── 03_compare_representations.ipynb
```

## Execution Order
1. **Discovery Tasks 1–3** (before writing any code)
2. **Module 1** (data prep — depends on discovery findings)
3. **Module 2** (architecture — can be written independently)
4. **Module 3** (training — depends on 1 + 2)
5. **Module 4** (features — depends on 3)
6. **Module 5** (clustering — depends on 4)
7. **Module 6** (comparison — depends on 4 + Task 2)

## Key Hyperparameters to Track
Document all of these in config.yaml:
- `input_shape: [1, 64, 160]`  # verify with discovery
- `bottleneck_filters: 8`
- `encoder_filters: [64, 32, 8]`
- `kernel_size: 3`
- `pool_size: 2`
- `loss: bce`
- `optimizer: adam`
- `lr: 0.001`
- `batch_size: 32`
- `max_epochs: 20`
- `early_stopping_patience: 5`
- `variance_threshold_factor: 1.2`
- `pca_variance_ratio: 0.95`
- `n_clusters: [4, 6, 8, 10]`  # try multiple
- `umap_n_neighbors: 15`
- `umap_min_dist: 0.1`

## Success Criteria
- Autoencoder reconstructions are visually recognizable (not blurry mush)
- Training loss converges (validation loss doesn't diverge)
- UMAP embedding shows visible structure (not uniform blob)
- At least one cluster count produces clusters where most USVs within a cluster share visual similarity
- Autoencoder features produce better clustering metrics than raw flattened spectrograms
