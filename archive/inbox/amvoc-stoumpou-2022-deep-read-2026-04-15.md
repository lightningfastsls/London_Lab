---
description: "Deep-read technical extraction of AMVOC (Stoumpou et al. 2022) covering autoencoder architecture, training details, spectrogram preprocessing, feature extraction pipeline, clustering methods, hand-crafted feature baselines, detection algorithm, and improvement gaps for our wild-mice pipeline"
source_type: paper
url: "https://github.com/tyiannak/amvoc"
author: "Stoumpou et al. (2022). Jarvis Neuroscience Lab (Rockefeller) + MagCIL (NCSR Demokritos)"
date_accessed: "2026-04-15"
status: unprocessed
research_tool: "manual"
research_query: "AMVOC convolutional autoencoder architecture, training, preprocessing, feature extraction pipeline for USV classification"
research_depth: "deep"
paper_citation: "Stoumpou et al. (2022). Analysis of Mouse Vocal Communication (AMVOC). Bioacoustics, 32(2), 199–229."
---

# AMVOC Deep-Read Technical Extraction (Stoumpou et al. 2022)

Full technical extraction from direct paper + code reading of the AMVOC system. The paper is open-access (CC-BY); repository is MIT-licensed (github.com/tyiannak/amvoc, 226 commits, Python 3.8, PyTorch).

This extraction focuses on the technical specifics needed to: (1) compare AMVOC's autoencoder with our VQ-VAE approach, (2) adapt AMVOC's preprocessing for our existing detected-USV pipeline, (3) identify what AMVOC does that we should adopt vs. improve.

---

## 1. Convolutional Autoencoder Architecture

### 1.1 Input Dimensions
- Input shape: `(1, 64, 160)` — single-channel (grayscale), 64 time frames × 160 frequency bins
- 64 time frames × 2 ms/frame = 128 ms temporal window
- 160 frequency bins: 80 kHz range (30–110 kHz) at 0.5 kHz resolution

### 1.2 Encoder (Layer-by-Layer)

| Layer | Type | Filters | Kernel | Padding | Activation | Output Shape |
|-------|------|---------|--------|---------|------------|-------------|
| Input | — | — | — | — | — | 1 × 64 × 160 |
| conv1 | Conv2d | 64 | 3×3 | 1 (same) | ReLU | 64 × 64 × 160 |
| pool1 | MaxPool2d | — | 2×2, stride 2 | — | — | 64 × 32 × 80 |
| conv2 | Conv2d | 32 | 3×3 | 1 (same) | ReLU | 32 × 32 × 80 |
| pool2 | MaxPool2d | — | 2×2, stride 2 | — | — | 32 × 16 × 40 |
| conv3 | Conv2d | **8** | 3×3 | 1 (same) | ReLU | 8 × 16 × 40 |
| pool3 | MaxPool2d | — | 2×2, stride 2 | — | — | **8 × 8 × 20** |

Bottleneck dimensions: 8 × 8 × 20 = **1,280 values** (flattened)

### 1.3 Decoder (Layer-by-Layer)

| Layer | Type | Filters | Kernel | Stride | Activation | Output Shape |
|-------|------|---------|--------|--------|------------|-------------|
| Code | — | — | — | — | — | 8 × 8 × 20 |
| t_conv1 | ConvTranspose2d | 32 | 2×2 | 2 | ReLU | 32 × 16 × 40 |
| t_conv2 | ConvTranspose2d | 64 | 2×2 | 2 | ReLU | 64 × 32 × 80 |
| t_conv3 | ConvTranspose2d | 1 | 2×2 | 2 | **Sigmoid** | 1 × 64 × 160 |

Upsampling via transposed convolutions (not nearest-neighbor + conv). Final activation is **sigmoid** to constrain output to [0, 1] for BCE loss.

### 1.4 Loss Function
- Binary Cross-Entropy (BCE): `nn.BCELoss()` in PyTorch
- NOT MSE — treats each pixel as a probability. Requires input normalized to [0, 1].

### 1.5 Code Discrepancy (Paper vs. Repository)
- `training_task.py`: Last encoder layer has **8 filters** → bottleneck = 1,280 (matches paper)
- `conv_autoencoder.py` (separate file): Last encoder layer has **4 filters** → bottleneck = 640
- Paper tested 2, 4, and 8 filters. 8 was chosen for final design; 4-filter version may be used for semi-supervised retraining pipeline

### 1.6 Compression Ratio
- Input: 64 × 160 = 10,240 values
- Bottleneck: 8 × 20 × 8 = 1,280 values
- **Compression factor: 8×** (10,240 → 1,280)

---

## 2. Training Details

### 2.1 Training Dataset
- Dataset D2: 26 recordings from 9 different male mice (B6D2F1/J and C57BL/6J strains — lab mice)
- **22,409 detected syllables** as training images
- 80/20 train/test split in `training_task.py`

### 2.2 Training Hyperparameters
- Optimizer: Adam, LR: 0.001
- Batch size: 32
- **Epochs: 2** (loss plateaus after ~3 epochs per Figure 5b)
- No data augmentation
- Shuffle: True for training DataLoader

### 2.3 Why Only 2–3 Epochs?
- Loss drops from ~0.125 to ~0.095 in first 2 epochs, then minimal improvement
- Deliberately avoiding overfitting: autoencoder is undercomplete (8× compression), so bottleneck acts as regularizer
- Reconstruction intentionally lossy — goal is feature extraction, not perfect reconstruction

### 2.4 Computational Requirements
- Training: Google Colab GPU
- Detection: ~5 ms per 750 ms window (21.2× faster than real-time)
- ~1,400 total iterations (22K images, 2 epochs, batch 32) — very fast on GPU

---

## 3. Spectrogram Preprocessing

### 3.1 FFT Parameters
- Window size (w): 2 ms (non-overlapping)
- Hop length: 2 ms (same as window — no overlap, ST_STEP = ST_WIN = 0.002)
- Frequency resolution: f_r = 1/w = 0.5 kHz
- Frequency range: 30–110 kHz (80 kHz bandwidth)
- Spectrogram via `pyAudioAnalysis.ShortTermFeatures.spectrogram()`
- Post-processing: Median filter with kernel (2, 3) applied via `ndimage.median_filter`

### 3.2 Normalization
- **Per-spectrogram max normalization:** `spectrogram[i] = spectrogram[i] / np.amax(spectrogram[i])`
- Scales all values to [0, 1] — NOT min-max, NOT z-score, NOT log-scale

### 3.3 Handling Variable-Duration USVs
- Fixed target width: **64 time frames** (128 ms)
- **Short USVs (< 64 frames):** Zero-padded symmetrically (center the USV, pad both sides)
- **Long USVs (> 64 frames):** Center-cropped (keep central portion, discard edges)
- Exact code logic:
  ```python
  if len(spec) > 64:
      spec = spec[int((len(spec)-64)/2) : int((len(spec)-64)/2)+64, :] / np.amax(spec)
  elif len(spec) < 64:
      spec = np.pad(spec/np.amax(spec), ((pad_before, pad_after), (0, 0)))
  ```
- Final shape for PyTorch: `(N, 1, 64, 160)`

---

## 4. Feature Extraction from the Autoencoder

### 4.1 Bottleneck Extraction
- Model in `eval()` mode
- Encoder output flattened: `outputs[i].detach().numpy().flatten()` → 1,280-dimensional vector per USV
- No global pooling — pure flatten

### 4.2 Post-Processing Pipeline (4 stages)

**Stage 1: Variance Thresholding**
- Remove features with low variance
- Threshold: v_t = 1.2 × mean(variance_per_feature)
- Reduces ~1,280 → ~320 dimensions (~4× reduction)
- Uses `sklearn.feature_selection.VarianceThreshold`

**Stage 2: Standard Scaling**
- Z-score normalization per feature
- Uses `sklearn.preprocessing.StandardScaler`

**Stage 3: PCA**
- Retain smallest number of components preserving 95% of variance
- Uses `sklearn.decomposition.PCA`

**Stage 4: Visualization (only)**
- 2D t-SNE for visualization — clustering operates in PCA-reduced space, NOT t-SNE space

### 4.3 Dimensionality Reduction Summary
```
Input image:         64 × 160       = 10,240
Bottleneck:          8 × 20 × 8     = 1,280   (8× reduction)
After var threshold: ~320           (~4× reduction)
After PCA (95%):     variable       (further reduction)
t-SNE for viz:       2              (display only)
```

---

## 5. Clustering Pipeline

### 5.1 Available Algorithms
- K-Means, Agglomerative, Birch, GMM, Mini-Batch K-Means (all via sklearn)

### 5.2 Number of Clusters
- User-selectable; paper used **6 clusters** for all tested configurations
- No automatic cluster number selection

### 5.3 Evaluation Metrics (Internal)
- Silhouette, Calinski-Harabasz, Davies-Bouldin scores

### 5.4 Human Evaluation (GUI)
- 3 levels: global annotations (1–5), cluster annotations (1–5), point annotations (~100 points per config)
- 4 annotators (2 domain experts + 2 non-experts), blinded to method

### 5.5 Semi-supervised Retraining Loop
- Pairwise constraints between USVs via GUI
- Combined loss: BCE reconstruction + KL divergence clustering + pairwise constraint
- Weights: gamma_1=0.5 (reconstruction), gamma_2=0.2 (KL), gamma_3=0.001 (pairwise)
- K-means centers updated via gradient descent during retraining
- Uncertain USVs (lowest max-probability of cluster assignment) prioritized for annotation

---

## 6. Hand-Crafted Features for Comparison

### 6.1 Baseline Feature Set (feature_mode == 2, 4 features)
1. Duration of the vocalisation
2. Normalized time position of minimum frequency
3. Normalized time position of maximum frequency
4. Normalized bandwidth: (freq_start − freq_end) / mean_frequency

### 6.2 Frequency Contour Detection
- Peak energy position per time frame, thresholded at >20th percentile of max values
- **SVM regression** (RBF kernel, C=1e3, gamma=0.1) maps time → frequency
- Feature mode 3: resampled 90-dimensional contour (closer to Didi's vectorization approach)

### 6.3 Deep vs. Simple Comparison Results
- Deep features: **37% higher** global annotation scores on average
- All three clustering configs: significant differences (p < 0.01)
- Cluster-level scores: 30% higher for deep features
- Point approval rate: higher for deep but not significant (p = 0.18)

### 6.4 Connection to Didi's Vectorization
- Feature mode 3 (90-dim resampled contour) is architecturally similar to Didi's peak-frequency-per-column approach, but AMVOC adds SVM smoothing first

---

## 7. Detection Method

### 7.1 Dynamic Spectral Thresholding (Two-Criterion)
**Criterion 1 — Time-based (TT):**
- S_i = sum of energy across 30–110 kHz for frame i
- Dynamic threshold T_i = 0.5×(global mean) + 0.5×(moving average of last K=100 frames)
- Frame passes if S_i > t × T_i (t = 0.5)

**Criterion 2 — Frequency-based (FT):**
- P_i = max energy in frame i
- M_i = mean energy in 60 kHz window around peak frequency
- Frame passes if P_i > f × M_i (f = 3.5)

Combined: BOTH criteria must pass. Post-processing: 20 ms box filter (L=10 frames), concatenate segments <11 ms, remove detections <5 ms.

### 7.2 Performance vs. Other Tools (Dataset D1: 245 syllables, 14 mice)
- AMVOC: Event F1 = 90.5%, Temporal F1 = 75.5%
- DeepSqueak: Event F1 = 87.0%, Temporal F1 = 79.5%
- AMVOC real-time ratio: 21.2× faster than real-time

---

## 8. Code Structure

Key files: `training_task.py` (autoencoder + training, 8 filters), `conv_autoencoder.py` (standalone, 4 filters), `audio_process.py` (spectrogram, detection, syllable extraction), `audio_recognize.py` (feature extraction, clustering, metrics).

Pre-trained models ship in `src/models/` as full PyTorch model objects (`torch.save(model, path)`), not just state_dicts.

Dependencies: PyTorch 1.7.1, scikit-learn 0.24, pyAudioAnalysis 0.3.6, Dash 1.18, scipy 1.5.4, numpy 1.19, umap-learn 0.5.1.

---

## 9. Reconstruction Quality

- Preserves: overall USV shape/contour, approximate frequency range/duration, general morphological category
- Loses: fine spectral detail, harmonics, precise amplitude variations, sharp edges
- 2 filters: poor; 4 filters: reasonable basic shapes; 8 filters: selected for full USV shape detail

---

## 10. Limitations and Gaps Relevant to Our Pipeline

### What AMVOC doesn't do that we need:
- No VAE variant (no KL regularization → latent space may be discontinuous)
- No UMAP — t-SNE for visualization only
- No systematic comparison of latent space sizes
- No investigation of compositional structure in sequences
- Shallow architecture (3 conv layers)
- No batch normalization or dropout
- No validation set monitoring (fixed 2 epochs)

### What we should adopt:
- 64×160 input size (well-justified, supports our pipeline)
- BCE loss with sigmoid output
- Variance thresholding → StandardScaler → PCA pipeline
- 8 filters for ~7.5K USVs (matches our dataset size)
- Per-spectrogram max normalization for BCE

### What we should improve:
- Add batch normalization (modern best practice)
- Use validation set with early stopping
- Try deeper architectures (4–5 conv layers) for wild-mouse variability
- Implement VAE variant for smoother latent space
- Use UMAP instead of (or in addition to) t-SNE
- Systematically compare autoencoder features vs. Didi's vectorization

---

## 11. PyTorch Architecture (Copy-Paste Ready)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMVOCAutoencoder(nn.Module):
    """Reproduction of AMVOC's convolutional autoencoder (Stoumpou et al. 2022).
    Input: (batch, 1, 64, 160) — single-channel USV spectrogram
    Bottleneck: (batch, 8, 8, 20) — 1,280 features when flattened
    """
    def __init__(self, n_bottleneck_filters=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, n_bottleneck_filters, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = nn.ConvTranspose2d(n_bottleneck_filters, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def encode(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def decode(self, z):
        z = F.relu(self.t_conv1(z))
        z = F.relu(self.t_conv2(z))
        z = torch.sigmoid(self.t_conv3(z))
        return z

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon
```

---

*Source: Published paper (CC-BY) + GitHub repository (MIT license). Deep read performed April 2026.*

## Processing Notes
- **Section 1** (architecture): 3–4 atomic notes expected — architecture spec, BCE loss choice, code discrepancy, compression ratio
- **Section 2** (training): 2–3 notes — training dataset, hyperparameters (esp. epoch count rationale)
- **Section 3** (preprocessing): 2–3 notes — zero-padding/crop handling, per-max normalization, no-overlap window
- **Section 4** (feature extraction): 2 notes — 4-stage post-processing pipeline, dimensionality reduction cascade
- **Section 5** (clustering): 1–2 notes — user-selectable k limitation, semi-supervised retraining loop
- **Section 6** (baselines): 1–2 notes — hand-crafted feature comparison results, connection to Didi vectorization
- **Section 7** (detection): 1 note — detection algorithm (lower priority, not in our critical path)
- **Section 10** (gaps): 2–3 notes — what to adopt vs. improve (actionable decisions for our pipeline)
- Check whether the existing AMVOC note needs enrichment vs. new notes needed
