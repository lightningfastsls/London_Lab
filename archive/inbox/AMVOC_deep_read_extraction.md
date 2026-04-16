# AMVOC Deep-Read Technical Extraction

**Paper:** Stoumpou et al. (2022). "Analysis of Mouse Vocal Communication (AMVOC)." Bioacoustics, 32(2), 199–229.
**Source:** https://github.com/tyiannak/amvoc (226 commits, Python 3.8, PyTorch, MIT license)
**Lab:** Jarvis Neuroscience Lab (Rockefeller) + MagCIL (NCSR Demokritos, Greece)

---

## 1. Convolutional Autoencoder Architecture (HIGHEST PRIORITY)

### 1.1 Input Dimensions
- **Input shape:** `(1, 64, 160)` — single-channel (grayscale), 64 time frames × 160 frequency bins
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

**Bottleneck dimensions:** 8 × 8 × 20 = **1,280** values (flattened)

### 1.3 Decoder (Layer-by-Layer)

| Layer | Type | Filters | Kernel | Stride | Activation | Output Shape |
|-------|------|---------|--------|--------|------------|-------------|
| Code | — | — | — | — | — | 8 × 8 × 20 |
| t_conv1 | ConvTranspose2d | 32 | 2×2 | 2 | ReLU | 32 × 16 × 40 |
| t_conv2 | ConvTranspose2d | 64 | 2×2 | 2 | ReLU | 64 × 32 × 80 |
| t_conv3 | ConvTranspose2d | 1 | 2×2 | 2 | **Sigmoid** | 1 × 64 × 160 |

The decoder mirrors the encoder. Upsampling is via transposed convolutions (not nearest-neighbor + conv). The final activation is **sigmoid** (not ReLU) to constrain output to [0, 1] for BCE loss.

### 1.4 Loss Function
- **Binary Cross-Entropy (BCE):** `nn.BCELoss()` in PyTorch
- Formula: L(y, ŷ) = −y·log(ŷ) − (1−y)·log(1−ŷ)
- Requires input normalized to [0, 1] (achieved via per-spectrogram max normalization)
- NOT MSE — this is a critical design choice. BCE treats each pixel as a probability.

### 1.5 CODE DISCREPANCY (Paper vs. Repository)
- **Paper + `training_task.py`:** Last encoder layer has **8 filters** → bottleneck = 8 × 8 × 20 = 1,280
- **`conv_autoencoder.py` (separate file):** Last encoder layer has **4 filters** → bottleneck = 4 × 8 × 20 = 640
- The `training_task.py` version with 8 filters matches the paper's final design choice
- The 4-filter version in `conv_autoencoder.py` may be used for the semi-supervised retraining pipeline
- **Paper tested 2, 4, and 8 filters.** 4 was sufficient for decent reconstruction; 8 was chosen to ensure all USV shape details are captured. The training notebook confirms 8 filters was the selected configuration.

### 1.6 Compression Ratio
- Input: 64 × 160 = 10,240 values
- Bottleneck: 8 × 20 × 8 = 1,280 values
- **Compression factor: 8×** (10,240 → 1,280)

---

## 2. Training Details

### 2.1 Training Dataset
- **Dataset D2:** 26 recordings from 9 different male mice (both B6D2F1/J and C57BL/6J strains)
- **22,409 detected syllables** used as training images
- 80/20 train/test split applied in `training_task.py` (`train_test_split(train_data, test_size=0.2)`)
- Data publicly available via Google Drive links in the paper

### 2.2 Training Hyperparameters
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Batch size:** 32
- **Epochs:** 2 (paper says 2–3 sufficient; loss plateaus after ~3 epochs per Figure 5b)
- They deliberately train for only 2 epochs to avoid overfitting
- **No data augmentation** mentioned or present in code
- **Shuffle:** True for training DataLoader

### 2.3 Why Only 2–3 Epochs?
- The loss curve (Figure 5b) shows rapid convergence — loss drops from ~0.125 to ~0.095 in first 2 epochs, then minimal improvement
- They explicitly wanted to avoid overfitting the training data
- The autoencoder is undercomplete (8× compression), so the bottleneck itself acts as a regularizer
- The reconstruction is intentionally lossy — the goal is feature extraction, not perfect reconstruction

### 2.4 Computational Requirements
- Training was done on Google Colab with GPU (`accelerator: GPU` in notebook metadata)
- Detection processing: ~5 ms per 750 ms window (very fast)
- Real-time processing ratio for detection: 21.2× faster than real-time
- No specific training time reported, but with 22K images, 2 epochs, batch 32 → ~1,400 iterations total — very fast on GPU

---

## 3. Spectrogram Preprocessing

### 3.1 FFT Parameters
- **Window size (w):** 2 ms (non-overlapping)
- **Hop length:** 2 ms (same as window — no overlap, ST_STEP = ST_WIN = 0.002)
- **Frequency resolution:** f_r = 1/w = 0.5 kHz
- **Frequency range:** 30–110 kHz (80 kHz bandwidth)
- **Spectrogram computation:** via `pyAudioAnalysis.ShortTermFeatures.spectrogram()`
- **Post-processing:** Median filter with kernel (2, 3) applied to spectrogram (`ndimage.median_filter`)

### 3.2 Spectrogram Extraction per USV
- After detection, each USV's spectrogram is extracted from the full recording spectrogram
- Sliced by time (start:end frames) and frequency (f1:f2 bins corresponding to 30–110 kHz)
- Each USV spectrogram has shape: (variable_time_frames, 160)

### 3.3 Normalization
- **Per-spectrogram max normalization:** Each spectrogram is divided by its own maximum value
- `spectrogram[i] = spectrogram[i] / np.amax(spectrogram[i])`
- This scales all values to [0, 1], which is required for BCE loss with sigmoid output
- **NOT min-max, NOT z-score, NOT log-scale** — simple division by max

### 3.4 Handling Variable-Duration USVs
- **Fixed target width:** 64 time frames (128 ms)
- Selected because it's larger than both mean and median of USV durations (Figure 4), and is a power of 2
- **Short USVs (< 64 frames):** Zero-padded symmetrically (center the USV, pad both sides)
- **Long USVs (> 64 frames):** Center-cropped (keep central portion, discard edges)
- **Exact code logic:**
  ```python
  if len(spec) > 64:
      spec = spec[int((len(spec)-64)/2) : int((len(spec)-64)/2)+64, :] / np.amax(spec)
  elif len(spec) < 64:
      spec = np.pad(spec/np.amax(spec), ((pad_before, pad_after), (0, 0)))
  ```
- Final shape after reshaping for PyTorch: `(N, 1, 64, 160)` — batch × channels × time × freq

---

## 4. Feature Extraction from the Autoencoder

### 4.1 Bottleneck Extraction
- Model is set to `eval()` mode
- Input spectrograms passed through encoder only (forward with `decode=False`)
- The encoder output (after final MaxPool) has shape `(batch, 8, 8, 20)`
- **Flattened** to 1D vector: `outputs[i].detach().numpy().flatten()` → 1,280-dimensional vector per USV
- No global pooling — pure flatten

### 4.2 Post-Processing Pipeline (Critical — 4 stages)

**Stage 1: Variance Thresholding**
- Remove features with low variance (unlikely to discriminate USVs)
- Threshold: `v_t = 1.2 × mean(variance_per_feature)`
- Features with variance < v_t are excluded
- Reduces dimensionality by ~4× (1,280 → ~320)
- Uses `sklearn.feature_selection.VarianceThreshold`

**Stage 2: Standard Scaling**
- Z-score normalization per feature: `X_transformed = (X - X_mean) / X_std`
- Uses `sklearn.preprocessing.StandardScaler`

**Stage 3: PCA**
- Retain smallest number of components preserving 95% of variance
- Number of components varies across recordings/datasets
- Uses `sklearn.decomposition.PCA`
- Code iteratively increases component count if 95% isn't reached with initial guess

**Stage 4: Visualization**
- 2D projection via t-SNE for visualization (not used for clustering)
- Clustering operates in the PCA-reduced space, not the t-SNE space

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

### 5.1 Available Clustering Algorithms
- **K-Means** (`sklearn.cluster.KMeans`)
- **Agglomerative** (`sklearn.cluster.AgglomerativeClustering`)
- **Birch** (`sklearn.cluster.Birch`)
- **Gaussian Mixture Models** (`sklearn.mixture.GaussianMixture`)
- **Mini-Batch K-Means** (`sklearn.cluster.MiniBatchKMeans`)

All use default sklearn distance metrics (Euclidean for K-Means, various for others).

### 5.2 Number of Clusters
- User-selectable via GUI
- Paper's evaluation used **6 clusters** for all three tested configurations (K-Means-6, GMM-6, Agg-6)
- No automatic cluster number selection (no elbow method, no silhouette optimization)

### 5.3 Evaluation Metrics (Internal)
- Silhouette score
- Calinski-Harabasz score
- Davies-Bouldin score

### 5.4 GUI Evaluation (Human)
- Three evaluation levels, all scored 1–5:
  1. **Global annotations:** Overall clustering quality score
  2. **Cluster annotations:** Per-cluster quality score
  3. **Point annotations:** Approve/reject individual USVs in their assigned cluster (~100 points per config per method)
- 4 annotators (2 domain experts, 2 non-experts)
- Annotators were blinded — didn't know which method (deep vs. simple) produced which clustering

### 5.5 Semi-supervised Retraining Loop
- Users can impose **pairwise constraints** between USVs via GUI
- The autoencoder is retrained with a combined loss:
  - BCE reconstruction loss
  - KL divergence clustering loss (soft cluster assignments vs. target distribution)
  - Pairwise constraint loss (penalizes distance between must-link pairs)
- Loss weights: `gamma_1=0.5` (reconstruction), `gamma_2=0.2` (KL), `gamma_3=0.001` (pairwise)
- K-means centers updated via gradient descent during retraining
- The uncertain USVs (lowest max-probability of cluster assignment) are prioritized for annotation

---

## 6. Hand-Crafted Features for Comparison

### 6.1 Baseline Feature Set (4 features)
The paper's `feature_mode == 2` (used for evaluation):
1. **Duration** of the vocalisation
2. **Normalized time position of minimum frequency** (position / duration)
3. **Normalized time position of maximum frequency** (position / duration)
4. **Normalized bandwidth:** (freq_start − freq_end) / mean_frequency

### 6.2 Frequency Contour Detection
- Peak energy position detected per time frame
- Thresholded: keep points where peak energy > 20th percentile of max values
- **SVM regression** (RBF kernel, C=1e3, gamma=0.1) maps time → frequency
- Predicted contour captures the dominant frequency trajectory

### 6.3 Extended Feature Set (in code but not default)
`feature_mode == 1` includes 12 features: duration, min/max/mean frequency, max/min frequency change, delta mean/std, delta2 mean/std, freq start/end.

### 6.4 Deep vs. Simple Comparison Results
- Deep features yielded **37% higher** global annotation scores on average
- All three clustering configs showed significant differences (p < 0.01)
- Cluster-level scores 30% higher for deep features
- Point approval rate higher for deep but not significantly (p = 0.18)

### 6.5 Connection to Didi's Vectorization
AMVOC's baseline feature extraction (peak frequency per column → SVM-smoothed contour) is conceptually similar to the peak-frequency-per-column approach, but AMVOC adds SVM smoothing and derives summary statistics rather than using the raw contour as the feature vector. Feature mode 3 in the code does use the resampled contour directly (90-dimensional), which is closer to the vectorization approach.

---

## 7. Detection Method (Secondary Priority)

### 7.1 Dynamic Spectral Thresholding
Two-criterion approach operating on 2 ms frames in the 30–110 kHz range:

**Criterion 1 — Time-based thresholding (TT):**
- Spectral energy S_i = sum of energy across 30–110 kHz for frame i
- Dynamic threshold T_i = 0.5 × (global mean energy) + 0.5 × (moving average of last K=100 frames)
- Frame passes if S_i > t × T_i (t = 0.5)

**Criterion 2 — Frequency-based thresholding (FT):**
- Peak energy P_i = max energy in frame i across 30–110 kHz
- Mean energy M_i = mean energy in 60 kHz window around peak frequency
- Frame passes if P_i > f × M_i (f = 3.5)

**Combined rule:** Frame is vocalization if BOTH criteria pass.

**Post-processing:**
- Smooth binary sequence V with 20 ms box filter (L=10 frames)
- Concatenate segments separated by < 11 ms
- Remove detections < 5 ms duration

### 7.2 Performance (Dataset D1: 245 syllables, 14 mice)

| Method | Event F1 (Mean) | Temporal F1 (Mean) |
|--------|----------------|-------------------|
| AMVOC offline | **90.5** | 75.5 |
| AMVOC online | **90.0** | 76.5 |
| MSA2 | 83.0 | 79.5 |
| DeepSqueak | 87.0 | 79.5 |
| VocalMat | 74.0 | 74.5 |
| MUPET | 75.0 | 69.0 |

AMVOC had the highest event F1 score, especially in noisy conditions. MSA2 and DeepSqueak had slightly better temporal F1. AMVOC's real-time processing ratio (21.2×) was intermediate — faster than DeepSqueak (8.2×) and VocalMat (4.3×) but slower than MUPET (32.4×).

---

## 8. Code Structure

### 8.1 Key Files
| File | Purpose |
|------|---------|
| `src/main.py` | Main offline GUI app (Dash). Detection + clustering + evaluation |
| `src/main_live.py` | Online/real-time detection |
| `src/training_task.py` | Autoencoder training script + ConvAutoencoder class (8 filters) |
| `src/conv_autoencoder.py` | Standalone ConvAutoencoder class (4 filters — older version?) |
| `src/audio_process.py` | Spectrogram computation, detection features, syllable extraction |
| `src/audio_recognize.py` | Feature extraction (deep + simple), clustering, metrics |
| `src/evaluation.py` | Detection evaluation and comparison |
| `src/syllables_comp.py` | Compare detected syllables between methods |
| `src/config.json` | Configuration parameters |
| `src/constraints.py` | Pairwise constraint handling for semi-supervised learning |
| `notebooks/training_task.ipynb` | Colab training notebook (original development environment) |

### 8.2 How to Run

**Train autoencoder:**
```bash
python3 training_task.py -i /path/to/wav/folder -ne 2 -s true
```

**Run detection only:**
```bash
python3 main.py -i data/recording.wav -c n
```

**Run detection + clustering GUI:**
```bash
python3 main.py -i data/recording.wav -c y -s 1
```

**Online detection:**
```bash
python3 main_live.py -i data/recording.wav
```

### 8.3 Pre-trained Models
Two pre-trained models ship in `src/models/`:
- `model_test` (~127 KB)
- `model_test_new_4` (~121 KB) — this is the default in config.json

Models are saved as full PyTorch model objects (`torch.save(model, path)`), not just state_dicts.

### 8.4 Dependencies
PyTorch 1.7.1, scikit-learn 0.24, pyAudioAnalysis 0.3.6, Dash 1.18, scipy 1.5.4, numpy 1.19, umap-learn 0.5.1, pyaudio 0.2.11

---

## 9. Reconstruction Quality

### 9.1 What the Autoencoder Preserves
- Overall USV shape and contour (the dominant frequency trajectory)
- Approximate frequency range and duration
- General morphological category (up-sweeps, down-sweeps, chevrons, flats)

### 9.2 What the Autoencoder Loses
- Fine spectral detail and harmonics
- Precise amplitude variations within the USV
- Sharp edges become blurred
- The reconstruction is described as "lossy" by the authors — this is by design

### 9.3 Effect of Filter Count
- **2 filters:** Loses significant information, poor reconstruction
- **4 filters:** Sufficient for reasonable reconstruction of basic shapes
- **8 filters:** Better detail preservation, selected as final design to capture all USV shape variations

---

## 10. Limitations and Improvement Suggestions

### 10.1 Acknowledged Limitations
- Classification relies on predefined number of clusters (user must choose)
- No automatic optimal cluster selection
- Semi-supervised loop requires human intervention
- Detection parameters (t, f) were optimized on D1 — may need tuning for very different recording conditions
- Single-syllable focus — no sequence-level analysis

### 10.2 Authors' Future Directions
- **Sequence-level analysis:** Examine temporal relationships between syllables across bouts (beyond pairwise transition probabilities)
- **Semi-supervised methodologies:** Human-in-the-loop for better clustering (already partially implemented)
- **Real-time closed-loop experiments:** Use saved clustering models for online syllable classification
- **Cross-strain generalization:** Test on different strains with different vocalization types

### 10.3 Gaps Relevant to Our Pipeline

**What AMVOC doesn't do that we need:**
- No VAE variant (no KL regularization on latent space → latent space may be discontinuous)
- No UMAP — they use t-SNE for visualization only (not for clustering)
- No systematic comparison of different latent space sizes
- No investigation of compositional structure in sequences
- The autoencoder is relatively shallow (3 conv layers) — may not capture complex hierarchical features
- No batch normalization or dropout in the autoencoder
- No validation set monitoring during training (just fixed 2 epochs)

**What we should adapt:**
- The 64×160 input size is well-justified and should be our starting point
- BCE loss with sigmoid output works well for spectrogram reconstruction
- The variance thresholding → StandardScaler → PCA pipeline is a sensible feature post-processing chain
- 8 filters in the bottleneck is reasonable for ~7.5K USVs (similar to our dataset size)
- Their per-spectrogram max normalization is simple but effective for BCE

**What we should change:**
- Add batch normalization (modern best practice)
- Use a validation set with early stopping instead of fixed 2 epochs
- Try deeper architectures (4–5 conv layers) given that our USVs may have more variability (wild mice)
- Implement VAE variant for smoother latent space (AMVOC authors themselves mention VAEs in their introduction)
- Use UMAP instead of (or in addition to) t-SNE for visualization
- Compare autoencoder features against Didi's vectorization systematically

---

## 11. PyTorch Architecture Summary (Copy-Paste Ready)

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
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, n_bottleneck_filters, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(n_bottleneck_filters, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def encode(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 64, 32, 80)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 32, 16, 40)
        x = self.pool(F.relu(self.conv3(x)))   # -> (B, 8, 8, 20)
        return x

    def decode(self, z):
        z = F.relu(self.t_conv1(z))            # -> (B, 32, 16, 40)
        z = F.relu(self.t_conv2(z))            # -> (B, 64, 32, 80)
        z = torch.sigmoid(self.t_conv3(z))     # -> (B, 1, 64, 160)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon

# Training setup:
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# batch_size = 32, epochs = 2
```

---

*Extracted from published paper + GitHub repository (commit depth=1, April 2026). The paper is open-access (CC-BY). Repository is MIT-licensed.*
