# Analysis Next Steps — Actionable Tasks

## Context
From meeting with Mickey London, April 15 2026. These are technical analysis directions for the existing 5970 dataset. Ordered by priority.

---

## Task 1: UMAP Visualization Improvements (URGENT — needed for presentation)

### 1A: Add spectrogram examples per cluster
On the existing UMAP plot, overlay actual spectrogram thumbnails so the audience can see what each region of the UMAP "looks like."

**Approach:**
- Find the existing UMAP + clustering code in this repo
- For each K-means cluster (or HDBSCAN cluster), select 3–5 representative USVs closest to cluster centroid
- Render their spectrograms as small thumbnails
- Place near their UMAP coordinates or in a connected legend panel
- Save as a presentation-ready figure

### 1B: K-means cluster overlay on UMAP
Color-code the UMAP scatter plot by K-means cluster assignment (the 27-cluster DeepSqueak classification). Shows whether k-means clusters map to density regions or cut across the continuum.

**Discovery steps:**
- Find where the UMAP embedding is computed and stored
- Find where the DeepSqueak 27-cluster assignments are stored
- Find the spectrogram extraction utility for individual USVs
- Then implement the visualization

---

## Task 2: Noise Removal + Re-clustering (After presentation)

### Problem
Some detections that passed FP filtering may still be noise. UMAP might reveal them as outliers.

### Approach
- Cross-reference UMAP outliers with CNN confidence scores (low confidence = likely noise)
- Also cross-reference with HDBSCAN's existing 37 noise-classified points
- Visually inspect suspected noise spectrograms
- Remove confirmed noise
- Re-run UMAP + K-means + HDBSCAN on cleaned data
- Mickey's hypothesis: "noise has more variability than USVs" — removing it should tighten USV clusters and possibly reveal sub-structure

---

## Task 3: Vectorization Approach (After deep-read of Didi's paper)

### What it is
Hand-crafted feature extraction from USV spectrograms:
1. For each spectrogram, go column by column (each column = one time step)
2. Find the pixel with highest amplitude in that column → that's the dominant frequency
3. Record the frequency value → produces a **pitch contour vector**
4. Also record the amplitude at each peak → concatenate to the vector
5. Result: fixed-length vector per USV = [freq_1, freq_2, ..., freq_N, amp_1, amp_2, ..., amp_N]

### Implementation notes
- Handle variable-length USVs: resample or interpolate to fixed number of columns
- Run PCA, UMAP, K-means on these vectors
- Compare resulting clusters to existing classifications and to autoencoder features (once built)

---

## Task 4: Kernel-Based Feature Extraction (Optional alternative to Task 3)

Apply filter banks to USV spectrograms:
- **Standard kernels**: Gabor filters at different orientations/frequencies
- **Special kernels**: designed for USV morphology — horizontal (Flat calls), descending diagonal (Down), V-shape (Chevron), etc.
- Pool/aggregate filter responses into feature vector per USV
- Essentially what CNN early layers do, but without training

---

## Task 5: Convolutional Autoencoder (After deep-read of AMVOC paper)

### What Mickey described
"Create an autoencoder that will try to learn how to create all the USVs and then you can run PCA and UMAP on its representations (in the middle layer)"

### Plan
- **Input**: spectrogram image of detected USV (standardized size)
- **Encoder**: conv layers → low-dimensional latent vector (bottleneck)
- **Decoder**: transposed conv layers → reconstructed spectrogram
- **Loss**: reconstruction error (MSE)
- **Output of interest**: latent vectors = learned USV representations

### What to do with latent representations
1. PCA → find principal axes of USV variation
2. UMAP → visualize learned representation space
3. Cluster in latent space → compare to acoustic-feature and vectorization clusters
4. Compare lab vs. wild (once lab data available) in same latent space

### Reference implementation
AMVOC (GitHub: https://github.com/tyiannak/amvoc) — PyTorch autoencoder for mouse USVs. Use architecture details from the AMVOC deep-read session.

### Future extensions
- VAE variant for smoother latent space + generative capability
- VQ-VAE + Transformer for compositional structure investigation (long-term goal)

### Training considerations
- Dataset: ~7,575 USVs from cage 5970 (+ possibly 3452)
- Try bottleneck dimensions: 8, 16, 32, 64
- Validate by visual inspection of reconstructions
- 2–3 epochs may suffice (per AMVOC findings)

---

## Priority Order
1. **UMAP improvements** → needed for presentation (URGENT)
2. Wait for lab data → run detection + classification
3. **Vectorization** → quick to implement, gives a baseline representation
4. **Noise removal + re-clustering** → improves all downstream analyses
5. **Autoencoder** → the most important longer-term deliverable
