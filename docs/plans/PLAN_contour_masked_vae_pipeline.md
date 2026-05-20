# Plan: Contour-Masked Spectrogram → VAE Pipeline

## Goal

Build a pipeline that uses DeepSqueak's tonal-ridge contour as a **spatial mask** on the spectrogram, producing clean fixed-size patches, then trains a convolutional VAE on those patches to learn a continuous latent representation of USV morphology — without requiring labels.

## Why

The current classification approaches fail for two reasons:
1. Rule-based Holy & Guo types impose discrete boundaries on a continuous manifold — k-means on summary features produces mixed clusters.
2. Training any supervised or unsupervised model on raw spectrogram patches fails because the model fits noise alongside signal — the error signal is corrupted.

The contour mask solves (2): it keeps spectral energy near the tonal ridge and zeros out everything else. The VAE solves (1): it learns continuous latent structure without labels.

## Pipeline Architecture

```
CNN detection → DeepSqueak contour extraction → 100ms windowing → contour mask → masked patch → VAE
```

Each step has one job:
- **CNN**: decides what's a call (detection)
- **DeepSqueak**: finds the tonal ridge within each call (contour extraction)
- **Windowing**: standardizes input size (100ms fixed windows)
- **Mask**: removes noise, keeps signal (denoising)
- **VAE**: learns latent representation (unsupervised feature learning)

---

## Phase 1: Extract Contour Coordinates from DeepSqueak

### Task
Find and extract the per-STFT-time-bin contour data that DeepSqueak computes internally. We need `(time_bin, frequency)` pairs for each detected call.

### Discovery Steps
- Find the DeepSqueak output files for the 5970 cohort. Look in the project directories for `.mat` files, detection tables, or exported CSVs. The relevant fields are likely called `Stats.ContourFreq`, `Stats.ContourTime`, or similar — but discover the actual structure, don't assume.
- Check if the existing pipeline already exports contour coordinates somewhere (CSV columns, intermediate files). We may already have this data without needing to touch MATLAB.
- If contour data is only inside `.mat` files, write a Python script using `scipy.io.loadmat` or `h5py` (depending on MATLAB version) to extract the contour arrays per call.
- Document the exact structure: how many time bins per call, what the frequency units are (Hz? kHz? bin index?), whether NaN values appear for non-tonal frames.

### Output
A per-call data structure (parquet or CSV) with columns: `call_id`, `time_bin_index`, `frequency_kHz`. One row per tonal time bin per call. Calls with non-tonal gaps should have those bins absent (not filled with NaN).

### Canonical Parameters (reference)
- SR = 300 kHz
- STFT hop = 128 samples → 1 time bin ≈ 0.4267 ms
- NFFT = 512 (processing)
- USV band: 20–120 kHz
- 100 ms window = ~234 time bins

---

## Phase 2: 100ms Windowing

### Logic
- **Short calls (< 100ms)**: Take a 100ms window centered on the call. The extra time captures surrounding recording context. After masking, the non-call portion becomes zeros — the mask handles padding naturally.
- **Long calls (> 100ms)**: Use a **moving window** with a fixed step size (start with ~10ms / ~23 bins). A 200ms call produces ~11 overlapping 100ms windows. Each window is a separate training example for the VAE. This preserves local morphology and acts as natural data augmentation.
- **Exactly 100ms calls**: One window, no special handling.

### Implementation Notes
- The window is cut from the **original spectrogram** (not a pre-masked version). Masking happens after windowing.
- Window boundaries that extend beyond the recording start/end should be zero-padded.
- Track provenance: each window should know its source `call_id`, window index, and absolute time offset in the recording.

### Output
A set of raw (unmasked) 100ms spectrogram patches, each with shape `(freq_bins, ~234)`, plus metadata linking each patch to its source call and temporal position.

---

## Phase 3: Contour Mask Application

### Logic
For each 100ms window:
1. Look up which contour points fall within this window's time range.
2. At each time bin that has a contour frequency, keep energy within some bandwidth around the contour. Zero out everything else.
3. Time bins with no contour point (no call present, or non-tonal gap) → entire column is zeroed.

### Mask Bandwidth — Experimental Parameter
Start with a few candidates and **generate visual comparisons** before committing:
- **Narrow**: ±2 kHz (~keeps just the fundamental)
- **Medium**: ±5 kHz (~preserves natural bandwidth variation)
- **Wide**: ±10 kHz (~captures more spectral context)
- **Soft falloff**: Gaussian weighting centered on contour frequency, σ = 3 kHz (no hard cutoff)

Generate a comparison figure: for ~20 example calls across different types, show side by side: raw spectrogram patch, contour overlay, and masked result at each bandwidth setting. This is the decision point — Shachar will choose bandwidth based on visual inspection.

### Output
Masked 100ms spectrogram patches, same shape as Phase 2 output but with noise energy removed. Ready for VAE input.

---

## Phase 4: VAE Training

### Architecture
Convolutional VAE. Nothing exotic — the goal is to learn a useful latent space, not to push reconstruction quality.

- **Encoder**: A few conv layers with stride-2 downsampling → flatten → FC to μ and log(σ²)
- **Latent dim**: Start with 8–16 dimensions. Too few = underfitting morphological variety. Too many = hard to visualize and may capture noise.
- **Decoder**: FC → reshape → transposed conv layers back to input dimensions
- **Loss**: Standard ELBO = reconstruction (MSE on masked patches) + KL divergence. β-VAE weighting is worth trying if the latent space is too entangled.

### Training Notes
- Train on 5970 cohort first (largest, best characterized).
- Input normalization: per-patch or global — try both, see which gives cleaner latent space.
- Keep training simple: Adam, LR ~1e-4, early stopping on validation reconstruction loss.
- The model trains on the **masked** patches, so it should never see noise. If reconstruction quality is poor on high-tonality calls, the architecture is too small; if it's poor on low-tonality calls, the mask might be too aggressive.

### Output
- Trained VAE checkpoint
- Per-window latent vectors (μ) for all 5970 calls
- Reconstruction examples (input vs. output) for quality check

---

## Phase 5: Latent Space Analysis & Visualization

### Core Visualizations
1. **UMAP on VAE latent vectors** — compare to existing UMAP on DeepSqueak summary features. Is there more structure? Cleaner separation?
2. **Color by Holy & Guo type** — do traditional types map to regions of the latent space? (They should, loosely, if the VAE captured morphology.)
3. **Color by DeepSqueak cluster ID** — same question for the k-means clusters.
4. **Long-call analysis** — for calls that produced multiple windows, plot all their latent vectors. Do they cluster together (call identity preserved) or spread out (local morphology dominates)? This tells us whether the moving window approach is reasonable.

### Cross-Cohort (stretch goal if time permits)
- Encode 3452 and 9252 cohort calls through the trained VAE (trained on 5970).
- Overlay on the 5970 UMAP. Do different cohorts occupy different regions?

### Deliverable for Lab Presentation
The key figure: UMAP of VAE latent space for 5970, colored by type, with annotation showing whether clusters are cleaner than the current feature-based UMAP. Side-by-side comparison would be the strongest visual.

---

## File & Directory Conventions

Discover the current project structure before creating new directories. Look for:
- Where existing DeepSqueak outputs live (`.mat` files, CSVs)
- Where the spectrogram gallery PNGs are stored
- Where the CNN detection outputs are
- What the existing CSV schema looks like (columns for `Cluster_NN`, `syllable_type`, detection features)

Place new outputs in a sensible location relative to existing structure. Propose a directory layout and confirm before writing files.

---

## Order of Operations

1. **Phase 1** first — everything depends on having contour coordinates.
2. **Phase 3 visualization** before Phase 4 — don't train the VAE until mask bandwidth is chosen.
3. Phases 2–4 can be built incrementally once Phase 1 is done.
4. Phase 5 is the payoff — prioritize getting here within the two-week window.

## Important Context
- 5970 is the reference cohort: one wild-mouse couple, animal lmt_034, 5 sessions, ~8k accepted detections.
- 3452 and 9252 are comparison cohorts.
- DeepSqueak is MATLAB-based; all new pipeline code should be Python.
- The existing pipeline CSV has both `Cluster_NN` (DeepSqueak k-means) and `syllable_type` (rule-based Holy & Guo) columns. The VAE latent codes will become a third representation alongside these two.
