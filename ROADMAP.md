# USV Detection & Analysis — Implementation Roadmap

> This file is the master plan for building the USV Detection & Analysis pipeline.
> It lives in the project root alongside CLAUDE.md and DECISIONS.md.
> Claude Code: read this file when asked "what's next", "check the roadmap", or "what should I build".
> Claude Code: read DECISIONS.md before making any architectural choice.
> Human: use the `/implement` commands below by copy-pasting them into Claude Code sessions.

---

## How to Use This File

1. Work through modules **in order** within each phase (dependencies are noted)
2. Each module has:
   - **What**: brief description of what to build
   - **`/implement` command**: copy-paste into Claude Code (or type `/implement <module description>`)
   - **Test plan**: how Claude Code should verify the module works
   - **Exit criteria**: what "done" looks like
3. After each module: commit, run `/review-all`, fix issues, commit again
4. Phase gates must pass before starting the next phase
5. **Phases 1–7** are all DONE. Current work starts at **Phase 8**.

## Status Key

- **DONE** — Implemented and tested
- **READY** — Dependencies met, can start
- **BLOCKED** — Waiting on dependency or external input
- **FUTURE** — Not yet prioritized

---

## Phase 1: Core Detection Pipeline

### 1.1 Energy Detector

**What:** Energy-based USV candidate detector optimized for high recall. Two-stage detection: energy thresholding finds candidates, then CNN filters false positives.
**Status:** DONE
**Review Tier:** 3
**Depends on:** None

**Key files:**
- `src/usv_spectrogram/detection/config.py` — DetectionConfig (frozen dataclass)
- `src/usv_spectrogram/detection/candidate.py` — Candidate dataclass
- `src/usv_spectrogram/detection/energy_detector.py` — EnergyDetector class
- `scripts/run_detection.py` — CLI entry point
- `tests/test_energy_detector.py` — 42 tests

**Key decisions:** ADR-001 (300 kHz), ADR-002 (STFT params), ADR-003 (thresholds), ADR-011 (auto sample rate), ADR-012 (peak energy mode), ADR-013 (segment continuity)

---

## Phase 2: Spectrogram Extraction

### 2.1 Spectrogram Extractor

**What:** Extract spectrogram PNG images from candidate segments for labeling and CNN training.
**Status:** DONE
**Review Tier:** 2
**Depends on:** Phase 1

**Key files:**
- `src/usv_spectrogram/detection/extraction_config.py` — ExtractionConfig
- `src/usv_spectrogram/detection/spectrogram_extractor.py` — SpectrogramExtractor
- `scripts/extract_spectrograms.py` — CLI entry point

**Key decisions:** ADR-002 (STFT params must match detection config)

---

## Phase 3: Labeling Tool

### 3.1 Streamlit Labeling App

**What:** Interactive Streamlit UI for human review and labeling of USV candidates (USV / Not USV / Uncertain). Includes keyboard shortcuts, progress tracking, session management.
**Status:** DONE
**Review Tier:** 2
**Depends on:** Phase 2

**Key files:**
- `src/usv_spectrogram/labeling/labeling_app.py` — Streamlit labeling UI
- `src/usv_spectrogram/labeling/noise_review_app.py` — Noise sample review app
- `scripts/usv_labeling_tool.py` — Launcher
- `scripts/extract_noise_samples.py` — Extract noise samples for balancing

**Dataset stats:** ~458 USV, ~374 Not USV, ~8 Uncertain (~840 total labels)

---

## Phase 4: Dataset Preparation

### 4.1 Splits, Quality Checks & Metadata

**What:** Train/val/test splitting by recording (not by candidate), quality checks for data leakage, class balance, and population coverage. Metadata extraction for traceability.
**Status:** DONE
**Review Tier:** 2
**Depends on:** Phase 3

**Key files:**
- `src/usv_spectrogram/dataset/splits.py` — Recording-based splitting
- `src/usv_spectrogram/dataset/quality_checks.py` — 7 pre-training checks
- `src/usv_spectrogram/dataset/metadata.py` — Metadata extraction
- `scripts/prepare_dataset.py` — CLI entry point
- `tests/test_dataset_splits.py` — 30 tests
- `tests/test_dataset_quality.py` — 41 tests

**Key decisions:** ADR-004 (split by recording, not candidate), ADR-008 (3-source negatives)

---

## Phase 5: CNN Classifier

### 5.1 CNN Architecture & Training Pipeline

**What:** Binary USV classifier: 3 conv blocks (32→64→128) + GlobalAvgPool + dense head. Training loop with early stopping, class weighting, evaluation metrics, threshold optimization.
**Status:** DONE
**Review Tier:** 3
**Depends on:** Phase 4

**Key files:**
- `src/usv_spectrogram/models/cnn_classifier.py` — USVClassifier (~101K params)
- `src/usv_spectrogram/models/config.py` — TrainingConfig, model size configs (small/medium/large)
- `src/usv_spectrogram/models/trainer.py` — USVTrainer with early stopping
- `src/usv_spectrogram/models/data_loader.py` — USVDataset, pad_collate_fn, data loaders
- `src/usv_spectrogram/models/evaluate.py` — Evaluation metrics
- `scripts/train_cnn.py` — Training CLI
- `scripts/evaluate_experiment.py` — Evaluation CLI
- `scripts/optimize_threshold.py` — Threshold sweep
- `scripts/plot_training_curves.py` — Training visualization
- `tests/test_cnn_model.py` — 38 tests

**Key decisions:** ADR-005 (3.0x USV class weight), ADR-006 (architecture), ADR-009 (model artifacts as .pt)

**Performance baseline:** 89.7% precision, 93.8% recall, F1 91.7% at optimal threshold 0.05

---

## Phase 6: Detection Desktop App

### 6.1 PyQt6 Detection & Labeling App

**What:** Desktop app for USV detection, spectrogram visualization, interactive threshold adjustment, boundary editing, progressive labeling presets, session tracking, and label management. Features auto-move of reviewed files.
**Status:** DONE
**Review Tier:** 3
**Depends on:** Phase 5

**Key files:**
- `src/usv_spectrogram/app/main_window.py` — Main window orchestration
- `src/usv_spectrogram/app/main.py` — Application entry point
- `src/usv_spectrogram/app/widgets/spectrogram_view.py` — Spectrogram display with boundary handles
- `src/usv_spectrogram/app/widgets/probability_view.py` — Probability curve display
- `src/usv_spectrogram/app/core/audio_loader.py` — WAV loading + full STFT
- `src/usv_spectrogram/app/core/sliding_inference.py` — CNN sliding window inference
- `src/usv_spectrogram/app/core/detection_logic.py` — Hysteresis thresholding
- `src/usv_spectrogram/app/core/label_storage.py` — Label persistence
- `src/usv_spectrogram/app/core/detection_exporter.py` — Detection export
- `src/usv_spectrogram/app/core/saved_detection_tracker.py` — Session tracking
- `src/usv_spectrogram/app/core/preset_config.py` — Progressive labeling presets
- `scripts/run_app.py` — Launcher

**Key decisions:** ADR-003 (thresholds: high 0.40, low 0.28 in app), ADR-010 (JSON label format), ADR-011 (auto sample rate)

---

## Phase 7: Clustering Exploration

### 7.1 USV Type Discovery via Unsupervised Clustering

**What:** Use the trained CNN as a feature extractor, then cluster and visualize the resulting feature space to discover natural USV type groupings. Includes t-SNE/UMAP visualization, k-means/HDBSCAN/GMM clustering, cross-recording comparison.
**Status:** DONE
**Review Tier:** 2
**Depends on:** Phase 5

**Key files:**
- `src/usv_spectrogram/clustering/feature_extractor.py` — CNN feature extraction
- `src/usv_spectrogram/clustering/visualizer.py` — t-SNE/UMAP visualization
- `src/usv_spectrogram/clustering/clusterer.py` — k-means, HDBSCAN, GMM
- `src/usv_spectrogram/clustering/analyzer.py` — Cluster analysis + exemplars
- `scripts/clustering_extract_features.py` — Feature extraction CLI
- `scripts/clustering_visualize.py` — Visualization CLI
- `scripts/clustering_cluster.py` — Clustering CLI
- `scripts/clustering_analyze.py` — Analysis CLI

---

## Phases 1–7 Gate (ALL DONE)

All foundation work is complete:
- [x] Energy detector with high recall (42 tests)
- [x] Spectrogram extraction pipeline
- [x] Streamlit labeling tool (~840 labels collected)
- [x] Dataset preparation with recording-based splits (ADR-004)
- [x] CNN classifier trained (89.7% precision, 93.8% recall, F1 91.7%)
- [x] Desktop detection app (PyQt6) with progressive labeling
- [x] Clustering exploration (feature extraction, k-means/HDBSCAN/GMM, visualization)
- [x] All tests passing (295+)

---

## Phase 8: Transformer + VQ-VAE for USV Concept Discovery

> **Architecture (v2 — two-phase pipeline):** This implements a fundamentally different approach from the v1 end-to-end VQ-VAE (d_model=64, K=512, ~437K params). The v2 pipeline trains a large autoregressive transformer first (~25-30M params) on raw spectrogram columns, then applies a VQ-VAE to its internal hidden states as a post-hoc interpretability tool. See `docs/plans/theoretical_guide.md` for the full scientific rationale.
>
> **Why this order:** Training the transformer first (without discretization) lets it freely learn whatever representations are most useful for prediction. The VQ-VAE then discovers discrete "concepts" within those learned representations. This avoids forcing discretization before the model knows what matters.
>
> **Existing v1 code:** The `usv_language/` directory contains the v1 implementation (63 tests). V1 code should be archived before v2 implementation begins. Some infrastructure (HDF5 loading, preprocessing, test patterns) may be reusable.

### 8.1 Data Preparation Pipeline

**What:** Extract bout-level spectrograms from raw WAV files using CNN detection results, normalize per-frequency-bin, and create PyTorch datasets with length-bucketed batching for transformer training. Bouts are continuous recording segments containing clusters of USV activity with surrounding context — preserving inter-USV timing, silence gaps, and transitions.
**Status:** DONE
**Review Tier:** 2
**Depends on:** Phase 1 (CNN detection results for bout grouping)

**Key files:**
- `usv_language/data/bout_extractor.py` — BoutExtractionConfig, Bout, BoutExtractor
- `usv_language/data/spectrogram.py` — BoutSpectrogramConfig, compute_bout_spectrogram()
- `usv_language/data/normalization.py` — NormalizationStats, per-frequency Welford's algorithm
- `usv_language/data/dataset.py` — USVBoutDataset (PyTorch), BucketedBatchSampler, augmentation
- `usv_language/data/prepare_data.py` — CLI end-to-end pipeline
- `usv_language/configs/default_config.yaml` — Master config reference

/implement USV Bout Data Preparation Pipeline

Create the full data pipeline for the transformer: bout extraction → spectrogram computation → normalization → PyTorch dataset with bucketed batching. This lives in `usv_language/` and replaces the v1 data pipeline.

**Context:** Per ADR-001, sample rate is 300 kHz. Per ADR-002, STFT uses n_fft=512, hop=128, 20-120 kHz (~170 freq bins). Use the EXACT same STFT implementation as the existing CNN pipeline for consistency. See `docs/plans/theoretical_guide.md` §"Input Data: Bout-Level Spectrograms" for design rationale.

**Files to create:**

1. `usv_language/data/bout_extractor.py` (NEW) — Bout extraction from detection results

```python
@dataclass
class BoutExtractionConfig:
    bout_gap_threshold_ms: float = 500.0    # max gap between USVs in same bout
    context_padding_ms: float = 200.0       # padding before first / after last USV
    min_bout_duration_ms: float = 50.0      # discard bouts shorter than this
    max_bout_duration_ms: float = 10000.0   # split bouts longer than this
```

Logic:
1. Load USV detection results (CSV/JSON from CNN pipeline) containing (file, start_time, end_time) for each detected USV
2. Group USVs into bouts: sort by start_time, merge USVs within `bout_gap_threshold` (500ms) into single bout
3. Each bout: bout_start = first_USV_start − padding, bout_end = last_USV_end + padding (clamped to file bounds)
4. Extract raw audio segment from WAV file for each bout

Output: List of bout audio segments with metadata (source file, start/end times, USV count per bout).

2. `usv_language/data/spectrogram.py` (NEW) — Bout audio → log-magnitude spectrogram

Logic:
1. Compute STFT: sr=300000, n_fft=512, hop_length=128 (→ 0.427 ms/frame)
2. Magnitude → log scale: `S_db = 20 * log10(|S| + 1e-10)`
3. Crop frequency axis to 20-120 kHz → ~170 frequency bins
4. Output shape: `(n_freq=170, n_frames=T)` where T varies by bout duration

**Important:** Use the EXACT same STFT implementation (librosa or scipy) and parameters as the existing CNN detection pipeline.

3. `usv_language/data/normalization.py` (NEW) — Per-frequency-bin normalization

Logic:
1. First pass: compute mean and std for each of 170 frequency bins across all TRAINING spectrograms
2. Save statistics to npz file for reproducibility
3. Normalize: `S_norm[f, t] = (S[f, t] - mean[f]) / (std[f] + 1e-8)`

**Important:** Compute statistics on TRAINING set only. Apply same stats to val/test.

4. `usv_language/data/dataset.py` (NEW) — PyTorch Dataset with bucketed batching

```python
@dataclass
class TransformerDataConfig:
    max_seq_len: int = 512
    overlap_ratio: float = 0.5      # overlap between chunks from same bout
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
```

Logic:
1. Take normalized spectrograms (170 × T), transpose to (T × 170) — each row = one "token"
2. Chunk into windows of `max_seq_len` frames (512) with configurable overlap (50%, stride=256)
3. Bouts shorter than `max_seq_len` → pad, create attention mask (1=real, 0=padding)
4. For next-column prediction: input = frames[0:T-1], target = frames[1:T]
5. Length-bucketed batching: sort chunks by length into ~6-8 buckets (64, 128, 192, 256, 384, 512), pad only to longest in batch

Data augmentation (applied p=0.5, training only):
- Gaussian noise: N(0, σ) at SNR ~15-20 dB
- Gain perturbation: multiply by 10^(g/20), g ~ Uniform(-3, 3) dB
- Frequency masking: zero out 1-2 bands of ~20-30 bins
- Time masking: zero out 1-2 spans of ~10% sequence length

5. `usv_language/data/prepare_data.py` (NEW) — End-to-end pipeline script

CLI that runs: bout extraction → spectrogram computation → normalization → save splits.
Accepts: path to WAV dir, path to detection results, output dir.
Saves: processed dataset splits, normalization stats, dataset summary.

**Test plan:**
```
1. Bout extraction correctly groups USVs within gap threshold on synthetic data
2. Bouts longer than max_bout_duration are split
3. Spectrogram output shape is (170, T) with correct frequency range
4. Normalization produces zero mean, unit variance on training set
5. Chunking with overlap=0.5 produces correct number of chunks for known-length bouts
6. Attention masks correctly mark padding positions
7. Bucketed batching pads to max-in-batch, not globally
8. Augmentation only applied during training
```

**Exit criteria:**
- [ ] `prepare_data.py` runs end-to-end on test data
- [ ] Output spectrograms visually match expected frequency content
- [ ] Normalization stats saved and reproducible
- [ ] DataLoader yields batches with correct shapes
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 8.2 Autoregressive Transformer

**What:** GPT-style autoregressive transformer for next-spectrogram-column prediction. Processes sequences of spectrogram columns (170-dim vectors) and predicts the next column given all previous columns. Includes training loop with warmup/cosine schedule and hidden state extraction for Phase 2 (VQ-VAE).
**Status:** DONE
**Review Tier:** 3
**Depends on:** Phase 8.1

**Key files:**
- `usv_language/models/transformer.py` — TransformerConfig, TransformerBlock, SpectrogramTransformer (~25.6M params)
- `usv_language/training/train_transformer.py` — CLI training with masked MSE, CosineWarmupScheduler, AdamW, early stopping
- `usv_language/training/extract_hidden_states.py` — Hidden state extraction as memory-mapped numpy arrays + metadata JSON
- 11 tests passing

**Key design:** Pre-norm architecture, GELU activation, causal masking, learned positional embeddings, True=padding mask convention.

/implement Spectrogram Autoregressive Transformer

Build the transformer model, training loop, and hidden state extraction pipeline. This is the core model — the transformer learns to predict "what comes next" in the acoustic stream, developing internal representations that capture USV structure.

**Context:** See `docs/plans/theoretical_guide.md` §"Phase 1: Transformer Architecture" for design decisions. Pre-norm architecture for training stability. Causal attention matches the scientific question: "given what came before, what comes next?"

**Files to create:**

1. `usv_language/models/transformer.py` (NEW) — SpectrogramTransformer

```python
@dataclass
class TransformerConfig:
    n_freq: int = 170
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ffn: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
```

Architecture (~25-30M params):
- **Input projection:** Linear(170 → 512) → GELU → LayerNorm
- **Positional embeddings:** Learned, nn.Embedding(max_seq_len, d_model)
- **8× Transformer blocks (pre-norm):**
  - LayerNorm → MultiheadAttention(512, 8 heads, dropout=0.1) with causal mask + residual
  - LayerNorm → FFN: Linear(512→2048) → GELU → Dropout → Linear(2048→512) → Dropout + residual
- **Output head:** LayerNorm → Linear(512 → 170)
- **Causal mask:** `torch.triu(ones(max_seq_len, max_seq_len), diagonal=1).bool()`, registered as buffer

Forward pass:
```python
def forward(self, x, attention_mask=None, return_hidden_states=False):
    # x: (batch, seq_len, 170)
    h = self.input_proj(x)                           # (batch, seq_len, 512)
    positions = torch.arange(x.size(1), device=x.device)
    h = h + self.pos_embed(positions)

    hidden_states = [] if return_hidden_states else None
    for block in self.blocks:
        h = block(h, causal_mask=self.causal_mask, padding_mask=attention_mask)
        if return_hidden_states:
            hidden_states.append(h)

    output = self.output_head(h)                     # (batch, seq_len, 170)
    return output, hidden_states
```

Critical: `return_hidden_states=False` by default to save memory during Phase 1 training. Only set True during Phase 2 extraction.

**If MSE produces blurry predictions:** Consider upgrading to a GMM output head (K=5-10 mixture components) per `docs/plans/theoretical_guide.md`.

2. `usv_language/training/train_transformer.py` (NEW) — Training script

Training configuration:
- **Loss:** MSE between predicted and actual next columns, masked for padding positions
- **Optimizer:** AdamW with parameter groups (weight decay=0.01 for weight matrices only, 0.0 for biases/norms)
- **LR schedule:** Linear warmup (2000 steps) → cosine decay to 1e-6, peak lr=1e-4
- **Gradient clipping:** max norm 1.0
- **Checkpointing:** Every 5 epochs + best model (by val loss) + training state for resume
- **Early stopping:** 20 epochs patience
- **Logging:** Training/val loss, LR, per-frequency-bin error distribution. Every 10 epochs: predicted vs actual spectrogram visualizations.
- **Multi-GPU:** Support DataParallel/DistributedDataParallel via CLI flags for HPC.

3. `usv_language/training/extract_hidden_states.py` (NEW) — Hidden state extraction

After transformer training, extract and save hidden states for VQ-VAE:
1. Load frozen transformer (best checkpoint), `model.eval()`, `torch.no_grad()`
2. Run every sequence chunk through transformer with `return_hidden_states=True`
3. Extract from target layers (default: layer 4, also extract 2, 6, 8 for comparison)
4. Save as memory-mapped numpy arrays (these can be large: ~1GB per layer for 500K frames)
5. Save metadata: bout_id, chunk_id, frame index, original timestamp per hidden state

Output format per layer L:
- `hidden_states_layer{L}.npy` — shape (total_frames, 512)
- `metadata.json` — mapping from frame index to (bout_id, chunk_id, frame_within_chunk, timestamp)

**Test plan:**
```
1. Forward pass on dummy data produces correct output shape (batch, seq_len, 170)
2. Causal mask prevents attending to future positions (verify with gradient check)
3. Parameter count matches expectations (~25-30M)
4. Training loss decreases on single-batch overfit (convergence sanity check)
5. Hidden states extraction produces correct shapes per layer
6. Padding mask correctly combined with causal mask
7. Checkpoint save/resume preserves optimizer and scheduler state
8. Gradient clipping is active (verify max grad norm)
```

**Exit criteria:**
- [ ] Model parameter count verified (~25-30M)
- [ ] Forward pass on dummy data: correct shapes, no errors
- [ ] 5-epoch training run: loss decreases, predicted spectrograms show structure
- [ ] Hidden state extraction produces correct output files with metadata
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 8.3 VQ-VAE on Hidden States

**What:** VQ-VAE that operates on transformer hidden state vectors to discover discrete "concepts." Each codebook entry becomes an interpretable recurring pattern the transformer has learned to recognize. Uses EMA codebook updates, dead code reinitialization, L2 normalization, and k-means initialization to prevent codebook collapse.
**Status:** DONE
**Review Tier:** 3
**Depends on:** Phase 8.2

**Key files:**
- `usv_language/models/vqvae.py` — VQVAEConfig, VectorQuantizerV2, HiddenStateVQVAE (~820K params)
- `usv_language/training/train_vqvae.py` — HiddenStateDataset, training CLI with CosineWarmupScheduler
- `usv_language/training/compare_layers.py` — Multi-layer comparison with weighted scoring report
- `usv_language/tests/test_hidden_state_vqvae.py` — 21 tests (config, forward pass, STE gradients, EMA, dead code reset, k-means init, overfit, utilization, checkpointing, dataset windowing)

**Key design:** Fresh VectorQuantizerV2 (not v1 import), L2-normalized codebook with re-normalization after EMA update, raw commitment loss (beta applied by caller), sequential val split (no temporal leakage), codebook excluded from optimizer (EMA-only), k-means++ init on GPU tensors.

/implement Hidden State VQ-VAE

Build the VQ-VAE model, training loop, and multi-layer comparison script. This is the interpretability tool — it compresses the transformer's continuous internal representations into a small discrete codebook.

**Context:** See `docs/plans/theoretical_guide.md` §"Phase 2: VQ-VAE on Transformer Hidden States" for design decisions. Start with layer 4 (middle layer = mid-level concepts). K=64 codebook entries (traditional USV taxonomy has ~10-15 types, K=64 gives headroom for finer subtypes). Codebook collapse is the #1 failure mode — multiple defenses required.

**Files to create:**

1. `usv_language/models/vqvae.py` (NEW) — HiddenStateVQVAE

```python
@dataclass
class VQVAEConfig:
    d_model: int = 512              # transformer hidden size (input dimension)
    codebook_size: int = 64         # K — number of codebook entries
    codebook_dim: int = 64          # D — dimension of each entry
    commitment_weight: float = 0.25 # β
    ema_decay: float = 0.99         # γ for EMA updates
    dead_code_threshold: float = 2.0
    use_conv_encoder: bool = True   # 1D conv for temporal context
    conv_kernel_size: int = 5
```

Architecture:
- **Encoder:** Conv1d(512→256, kernel=5, padding=2) → GELU → Linear(256→64) → L2-normalize
- **Vector Quantizer:** K=64 entries of dimension D=64
  - Nearest-neighbor lookup: k* = argmin_k ||z_e - e_k||²
  - Straight-through estimator: z_q = z_e + (e_{k*} - z_e).detach()
  - EMA codebook updates (γ=0.99) instead of gradient-based learning
  - Dead code reinitialization: entries with cluster_size < threshold → reinitialize from encoder outputs
  - Commitment loss: ||z_e - sg(e_{k*})||²
- **Decoder:** Linear(64→256) → GELU → Linear(256→512)

Forward pass:
```python
def forward(self, hidden_states):
    # hidden_states: (batch, seq_len, 512)
    z_e = self.encoder(hidden_states)           # (batch, seq_len, 64)
    z_e = F.normalize(z_e, dim=-1)              # L2 normalize
    z_q, indices, commit_loss = self.quantizer(z_e)
    decoded = self.decoder(z_q)                 # (batch, seq_len, 512)

    recon_loss = F.mse_loss(decoded, hidden_states)
    total_loss = recon_loss + self.config.commitment_weight * commit_loss

    return {
        'decoded': decoded,
        'indices': indices,         # (batch, seq_len) — codebook assignments
        'z_e': z_e,                 # encoder outputs (for analysis)
        'z_q': z_q,                 # quantized outputs
        'recon_loss': recon_loss,
        'commit_loss': commit_loss,
        'total_loss': total_loss,
    }
```

K-means initialization:
```python
def initialize_codebook_from_data(self, encoder_outputs):
    """Run k-means on ~5000 encoder outputs to initialize codebook before training."""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=self.config.codebook_size, n_init=10)
    kmeans.fit(encoder_outputs.numpy())
    self.quantizer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_)
```

**Codebook collapse prevention (all used simultaneously):**
1. EMA codebook updates (γ=0.99)
2. Dead code reinitialization (threshold=2.0)
3. K-means initialization before training
4. L2-normalization of encoder outputs and codebook vectors
5. Optional: entropy regularization to encourage uniform utilization

**Fallback — Finite Scalar Quantization (FSQ):** If collapse persists despite all defenses, FSQ (Mentzer et al., ICLR 2024) achieves 100% utilization by design. Rounds each scalar channel to fixed levels instead of nearest-neighbor lookup. See `docs/plans/theoretical_guide.md` for details.

2. `usv_language/training/train_vqvae.py` (NEW) — VQ-VAE training

Training configuration:
- Load pre-extracted hidden states from 8.2 (chosen layer, default: layer 4)
- K-means initialization: encode ~5000 samples, run k-means, set codebook
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01), warmup 500 steps, cosine decay
- Gradient clipping: max norm 1.0
- Batch size: 256 (large batches OK — inputs are just 512-dim vectors)
- Expected convergence: 50-100 epochs, minutes on GPU

Monitoring (every N steps):
- `recon_loss`: MSE between decoded and original hidden states
- `commit_loss`: commitment loss
- `codebook_perplexity`: exp(entropy of code usage distribution) — target > 0.5 × K
- `codebook_utilization`: fraction of codes used in batch — target > 90%
- `code_usage_histogram`: distribution across entries (should be roughly uniform)
- Dead code count: entries reinitialized

3. `usv_language/training/compare_layers.py` (NEW) — Multi-layer comparison

Train VQ-VAE separately on hidden states from layers 2, 4, 6, 8 with identical hyperparameters. Generate comparison report: table of metrics per layer, recommendation for best layer. Criterion: highest perplexity + highest utilization + lowest recon_loss, with emphasis on perplexity (interpretability).

**Test plan:**
```
1. VQ-VAE forward pass produces correct output shapes
2. Codebook indices are valid (0 to K-1)
3. Straight-through estimator: gradients flow through to encoder (verify with backward pass)
4. Dead code reinitialization triggers when assignments fall below threshold
5. K-means initialization sets codebook weights from data
6. Reconstruction loss decreases on single-batch overfit
7. EMA updates modify codebook vectors during training
8. Multi-layer comparison produces report with all expected metrics
```

**Exit criteria:**
- [ ] Forward pass on dummy hidden states: correct shapes, valid indices
- [ ] Gradient flow verified through straight-through estimator
- [ ] Single-batch overfit: reconstruction loss < 0.01
- [ ] Codebook utilization > 50% on synthetic data
- [ ] Compare_layers script produces comparison table for 4 layers
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 8.4 Analysis & Interpretation Tools

**What:** Comprehensive analysis suite for probing the learned representations: decode codebook entries to spectrograms, visualize exemplar galleries, analyze sequential structure (Zipf's law, transition matrices, entropy rate, mutual information), concept manipulation experiments, context-dependent analysis, and compositionality tests.
**Status:** DONE (2026-02-21)
**Review Tier:** 2
**Depends on:** Phase 8.3

/implement VQ-VAE Analysis & Interpretation Tools

Build the full analysis pipeline for interpreting VQ-VAE codebook entries and code sequences. These tools answer the core research question: do mouse USVs contain language-like structure?

**Context:** See `docs/plans/theoretical_guide.md` §"Phase 3: Analysis and Interpretation" for scientific background. Key tests: Zipf's law (α ≈ 1 for natural language), transition entropy vs. maximum entropy, excess entropy for long-range structure.

**Files to create:**

1. `usv_language/analysis/codebook_viz.py` (NEW) — Codebook visualization

Logic:
- **Decode codebook entries through full pipeline:** For each entry e_k → VQ-VAE decoder → reconstructed hidden state h_k → pass through remaining transformer layers → output head → predicted spectrogram column. This reveals what acoustic continuation each concept implies.
- **Exemplar galleries:** For each entry k, find N=10 nearest encoder outputs (L2 distance), extract surrounding spectrogram context (±50 frames) for each exemplar.
- **t-SNE/UMAP visualization:** Project all K codebook vectors to 2D, color by mean frequency of exemplars, annotate with decoded spectrogram thumbnails.

Output: PNG figures + HTML gallery for browsing.

2. `usv_language/analysis/sequence_analysis.py` (NEW) — Sequential structure analysis

Logic:
- **Extract code sequences:** All bouts → frozen transformer → hidden states → frozen VQ-VAE → code indices per frame. Each bout becomes a sequence of integers [c_0, c_1, ..., c_T].
- **Zipf's law:** Count code frequencies, rank by frequency, plot log(freq) vs log(rank), fit power law (frequency ∝ rank^(-α)), report α. Compare to α ≈ 1 (natural language).
- **Transition analysis:** Compute bigram transition matrix P(c_{t+1}|c_t), transition entropy H(C_{t+1}|C_t), mutual information I(C_t; C_{t+1}). Extend to trigrams.
- **Entropy rate:** Approximate h = lim H(C_n|C_{n-1},...,C_1) with increasing context (1-gram through 8-gram). Plot entropy rate vs. context length — should decrease and plateau.
- **Excess entropy:** Mutual information between past and future halves of sequences. Higher = more complex long-range structure.

Output: PNGs + JSON numerical results + console summary.

3. `usv_language/analysis/concept_manipulation.py` (NEW) — Concept injection experiments

Logic:
- **Single concept injection:** Take real sequence, replace hidden state at time t with decoded codebook entry k, pass through remaining layers, generate predictions autoregressively for t+1..t+N. Visualize predicted spectrogram.
- **Concept scanning:** For fixed context (0..t-1), inject each of K entries at position t, record predicted next frame for each. Create K × 170 matrix showing what each concept predicts. Cluster predictions.
- **Top-k analysis:** For each time step, record distances to all K entries, identify top-4 competing concepts and their distances. Visualize transition points where winning concept changes.

Output: Interactive matplotlib/plotly visualizations, saved as HTML.

4. `usv_language/analysis/context_analysis.py` (NEW) — Context-dependent analysis

Logic (requires metadata: mouse ID, sex, strain, social context):
- Group code sequences by metadata variable
- Compare code frequency distributions across groups (chi-squared, KL divergence)
- Compare transition matrices across groups
- Report most differentially used codes
- If no metadata available: skip, note in output.

5. `usv_language/analysis/compositionality.py` (NEW) — Compositionality tests

Logic:
- **Bigram productivity:** Unique observed bigrams vs. theoretically possible (K²)
- **Held-out bigram test:** Can VQ-VAE decode unseen bigram combinations into meaningful spectrograms?
- **Positional independence:** Do codes maintain identity regardless of position? Compare exemplars of same code at different positions.

**Test plan:**
```
1. Codebook decoding produces spectrogram columns of shape (170,)
2. Zipf analysis correctly identifies power-law distribution in synthetic data
3. Transition matrix is K×K and rows sum to ~1.0
4. Entropy rate decreases with increasing context length on synthetic Markov chain
5. Code sequence extraction produces valid integer sequences (0 to K-1)
6. Concept injection produces spectrogram of correct shape
7. n-gram extraction handles sequences shorter than n gracefully
```

**Exit criteria:**
- [ ] All visualizations generate without errors on synthetic data
- [ ] Zipf analysis correctly recovers known α from synthetic power-law data
- [ ] Transition matrix heatmap is readable and informative
- [ ] Concept manipulation produces interpretable predicted spectrograms
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### Phase 8 Project Structure

```
usv_language/
├── configs/
│   ├── default_config.yaml          # All hyperparameters in one place
│   └── experiment_configs/          # Per-experiment overrides
├── data/
│   ├── bout_extractor.py           # 8.1: Bout extraction
│   ├── spectrogram.py              # 8.1: STFT computation
│   ├── normalization.py            # 8.1: Per-freq-bin normalization
│   ├── dataset.py                  # 8.1: PyTorch Dataset + bucketed batching
│   └── prepare_data.py             # 8.1: End-to-end pipeline script
├── models/
│   ├── transformer.py              # 8.2: SpectrogramTransformer
│   └── vqvae.py                    # 8.3: HiddenStateVQVAE
├── training/
│   ├── train_transformer.py        # 8.2: Transformer training loop
│   ├── extract_hidden_states.py    # 8.2: Hidden state extraction
│   ├── train_vqvae.py              # 8.3: VQ-VAE training loop
│   └── compare_layers.py           # 8.3: Multi-layer comparison
├── analysis/
│   ├── codebook_viz.py             # 8.4: Codebook visualization
│   ├── sequence_analysis.py        # 8.4: Zipf, transitions, entropy
│   ├── concept_manipulation.py     # 8.4: Concept injection experiments
│   ├── context_analysis.py         # 8.4: Context-dependent analysis
│   └── compositionality.py         # 8.4: Compositionality tests
├── utils/
│   ├── logging.py                  # Tensorboard/wandb integration
│   ├── checkpointing.py            # Save/load model checkpoints
│   └── visualization.py            # Shared plotting utilities
└── tests/
```

### Phase 8 Master Configuration

**File:** `usv_language/configs/default_config.yaml`

All hyperparameters in a single YAML config for reproducibility:

```yaml
data:
  bout_gap_threshold_ms: 500
  context_padding_ms: 200
  min_bout_duration_ms: 50
  max_bout_duration_ms: 10000
  sr: 300000
  n_fft: 512
  hop_length: 128
  freq_min_hz: 20000
  freq_max_hz: 120000
  max_seq_len: 512
  overlap_ratio: 0.5
  batch_size: 32
  num_workers: 4
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

augmentation:
  enabled: true
  probability: 0.5
  gaussian_noise_snr_db: 17.5
  gain_perturbation_db: 3.0
  freq_mask_bins: 25
  freq_mask_count: 2
  time_mask_ratio: 0.1
  time_mask_count: 2

transformer:
  n_freq: 170
  d_model: 512
  n_heads: 8
  n_layers: 8
  d_ffn: 2048
  dropout: 0.1
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 2000
  min_lr: 1e-6
  max_epochs: 200
  early_stopping_patience: 20
  gradient_clip_norm: 1.0
  checkpoint_every_n_epochs: 5

vqvae:
  codebook_size: 64
  codebook_dim: 64
  commitment_weight: 0.25
  ema_decay: 0.99
  dead_code_threshold: 2.0
  use_conv_encoder: true
  conv_kernel_size: 5
  extract_from_layers: [2, 4, 6, 8]
  default_layer: 4
  learning_rate: 3e-4
  warmup_steps: 500
  max_epochs: 100
  batch_size: 256

analysis:
  n_exemplars: 10
  context_frames: 50
  zipf_min_count: 5
  max_ngram_order: 8
  manipulation_n_future_steps: 50
```

### Phase 8 Implementation Order

Execute tasks in this order, verifying each before moving on:

1. **8.1a** — Spectrogram extraction (reuse existing STFT code where possible)
2. **8.1b** — Bout extraction
3. **8.1c** — Normalization
4. **8.1d** — Dataset and DataLoader
5. **8.1e** — Data preparation integration script
6. **Verify:** Run prepare_data.py, inspect output spectrograms visually, confirm shapes
7. **8.2a** — Transformer model
8. **Verify:** Parameter count (~25-30M), forward pass on dummy data, output shapes
9. **8.2b** — Transformer training loop
10. **Verify:** Train ~5 epochs, loss decreases, inspect predicted vs actual spectrograms
11. **8.2c** — Hidden state extraction
12. **8.3a** — VQ-VAE model
13. **Verify:** Forward pass on dummy data, codebook assignment, gradient flow through STE
14. **8.3b** — VQ-VAE training loop
15. **8.3c** — Multi-layer comparison
16. **8.4a** — Codebook visualization
17. **8.4b** — Sequence analysis
18. **8.4c** — Concept manipulation
19. **8.4d** — Context analysis (if metadata available)
20. **8.4e** — Compositionality tests

### Phase 8 Dependencies

```
torch >= 2.0
torchaudio
librosa
scikit-learn       # k-means initialization
matplotlib
seaborn
plotly             # optional, interactive visualizations
umap-learn         # UMAP projections
tensorboard        # or wandb
pyyaml             # config loading
```

**Key decisions:** ADR-001 (300 kHz), ADR-002 (STFT), ADR-004 (split by recording). Note: ADR-007 (VQ-VAE codebook approach) needs updating to reflect v2 architecture.

---

## Phase 8 Gate

Before starting Phase 9:
- [ ] Data preparation pipeline (8.1) runs end-to-end on test data
- [ ] Transformer architecture (8.2) verified: correct param count, forward pass, loss decreases
- [ ] VQ-VAE (8.3) verified: codebook utilization > 50%, gradient flow through STE
- [ ] Analysis tools (8.4) generate visualizations without errors on synthetic data
- [ ] All Phase 8 tests pass
- [ ] py_compile passes on all new files

---

## Phase 9: Training Data Assembly Pipeline

### 9.1 Unified Dataset Assembly

**What:** Automate the full training data preparation cycle: collect app labels → generate jittered positive spectrograms → generate negative spectrograms from 3 sources → combine into a unified train/val/test dataset with recording-based splits and quality validation. Currently this requires manually running 3-4 separate scripts.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 5, Phase 6

/implement Training Data Assembly Pipeline

Create a unified training data assembly module that chains the existing individual steps into a single reproducible pipeline. Currently, preparing training data requires manually running several scripts (`generate_comprehensive_negatives.py`, `create_full_training_dataset.py`, etc.) and combining their outputs. This module orchestrates them into one command.

**Context:** Per ADR-004, splits must be by recording. Per ADR-008, negatives must come from 3 sources (random, inter-USV gap, low-energy). Per ADR-005, class weighting handles remaining imbalance during training.

**Files to create:**

1. `src/usv_spectrogram/dataset/assembler.py` (NEW) — Core assembly logic

```python
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class AssemblyConfig:
    """Configuration for training data assembly."""
    # Input paths
    labels_dir: Path            # Directory with app label JSON files
    wav_dir: Path               # WAV file directory

    # Jittering parameters
    jitter_n_samples: int = 5   # Jittered versions per positive
    jitter_window_ms: float = 40.0
    jitter_context_padding_ms: float = 20.0
    jitter_min_overlap: float = 0.5

    # Negative sampling (ADR-008: 3-source mix)
    neg_random_frac: float = 0.5     # 50% random chunks
    neg_inter_usv_frac: float = 0.3  # 30% inter-USV gaps
    neg_low_energy_frac: float = 0.2 # 20% low-energy regions
    neg_ratio: float = 1.0           # Negatives per positive

    # Splitting (ADR-004: by recording)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # Output
    output_dir: Path = Path("data/training/assembled")


@dataclass
class AssemblyReport:
    """Summary of assembly results."""
    total_positives: int
    total_negatives: int
    train_count: int
    val_count: int
    test_count: int
    n_recordings: int
    warnings: list[str]
    output_dir: Path


class DatasetAssembler:
    """Orchestrate full training data assembly."""

    def __init__(self, config: AssemblyConfig): ...

    def assemble(self) -> AssemblyReport:
        """
        Run full pipeline: collect → jitter → negate → split → validate.

        Steps:
        1. Collect labels from app JSON files (ADR-010 format)
        2. Group detections by source recording (ADR-004)
        3. Generate jittered positive spectrograms
        4. Generate negative spectrograms from 3 sources (ADR-008)
        5. Split recordings into train/val/test
        6. Write train.csv, val.csv, test.csv
        7. Run quality checks (no leakage, class balance, all files exist)
        8. Return AssemblyReport with statistics
        """
        ...

    def _collect_labels(self) -> pd.DataFrame: ...
    def _generate_positives(self, labels_df: pd.DataFrame) -> pd.DataFrame: ...
    def _generate_negatives(self, labels_df: pd.DataFrame, n_negatives: int) -> pd.DataFrame: ...
    def _create_splits(self, combined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...
    def _validate(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]: ...
```

2. `scripts/assemble_training_data.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/assemble_training_data.py \
      --labels-dir USV_Detections \
      --wav-dir "5970 USV" \
      --output-dir data/training/milestone_1 \
      --jitter-samples 5 \
      --neg-ratio 1.0 \
      --seed 42
```

Arguments:
- `--labels-dir` (required): Directory containing app label JSON files
- `--wav-dir` (required): Root WAV file directory
- `--output-dir` (default: `data/training/assembled`): Output directory
- `--jitter-samples` (default: 5): Jittered versions per positive
- `--neg-ratio` (default: 1.0): Negative-to-positive ratio
- `--seed` (default: 42): Random seed for reproducibility
- `--dry-run`: Show what would be assembled without writing files

Output structure:
```
data/training/milestone_1/
├── spectrograms/          # All spectrogram PNGs (positives + negatives)
├── train.csv              # Training split
├── val.csv                # Validation split
├── test.csv               # Test split
└── assembly_report.json   # Statistics and config used
```

3. `tests/test_dataset_assembler.py` (NEW) — Tests

**Integration points:**
- Reads label JSONs from `LabelStorage` format (ADR-010)
- Uses `SpectrogramExtractor` for spectrogram generation
- Uses recording-based splitting logic from `dataset/splits.py` (ADR-004)
- Output CSV format must match what `train_cnn.py` expects as input

**Test plan:**
```
1. Assembly with 3 mock recordings produces non-empty train/val/test CSVs
2. No recording appears in multiple splits (ADR-004 compliance)
3. Negatives come from all 3 source types (ADR-008 compliance)
4. Jittered positive count equals n_originals × jitter_n_samples
5. Dry-run mode produces no output files
6. AssemblyReport statistics are accurate (counts match actual files)
7. Assembly with empty labels_dir fails gracefully with clear error
8. All spectrogram paths in output CSVs point to existing files
```

**Exit criteria:**
- [ ] `assemble_training_data.py --dry-run` runs without error on real label data
- [ ] Full assembly produces train.csv, val.csv, test.csv in correct format
- [ ] Quality checks pass: no leakage, acceptable class balance, all files exist
- [ ] Output can be fed directly to `train_cnn.py` and training starts successfully
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 10: Active Learning Cycle Runner

### 10.1 Active Learning Automation

**What:** Orchestrate one complete active learning cycle: assemble training data → train CNN → evaluate → optimize threshold → mine hard negatives → generate comparison report. Automates the manual milestone workflow from `SCALING_TO_30K_ROADMAP.md`.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 9

/implement Active Learning Cycle Runner

Create a script that runs one complete active learning cycle. The scaling roadmap defines 5 milestones (2K → 5K → 10K → 20K → 30K labels), each requiring the same steps. This automates everything after labeling (which remains human work).

**Context:** Each milestone cycle is: label new data (human) → assemble training data → train CNN → evaluate → optimize threshold → mine hard negatives → report. Currently each step is a separate manual command. This module chains them.

**Files to create:**

1. `scripts/run_training_cycle.py` (NEW) — Orchestration script

```
Usage:
  .\.venv\Scripts\python.exe scripts/run_training_cycle.py \
      --labels-dir USV_Detections \
      --wav-dir "5970 USV" \
      --cycle-name milestone_1 \
      --model-size small \
      --output-dir runs/milestone_1 \
      --previous-model models/full_retrained_cnn/best_model.pt
```

The script chains these steps in sequence:
1. **Assemble** training data (uses `DatasetAssembler` from Phase 9)
2. **Train** CNN with specified model size (uses `USVTrainer`)
3. **Evaluate** on test set (uses `evaluate_model`)
4. **Optimize** threshold (threshold sweep on validation set)
5. **Mine** hard negatives from unlabeled recordings (uses existing mining logic)
6. **Compare** with previous model if `--previous-model` provided
7. **Report** — writes markdown summary with metrics and plots

Arguments:
- `--labels-dir` (required): App label JSON directory
- `--wav-dir` (required): WAV file directory
- `--cycle-name` (required): Name for this cycle (e.g., "milestone_1")
- `--model-size` (default: "small"): CNN model size per ADR-006 — "small" (101K params), "medium" (~400K), "large" (~1.6M)
- `--output-dir` (required): Where to write all outputs
- `--previous-model` (optional): Path to previous cycle's model for comparison
- `--epochs` (default: 50): Training epochs
- `--patience` (default: 10): Early stopping patience
- `--skip-mining` (flag): Skip hard negative mining step
- `--seed` (default: 42): Random seed

Output structure:
```
runs/milestone_1/
├── data/                        # Assembled training data
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── spectrograms/
├── model/                       # Trained model
│   ├── best_model.pt
│   └── training_history.json
├── eval/                        # Evaluation results
│   ├── metrics.json             # {precision, recall, f1, threshold}
│   ├── confusion_matrix.png
│   └── probability_distributions.png
├── threshold/                   # Threshold optimization
│   ├── threshold_sweep.png
│   └── optimal_threshold.json
├── hard_negatives/              # Mined hard negatives (if not skipped)
│   └── candidates.csv
├── comparison/                  # Comparison with previous model
│   └── model_comparison.json    # {f1_delta, precision_delta, recall_delta}
└── cycle_report.md              # Full markdown report
```

2. `src/usv_spectrogram/training/__init__.py` (NEW)
3. `src/usv_spectrogram/training/cycle_report.py` (NEW) — Report generator

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CycleMetrics:
    """Metrics from one training cycle."""
    cycle_name: str
    timestamp: str
    label_count: int
    model_size: str
    train_samples: int
    val_samples: int
    test_samples: int
    precision: float
    recall: float
    f1: float
    optimal_threshold: float
    hard_negatives_found: int
    previous_f1: float | None       # From comparison
    f1_delta: float | None          # Improvement over previous

def generate_cycle_report(metrics: CycleMetrics, output_path: Path) -> None:
    """
    Write markdown report summarizing the training cycle.

    Sections:
    - Dataset summary (label count, split sizes, class balance)
    - Training results (model size, convergence, best epoch)
    - Evaluation metrics (precision, recall, F1, threshold)
    - Comparison with previous model (if applicable)
    - Hard negative mining results
    - Recommended next steps
    """
    ...
```

**Integration points:**
- Uses `DatasetAssembler` from Phase 9 for data assembly
- Uses `USVTrainer` from `models/trainer.py` for training
- Uses `evaluate_model` from `models/evaluate.py` for evaluation
- Uses threshold sweep logic from `scripts/optimize_threshold.py`
- Uses hard negative mining from `scripts/mine_hard_negatives.py`
- Model size configs defined in `models/config.py` (ADR-006)

**Test plan:**
```
1. Cycle runner chains steps in correct order using mock components
2. Output directory structure is created with all expected subdirectories
3. cycle_report.md is generated with all required sections
4. Comparison metrics computed correctly when previous model provided
5. --skip-mining flag correctly skips the mining step
6. Failure in one step produces clear error and preserves partial results
```

**Exit criteria:**
- [ ] Full cycle completes on real data with small model
- [ ] `cycle_report.md` contains: label count, train/val/test sizes, precision/recall/F1, threshold, comparison delta
- [ ] Output model loadable by the detection app (`run_app.py`)
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 9–10 Gate

Before starting Phase 11:
- [ ] Dataset assembly (9.1) produces correct train/val/test splits from real labels
- [ ] Active learning cycle runner (10.1) completes end-to-end on real data
- [ ] At least one training cycle report generated with precision/recall/F1
- [ ] Output model loadable by the detection app
- [ ] All tests pass

---

## Phase 11: Transformer + VQ-VAE Real Data Execution

> This phase executes the v2 pipeline (Phase 8) on real data. The transformer is large (~25-30M params) and requires HPC/cloud GPU. Data preparation can run locally.

### 11.1 Bout Extraction & Spectrogram Preprocessing

**What:** Run the data preparation pipeline (Phase 8.1) on the full WAV dataset (~6,500 files). Extract bouts using CNN detection results, compute spectrograms, normalize, and save as HDF5 for efficient loading.
**Status:** BLOCKED (needs CNN detection results from batch detection; preprocessing can run locally)
**Review Tier:** 1
**Depends on:** Phase 8.1, Phase 13 (batch detection results)

/implement Bout Extraction & Preprocessing on Real Data

Run the data preparation pipeline on the real WAV dataset. Adapt for the project's WAV directory structure (nested `USV_1/` through `USV_5/` folders).

**Context:** Per ADR-001, sample rate is 300 kHz. Per ADR-002, STFT uses n_fft=512, hop=128, 20-120 kHz band (~170 freq bins). Bout extraction requires detection results (CSV/JSON) from the batch detection pipeline.

**Steps:**
1. Run `prepare_data.py` on full WAV dataset with detection results
2. Verify bout extraction statistics: number of bouts, duration distribution, USVs per bout
3. Verify spectrogram shapes: (170, T) per bout
4. Verify normalization: per-frequency-bin stats computed on training set only
5. Save processed dataset splits as HDF5 for efficient loading

**Files to create:**
- `usv_language/scripts/validate_preprocessing.py` (NEW) — Spot-check outputs

Validation script:
1. Load 5 random processed bouts
2. Verify spectrogram shape (170, T) and value ranges
3. Verify normalization statistics stored
4. Plot spectrograms as sanity check images
5. Print dataset summary: total bouts, total frames, duration stats, frames per split

**Exit criteria:**
- [ ] All WAV files with detections processed into bouts
- [ ] Validation script reports no issues on 5 random samples
- [ ] Dataset summary shows expected bout count and duration distribution
- [ ] DataLoader yields batches with correct shapes from processed data

---

### 11.2 Train Transformer on HPC

**What:** Execute transformer training on a GPU cluster following a staged approach to catch issues early.
**Status:** BLOCKED (needs HPC or cloud GPU — AMD RX 5700 insufficient for ~25-30M param model)
**Review Tier:** 3
**Depends on:** Phase 11.1

This is an **execution task**, not a code-writing task. Follow the staged training approach:

**Stage A: Overfit on 1 bout** (~1 hour on GPU)
```
python usv_language/training/train_transformer.py --max-bouts 1 --epochs 50
```
Verify: loss decreases, predicted spectrograms show structure, no NaN/Inf

**Stage B: Overfit on 10 bouts** (~few hours)
```
python usv_language/training/train_transformer.py --max-bouts 10 --epochs 30
```
Verify: loss generalizes slightly to held-out bouts, attention maps attend to relevant context

**Stage C: Train on 100 bouts** (~half day)
```
python usv_language/training/train_transformer.py --max-bouts 100 --epochs 50
```
Verify: validation loss tracks training loss, predicted spectrograms visually recognizable

**Stage D: Train on full dataset** (~1-2 days on A100)
```
python usv_language/training/train_transformer.py --epochs 200
```
Verify: validation loss plateaus, early stopping triggers

**Monitoring per stage:**
- Training/validation MSE loss curves
- Per-frequency-bin error distribution
- Predicted vs. actual spectrogram visualizations (5 random val samples every 10 epochs)
- Attention pattern visualization (verify model attends beyond immediately preceding frame)

**Exit criteria:**
- [ ] Stage A: loss converges, no numerical issues
- [ ] Stage C: validation loss tracks training loss, predictions visually meaningful
- [ ] Stage D: validation loss plateaus, best model saved, predicted spectrograms match structure of real USVs

---

### 11.3 Extract Hidden States & Train VQ-VAE

**What:** Extract hidden states from the frozen trained transformer, then train VQ-VAE on them. Compare layers 2, 4, 6, 8 to find the most interpretable representations.
**Status:** BLOCKED (needs trained transformer from 11.2)
**Review Tier:** 3
**Depends on:** Phase 11.2

**Step 1: Extract hidden states** (GPU, ~hours)
```
python usv_language/training/extract_hidden_states.py \
    --checkpoint best_model.pt \
    --layers 2 4 6 8 \
    --output-dir data/hidden_states/
```

Verify: Output files exist for each layer, shapes correct (total_frames, 512), metadata JSON valid.

**Storage estimate:** ~500K frames × 512 floats × 4 bytes = ~1GB per layer. For 4 layers: ~4GB. Use memory-mapped files.

**Step 2: Train VQ-VAE per layer** (GPU, ~minutes each)
```
python usv_language/training/compare_layers.py \
    --hidden-states-dir data/hidden_states/ \
    --layers 2 4 6 8 \
    --output-dir models/vqvae_comparison/
```

Verify per layer: codebook perplexity > 0.5×K (>32), utilization > 90%, recon_loss low.

**Step 3: Select best layer and train final VQ-VAE**
```
python usv_language/training/train_vqvae.py \
    --hidden-states data/hidden_states/hidden_states_layer4.npy \
    --output-dir models/vqvae_final/
```

**Exit criteria:**
- [ ] Hidden states extracted for layers 2, 4, 6, 8
- [ ] Layer comparison report generated with metrics table
- [ ] Best layer VQ-VAE: codebook perplexity > 32, utilization > 90%
- [ ] No codebook collapse (all entries assigned at least once per epoch)
- [ ] Final VQ-VAE model saved with codebook state

---

### 11.4 Run Analysis Pipeline

**What:** Run the full analysis suite (Phase 8.4) on the trained VQ-VAE to examine codebook entries, sequential structure, and language-like properties.
**Status:** BLOCKED (needs trained VQ-VAE from 11.3)
**Review Tier:** 2
**Depends on:** Phase 11.3

/implement VQ-VAE Analysis Report Generator

Run analysis tools and produce comprehensive report.

**Files to create:**

1. `usv_language/scripts/generate_analysis_report.py` (NEW)

This script should:
1. Load trained transformer + VQ-VAE checkpoints
2. Run codebook visualization (decode all K=64 entries, exemplar galleries, t-SNE/UMAP)
3. Encode full dataset to code sequences
4. Compute Zipf's law analysis (α exponent, log-log plot)
5. Compute transition matrix and per-code entropy
6. Find top-20 n-grams (bigrams, trigrams, 4-grams)
7. Compute entropy rate at context lengths 1-8
8. Compute excess entropy (mutual information between past and future halves)
9. Run concept scanning on 5 representative sequences
10. If wild/lab population labels available: compute metrics separately, test for differences
11. Write markdown report with all figures and statistics

Output structure:
```
usv_language/results/
├── codebook/
│   ├── decoded_entries.png           # Decoded spectrogram columns for all K entries
│   ├── exemplar_galleries/           # N=10 nearest exemplars per entry
│   ├── codebook_usage.png            # Usage histogram
│   └── codebook_tsne.png             # 2D projection of codebook vectors
├── sequences/
│   ├── zipf_plot.png                 # Log(freq) vs log(rank)
│   ├── transition_matrix.png         # K × K heatmap
│   ├── transition_entropy.png        # Per-code entropy
│   ├── entropy_rate.png              # Entropy rate vs context length
│   ├── excess_entropy.json           # Excess entropy value
│   └── top_ngrams.txt               # Most common n-grams
├── manipulation/
│   ├── concept_scanning/             # K × 170 prediction matrices
│   └── top_k_analysis/              # Competing concepts over time
├── population_comparison/            # Wild vs lab (if labels available)
│   ├── code_usage_comparison.png
│   └── transition_comparison.png
└── analysis_report.md                # Full report with all figures
```

**Exit criteria:**
- [ ] Analysis report generated without errors
- [ ] Codebook utilization > 50% confirmed
- [ ] Zipf's law α exponent reported (compare to α ≈ 1)
- [ ] Transition matrix shows non-uniform structure
- [ ] At least some n-grams appear significantly more often than chance (z-score > 3)
- [ ] Entropy rate decreases with context length (evidence of sequential structure)
- [ ] analysis_report.md is self-contained and interpretable

---

## Phase 11 Gate

Before starting Phase 12:
- [ ] Transformer trained on real bout data (Stage D), validation loss plateaued
- [ ] Hidden states extracted for layers 2, 4, 6, 8
- [ ] VQ-VAE trained on best layer: codebook perplexity > 32, utilization > 90%
- [ ] Analysis report generated: Zipf exponent, transition matrix, entropy rate
- [ ] At least some n-grams appear significantly more often than chance (z-score > 3)
- [ ] Batch detection results available for all WAV files (Phase 13)

---

## Phase 12: Cross-Population USV Comparison

### 12.1 Population-Level USV Analysis

**What:** Statistical comparison of USV repertoires between wild and lab mouse populations. This is the core research question: do wild mice vocalize differently than lab mice? Uses CNN features (Phase 7) and optionally transformer/VQ-VAE codes (Phase 11).
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 7 (clustering); Phase 11.4 optional

/implement Cross-Population USV Comparison

Create an analysis module that compares USV characteristics across mouse populations (wild vs lab). Uses two complementary approaches:
1. CNN feature space analysis (clustering-based, from Phase 7)
2. Transformer/VQ-VAE code analysis (language-model-based, from Phase 11, optional)

**Context:** The project records USVs from both wild-caught and lab-bred mice. Comparing vocalizations reveals whether domestication altered vocal repertoires. Population labels come from the WAV directory structure or metadata CSV.

**Files to create:**

1. `src/usv_spectrogram/analysis/__init__.py` (NEW)
2. `src/usv_spectrogram/analysis/population_comparison.py` (NEW) — Core analysis

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class PopulationComparisonConfig:
    features_path: Path               # CNN features .npy from Phase 7
    metadata_path: Path               # Metadata CSV with population column
    population_column: str = "population"
    populations: tuple[str, ...] = ("wild", "lab")
    n_permutations: int = 1000        # For permutation tests
    random_seed: int = 42
    output_dir: Path = Path("analysis/population_comparison")


@dataclass
class ComparisonReport:
    feature_distribution_test: dict   # {statistic, p_value, method}
    call_characteristics: dict        # {duration: {wild_median, lab_median, p_value}, ...}
    cluster_usage: dict | None        # Cluster proportion chi-squared test
    vqvae_codes: dict | None          # VQ-VAE comparison (if available)
    summary: str                      # Plain-language summary


class PopulationComparison:
    """Compare USV repertoires across mouse populations."""

    def __init__(self, config: PopulationComparisonConfig): ...

    def run_full_comparison(self) -> ComparisonReport:
        """Run all analyses and generate report."""
        ...

    def compare_feature_distributions(self) -> dict:
        """
        Test if CNN feature distributions differ between populations.
        Methods:
        - MANOVA on first 10 PCA components
        - Permutation test on centroid distance
        - Per-cluster proportion comparison (chi-squared)
        """
        ...

    def compare_call_characteristics(self) -> dict:
        """
        Compare basic USV properties: duration, peak frequency, bandwidth, call rate.
        Uses Mann-Whitney U test (non-parametric) for each property.
        """
        ...

    def compare_cluster_usage(self, cluster_labels: np.ndarray) -> dict:
        """
        Compare which USV types (clusters) each population uses.
        Chi-squared test on cluster frequency distributions.
        """
        ...

    def compare_vqvae_codes(self, code_sequences: dict) -> dict:
        """
        Compare VQ-VAE code usage patterns between populations.
        Only available if VQ-VAE has been trained (Phase 11).
        - Code frequency distribution comparison
        - Transition matrix differences (Frobenius norm + permutation test)
        - Population-unique n-grams
        """
        ...
```

3. `scripts/compare_populations.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/compare_populations.py \
      --features analysis/clustering/features.npy \
      --metadata analysis/clustering/metadata.csv \
      --population-column population \
      --output analysis/population_comparison
```

Arguments:
- `--features` (required): CNN features .npy from clustering pipeline
- `--metadata` (required): Metadata CSV with population labels
- `--population-column` (default: "population"): Column name for population labels
- `--cluster-assignments` (optional): Cluster assignments CSV from Phase 7
- `--vqvae-codes` (optional): VQ-VAE code sequences directory from Phase 11
- `--output` (default: `analysis/population_comparison`): Output directory

Output:
```
analysis/population_comparison/
├── feature_space_by_population.png      # t-SNE/UMAP colored by population
├── call_characteristics.png             # Box plots of duration, frequency, etc.
├── cluster_usage_comparison.png         # Bar chart of cluster proportions per population
├── statistical_tests.json               # All test results with p-values
└── comparison_report.md                 # Plain-language summary
```

4. `tests/test_population_comparison.py` (NEW) — Tests

**Test plan:**
```
1. Two synthetic populations with identical distributions -> p-value > 0.05 (no significant difference)
2. Two synthetic populations with very different distributions -> p-value < 0.05 (detected difference)
3. Missing population column raises clear error
4. Single-population data raises clear error (need at least 2 populations)
5. ComparisonReport contains all required fields
6. Permutation test produces p-values in [0, 1]
7. compare_call_characteristics detects known duration differences in synthetic data
```

**Exit criteria:**
- [ ] Comparison runs without error on clustering features from Phase 7
- [ ] Report includes statistical tests with p-values for each comparison dimension
- [ ] Visualizations show population-colored feature space
- [ ] `comparison_report.md` provides plain-language interpretation ("wild mice produce significantly more X-type calls, p < 0.01")
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 13: Batch Detection Pipeline

### 13.1 Large-Scale USV Detection

**What:** Run the full detection workflow (energy detection → CNN classification → export) on thousands of WAV files non-interactively. The detection app handles one file at a time; this processes entire collections with progress tracking and error recovery.
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 10 (for a well-trained model)

/implement Batch Detection Pipeline

Create a batch processing pipeline that runs headless detection on large WAV collections. Produces per-file detection JSONs (ADR-010 format) and a summary CSV for downstream analysis.

**Context:** The project has ~6,500 WAV files. Research analysis requires detection results for all of them. The desktop app is interactive (one-at-a-time); this pipeline processes all files with a single command, skipping already-processed files for incremental runs.

**Files to create:**

1. `src/usv_spectrogram/pipeline/__init__.py` (NEW)
2. `src/usv_spectrogram/pipeline/batch_detector.py` (NEW) — Core batch logic

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BatchConfig:
    wav_dir: Path                     # Root WAV directory (searched recursively)
    model_path: Path                  # Trained CNN model (.pt)
    output_dir: Path                  # Output directory

    # Detection parameters (ADR-003)
    high_threshold: float = 0.10      # CNN high threshold for hysteresis
    low_threshold: float = 0.05       # CNN low threshold for hysteresis

    # Energy detector params (passed through to DetectionConfig)
    energy_threshold_db: float = -60.0  # ADR-003
    energy_mode: str = "peak"           # ADR-012
    max_bandwidth_hz: float = 20_000

    # Processing
    batch_size: int = 32              # CNN inference batch size
    skip_existing: bool = True        # Skip already-processed files
    save_probability_curve: bool = True

    # Sample rate (ADR-001, ADR-011)
    auto_sample_rate: bool = True     # Read from WAV header


@dataclass
class BatchReport:
    total_files: int
    processed_files: int
    skipped_files: int
    failed_files: int
    total_detections: int
    processing_time_s: float
    detections_per_file: float        # Mean
    errors: list[tuple[str, str]]     # (filename, error message)


class BatchDetector:
    """Run detection pipeline on multiple WAV files."""

    def __init__(self, config: BatchConfig): ...

    def process_directory(self, progress_callback=None) -> BatchReport:
        """
        Process all WAV files in directory tree.

        For each WAV file:
        1. Run energy detection (high recall, ADR-003)
        2. Run CNN sliding window inference
        3. Apply hysteresis thresholding
        4. Export detections to JSON (ADR-010 format)
        5. Optionally save probability curve

        Skips files that already have output (if skip_existing=True).
        Continues processing if individual files fail (logged to errors).
        """
        ...

    def process_file(self, wav_path: Path) -> dict:
        """Process a single WAV file and return detection results."""
        ...
```

3. `scripts/batch_detect.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/batch_detect.py \
      --wav-dir "5970 USV" \
      --model models/full_retrained_cnn/best_model.pt \
      --output-dir analysis/batch_detections \
      --high-threshold 0.10 \
      --low-threshold 0.05 \
      --skip-existing
```

Output structure:
```
analysis/batch_detections/
├── results/
│   ├── 2024-09-30_11-18-17_0000001.json   # Per-file JSON (ADR-010 format)
│   ├── 2024-09-30_11-18-27_0000003.json
│   └── ...
├── summary.csv                             # One row per file: filename, n_detections, duration_s
├── batch_report.md                         # Processing summary
└── errors.log                              # Failed files with error messages
```

4. `tests/test_batch_detector.py` (NEW)

**Integration points:**
- Uses `AudioLoader` for WAV loading and STFT computation
- Uses `SlidingInference` for CNN inference
- Uses `HysteresisDetector` (detection_logic.py) for thresholding
- Output JSON format matches `LabelStorage` (ADR-010) so files can be loaded in the desktop app
- Uses `DetectionConfig` defaults (ADR-003, ADR-011, ADR-012)

**Test plan:**
```
1. Processing 3 synthetic WAV files produces 3 output JSONs in results/
2. skip_existing=True skips files that already have output
3. A corrupted WAV file doesn't halt batch processing; error logged
4. Output JSON format matches LabelStorage.load() expectations (ADR-010)
5. summary.csv has correct columns (filename, n_detections, duration_s) and row count
6. BatchReport statistics match actual processing results
```

**Exit criteria:**
- [ ] Can process 100+ WAV files without crashing
- [ ] Skip-existing works correctly for incremental processing
- [ ] Output JSONs loadable by the detection app's `LabelStorage`
- [ ] `summary.csv` enables quick filtering (e.g., "show files with > 10 USVs")
- [ ] Processing rate > 5 files/minute on CPU
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Dependency Graph

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
                                              ↓          ↓
                                          Phase 7    Phase 9 → Phase 10 → Phase 13
                                              ↓                              ↓
                                          Phase 12 ←←←←←←←←←←←←← (detection results)

Phase 8.1 → 8.2 → 8.3 → 8.4
  ↓
Phase 11.1 → 11.2 → 11.3 → 11.4
                                ↓
                            Phase 12 (VQ-VAE codes, optional)

Phase 13 → Phase 11.1 (bout extraction needs detection results)
```

---

## Recommended Execution Order

For upcoming work, prioritize based on research impact:

| Priority | Module | Why |
|----------|--------|-----|
| **1** | Phase 9 (Assembly) | Foundation for all future CNN training cycles |
| **2** | Phase 10 (Active Learning) | Enables scaling to 30K labels efficiently |
| **3** | Phase 8.1 (Data Pipeline) | Code can be written locally, no GPU needed |
| **4** | Phase 13 (Batch Detection) | Needed for bulk analysis + bout extraction input |
| **5** | Phase 8.2-8.3 (Transformer + VQ-VAE code) | Code can be written/tested locally with dummy data |
| **6** | Phase 11.1 (Preprocessing) | Can run locally after batch detection results exist |
| **7** | Phase 11.2-11.4 (Training + Analysis) | Requires HPC access |
| **8** | Phase 8.4 (Analysis tools) | Can be written in parallel with training |
| **9** | Phase 12 (Population Comparison) | Requires sufficient data from Phases 10 + 11 |

---

## Model Size Selection Guide

Per ADR-006 and `SCALING_TO_30K_ROADMAP.md`:

| Label Count | Model Size | Parameters | Flag |
|-------------|------------|------------|------|
| < 5,000 | Small | ~101K | `--model-size small` |
| 5,000 – 15,000 | Medium | ~400K | `--model-size medium` |
| 15,000+ | Large | ~1.6M | `--model-size large` |

Signs you need to scale up: both train and val loss remain high (underfitting).
Signs you need to scale down: train loss low, val loss high and diverging (overfitting).

---

## Reference Documents

| Document | Purpose |
|----------|---------|
| `docs/plans/theoretical_guide.md` | Scientific rationale for v2 transformer + VQ-VAE architecture |
| `docs/plans/implementation_plan.md` | Detailed pseudocode and architecture specs for Phase 8 |
| `DECISIONS.md` | Architectural decisions (ADRs) |
| `IMPLEMENTATION_PROGRESS.md` | Current implementation status |
