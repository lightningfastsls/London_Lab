# ⚠️ SUPERSEDED — v1 Architecture

> **This plan has been superseded by the v2 two-phase architecture (2026-02-18).**
> The v2 approach trains a large autoregressive transformer first, then applies a VQ-VAE
> to its hidden states. See:
> - `docs/plans/implementation_plan.md` — v2 implementation plan
> - `docs/plans/theoretical_guide.md` — v2 scientific rationale
> - `ROADMAP.md` Phase 8 — v2 roadmap entries
> - `DECISIONS.md` ADR-007 — updated architectural decision
>
> This file is retained for historical reference only. Do not implement from this plan.

---

# USV Language Structure Discovery: VQ-VAE + Transformer Plan (v1 — ARCHIVED)

## Project Goal

Investigate whether mouse ultrasonic vocalizations (USVs) have language-like compositional structure by training a **VQ-VAE with a Transformer backbone** on continuous spectrogram sequences containing multiple USVs. The model should learn discrete "codebook entries" (analogous to phonemes or concepts) that capture meaningful structure — which we can then analyze, manipulate, and compare between wild and lab mouse populations.

---

## Conceptual Foundation (Read This First)

### What Are We Actually Building?

Imagine you have a long spectrogram — a 2D image where the x-axis is time and the y-axis is frequency. USVs appear as bright contours scattered across it. We want to build a model that:

1. **Reads** the spectrogram left-to-right, column by column (like reading text)
2. **Compresses** what it sees into discrete symbols from a learned vocabulary
3. **Predicts** what comes next based on those symbols
4. **Reconstructs** the spectrogram from just the discrete symbols

If USVs carry compositional meaning, the discrete symbols should capture meaningful units — syllable types, transition patterns, or contextual states. The key insight is that by *forcing* the representation through a discrete bottleneck, we make the model discover categorical structure rather than just memorizing pixel patterns.

### Architecture Components Explained

**Transformer (the sequence modeler)**
- Processes spectrogram columns as a sequence of "tokens"
- Uses self-attention: each column can "look at" all previous columns to decide what matters
- Learns temporal patterns: "after this type of chirp, this type of sweep usually follows"
- Think of it as the "language model" part — it models the grammar of USV sequences

**Autoencoder (the compressor)**
- Encoder: takes a chunk of spectrogram → compresses to small representation
- Decoder: takes small representation → reconstructs the spectrogram chunk
- The bottleneck forces the model to keep only what matters
- Like PCA but nonlinear — can capture complex patterns that PCA misses

**VQ (Vector Quantization — the discretizer)**
- The bottleneck doesn't output continuous values — it outputs *indices into a codebook*
- The codebook is a lookup table of K learned vectors (e.g., K=512 "concepts")
- Encoder output gets matched to nearest codebook entry (winner-takes-all)
- Only that discrete index passes through — forces categorical thinking
- Forward pass: argmin (non-differentiable) → uses "straight-through estimator" for gradients
- This is what makes it interpretable: we can enumerate all K concepts the model learned

**VQ-VAE + Transformer together:**
- Transformer provides the sequential context (what came before)
- VQ-VAE compresses each timestep into discrete codes
- The transformer operates on the discrete codes to predict next codes
- Result: a discrete "language" that the model invented to describe USV sequences

### Why This Approach Is Right for Your Question

- **Discrete codes → testable vocabulary**: You get a finite set of "USV concepts" you can enumerate
- **Sequence modeling → grammar discovery**: Transition probabilities between codes reveal structure
- **Reconstruction → verification**: If the decoder reproduces USVs well from codes, the codes capture real info
- **Wild vs lab comparison**: Different code usage patterns = different "dialects"
- **Manipulation → causal probing**: Change codes, see what the decoder produces, understand meaning

---

## Current State Assessment

### What You Have
- **Detection pipeline**: CNN-based USV detector (working, scaling to 30k labels)
- **Spectrogram generation**: 170 freq bins × variable time frames, 20-120 kHz, sr=300k, hop=128, n_fft=512
- **Data**: ~6.5k WAV files (102 reviewed, 6491 unreviewed), ~688 labeled detections (188 exported + 500 from CNN training)
- **Detection app**: PyQt6 app that can detect USVs in continuous recordings
- **Hardware**: AMD RX 5700 (4GB, no CUDA) — will need cloud GPU for training

### What You Need
- **Continuous sequence data**: Long spectrograms with multiple USVs and their positions
- **Cloud GPU access**: For transformer + VQ-VAE training (minimum 16GB VRAM recommended)
- **New data pipeline**: Raw WAV → continuous spectrogram → sequence of column vectors
- **The model itself**: Transformer + VQ-VAE architecture in PyTorch

---

## Phase 0: Prerequisites & Environment Setup (Do This Before May)

### 0.1 Cloud GPU Setup

**WHY**: Your AMD GPU won't work for PyTorch training in any practical way. You need NVIDIA CUDA.

**Options (ranked by cost-effectiveness for a research student):**
1. **Google Colab Pro+** (~$50/mo) — easiest, A100 access, good for prototyping
2. **Lambda Cloud** (~$1-2/hr for A10) — better for long training runs
3. **University HPC cluster** — check if your institution has GPU nodes (free!)
4. **vast.ai** (~$0.30-0.80/hr) — cheapest, less reliable

**Action items:**
- [ ] Check with your PI/university about HPC cluster access
- [ ] Set up Google Colab Pro as a fallback
- [ ] Create a simple PyTorch GPU test script to verify your setup works

### 0.2 Project Structure

```
usv_language/
├── configs/                  # Hyperparameter configs (YAML)
│   ├── data.yaml
│   ├── model.yaml
│   └── training.yaml
├── data/
│   ├── raw/                  # Symlinks to WAV files
│   ├── processed/            # Preprocessed spectrograms (.npy)
│   └── sequences/            # Chunked sequences ready for training
├── src/
│   ├── data/
│   │   ├── spectrogram.py    # Spectrogram generation (reuse your existing code)
│   │   ├── dataset.py        # PyTorch Dataset for sequences
│   │   └── preprocessing.py  # WAV → continuous spectrogram → chunks
│   ├── model/
│   │   ├── encoder.py        # Spectrogram column encoder
│   │   ├── decoder.py        # Spectrogram column decoder
│   │   ├── quantizer.py      # Vector quantization layer
│   │   ├── transformer.py    # Transformer sequence model
│   │   └── vqvae.py          # Full VQ-VAE + Transformer model
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   ├── losses.py         # Reconstruction + commitment + codebook losses
│   │   └── scheduler.py      # Learning rate scheduling
│   └── analysis/
│       ├── codebook_viz.py   # Visualize what each codebook entry represents
│       ├── sequence_analysis.py  # Transition matrices, entropy, etc.
│       └── comparison.py     # Wild vs lab code usage comparison
├── notebooks/                # Jupyter notebooks for exploration
├── scripts/                  # Training/evaluation launch scripts
└── tests/                    # Unit tests for model components
```

### 0.3 Dependency Setup

```
# Core
torch>=2.0
torchaudio
einops              # Tensor reshaping (makes transformer code readable)
vector-quantize-pytorch  # Lucidrains' VQ implementation (excellent, well-tested)

# Data
numpy
scipy
librosa             # Audio processing (you may already have this)
h5py                # Efficient storage for large spectrogram arrays

# Training
wandb               # Experiment tracking (free for academics)
accelerate          # Hugging Face's distributed training helper

# Analysis
matplotlib
seaborn
scikit-learn        # For downstream clustering analysis
umap-learn          # For codebook embedding visualization
```

---

## Phase 1: Data Pipeline for Continuous Sequences (Weeks 1-2)

### Conceptual Goal
Transform your raw WAV recordings into training-ready sequences. This is the **most important phase** — garbage data means garbage model.

### 1.1 Continuous Spectrogram Generation

**WHY**: Your current pipeline exports individual USV snippets. For language modeling, we need the full temporal context — silence between USVs is informative (it's like spaces between words).

```
Input:  Raw WAV file (300 kHz, variable length)
Output: Full spectrogram array, shape (170, T) where T = total time frames
        Saved as .npy or in HDF5 file
```

**Key decisions:**
- **Reuse your existing spectrogram parameters**: n_fft=512, hop_length=128, 20-120 kHz band → 170 freq bins. These work and changing them adds unnecessary complexity.
- **Normalization**: Per-file normalize to [0, 1] using min-max on the dB-scale spectrogram. Store the normalization parameters so you can invert later.
- **Storage**: Use HDF5 — your files can be up to 170×38k, and you have 6.5k of them. HDF5 handles this efficiently with chunked storage and lazy loading.

**Implementation task for Claude Code:**
> "Create `src/data/preprocessing.py`. It should:
> 1. Load a WAV file using the same parameters as `src/usv_spectrogram/app/core/audio_loader.py` (sr=300000, n_fft=512, hop_length=128, freq band 20-120 kHz)
> 2. Generate the full mel/linear spectrogram as a numpy array (170 × T)
> 3. Convert to dB scale, then min-max normalize to [0, 1], storing normalization params
> 4. Save to an HDF5 file with datasets: 'spectrogram', 'norm_min', 'norm_max', 'sample_rate', 'source_wav'
> 5. Include a batch processing function that processes all WAVs in a directory
> Make sure the spectrogram generation exactly matches the existing app code so detections are compatible."

### 1.2 Sequence Chunking

**WHY**: Transformers need fixed-length inputs. We cut continuous spectrograms into overlapping chunks.

**Key decisions:**
- **Chunk length**: Start with 256 columns (~109 ms at hop=128, sr=300k). This should capture 1-3 USVs plus silence. You can experiment with 512 or 1024 later.
  - Calculation: each column = hop_length/sr = 128/300000 ≈ 0.427 ms. So 256 columns ≈ 109 ms.
  - A typical USV is 5-50 ms, so 109 ms captures a few USVs with context.
- **Overlap**: 50% overlap (stride of 128 columns) — ensures USVs near chunk boundaries appear fully in at least one chunk
- **Filtering**: Optionally skip chunks that are pure silence (below energy threshold). But keep *some* silence chunks — the model needs to learn what silence looks like too. Aim for ~20% silence chunks.

**Implementation task for Claude Code:**
> "Create `src/data/dataset.py` with a PyTorch Dataset class called `USVSequenceDataset`. It should:
> 1. Take a directory of HDF5 spectrogram files
> 2. Cut each spectrogram into fixed-length chunks (configurable length, default 256 columns)
> 3. Use configurable overlap (default 50%)
> 4. Optionally filter pure-silence chunks based on mean energy threshold, but keep a configurable fraction of silence chunks (default 20%)
> 5. Return chunks as tensors of shape (170, chunk_length) — freq × time
> 6. Support lazy loading from HDF5 (don't load all into RAM)
> 7. Include a `__repr__` that prints dataset stats: total chunks, silence ratio, source files count"

### 1.3 Detection Overlay (Optional but Valuable)

**WHY**: You already have USV detections. Marking where USVs are in continuous spectrograms lets you validate later whether the model's learned codes align with detected USVs.

**Implementation task:**
> "Create a utility in `src/data/preprocessing.py` that, given a spectrogram HDF5 file and its corresponding detection labels (from the app's JSON exports or saved_tracking.json), produces a binary mask array of the same width as the spectrogram indicating which columns contain a detected USV. Save this as an additional dataset 'usv_mask' in the HDF5 file."

---

## Phase 2: The Column Encoder/Decoder (Weeks 2-3)

### Conceptual Goal
Before building the full model, build and validate the piece that compresses a single spectrogram column (170 values) into a small vector, and reconstructs it. This is the "per-token" part of the architecture.

### 2.1 Why Start Here

The full model has many moving parts. Starting with the column encoder/decoder lets you:
- Verify your data pipeline works (does reconstruction look right?)
- Get intuition for what compression level is appropriate
- Debug in isolation before adding transformer complexity

### 2.2 Column Encoder Architecture

```
Input:  (batch, 170)      — one frequency column
   → Linear(170, 256) + ReLU
   → Linear(256, 128) + ReLU
   → Linear(128, d_model)           — d_model is the embedding dimension (e.g., 64 or 128)
Output: (batch, d_model)  — compressed column representation
```

**WHY these dimensions**: 170 → 256 first *expands* before compressing. This gives the network room to create useful intermediate features before being forced to compress. It's a common pattern — going up before going down. `d_model` should match what the transformer will use (64 is a good start, 128 if your GPU allows it).

### 2.3 Column Decoder Architecture

```
Input:  (batch, d_model)
   → Linear(d_model, 128) + ReLU
   → Linear(128, 256) + ReLU
   → Linear(256, 170) + Sigmoid     — Sigmoid because your spectrograms are normalized to [0,1]
Output: (batch, 170)      — reconstructed frequency column
```

### 2.4 Standalone Validation

**Implementation task for Claude Code:**
> "Create `src/model/encoder.py` and `src/model/decoder.py` with the column encoder and decoder as described. Then create a notebook `notebooks/01_column_autoencoder.ipynb` that:
> 1. Loads some spectrogram chunks from the dataset
> 2. Trains a simple autoencoder (encoder + decoder, no VQ, no transformer) to reconstruct individual columns
> 3. Plots original vs reconstructed columns and full spectrogram chunks
> 4. Reports reconstruction MSE
> 5. Experiments with d_model = [32, 64, 128] to see the quality/compression tradeoff
> This is a sanity check — we want to see that d_model=64 gives decent reconstruction before proceeding."

---

## Phase 3: Vector Quantization Layer (Weeks 3-4)

### Conceptual Goal
Insert the discrete bottleneck between encoder and decoder. This is where the "concept discovery" happens.

### 3.1 How VQ Works (In Detail)

This is the most important concept in the whole project. Let's trace a forward pass:

```
1. Encoder produces:       z_e = encoder(x)           shape: (batch, d_model)
2. Codebook has K entries:  codebook = [e_1, e_2, ..., e_K]   each shape: (d_model,)
3. Find nearest neighbor:   k* = argmin_k ||z_e - e_k||²
4. Quantized output:        z_q = e_{k*}               shape: (batch, d_model)
5. Decoder reconstructs:    x_hat = decoder(z_q)       shape: (batch, 170)
```

**The problem**: Step 3 (argmin) is not differentiable! You can't backpropagate through a nearest-neighbor lookup.

**The solution — Straight-Through Estimator (STE):**
- Forward pass: use z_q (the codebook entry)
- Backward pass: pretend z_q = z_e, so gradients flow through as if quantization didn't happen
- In code: `z_q = z_e + (z_q - z_e).detach()`
- This means the encoder gets gradient signal to move toward its assigned codebook entry

**Three losses work together:**
1. **Reconstruction loss**: `||x - x_hat||²` — make the output look like the input
2. **Codebook loss (dictionary learning)**: `||z_e.detach() - e_{k*}||²` — move codebook entries toward encoder outputs
3. **Commitment loss**: `||z_e - e_{k*}.detach()||²` — move encoder outputs toward codebook entries

The balance is: `L = L_recon + L_codebook + β * L_commitment` where β ≈ 0.25 is standard.

### 3.2 Codebook Collapse (The Main Failure Mode)

**WHAT**: The model uses only a few codebook entries and ignores the rest. Instead of 512 "concepts," it might only use 10.

**WHY**: If the encoder converges to a small region of space faster than the codebook spreads out, most entries never get updated (they're never anyone's nearest neighbor).

**SOLUTIONS** (implement all of them):
1. **EMA (Exponential Moving Average) updates**: Instead of gradient descent on codebook entries, use running averages of assigned encoder outputs. More stable.
2. **Codebook reset**: If an entry hasn't been used in N batches, re-initialize it to a random encoder output from the current batch.
3. **Entropy regularization**: Add a loss term that encourages uniform codebook usage.

### 3.3 Implementation

**Implementation task for Claude Code:**
> "Create `src/model/quantizer.py` implementing vector quantization. I recommend using the `vector-quantize-pytorch` library by lucidrains as the core, but wrapping it in our own module for clarity. The module should:
> 1. Take a codebook size K (default 512) and embedding dimension d_model
> 2. Implement EMA codebook updates (not gradient-based)
> 3. Include codebook reset for dead entries (threshold: not used in 100 batches)
> 4. Return: quantized vectors, indices (which codebook entry was chosen), and all three loss components
> 5. Track and expose codebook utilization metrics (how many entries used, entropy of usage)
> 6. Include a method `encode(x) → indices` and `decode(indices) → vectors` for later analysis
>
> Also create `notebooks/02_vq_autoencoder.ipynb` that:
> 1. Trains encoder + VQ + decoder (no transformer yet) on spectrogram columns
> 2. Monitors codebook utilization over training (plot: how many of K=512 entries are actually used?)
> 3. Plots reconstruction quality compared to the non-VQ autoencoder from Phase 2
> 4. Visualizes what each active codebook entry 'means' by decoding each entry and plotting the frequency profile
> This validates the VQ layer in isolation before we add the transformer."

---

## Phase 4: The Transformer Backbone (Weeks 4-6)

### Conceptual Goal
Add temporal modeling. The transformer processes sequences of quantized codes and learns to predict what comes next — this is where "language structure" emerges.

### 4.1 Architecture Overview

```
Input spectrogram chunk: (batch, 170, T)    — T columns of 170 freq bins

For each column t:
    z_e[t] = encoder(column[t])              — (batch, d_model)
    z_q[t], idx[t] = quantize(z_e[t])        — discrete code

Sequence of codes: z_q = [z_q[0], z_q[1], ..., z_q[T-1]]    — (batch, T, d_model)

Transformer processes sequence:
    h = transformer(z_q)                      — (batch, T, d_model)
    h[t] = contextualized representation of position t, attending to all positions ≤ t

For each position t:
    next_pred[t] = decoder(h[t])              — predict column t+1
    code_pred[t] = codebook_predictor(h[t])   — predict which code comes at t+1
```

### 4.2 Transformer Details

**Use a causal (decoder-only) transformer** — like GPT. Each position can only attend to itself and previous positions. This enforces the "predict next" objective.

**Hyperparameters to start with:**
- `d_model`: 64 or 128 (match encoder output dimension)
- `n_heads`: 4 or 8 (must divide d_model evenly)
- `n_layers`: 4 (start small, scale up later if needed)
- `d_ff`: 256 (feedforward hidden dim, typically 2-4x d_model)
- `max_seq_len`: 256 (match your chunk length)
- `dropout`: 0.1

**Positional encoding**: Use standard sinusoidal positional encoding. The transformer has no inherent sense of position (attention is permutation-invariant), so positional encoding tells it "this is column 47" vs "this is column 48."

### 4.3 Training Objectives (Two Losses)

**1. Reconstruction loss**: The decoder takes the transformer's output at position t and tries to reconstruct column t+1. This forces the transformer to encode useful information.

**2. Next-code prediction loss**: A classification head on the transformer output predicts which codebook index appears at position t+1. Cross-entropy loss with K classes. This is the most "language-model-like" objective — literally predicting the next "token."

Combined: `L = L_recon + L_VQ + α * L_next_code` where α balances the objectives.

### 4.4 Implementation

**Implementation task for Claude Code:**
> "Create `src/model/transformer.py` with a causal transformer. Use PyTorch's `nn.TransformerDecoder` or implement from scratch using `nn.MultiheadAttention`. Include:
> 1. Sinusoidal positional encoding
> 2. Causal masking (each position attends only to current and past)
> 3. Configurable depth, heads, dimensions via a config dataclass
> 4. A next-code prediction head (linear layer → K logits)
>
> Then create `src/model/vqvae.py` that assembles the full model:
> 1. Column encoder → VQ layer → Transformer → Column decoder
> 2. Forward pass returns: reconstructed spectrogram, VQ losses, next-code prediction logits
> 3. A `generate` method: given a seed sequence of columns, autoregressively generate new columns
> 4. An `encode_to_codes` method: input spectrogram → sequence of codebook indices (for analysis)
>
> Also create `notebooks/03_full_model_test.ipynb` that:
> 1. Instantiates the full model with small hyperparameters
> 2. Runs a forward pass on a single batch to verify shapes and no errors
> 3. Runs a backward pass to verify gradients flow through the straight-through estimator
> 4. Prints parameter count and estimated memory usage"

---

## Phase 5: Training Pipeline (Weeks 5-7)

### 5.1 Training Strategy

**Start tiny, then scale.** This is crucial — don't try to train the full model on all data immediately.

**Stage A: Overfit on 1 file** (~1 hour)
- Take a single WAV with clear USVs
- Train until reconstruction loss plateaus
- **What you learn**: Does the architecture work at all? Can it reconstruct spectrograms?

**Stage B: Overfit on 10 files** (~few hours)
- Verify the model generalizes slightly
- Monitor codebook utilization — it should start increasing

**Stage C: Train on all reviewed data** (~half day)
- 102 reviewed WAVs
- This is your first "real" model
- Analyze learned codebook entries

**Stage D: Train on full dataset** (~1-2 days)
- All 6.5k WAVs
- Full hyperparameter tuning

### 5.2 Training Hyperparameters

```yaml
# Start with these, adjust based on experiments
optimizer: AdamW
learning_rate: 3e-4
weight_decay: 0.01
lr_scheduler: cosine_with_warmup
warmup_steps: 1000
batch_size: 32          # Adjust based on GPU memory
max_epochs: 100         # With early stopping
gradient_clip: 1.0

# Loss weights
vq_commitment_weight: 0.25
next_code_prediction_weight: 0.1    # Start low, increase if codebook is good

# VQ specific
codebook_size: 512
codebook_dim: 64        # Same as d_model
ema_decay: 0.99
dead_code_threshold: 100  # Reset after 100 batches unused
```

### 5.3 What to Monitor (W&B Dashboard)

**Critical metrics:**
- `train/reconstruction_loss` — should decrease steadily
- `train/codebook_utilization` — fraction of K entries used. Alarm if < 10%. Goal: > 50%
- `train/codebook_entropy` — higher = more uniform usage = better
- `train/perplexity` — exp(entropy). Should approach K if all entries used equally
- `train/next_code_accuracy` — how well the transformer predicts the next code
- `val/reconstruction_loss` — overfitting check

**Visualizations (log periodically):**
- Original vs reconstructed spectrograms
- Codebook entry frequency histogram
- t-SNE/UMAP of codebook entries colored by frequency of use

### 5.4 Implementation

**Implementation task for Claude Code:**
> "Create `src/training/trainer.py` with a training class that:
> 1. Takes model, dataset, and config
> 2. Implements the training loop with mixed precision (fp16) for memory efficiency
> 3. Logs all metrics listed in section 5.3 to Weights & Biases
> 4. Implements early stopping based on validation reconstruction loss
> 5. Saves checkpoints with best model tracking
> 6. Includes a `fit` method with the staged approach: can accept a `max_files` param to limit data
>
> Create `src/training/losses.py` with the combined loss function:
> 1. Reconstruction MSE
> 2. VQ losses (from quantizer module)
> 3. Next-code cross-entropy
> 4. Weighted combination with configurable weights
>
> Create `scripts/train.py` as the entry point:
> 1. Loads config from YAML
> 2. Sets up data, model, trainer
> 3. Supports resuming from checkpoint
> 4. Command-line overrides for key hyperparameters"

---

## Phase 6: Analysis & Interpretation (Weeks 7-10)

### This Is The Payoff

Once the model is trained, the real science begins. You have a model that compressed USV sequences into discrete codes. Now: **what do those codes mean?**

### 6.1 Codebook Analysis

**For each codebook entry k (0 to K-1):**
- Decode it: what frequency pattern does it represent? (e.g., rising chirp, flat tone, silence)
- When does it appear: beginning/middle/end of USVs? During silence?
- How common is it: frequency of occurrence across the dataset
- Overlay with your CNN detections: do certain codes correspond to detected USVs?

**Implementation task for Claude Code:**
> "Create `src/analysis/codebook_viz.py` that:
> 1. Loads a trained model checkpoint
> 2. Decodes every codebook entry through the decoder → plots each as a frequency column
> 3. Runs the encoder on the full dataset → computes code frequency histogram
> 4. For each code, finds example spectrogram locations where it activates → creates a montage
> 5. If USV detection masks are available, computes correlation: which codes fire during USVs vs silence?
> 6. Clusters codebook entries using their decoded frequency patterns → shows natural groupings"

### 6.2 Sequence/Grammar Analysis

**Transition matrix**: P(code_j at t+1 | code_i at t). This is literally a first-order "grammar" of USV codes.
- High entropy rows = unpredictable transitions (beginning of new USV?)
- Low entropy rows = predictable sequences (within a stereotyped call?)
- Compare transition matrices between wild vs lab mice

**Higher-order patterns**:
- Look for common n-grams (code sequences that appear frequently)
- Compute mutual information between codes at various lags
- Identify "motifs" — recurring code sequences that might be meaningful units

**Implementation task for Claude Code:**
> "Create `src/analysis/sequence_analysis.py` that:
> 1. Encodes the full dataset to code sequences
> 2. Computes the code transition matrix and visualizes as heatmap
> 3. Computes transition entropy per code (which codes lead to predictable vs unpredictable continuations)
> 4. Finds the top-N most common code bigrams, trigrams, and 4-grams
> 5. Computes mutual information between codes at lags 1, 2, 5, 10, 20
> 6. If wild/lab labels are available, computes all of the above separately and tests for significant differences"

### 6.3 Generative Probing (The Fun Part)

Feed the model a seed sequence and let it generate. Then **manipulate**:
- Start from real USV beginning, let model continue — does it produce realistic USVs?
- Swap one code in a sequence — how does the output change?
- Force a specific code at a position — what does the model predict next?
- Interpolate between two codebook entries (in the continuous space before quantization) — what's "between" two USV types?

This is what your PI meant by "artificially give different orders of binary signal and see what the LLM starts predicting."

**Implementation task for Claude Code:**
> "Create `src/analysis/generative_probing.py` and `notebooks/04_generative_exploration.ipynb` that:
> 1. Load a trained model
> 2. Take a real spectrogram, encode it to codes, then decode back (round-trip test)
> 3. Implement code-swap experiments: change one code in a sequence, decode, show difference
> 4. Implement autoregressive generation from a seed: encode first 32 columns, generate the next 224
> 5. Implement 'concept scan': for a fixed context, try every possible code at position t, show how the continuation changes
> 6. All results should be visualized as spectrograms with annotations"

---

## Phase 7: Scaling Up & Refinements (Weeks 10+)

Once the basic pipeline works, these are natural extensions:

### 7.1 Multi-Scale VQ (Residual VQ)
Instead of one codebook, use a hierarchy: first codebook captures coarse structure, second captures residual detail, etc. This is how modern audio models (SoundStream, EnCodec) work.

### 7.2 Longer Context
Increase chunk length to 1024 or 2048 columns to capture longer multi-USV sequences. May require more GPU memory → use gradient checkpointing.

### 7.3 Conditional Generation
Condition the model on metadata (wild vs lab, male vs female, behavioral context). Then you can ask: "generate a USV sequence that a wild male mouse would produce."

### 7.4 Cross-Animal Comparison
Train separate models on wild vs lab data, or a single model with population labels. Compare codebook usage statistically.

---

## Quick Reference: Week-by-Week Timeline

| Week | Phase | Milestone | Validation Check |
|------|-------|-----------|-----------------|
| 1 | 0 + 1.1 | Environment setup, continuous spectrogram pipeline | Can load WAV → save HDF5 → load back |
| 2 | 1.2 + 1.3 | Sequence chunking, detection overlay | Dataset prints correct stats, chunks look right |
| 3 | 2 | Column autoencoder working | Reconstruction visually recognizable at d_model=64 |
| 4 | 3 | VQ layer integrated | >50% codebook utilization, reconstruction still OK |
| 5-6 | 4 | Full model assembled | Forward+backward pass works, shapes correct |
| 6-7 | 5 | Training pipeline complete | Can overfit on 1 file, loss decreases |
| 7-8 | 5 (cont) | Train on real data | Codebook entries look like real spectral patterns |
| 8-10 | 6 | Analysis pipeline | Transition matrix reveals non-random structure |
| 10+ | 7 | Scaling and extensions | Wild vs lab differences emerge |

---

## Key Gotchas and Tips

1. **Don't skip the staged training** (Phase 5.1). Training the full model on all data first is the #1 mistake. Overfit on tiny data first.

2. **Codebook collapse is your main enemy.** If utilization drops below 10%, stop training and debug. Common fixes: decrease learning rate, increase commitment loss weight, enable more aggressive dead code resets.

3. **Reconstruction quality sets the ceiling.** If the column autoencoder can't reconstruct well at d_model=64, the VQ-VAE won't either. Fix Phase 2 before moving on.

4. **Log everything.** You'll want to compare runs. Use W&B from day one.

5. **The 170-dim frequency columns are actually small** compared to typical VQ-VAE inputs (images are 256×256). This is good — it means your model can be relatively small and still work. Don't over-parameterize.

6. **Silence is data.** Don't filter it all out. The model needs to learn the structure of silence → USV → silence transitions. Think of silence as the "space between words."

7. **Your CNN detector is complementary.** Use it to validate the VQ-VAE's discoveries: if a codebook entry activates exactly when the CNN detects a USV, that's convergent evidence. If the VQ-VAE finds structure the CNN misses, that's a discovery.

8. **Version control your configs.** Every training run should have its config saved alongside the checkpoint. You will forget what hyperparameters you used.
