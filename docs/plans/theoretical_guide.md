# Transformer + VQ-VAE for Mouse USV Analysis: Theoretical & Architectural Guide

## Project Goal

Investigate whether mouse ultrasonic vocalizations (USVs) contain language-like structure by training an autoregressive transformer on spectrogram data, then using a VQ-VAE on the transformer's internal representations to discover discrete "concepts" that the model uses for prediction.

---

## Core Architecture: Two-Phase Pipeline

### Why this specific order matters

The pipeline trains a **transformer first**, then applies a **VQ-VAE to its internal representations**. This order is critical and differs from the DALL-E/AudioLM approach (which tokenizes first, then models tokens).

**Phase 1 — Transformer (self-supervised, no bottleneck):** The transformer receives raw spectrogram columns as input and learns to predict the next column autoregressively. It develops internal representations freely, without any discretization constraint. The deeper layers encode increasingly abstract patterns — not just "what frequency is active" but "what kind of acoustic event is happening" and "what should come next."

**Phase 2 — VQ-VAE on hidden states (interpretability tool):** After the transformer is frozen, hidden states from a chosen middle layer are extracted. A VQ-VAE compresses these continuous representations into a small discrete codebook. Each codebook entry becomes an interpretable "concept" — a recurring pattern the transformer has learned to recognize.

**Why not end-to-end?** Building the VQ-VAE into the transformer from the start forces discretization before the model has learned what matters. The discrete bottleneck constrains what the model can represent, potentially preventing it from discovering subtle patterns. By training the transformer first, we let it freely learn whatever representations are most useful, then use the VQ-VAE as a post-hoc analysis tool.

**Why not VQ-VAE first (DALL-E style)?** Training a VQ-VAE directly on spectrograms would discover acoustic categories (call shapes), but the resulting discrete tokens would only capture local spectral patterns. The transformer's hidden states capture something richer — contextual, predictive representations that encode what the model "thinks" is happening given everything it has seen so far. These are the representations where "concepts" live.

### Why sub-USV resolution (column-level) is correct

We cannot assume a single USV is a meaningful unit. The meaningful structure might be:
- **Within USVs** (like phonemes within words)
- **Across USV boundaries** (like how word combinations create meaning)
- **In the timing/silence patterns** between calls

By processing at the spectrogram-column level (~0.427ms per frame), the model sees a continuous acoustic stream and discovers structure at whatever granularity is informative for prediction. The VQ-VAE then reveals what temporal patterns the transformer identified as important — some codebook entries might correspond to within-call features, others to transitions, others to silence patterns.

---

## Input Data: Bout-Level Spectrograms

### What is a "bout"?

Rather than feeding isolated USV crops (which discards inter-call context) or entire wav files (which are mostly silence), we extract **bouts** — continuous segments of recording containing clusters of USV activity with surrounding context.

**Bout extraction logic:**
1. Use the existing CNN detection pipeline to identify USV locations (start_time, end_time) in each recording
2. Group USVs that occur within 500ms of each other into bouts
3. For each bout, extract the spectrogram from (first_USV_start − 200ms) to (last_USV_end + 200ms)
4. This preserves inter-USV timing, silence gaps between calls, and transition patterns

**Why 200ms padding?** It provides enough pre/post context for the transformer to learn what precedes and follows USV activity, without including long dead stretches. This is tunable.

### Spectrogram parameters (reuse existing pipeline)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 300 kHz | Standard for mouse USV recording |
| n_fft | 512 | Gives 257 freq bins, ~170 in the 20-120 kHz range |
| hop_length | 128 | 0.427 ms per frame — fine temporal resolution |
| Frequency range | 20-120 kHz | Standard mouse USV range |
| Frequency bins | ~170 | After cropping to 20-120 kHz |
| Representation | Log-magnitude (dB) | Better dynamic range than linear |
| Normalization | Per-frequency-bin, global stats | Zero mean, unit variance per freq bin across dataset |

**Key change from CNN pipeline:** Instead of extracting fixed 40ms windows around detected USVs, apply the same STFT to bout-length segments (which may be seconds long, producing hundreds to thousands of frames).

---

## Phase 1: Transformer Architecture

### Model specification

```
Input: spectrogram column (170-dim vector)
    ↓
Input projection: Linear(170 → 512) → GELU → LayerNorm
    ↓
+ Learned positional embeddings (max_seq_len × 512)
    ↓
8× Transformer decoder blocks:
    ├── Causal multi-head self-attention (8 heads, d_head=64)
    ├── LayerNorm + residual connection
    ├── FFN: Linear(512 → 2048) → GELU → Linear(2048 → 512)
    └── LayerNorm + residual connection
    ↓
Output head: Linear(512 → 170)
    ↓
Loss: MSE between predicted and actual next column
```

### Key design decisions

**Causal (GPT-style) attention:** Each position can only attend to previous positions. This is required for autoregressive next-column prediction and matches the scientific question — "given what came before, what comes next?"

**Input projection (170 → 512):** Raw frequency vectors are projected into a higher-dimensional embedding space. The 2-layer projection (Linear → GELU → LayerNorm) is more expressive than a single linear layer and follows the Tacotron 2 pre-net approach for spectrogram inputs.

**Learned positional embeddings:** For bounded sequences (max_seq_len=512), learned embeddings work well and are validated by the Audio Spectrogram Transformer (AST). Alternative: Rotary Position Embeddings (RoPE) for better length generalization — use if sequences regularly approach or exceed 512.

**Output is continuous (170-dim):** The model predicts the next spectrogram column as a real-valued vector, not a discrete token. The loss is MSE (or MSE + L1) between predicted and actual next columns. This is appropriate because spectrogram values are inherently continuous.

**If MSE produces blurry predictions:** Upgrade to a Gaussian Mixture Model (GMM) head with K=5-10 components. Instead of outputting a single 170-dim prediction, the head outputs K sets of (mixing_weight, mean_vector, variance_vector). This captures multimodal distributions — when multiple "next columns" are plausible, the model can assign probability to each.

### Handling variable-length sequences

**max_seq_len = 512 frames** (~218ms at 0.427ms/frame). Bouts longer than 512 frames are chunked with 50% overlap, so every frame appears as both context and prediction target. Bouts shorter than 512 are padded with a learned `[PAD]` embedding, with an attention mask preventing the model from attending to padding.

**Length bucketing:** Sort sequences by length into ~6-8 buckets (64, 128, 192, 256, 384, 512 frames) and pad only to the longest sequence within each batch. This reduces wasted computation from padding by 30-50%.

### Training configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 (peak) |
| LR schedule | Linear warmup (2000 steps) → cosine decay to 1e-6 |
| Weight decay | 0.01 (exclude biases, LayerNorm, embeddings) |
| Batch size | 32-64 per GPU, accumulate to effective 128-256 |
| Gradient clipping | Max norm 1.0 |
| Dropout | 0.1 |
| Epochs | Until validation loss plateaus (expect 50-200 epochs) |

**Input normalization:** Compute per-frequency-bin mean and standard deviation across the entire training set. Normalize all spectrograms to zero mean, unit variance per bin. Store these statistics for inference.

**Data augmentation (use sparingly):**
- Gaussian noise: SNR 15-20 dB
- Gain perturbation: ±3 dB
- SpecAugment-style frequency masking: mask 1-2 bands of ~20-30 bins
- Time masking: mask 1-2 spans of ~10% of sequence length
- Apply augmentation with p=0.5 per sample

### What to monitor during training

- **Training/validation MSE loss** (should decrease smoothly)
- **Per-frequency-bin error distribution** (check no frequency range is systematically worse)
- **Visual inspection:** plot predicted vs. actual next columns for sample sequences
- **Attention patterns:** visualize attention maps to verify the model attends to relevant context (not just the immediately preceding frame)

---

## Phase 2: VQ-VAE on Transformer Hidden States

### When to start Phase 2

Only after the transformer is fully trained and frozen. "Fully trained" means validation loss has plateaued for at least 10 epochs with no improvement.

### Which layer to extract from

**Start with layer 4 of 8 (the middle).** The intuition:
- Layers 1-2: Low-level features (spectral shape, local patterns)
- Layers 3-5: Mid-level concepts (acoustic motifs, call-type-like patterns)
- Layers 6-8: Highly specialized for next-token prediction

**Experiment with layers 2, 4, 6, and 8.** Compare codebook quality (utilization, perplexity) and interpretability of decoded entries. Middle layers typically give the most interpretable concepts.

### VQ-VAE architecture

```
Input: h_t from transformer layer L (512-dim vector per time step)
    ↓
Encoder:
    1D Conv(kernel=5, in=512, out=256) → GELU    ← captures local temporal context
    Linear(256 → D)                               ← D = codebook dimension (64)
    L2-normalize                                   ← stabilizes quantization
    ↓
Vector Quantization:
    Codebook: K entries, each of dimension D
    For each encoded vector z_e:
        Find nearest codebook entry: k* = argmin_k ||z_e - e_k||²
        Output: z_q = e_{k*}  (the codebook vector itself)
        Straight-through estimator: z_q = z_e + (z_q - z_e).detach()
    ↓
Decoder:
    Linear(D → 256) → GELU
    Linear(256 → 512)
    ↓
Loss: MSE(decoded, h_t) + β × ||z_e - sg[z_q]||²
      reconstruction        commitment loss (β=0.25)
```

### Codebook configuration

| Parameter | Start value | Range to explore |
|-----------|-------------|-----------------|
| K (codebook size) | 64 | 32, 64, 128, 256 |
| D (codebook dimension) | 64 | 32, 64, 128 |
| β (commitment weight) | 0.25 | 0.1 - 1.0 |
| EMA decay (γ) | 0.99 | 0.95 - 0.999 |

**Start with K=64.** Traditional USV taxonomy has ~10-15 types, but the model may discover finer subtypes or alternative organizing principles. K=64 gives headroom. If most entries go unused → try K=32. If utilization is high and perplexity is near K → try K=128.

### The "winner takes all" mechanism

For each time step, exactly one codebook entry wins (the nearest one). The output is effectively a one-hot vector of length K — binary and sparse, as Mickey described. During the forward pass, this is hard/discrete. During backpropagation, the straight-through estimator copies gradients from the decoder input directly to the encoder output, bypassing the non-differentiable argmin.

**Getting top-k concepts:** For analysis (not training), record the distances to all K codebook entries for each time step. The top-4 nearest entries reveal which concepts are "competing" — useful for understanding ambiguous or transitional moments in the signal.

### Preventing codebook collapse

This is the #1 failure mode. Use all of these simultaneously:

1. **EMA codebook updates (γ=0.99):** Instead of gradient-based learning, update codebook vectors as exponential moving averages of assigned encoder outputs. More stable than SGD on the codebook.

2. **Dead code reinitialization:** Monitor assignment counts. If any codebook entry receives fewer than ~2 assignments per batch (averaged over 100 batches), reinitialize it by sampling from current encoder outputs. The `vector-quantize-pytorch` library implements this.

3. **K-means initialization:** Before training begins, run a few batches through the encoder, collect outputs, and initialize codebook vectors via k-means clustering.

4. **L2-normalization:** Normalize both encoder outputs and codebook vectors before distance computation. Prevents any single vector from dominating.

5. **Entropy regularization (optional):** Add a small loss term that maximizes the entropy of the code usage distribution, encouraging uniform utilization.

**Fallback — Finite Scalar Quantization (FSQ):** If collapse persists despite all the above, FSQ (Mentzer et al., ICLR 2024) achieves 100% utilization by design. It rounds each scalar channel to fixed levels instead of doing nearest-neighbor lookup, creating an implicit codebook with no auxiliary losses needed.

### VQ-VAE training configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR schedule | Warmup (500 steps) → cosine decay |
| Batch size | 128-256 (can be large since inputs are just 512-dim vectors) |
| Epochs | 50-100 (convergence is fast on pre-extracted hidden states) |

---

## Phase 3: Analysis and Interpretation

### Visualizing codebook entries

**Decode through the full pipeline:** Take each codebook entry e_k → VQ-VAE decoder → get reconstructed hidden state h_k → pass h_k through remaining transformer layers → read off what spectrogram columns the transformer predicts. This reveals what acoustic pattern each concept corresponds to.

**Exemplar galleries:** For each codebook entry, retrieve the N=10 nearest training examples (by encoder distance). Show their spectrograms side by side to see what real-world USVs each concept captures.

**t-SNE/UMAP of codebook:** Project all K codebook vectors to 2D. Color by most frequent associated USV category (if labels exist) or by acoustic features (mean frequency, duration, bandwidth).

### The manipulation experiment

This is the key interpretability test Mickey described:
1. Take a sequence of hidden states from a real recording
2. At a chosen time step, **replace** the hidden state with a specific codebook entry (decoded back to 512-dim)
3. Pass through the remaining transformer layers
4. See what the transformer predicts next

This answers: "If the transformer were thinking concept X right now, what would it expect to hear next?" By systematically varying which concept you inject, you map out the transformer's predictive model of USV sequences.

**Extension — combining concepts:** If using top-k (e.g., k=4), try injecting weighted combinations of codebook entries and observe how predictions change. This tests whether concepts compose.

### Testing for language-like structure

**Zipf's law:** Plot code frequency vs. rank on a log-log scale. Natural language follows a power law with exponent α ≈ 1. If the code distribution follows Zipf's law, this is a necessary (but not sufficient) condition for language-like organization.

**Transition statistics:** Compute the bigram transition matrix P(c_{t+1} | c_t). Non-uniform transitions indicate sequential structure. Extend to trigrams and 4-grams. Compare transition entropy to what would be expected from a random (maximum entropy) process.

**Context-dependent usage:** If you have metadata (social context, mouse identity, sex), test whether code distributions or transition patterns change with context. This parallels Chabout et al. (2015) who found male mice change USV syntax depending on social context.

**Entropy rate and excess entropy:** The entropy rate measures intrinsic unpredictability of code sequences. Excess entropy measures long-range temporal structure. Higher excess entropy = more complex organization.

**Compositionality tests:** Do code combinations follow compositional rules? Test with Tree Reconstruction Error (TRE). Try generating code combinations unseen in training and decode them — do they produce meaningful spectrograms?

---

## Relevant Prior Work

### Mouse USV analysis tools
- **DeepSqueak:** Faster R-CNN detection + k-means clustering
- **VocalMat:** AlexNet CNN, ~86% on 11 categories
- **MUPET:** Gammatone + unsupervised k-means, discovers 100-140 types
- **AVA (Goffinet et al., 2021, eLife):** VAE-based, found USVs form a continuum rather than discrete clusters
- **AMVOC:** Convolutional autoencoders for unsupervised clustering

### Key finding from prior work
Goffinet et al. found that mouse USVs form a **continuum** rather than discrete clusters. This means VQ-VAE codebook entries may not map to the traditional 10-category taxonomy but could discover alternative organizing principles — potentially more scientifically interesting.

### Relevant ML work
- **Transformer VQ-VAE (Tjandra et al., Interspeech 2020):** VQ-VAE + transformer for unsupervised phoneme-like unit discovery in human speech. K=128 codes. Closest analog to this project.
- **AudioLM, SoundStream, EnCodec:** Tokenize audio → model with transformer. Use VQ-VAE first (opposite of our order), optimized for generation quality rather than interpretability.
- **VQLC (2025):** Vector Quantized Latent Concepts — applies VQ to transformer hidden states for concept discovery. Directly validates our approach.
- **Earth Species Project:** NatureLM-audio, adapting HuBERT for bioacoustics. Uses masked prediction + k-means clustering for unit discovery.

### Mouse USV sequential structure evidence
- **Chabout et al. (2015):** Males change syllable syntax with social context
- **Hertz et al. (2020):** Sequence statistics carry predictive information about temporal structure
- **Current consensus:** USVs have some sequential structure (non-random ordering) but no evidence yet for compositionality or hierarchical syntax

---

## Key References

1. van den Oord et al. (2017) — "Neural Discrete Representation Learning" (original VQ-VAE)
2. Mentzer et al. (ICLR 2024) — "Finite Scalar Quantization: VQ-VAE Made Simple"
3. Goffinet et al. (eLife 2021) — "Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires"
4. Chabout et al. (2015) — "Male mice song syntax depends on social contexts"
5. Tjandra et al. (Interspeech 2020) — "Transformer VQ-VAE for Unsupervised Unit Discovery"
6. Park et al. (2019) — "SpecAugment" data augmentation for spectrograms
7. Ivanenko et al. (2020) — "Classifying sex and strain from mouse USVs using deep learning"
8. Kershenbaum et al. (2021) — "Shannon entropy as a robust estimator of Zipf's Law in animal vocal communication"
9. Gong et al. (2021) — "Audio Spectrogram Transformer" (AST)
10. Henschel et al. (2025) — "Continuous Autoregressive Modeling" (GMM-LM, ICLR 2025)
