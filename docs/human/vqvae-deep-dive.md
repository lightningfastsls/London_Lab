# VQ-VAE Deep Dive: How Our USV Pipeline Discovers Discrete Concepts

This document explains the Vector Quantized Variational Autoencoder (VQ-VAE) used in our ultrasonic vocalization (USV) analysis pipeline. It's written as a teaching resource — start-to-finish, with math and intuition side by side.

---

## 1. The Scientific Question

We record mice at 300 kHz and capture ultrasonic vocalizations (USVs) — calls in the 25-125 kHz range. The question: **do these calls have discrete, reusable building blocks — like phonemes in speech?**

If they do, we want to find them automatically. VQ-VAE is the tool that forces continuous neural representations into a finite vocabulary of discrete codes, then lets us see if those codes capture meaningful acoustic categories.

---

## 2. Where VQ-VAE Sits in the Pipeline

We use a two-phase architecture:

```
Phase 1: Train an autoregressive transformer on bout spectrograms
         (learns rich, contextual representations of USV sequences)

Phase 2: Apply VQ-VAE to the transformer's frozen hidden states
         (discovers discrete "concepts" the transformer uses internally)
```

The VQ-VAE is a **post-hoc interpretability tool**. It doesn't touch raw audio — it operates on the internal representations of a transformer that has already learned to predict USV sequences. This means the codes it discovers reflect what the transformer found important, not just raw acoustic features.

### Data flow:

```
WAV recording (300 kHz)
  → bout extraction (isolate vocalization sequences)
  → spectrogram computation (STFT → frequency × time matrix)
  → train autoregressive transformer
  → extract hidden states from a chosen layer → .npy file (N_frames, 512)
  → train VQ-VAE on those hidden states
  → discrete code sequences: [14, 7, 7, 32, 5, ...]
  → analysis: transition matrices, n-grams, frequency profiles
```

---

## 3. Architecture Overview

The VQ-VAE has three components in series:

```
Input: Transformer hidden states  (B, S, 512)
       B = batch size, S = sequence length, 512 = transformer hidden dim

         ┌──────────────────────────────────────────┐
         │              ENCODER                      │
         │  Conv1d(512→256, kernel=5) + GELU         │
         │  Linear(256→64)                           │
         │  L2 normalize → project onto unit sphere  │
         └──────────────────────────────────────────┘
                          ↓
              z_e ∈ ℝ^(B×S×64), ||z_e|| = 1

         ┌──────────────────────────────────────────┐
         │          VECTOR QUANTIZER                 │
         │  Codebook: K=64 vectors on unit sphere    │
         │  For each z_e, find nearest codebook vec  │
         │  Replace z_e with that codebook vector    │
         │  (straight-through gradient trick)        │
         └──────────────────────────────────────────┘
                          ↓
              z_q ∈ ℝ^(B×S×64), indices ∈ {0,...,63}^(B×S)

         ┌──────────────────────────────────────────┐
         │              DECODER                      │
         │  Linear(64→256) + GELU                    │
         │  Linear(256→512)                          │
         └──────────────────────────────────────────┘
                          ↓
Output: Reconstructed hidden states  (B, S, 512)
```

### The key idea:

The encoder compresses 512 dimensions down to 64. The quantizer replaces each 64-dim vector with the nearest entry from a fixed codebook of K=64 entries. The decoder reconstructs back to 512. The system is trained end-to-end so that the reconstruction is as good as possible — forcing the codebook to learn 64 "prototype" representations that cover the full range of transformer hidden states.

---

## 4. The Encoder

### What it does
Compresses each 512-dimensional hidden state vector down to a 64-dimensional vector on the unit hypersphere.

### Architecture (Conv1d variant)

```python
Encoder:
  _Transpose()                    # (B, S, 512) → (B, 512, S)
  Conv1d(512, 256, kernel=5, pad=2)  # temporal context over 5 frames
  GELU()                          # activation function
  _Transpose()                    # (B, 256, S) → (B, S, 256)
  Linear(256, 64)                 # project to codebook dimension
  L2 normalize                    # project onto unit sphere
```

### Why Conv1d?

A Linear encoder would process each frame independently. Conv1d with kernel_size=5 means each output frame sees **5 input frames** (2 before, itself, 2 after). This captures local temporal context — important because a USV's identity depends not just on its instantaneous spectrum but on how it changes over a few milliseconds.

### L2 Normalization

After the encoder, we normalize every vector to unit length:

$$\hat{z} = \frac{z}{||z||_2}$$

This maps all encoder outputs onto the surface of a unit hypersphere in 64 dimensions. The codebook vectors are also on this sphere. This makes distance computations well-behaved — you're always comparing vectors of the same magnitude, so distance is essentially cosine similarity.

---

## 5. The Vector Quantizer (The Heart of VQ-VAE)

This is where continuous becomes discrete. It's the mathematical core of the whole system.

### 5.1 The Codebook

A codebook is a set of K learnable vectors, each of dimension D:

$$\mathbf{e} = \{e_1, e_2, \ldots, e_K\}, \quad e_k \in \mathbb{R}^D, \quad ||e_k|| = 1$$

In our system: K=64 entries, each 64-dimensional, all on the unit sphere.

Think of these as K "prototype" representations. After training, each one will correspond to a distinct "concept" the transformer uses.

### 5.2 Nearest Neighbor Lookup (Quantization)

For each encoder output z_e, find the closest codebook entry:

$$k^* = \arg\min_k ||z_e - e_k||^2$$

Since both z_e and e_k are unit-normalized, this simplifies:

$$||z_e - e_k||^2 = ||z_e||^2 - 2 z_e \cdot e_k + ||e_k||^2 = 2 - 2(z_e \cdot e_k)$$

So minimizing L2 distance is equivalent to **maximizing the dot product** (cosine similarity), which makes intuitive sense on the unit sphere.

The quantized output is:

$$z_q = e_{k^*}$$

**This is the discrete bottleneck.** Each 64-dimensional continuous vector gets replaced by one of only K=64 possible vectors. Each frame in the sequence becomes a single integer index k*.

### 5.3 The Straight-Through Estimator

**The problem:** The argmin operation is not differentiable. You can't compute ∂k*/∂z_e because argmin has zero gradient everywhere (it's piecewise constant) and undefined gradient at boundaries.

**The solution (Bengio et al., 2013):** During the forward pass, use z_q (the codebook entry). During the backward pass, pretend the quantization didn't happen and pass gradients straight through to z_e.

In code:
```python
z_q_st = z_e + (z_q - z_e).detach()
```

**Why this works:**
- Forward: `z_q_st = z_e + (z_q - z_e) = z_q` ✓ (uses the codebook entry)
- Backward: gradient of `.detach()` is 0, so `∂z_q_st/∂z_e = 1` (gradient passes through)

The encoder receives gradients as if it directly produced the decoder's input. This is an approximation — the true gradient is undefined — but it works remarkably well in practice. The intuition is: "if the encoder moved its output slightly, the same codebook entry would likely still be selected, so the decoder's output would change as if the encoder's output passed through directly."

### 5.4 Why Not Use a Regular Autoencoder?

A regular autoencoder would just have: Encoder → bottleneck → Decoder, with continuous values in the bottleneck. Why force discreteness?

1. **Interpretability**: A 64-dimensional continuous bottleneck has infinite possible states. A discrete codebook has exactly K=64 states, each of which you can inspect, name, and analyze.

2. **The scientific question requires it**: We're asking "do USVs have discrete categories?" A continuous bottleneck can't answer this — it would always find a continuum. Discreteness is the hypothesis we're testing.

3. **Sequence analysis**: Once you have discrete codes, you can apply information-theoretic tools — entropy, mutual information, n-grams, transition matrices — that require a finite alphabet.

---

## 6. Loss Function

The total loss has two components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{commit}}$$

### 6.1 Reconstruction Loss

$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} ||x_i - \hat{x}_i||^2$$

Standard MSE between original hidden states (x) and reconstructed hidden states (x̂). This trains both the encoder and decoder to minimize information loss through the bottleneck.

### 6.2 Commitment Loss

$$\mathcal{L}_{\text{commit}} = \frac{1}{N} \sum_{i=1}^{N} ||z_e^{(i)} - \text{sg}[e_{k^*}]||^2$$

Where sg[·] means "stop gradient" (treat as a constant during backprop).

**What this does:** Pushes encoder outputs toward their assigned codebook entries. Without it, the encoder might produce outputs that drift far from all codebook entries, making the quantization error large.

**Why stop-gradient on e_k*?** The codebook is NOT trained by this loss — it's trained by EMA (see below). The commitment loss only affects the encoder. You're saying: "encoder, move your outputs closer to wherever the codebook currently is."

### 6.3 The β (beta) Hyperparameter

β = 0.25 in our system. It balances:
- **High β**: encoder outputs tightly cluster around codebook entries (good quantization), but less freedom to explore
- **Low β**: encoder explores freely, but quantization error may be high
- **β = 0**: no commitment pressure — encoder outputs may drift away from codebook entries entirely

The original VQ-VAE paper (van den Oord et al., 2017) recommends β between 0.1 and 2.0, with 0.25 being a common default.

### 6.4 What About a Codebook Loss?

The original VQ-VAE paper also includes a "VQ loss" that moves codebook entries toward encoder outputs:

$$\mathcal{L}_{\text{VQ}} = \frac{1}{N} \sum_{i=1}^{N} ||\text{sg}[z_e^{(i)}] - e_{k^*}||^2$$

**Our implementation doesn't use this.** Instead, it uses Exponential Moving Average (EMA) updates for the codebook (Section 7), which is more stable.

---

## 7. Codebook Learning via EMA

The codebook vectors are NOT updated by gradient descent. Instead, they're updated using an Exponential Moving Average of the encoder outputs assigned to them. Think of it as a continuous, online version of k-means clustering.

### 7.1 The Update Equations

For each codebook entry e_k, track two running averages:

**Cluster size** (how many frames are assigned to code k):
$$N_k \leftarrow \gamma \cdot N_k + (1 - \gamma) \cdot n_k$$

Where n_k is the number of frames assigned to code k in the current batch, and γ = 0.99 (the EMA decay rate).

**Embedding sum** (sum of encoder outputs assigned to code k):
$$m_k \leftarrow \gamma \cdot m_k + (1 - \gamma) \cdot \sum_{i: k_i^* = k} z_e^{(i)}$$

**Updated codebook entry** (centroid of assigned frames):
$$e_k \leftarrow \text{normalize}\left(\frac{m_k}{\hat{N}_k}\right)$$

Where N̂_k uses Laplace smoothing to avoid division by zero:
$$\hat{N}_k = \frac{N_k + \epsilon}{\sum_j N_j + K\epsilon} \cdot \sum_j N_j$$

The final normalize() step re-projects onto the unit sphere (because the weighted average of unit vectors is NOT itself a unit vector).

### 7.2 Why EMA Instead of Gradient Descent?

1. **The argmin is non-differentiable.** You can't backprop through "find the nearest codebook entry" to update the codebook via gradients on the reconstruction loss. The VQ loss (Section 6.4) works around this but is less stable.

2. **EMA is equivalent to online k-means.** Each codebook entry tracks the centroid of the encoder outputs assigned to it. This is a well-understood, stable algorithm.

3. **The γ = 0.99 decay rate** means codebook entries move slowly — each update is 1% new data, 99% history. This prevents oscillation and gives the encoder time to adapt.

### 7.3 Intuition

Imagine K=64 magnets floating on the surface of a 64-dimensional sphere. Encoder outputs land on the sphere, and each magnet slowly drifts toward the points assigned to it. Over training, the magnets spread out to cover the regions where encoder outputs cluster — each magnet becomes the "representative" of a cluster of similar hidden states.

---

## 8. Dead Code Reset

### The Problem: Codebook Collapse

A common failure mode: some codebook entries never get "chosen" as nearest neighbors. Their EMA cluster sizes shrink toward zero. Once a code is dead, it stays dead — no encoder outputs are close to it, so it never gets updated, so it drifts further away. Eventually you might have K=64 entries but only 30 are actually used.

### The Solution

When a code's EMA cluster size drops below a threshold (default: 2.0):
1. Pick a random encoder output from the current batch
2. Replace the dead code's vector with that encoder output (normalized)
3. Reset its EMA statistics

This gives the dead code a second chance — it's now positioned near actual data, so encoder outputs will start getting assigned to it again.

### Monitoring

The **perplexity** metric tells you how many codes are effectively in use:

$$\text{perplexity} = \exp\left(-\sum_{k=1}^{K} p_k \log p_k\right)$$

Where p_k is the fraction of frames assigned to code k. If all codes are used equally, perplexity = K (maximum). If only one code is used, perplexity = 1.

**Codebook utilization** is simpler: the fraction of codes with any assignments at all.

A healthy codebook has perplexity close to K and utilization near 1.0.

---

## 9. K-Means Initialization

Before training begins, the codebook is initialized using k-means clustering on encoder outputs (not random):

### Algorithm
1. **k-means++ seeding** (Arthur & Vassilvitskii, 2007):
   - Pick first center randomly from data
   - For each subsequent center: pick with probability proportional to D² (squared distance to nearest existing center)
   - This spreads initial centers apart
2. **Lloyd's algorithm** (20 iterations):
   - Assign each data point to nearest center
   - Move each center to the mean of its assigned points
3. **Normalize** centers to unit sphere and set as codebook

### Why Not Random?

Random initialization on a 64-dimensional sphere means codebook entries could be clustered in one region, leaving other regions uncovered. K-means ensures the initial codebook already covers the data distribution, so training starts from a much better position. This reduces training time and the risk of dead codes.

---

## 10. Training Details

### Dataset

Hidden states are pre-extracted as .npy files. The training dataset chunks these into overlapping windows:
- **Window size**: 128 frames (each frame is a 512-dim vector)
- **Stride**: 64 frames (50% overlap between windows)
- **Tail handling**: final window anchored to end of sequence to avoid dropping frames

### Optimizer

- **AdamW** (Adam with decoupled weight decay)
- Learning rate: 3×10⁻⁴
- Weight decay: 0.01
- The codebook parameters are EXCLUDED from the optimizer (they're updated by EMA, not gradients)

### Learning Rate Schedule

- **Cosine warmup**: LR ramps from 0 to 3×10⁻⁴ over 500 steps, then follows cosine decay
- This prevents large, destructive gradient updates at the start when the codebook is still settling

### Early Stopping

- **Patience**: 15 epochs without improvement in validation loss → stop training
- Prevents overfitting and saves compute

### Gradient Clipping

- Max gradient norm: 1.0
- Prevents gradient explosions from the straight-through estimator (which can occasionally produce large gradients)

---

## 11. What the Codes Mean (Analysis)

After training, each codebook entry is a point on the unit sphere in 64-dimensional space. But what does it "mean" acoustically?

### 11.1 Decoded Frequency Profiles

Pass each codebook entry through the decoder, then through the transformer's output projection:

```
e_k (codebook entry, 64-dim)
  → decoder → reconstructed hidden state (512-dim)
  → transformer output layer → frequency distribution (n_freq bins)
```

This gives you a spectrogram-like profile for each code: which frequencies does this code "light up"?

### 11.2 Exemplar Frames

For each code k, find the actual frames in your dataset where that code was assigned. Look at the original spectrograms at those time points. This shows you real USV examples that the model groups together under the same code.

### 11.3 Transition Matrices

Once you have a sequence of codes [14, 7, 7, 32, 5, ...], compute:

$$P(k_{\text{next}} = j \mid k_{\text{current}} = i) = \frac{\text{count}(i \to j)}{\text{count}(i \to \text{any})}$$

This K×K matrix reveals which codes tend to follow which. High structure here (non-uniform rows) means vocalizations follow predictable sequential rules — evidence of grammar-like organization.

### 11.4 Entropy and Information Theory

- **Transition entropy per code**: H(next | current = k) — low means code k has predictable successors
- **Mutual information**: I(current; next) — how much knowing the current code tells you about the next
- **Excess entropy**: I(past; future) — overall sequential complexity
- **N-grams**: recurring 2-3 code patterns (analogous to syllable clusters)

If USV codes show low transition entropy, recurring n-grams, and high mutual information, that's evidence of compositional structure — not random sequences, not fixed patterns, but something with productive combinatorial properties.

---

## 12. Key Hyperparameters Summary

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `d_model` | 512 | Input dimension (must match transformer) |
| `codebook_size` (K) | 64 | Number of discrete concepts |
| `codebook_dim` | 64 | Bottleneck dimension (information per code) |
| `commitment_weight` (β) | 0.25 | How tightly encoder hugs codebook |
| `ema_decay` (γ) | 0.99 | How slowly codebook entries move |
| `dead_code_threshold` | 2.0 | When to reinitialize unused codes |
| `conv_kernel_size` | 5 | Temporal context window (frames) |
| `learning_rate` | 3×10⁻⁴ | Optimizer step size |
| `batch_size` | 256 | Windows per training step |
| `window_size` | 128 | Frames per training window |
| `patience` | 15 | Early stopping epochs |

---

## 13. Design Question: How Many Codes Activate Per Frame?

### The Answer: Exactly One

In the current implementation, each frame maps to **exactly one** codebook entry. The quantizer computes the distance from the encoder output to all K=64 codebook entries and picks the single closest one via `argmin`:

$$k^* = \arg\min_k ||z_e - e_k||^2$$

The output `indices` has shape `(B, S)` — one integer per frame per batch. A sequence of 100 frames becomes 100 code IDs: `[14, 7, 7, 32, 5, ...]`.

### What This Means

This is **"hard" vector quantization** — each frame is represented by exactly one codebook entry, with no mixing or blending. The entire 512-dimensional hidden state gets collapsed into a single integer from 0 to 63.

**Pros:** Clean interpretability. Each code maps to one clear meaning. You can build transition matrices, compute n-grams, and do sequence analysis with a simple finite alphabet.

**Cons:** Strong information bottleneck. With K=64 and 1 code per frame, there are only 64 possible representations for any frame. If the data has more than 64 meaningfully distinct states, some get merged.

The alternative — using multiple codes per frame — is discussed in Section 15 below.

---

## 14. Design Question: Can We Make the Codebook Bigger?

### Yes — It's Just a Config Parameter

The codebook size is controlled by a single field in `VQVAEConfig`:

```python
codebook_size: int = 64   # ← this is K
```

You can set it to 128, 256, 512, or any positive integer. There's no architectural constraint.

### The Trade-Off

| K (codebook size) | Pros | Cons |
|---|---|---|
| **Small (32-64)** | High utilization, interpretable, robust | May merge distinct concepts into one code |
| **Medium (128-256)** | Finer distinctions, captures subtypes | Harder to interpret, some codes may die |
| **Large (512+)** | Maximum expressiveness | Codebook collapse risk, codes become noisy, harder analysis |

### Three Real Risks of Going Too Big

**1. Codebook collapse.** With more codes available, many may go unused. The EMA cluster sizes drop below the dead code threshold, they get reset, and you enter a cycle of resets. The **perplexity** metric tells you this: if K=256 but perplexity is 40, you're effectively only using ~40 codes anyway, and the extra ones are noise.

**2. Interpretability degrades.** With 64 codes, you can visually inspect all of them (decode each to a frequency profile, look at exemplars). With 512, that becomes unwieldy. The analysis tools scale too: a transition matrix is K×K, so going from 64→256 means going from 4,096→65,536 entries.

**3. Statistical coverage.** Each code needs enough training examples to learn a stable representation. If your dataset has N frames and K codes, each code gets ~N/K frames on average. With small datasets, large K means some codes see too few examples to be meaningful.

### How to Choose K Empirically

Don't guess — **sweep it and let the data tell you:**

1. Train with K = 32, 64, 128, 256
2. Compare **perplexity** (how many codes are actually used)
3. Compare **reconstruction loss** (how much information is preserved)
4. Compare **codebook utilization** (fraction of codes that are alive)

The sweet spot is where:
- **Perplexity ≈ K** (nearly all codes used) — if perplexity at K=64 is already ~60, the codebook is saturated and going bigger is justified
- **Reconstruction loss plateaus** — more codes stop helping
- **Utilization stays high** (> 0.8)

If perplexity at K=64 is only 30, the model doesn't even need 64 codes — making K bigger won't help.

### Codebook Dimension vs. Codebook Size

There's a related but separate axis: `codebook_dim` (currently 64). This controls how much information each individual code **can carry**, not how many codes there are. You could independently increase this too, though the main lever for "number of concepts" is `codebook_size`, not `codebook_dim`.

---

## 15. Design Question: What If Multiple Codes Activated Per Frame?

### The Combinatorial Argument

With K=64 and 1 code per frame, you have **64 possible representations**.

If you used **3 codes per frame** with K=64, you'd get up to **64³ = 262,144 combinations** — without adding a single codebook entry. That's a massive expressiveness increase at almost no cost.

This is exactly the intuition behind human language: English has ~44 phonemes, but by *combining* them you get hundreds of thousands of words. Single-code VQ is like having a language where every word is one phoneme — you can say "ah" or "ee" but never "hello." Multi-code VQ lets you compose.

For USV research specifically, this matters: a vocalization might simultaneously encode *frequency band* + *temporal contour* + *amplitude envelope*. Forcing that into a single code merges those dimensions. Three codes could learn to separate them.

### Approach 1: Residual VQ (RVQ) — Coarse-to-Fine

Used by Meta's EnCodec, Google's SoundStream — the state of the art for neural audio compression.

**How it works:**

```
Hidden state (512-dim)
    ↓ Encoder
z_e (64-dim)
    ↓
Quantizer 1: find nearest code → z_q1, get residual r1 = z_e - z_q1
    ↓
Quantizer 2: quantize r1 → z_q2, get residual r2 = r1 - z_q2
    ↓
Quantizer 3: quantize r2 → z_q3
    ↓
Final quantized: z_q = z_q1 + z_q2 + z_q3
    ↓ Decoder
Reconstructed hidden state (512-dim)

Output per frame: 3 code indices, e.g. [14, 32, 7]
```

**What the codes mean:**
- **Code 1** (from Quantizer 1): the **coarse concept** — e.g., "this is a downward FM sweep"
- **Code 2** (from Quantizer 2): a **refinement** — e.g., "...with a sharp onset"
- **Code 3** (from Quantizer 3): **fine detail** — e.g., "...and moderate amplitude"

**The math of residuals:**

After the first quantizer picks the closest codebook entry z_q1, the residual is:

$$r_1 = z_e - z_{q1}$$

This is the "error" — what the first quantizer missed. The second quantizer then tries to encode *that error*:

$$r_2 = r_1 - z_{q2}$$

And so on. Each quantizer captures progressively finer details. The total reconstruction is:

$$z_q = z_{q1} + z_{q2} + z_{q3}$$

This is guaranteed to be at least as good as single-code VQ (since the residual quantizers can only reduce error, never increase it).

**Pros:**
- Natural hierarchy — first code is most important, you can drop later codes for lossy compression
- Interpretability is layered: analyze code 1 alone for broad categories, drill into codes 2-3 for subtypes
- Proven for audio (EnCodec, SoundStream use RVQ with 8-32 quantizers)

**Cons:**
- Codes are ordered/dependent — code 2 only makes sense *given* code 1
- Training is slightly more complex (each quantizer sees the residual from the previous one)

### Approach 2: Product Quantization (PQ) — Independent Aspects

Split the bottleneck vector into groups, quantize each independently.

**How it works:**

```
Encoder output (64-dim)
    ↓  split into 3 groups
Group A (dims 0-20)  → Quantizer A (K_a entries) → code_a
Group B (dims 21-42) → Quantizer B (K_b entries) → code_b
Group C (dims 43-63) → Quantizer C (K_c entries) → code_c
    ↓  concatenate quantized groups
Reconstructed bottleneck (64-dim)
    ↓ Decoder
Reconstructed hidden state (512-dim)

Output per frame: 3 independent code indices, e.g. [5, 18, 11]
```

**What the codes mean:**
- Each code captures a **different independent aspect** simultaneously
- Ideally the network learns meaningful factorizations: one group for frequency content, one for temporal shape, one for amplitude dynamics

**Pros:**
- Codes are truly independent — you can study each dimension separately
- Great for testing disentanglement hypotheses ("what aspects of USVs vary independently?")

**Cons:**
- No guarantee the groups learn semantically clean splits without additional pressure (like disentanglement losses)
- Each sub-codebook has fewer dimensions to work with

### Which Is Better for USV Research?

**RVQ is the stronger starting point** for three reasons:

1. **You don't know how many meaningful dimensions USVs have.** RVQ lets you add quantizers incrementally and stop when reconstruction loss plateaus — the number of quantizers needed *tells you* how many layers of structure exist. If 2 quantizers capture 95% of variance, USVs have ~2 layers of structure. If you need 6, they're more complex than expected.

2. **The first code is backwards-compatible.** You can run all existing analysis tools (transition matrices, n-grams, exemplar galleries) on just code 1 — it works exactly like the current single-code setup. Codes 2-3 are bonus detail you can analyze separately or jointly.

3. **It's proven for audio.** EnCodec and SoundStream use RVQ with 8-32 quantizers on audio that's structurally similar to what we're working with (spectral content evolving over time). The approach is well-understood for this domain.

**PQ becomes interesting later** if you want to test a specific hypothesis like "frequency content and temporal dynamics are independently controlled in mouse communication." That's a stronger scientific claim that needs its own experimental design.

### Comparison Table

| | Current (single VQ) | RVQ (3 codes) | PQ (3 groups) |
|---|---|---|---|
| Representations per frame | 64 | 64³ = 262K | ~21³ ≈ 9.3K* |
| Code relationship | N/A | Hierarchical (coarse→fine) | Independent aspects |
| Interpretability | One clear label | Layered analysis | Factored analysis |
| Analysis complexity | K×K transition matrix | 3 separate matrices + joint | 3 separate matrices |
| Best for | Broad categories | "How many layers of structure?" | "What varies independently?" |

*With PQ using 3 sub-codebooks of ~21 entries each.

---

## 16. Open Questions for Discussion

These are good starting points for a teaching conversation:

1. **Is K=64 saturated on our data?** Check perplexity after training. If perplexity ≈ 60, we need more codes.
2. **Which transformer layer produces the best codes?** The `compare_layers.py` script tests layers 2, 4, 6, 8 — lower layers capture acoustics, higher layers capture context.
3. **Would RVQ reveal hierarchical structure in USVs?** If the first quantizer captures call type and the second captures modulation, that's evidence of compositional structure.
4. **How does codebook structure differ between social contexts?** Same codes used differently (different transition matrices) across mating vs. pup isolation vs. same-sex interaction would be strong evidence for functional communication.

---

## 17. References

- **van den Oord et al. (2017)**: "Neural Discrete Representation Learning" — the original VQ-VAE paper
- **Bengio et al. (2013)**: "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation" — straight-through estimator
- **Arthur & Vassilvitskii (2007)**: "k-means++: The Advantages of Careful Seeding" — k-means++ initialization
- **Razavi et al. (2019)**: "Generating Diverse High-Fidelity Images with VQ-VAE-2" — hierarchical VQ-VAE
- **Défossez et al. (2022)**: "High Fidelity Neural Audio Compression" (EnCodec) — RVQ for audio

---

## 18. Glossary

| Term | Meaning |
|------|---------|
| **Codebook** | The set of K learnable prototype vectors |
| **Code / Index** | The integer ID (0 to K-1) of a selected codebook entry |
| **Quantization** | Replacing a continuous vector with the nearest discrete codebook entry |
| **Straight-through estimator** | Trick to pass gradients through a non-differentiable discrete step |
| **EMA** | Exponential Moving Average — a running average with decay |
| **Commitment loss** | Penalty encouraging encoder outputs to stay close to codebook entries |
| **Perplexity** | Effective number of codes in use (exp of entropy of usage distribution) |
| **Dead code** | A codebook entry that no encoder outputs get assigned to |
| **L2 normalization** | Projecting a vector onto the unit sphere: z/‖z‖ |
| **Unit hypersphere** | The surface of a sphere in high dimensions (all points at distance 1 from origin) |
| **Bout** | A cluster of USV calls close together in time |
| **Hidden state** | A transformer's internal representation at a given layer and position |
| **RVQ** | Residual Vector Quantization — multiple quantizers in series, each encoding the error from the previous one |
