# Implementation Plan: Transformer + VQ-VAE for Mouse USV Concept Discovery

## Overview

This plan implements a two-phase pipeline for investigating language-like structure in mouse USVs:
1. **Phase 1:** Train an autoregressive transformer on raw spectrogram columns (next-column prediction)
2. **Phase 2:** Train a VQ-VAE on the transformer's internal hidden states to discover discrete "concepts"
3. **Phase 3:** Analysis tools for probing and interpreting the learned representations

The code lives in the existing `usv_language/` module within the project repository.

---

## Phase 0: Data Preparation Pipeline

### Task 0.1: Bout Extraction Module

**File:** `usv_language/data/bout_extractor.py`

**Purpose:** Extract bout-level spectrograms from raw wav files using existing USV detection results.

**Logic:**
1. Load USV detection results (CSV/JSON from CNN pipeline) containing (file, start_time, end_time) for each detected USV
2. Group USVs into bouts:
   - Sort USVs by start_time within each file
   - Merge USVs that are within `bout_gap_threshold` (default: 500ms) of each other into a single bout
   - Each bout has: file, bout_start = first_USV_start − `context_padding` (default: 200ms), bout_end = last_USV_end + `context_padding`
   - Clamp bout_start ≥ 0 and bout_end ≤ file_duration
3. For each bout, extract the raw audio segment from the wav file

**Config dataclass:** `BoutExtractionConfig`
```python
@dataclass
class BoutExtractionConfig:
    bout_gap_threshold_ms: float = 500.0    # max gap between USVs in same bout
    context_padding_ms: float = 200.0       # padding before first / after last USV
    min_bout_duration_ms: float = 50.0      # discard bouts shorter than this
    max_bout_duration_ms: float = 10000.0   # split bouts longer than this
```

**Output:** A list/dataset of bout audio segments with metadata (source file, start/end times, number of USVs in bout).

### Task 0.2: Spectrogram Extraction Module

**File:** `usv_language/data/spectrogram.py`

**Purpose:** Convert bout audio segments to normalized log-magnitude spectrograms.

**Logic:**
1. Compute STFT using the same parameters as the existing CNN pipeline:
   - `sr = 300_000` (300 kHz)
   - `n_fft = 512`
   - `hop_length = 128` (→ 0.427 ms per frame)
2. Take magnitude, convert to log scale: `S_db = 20 * log10(|S| + 1e-10)`
3. Crop frequency axis to 20-120 kHz range → ~170 frequency bins
4. The resulting spectrogram has shape `(n_freq=170, n_frames=T)` where T varies by bout duration

**Important:** Use the EXACT same STFT implementation (librosa or scipy) and parameters as the existing CNN detection pipeline for consistency. If the existing pipeline uses `librosa.stft`, use the same. Do NOT introduce a different STFT implementation.

**Output:** Spectrograms as numpy arrays or torch tensors, shape `(170, T)`.

### Task 0.3: Dataset Normalization

**File:** `usv_language/data/normalization.py`

**Purpose:** Compute and apply per-frequency-bin normalization statistics.

**Logic:**
1. First pass over all bout spectrograms: compute mean and std for each of the 170 frequency bins across the entire dataset
2. Save statistics to a JSON/npz file for reproducibility
3. Normalize: `S_norm[f, t] = (S[f, t] - mean[f]) / (std[f] + 1e-8)` for each frequency bin f

**Important:** Compute statistics on the TRAINING set only. Apply the same statistics to validation and test sets.

### Task 0.4: PyTorch Dataset and DataLoader

**File:** `usv_language/data/dataset.py`

**Purpose:** PyTorch Dataset that yields chunked, padded spectrogram sequences ready for the transformer.

**Logic:**
1. Take normalized bout spectrograms (170 × T each)
2. Transpose to (T × 170) — each row is one "token" (a spectrogram column)
3. Chunk into windows of `max_seq_len` frames (default: 512) with configurable overlap (default: 50%, i.e., stride = 256)
   - Bouts shorter than `max_seq_len` are kept as-is and padded
   - Bouts longer than `max_seq_len` produce multiple overlapping chunks
4. Create attention mask: 1 for real frames, 0 for padding
5. For next-column prediction: input = frames[0:T-1], target = frames[1:T]

**Config dataclass:** `TransformerDataConfig`
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

**DataLoader:** Implement length-bucketed batching:
- Sort all chunks by length
- Create ~6-8 buckets (64, 128, 192, 256, 384, 512)
- Within each batch, pad to the longest sequence in that batch (not globally to 512)
- Use a BucketBatchSampler that samples batches from within buckets

**Data augmentation** (applied with p=0.5 during training only):
- Gaussian noise: add N(0, σ) where σ gives SNR ~15-20 dB
- Gain perturbation: multiply by 10^(g/20) where g ~ Uniform(-3, 3) dB
- Frequency masking: zero out 1-2 random contiguous bands of ~20-30 bins
- Time masking: zero out 1-2 random contiguous spans of ~10% of sequence length

### Task 0.5: Data Pipeline Integration Script

**File:** `usv_language/data/prepare_data.py`

**Purpose:** End-to-end script that runs the full data preparation pipeline.

**Logic:**
1. Accept command-line args: path to wav files directory, path to detection results, output directory
2. Run bout extraction → spectrogram extraction → normalization → save processed dataset
3. Save dataset splits (train/val/test) as separate files or a single file with split indices
4. Save normalization statistics
5. Print dataset summary: number of bouts, total frames, duration statistics, frames per split

---

## Phase 1: Autoregressive Transformer

### Task 1.1: Transformer Model

**File:** `usv_language/models/transformer.py`

**Purpose:** GPT-style autoregressive transformer for next-column prediction on spectrograms.

**Architecture:**

```python
class SpectrogramTransformer(nn.Module):
    """
    Autoregressive transformer for spectrogram next-column prediction.
    
    Input: (batch, seq_len, n_freq) — sequence of spectrogram columns
    Output: (batch, seq_len, n_freq) — predicted next columns
    
    The model predicts frame t+1 given frames 0..t using causal attention.
    """
```

**Components:**

1. **Input projection:**
   ```python
   self.input_proj = nn.Sequential(
       nn.Linear(n_freq, d_model),    # 170 → 512
       nn.GELU(),
       nn.LayerNorm(d_model),
   )
   ```

2. **Positional embeddings:**
   ```python
   self.pos_embed = nn.Embedding(max_seq_len, d_model)  # learned, (512, 512)
   ```

3. **Transformer blocks (×8):**
   Each block contains:
   - `nn.MultiheadAttention(d_model=512, num_heads=8, dropout=0.1, batch_first=True)` with causal mask
   - Pre-norm architecture: LayerNorm BEFORE attention and FFN (more stable training)
   - FFN: `Linear(512, 2048) → GELU → Dropout → Linear(2048, 512) → Dropout`
   - Residual connections around both attention and FFN

4. **Output head:**
   ```python
   self.output_head = nn.Sequential(
       nn.LayerNorm(d_model),
       nn.Linear(d_model, n_freq),    # 512 → 170
   )
   ```

5. **Causal attention mask:**
   ```python
   # Generate once, register as buffer
   causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
   self.register_buffer('causal_mask', causal_mask)
   ```

**Forward pass:**
```python
def forward(self, x, attention_mask=None):
    # x: (batch, seq_len, 170)
    h = self.input_proj(x)                           # (batch, seq_len, 512)
    positions = torch.arange(x.size(1), device=x.device)
    h = h + self.pos_embed(positions)                # add positional embeddings
    
    # Store hidden states for later VQ-VAE extraction
    hidden_states = []
    for i, block in enumerate(self.blocks):
        h = block(h, causal_mask=self.causal_mask, padding_mask=attention_mask)
        hidden_states.append(h)
    
    output = self.output_head(h)                     # (batch, seq_len, 170)
    return output, hidden_states
```

**Config dataclass:**
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

**Critical implementation details:**
- Use **pre-norm** (LayerNorm before attention/FFN), not post-norm. Pre-norm is more stable for training.
- The causal mask must be combined with the padding attention mask when both are present.
- The `hidden_states` list is returned but only used during Phase 2 (VQ-VAE extraction). During Phase 1 training, it can be ignored to save memory — add a flag `return_hidden_states=False` that skips storing them.
- Total parameters: ~25-30M. Verify this matches expectations after implementation.

### Task 1.2: Training Loop

**File:** `usv_language/training/train_transformer.py`

**Purpose:** Training script for Phase 1 transformer.

**Logic:**

1. **Loss function:** MSE between predicted next columns and actual next columns
   ```python
   # input_seq:  frames[0:T-1]  (what the model sees)
   # target_seq: frames[1:T]    (what the model predicts)
   # Mask out padding positions in the loss
   loss = F.mse_loss(predictions * mask, targets * mask, reduction='sum') / mask.sum()
   ```

2. **Optimizer:** AdamW
   ```python
   # Separate parameter groups: apply weight decay only to weight matrices
   decay_params = [p for n, p in model.named_parameters() if 'weight' in n and p.dim() >= 2]
   no_decay_params = [p for n, p in model.named_parameters() if 'bias' in n or p.dim() < 2]
   optimizer = torch.optim.AdamW([
       {'params': decay_params, 'weight_decay': 0.01},
       {'params': no_decay_params, 'weight_decay': 0.0},
   ], lr=1e-4, betas=(0.9, 0.999))
   ```

3. **Learning rate schedule:** Linear warmup (2000 steps) → cosine decay to 1e-6
   ```python
   # Use torch.optim.lr_scheduler.OneCycleLR or custom:
   # warmup: linear from 0 to 1e-4 over 2000 steps
   # decay: cosine from 1e-4 to 1e-6 over remaining steps
   ```

4. **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

5. **Checkpointing:**
   - Save model checkpoint every N epochs (default: 5)
   - Save best model (by validation loss)
   - Save training state (optimizer, scheduler, epoch, step) for resume
   - Early stopping: stop if validation loss hasn't improved for 20 epochs

6. **Logging:**
   - Log training loss, validation loss, learning rate every N steps
   - Log per-frequency-bin error distribution every epoch
   - Use tensorboard or wandb (check which is already used in the project)
   - Every 10 epochs: save visualization of predicted vs. actual spectrograms for 5 random validation samples

7. **Multi-GPU:** Use `torch.nn.DataParallel` or `DistributedDataParallel` if multiple GPUs are available on HPC. Add command-line flags for distributed training.

**Expected training time:** With ~30,000 USVs grouped into bouts, producing roughly 100k-500k sequence chunks, expect 50-200 epochs. On a single A100 GPU, each epoch should take minutes. Total training: hours to a day.

### Task 1.3: Hidden State Extraction

**File:** `usv_language/training/extract_hidden_states.py`

**Purpose:** After transformer training, extract and save hidden states for VQ-VAE training.

**Logic:**
1. Load the trained, frozen transformer (best checkpoint)
2. Set `model.eval()` and `torch.no_grad()`
3. For every sequence chunk in the dataset (train + val + test):
   a. Run through transformer with `return_hidden_states=True`
   b. Extract hidden states from the target layer(s) — default: layer 4, but also extract layers 2, 6, 8 for comparison
   c. Save as numpy arrays or memory-mapped files (these can be large)
4. Save metadata: which bout, which chunk, which frames correspond to each hidden state

**Output format:** For each layer L, save:
- `hidden_states_layer{L}.npy` — shape `(total_frames, 512)` or chunked into manageable files
- `metadata.json` — mapping from frame index to (bout_id, chunk_id, frame_within_chunk, original_timestamp)

**Storage estimate:** If total frames across all chunks ≈ 500k, and each hidden state is 512 floats (2048 bytes), that's ~1GB per layer. For 4 layers: ~4GB. Use memory-mapped files if this exceeds available RAM.

---

## Phase 2: VQ-VAE on Hidden States

### Task 2.1: VQ-VAE Model

**File:** `usv_language/models/vqvae.py`

**Purpose:** VQ-VAE that operates on transformer hidden state vectors.

**Architecture:**

```python
class HiddenStateVQVAE(nn.Module):
    """
    VQ-VAE operating on transformer hidden states.
    
    Input: (batch, seq_len, d_model) — hidden states from transformer layer L
    Output: reconstructed hidden states, codebook indices, losses
    """
```

**Components:**

1. **Encoder:**
   ```python
   self.encoder = nn.Sequential(
       # Optional: 1D conv for local temporal context
       # Rearrange (batch, seq_len, 512) → (batch, 512, seq_len) for Conv1d
       nn.Conv1d(d_model, 256, kernel_size=5, padding=2),  # preserves seq_len
       nn.GELU(),
       # Rearrange back to (batch, seq_len, 256)
       nn.Linear(256, codebook_dim),  # 256 → D (default 64)
   )
   ```
   After encoding, **L2-normalize** the output vectors.

2. **Codebook / Vector Quantization:**
   ```python
   class VectorQuantizer(nn.Module):
       def __init__(self, num_codes, codebook_dim, ema_decay=0.99, commitment_weight=0.25,
                    dead_code_threshold=2.0):
           self.embedding = nn.Embedding(num_codes, codebook_dim)  # K × D
           # EMA tracking
           self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
           self.register_buffer('ema_embed_sum', torch.zeros(num_codes, codebook_dim))
           self.register_buffer('usage_count', torch.zeros(num_codes))
           
       def forward(self, z_e):
           # z_e: (batch, seq_len, D) — L2-normalized encoder outputs
           # 1. Compute distances to all codebook entries
           # 2. Find nearest: k* = argmin ||z_e - e_k||²
           # 3. Straight-through: z_q = z_e + (e_{k*} - z_e).detach()
           # 4. Compute losses:
           #    commitment_loss = ||z_e - sg(e_{k*})||²
           # 5. EMA update codebook (during training only):
           #    Update ema_cluster_size and ema_embed_sum
           # 6. Dead code reinitialization:
           #    If any entry has ema_cluster_size < threshold, reinitialize
           #    from random encoder output in the current batch
           # Returns: z_q, indices, commitment_loss
   ```

3. **Decoder:**
   ```python
   self.decoder = nn.Sequential(
       nn.Linear(codebook_dim, 256),   # D → 256
       nn.GELU(),
       nn.Linear(256, d_model),        # 256 → 512
   )
   ```

**Config dataclass:**
```python
@dataclass
class VQVAEConfig:
    d_model: int = 512              # input dimension (transformer hidden size)
    codebook_size: int = 64         # K — number of codebook entries
    codebook_dim: int = 64          # D — dimension of each entry
    commitment_weight: float = 0.25 # β
    ema_decay: float = 0.99         # γ for EMA updates
    dead_code_threshold: float = 2.0
    use_conv_encoder: bool = True   # whether to use 1D conv for temporal context
    conv_kernel_size: int = 5
```

**Forward pass:**
```python
def forward(self, hidden_states):
    # hidden_states: (batch, seq_len, 512)
    z_e = self.encoder(hidden_states)           # (batch, seq_len, D)
    z_e = F.normalize(z_e, dim=-1)              # L2 normalize
    z_q, indices, commit_loss = self.quantizer(z_e)  # quantize
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

**K-means initialization method:**
```python
def initialize_codebook_from_data(self, encoder_outputs):
    """
    Run k-means on a batch of encoder outputs to initialize codebook.
    Call this before training begins with outputs from ~10-20 batches.
    """
    from sklearn.cluster import KMeans
    # encoder_outputs: (N, D) — collected from first few batches
    kmeans = KMeans(n_clusters=self.config.codebook_size, n_init=10)
    kmeans.fit(encoder_outputs.numpy())
    self.quantizer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_)
```

### Task 2.2: VQ-VAE Training Loop

**File:** `usv_language/training/train_vqvae.py`

**Purpose:** Train the VQ-VAE on extracted hidden states.

**Logic:**

1. Load pre-extracted hidden states from Task 1.3 for the chosen layer (default: layer 4)
2. Create a simple Dataset that yields chunks of hidden states
3. **K-means initialization:** Before training, run encoder on ~5000 samples, collect outputs, run k-means, set codebook
4. Train with AdamW (lr=3e-4, weight decay=0.01), warmup 500 steps, cosine decay
5. Gradient clipping at norm 1.0

**Monitoring (log every N steps):**
- `recon_loss`: MSE between decoded and original hidden states
- `commit_loss`: commitment loss
- `codebook_perplexity`: `exp(entropy of code usage distribution)` — target: > 0.5 × K
- `codebook_utilization`: fraction of codes used at least once in the batch — target: > 90%
- `code_usage_histogram`: distribution of assignments across codebook entries (should be roughly uniform)
- Dead code count: how many entries were reinitialized

**Checkpointing:**
- Save best model (by validation recon_loss)
- Save codebook state separately for easy loading during analysis

**Expected training:** Fast — hidden states are pre-extracted, model is small. Expect convergence in 50-100 epochs, minutes on GPU.

### Task 2.3: Multi-Layer Comparison Script

**File:** `usv_language/training/compare_layers.py`

**Purpose:** Train VQ-VAE separately on hidden states from layers 2, 4, 6, 8 and compare.

**Logic:**
1. For each layer in [2, 4, 6, 8]:
   a. Train VQ-VAE with identical hyperparameters
   b. Record: final recon_loss, codebook perplexity, codebook utilization
   c. Save trained VQ-VAE
2. Generate comparison report:
   - Table of metrics per layer
   - Recommendation for which layer to use for Phase 3 analysis
   - Criterion: highest perplexity + highest utilization + lowest recon_loss, with emphasis on perplexity (interpretability)

---

## Phase 3: Analysis and Interpretation Tools

### Task 3.1: Codebook Visualization

**File:** `usv_language/analysis/codebook_viz.py`

**Purpose:** Decode codebook entries to spectrograms and create visual catalogs.

**Logic:**

1. **Decode codebook entries through the full pipeline:**
   For each codebook entry e_k (dimension D=64):
   a. Pass through VQ-VAE decoder → get reconstructed hidden state h_k (512-dim)
   b. Pass h_k through the remaining transformer layers (layers L+1 through 8) as if it were the hidden state at some position
   c. Read off the transformer's output head prediction → 170-dim spectrogram column
   
   **Note:** This gives the spectrogram column the transformer would PREDICT NEXT if its hidden state were e_k. This reveals what acoustic continuation each concept implies.

2. **Exemplar galleries:**
   For each codebook entry k:
   a. Find the N=10 time steps in the dataset whose encoder output z_e is closest (L2 distance) to e_k
   b. Extract the surrounding spectrogram context (±50 frames) for each exemplar
   c. Create a figure: codebook entry decoded spectrogram + 10 exemplar spectrograms

3. **t-SNE/UMAP visualization:**
   a. Project all K codebook vectors to 2D using t-SNE or UMAP
   b. Color by: mean frequency of exemplars, mean duration of surrounding USVs, or known USV category (if labels exist)
   c. Annotate with decoded spectrogram thumbnails

**Output:** Save all visualizations as PNG files. Create an HTML gallery for easy browsing.

### Task 3.2: Code Sequence Analysis

**File:** `usv_language/analysis/sequence_analysis.py`

**Purpose:** Analyze the sequential structure of codebook index sequences.

**Logic:**

1. **Extract code sequences:** Run all bout spectrograms through frozen transformer → extract hidden states → run through frozen VQ-VAE → get code index per frame. Each bout becomes a sequence of integers [c_0, c_1, ..., c_T].

2. **Zipf's law analysis:**
   a. Count frequency of each code across the entire dataset
   b. Rank codes by frequency (most frequent = rank 1)
   c. Plot log(frequency) vs. log(rank)
   d. Fit power law: frequency ∝ rank^(-α), report α
   e. Compare to α ≈ 1 (Zipf's law for natural language)

3. **Transition analysis:**
   a. Compute bigram transition matrix P(c_{t+1} | c_t) — K × K matrix
   b. Compute transition entropy: H(C_{t+1} | C_t) = -Σ_i P(c_i) Σ_j P(c_j|c_i) log P(c_j|c_i)
   c. Compare to maximum entropy (uniform transitions): H_max = log(K)
   d. Compute mutual information: I(C_t; C_{t+1}) = H(C_{t+1}) - H(C_{t+1} | C_t)
   e. Extend to trigrams: P(c_{t+2} | c_t, c_{t+1})
   f. Visualize transition matrix as heatmap

4. **Entropy rate:**
   a. Estimate entropy rate h = lim_{n→∞} H(C_n | C_{n-1}, ..., C_1)
   b. Approximate with increasing context lengths (1-gram, 2-gram, ..., 8-gram)
   c. Plot entropy rate vs. context length — should decrease and plateau

5. **Excess entropy:**
   a. Mutual information between past and future halves of sequences
   b. Higher values = more complex long-range structure

**Output:** Save all plots as PNGs, save numerical results as JSON. Print summary to console.

### Task 3.3: Concept Manipulation Tool

**File:** `usv_language/analysis/concept_manipulation.py`

**Purpose:** Inject specific codebook entries into the transformer and observe predictions.

**Logic:**

1. **Single concept injection:**
   a. Take a real sequence of hidden states from a bout
   b. At a chosen time step t, replace h_t with the VQ-VAE-decoded version of codebook entry k
   c. Pass through remaining transformer layers
   d. Collect predictions for steps t+1, t+2, ..., t+N (autoregressive generation)
   e. Visualize the resulting predicted spectrogram

2. **Concept scanning:**
   a. For a fixed context (frames 0..t-1), inject each of the K codebook entries at position t
   b. For each injection, record the predicted next frame
   c. Create a K × 170 matrix showing what each concept predicts in this context
   d. Cluster these predictions to find which concepts lead to similar continuations

3. **Top-k concept analysis:**
   a. For each time step, record distances to ALL K codebook entries
   b. Identify the top-4 nearest entries
   c. Visualize: at each time step, show the 4 "competing" concepts and their distances
   d. Look for transition points where the winning concept changes

**Output:** Interactive visualizations (matplotlib or plotly). Save as HTML for easy sharing.

### Task 3.4: Context-Dependent Analysis

**File:** `usv_language/analysis/context_analysis.py`

**Purpose:** Test whether code distributions change with social/biological context.

**Logic:**

1. If metadata is available (mouse ID, sex, strain, social context):
   a. Group code sequences by metadata variable
   b. Compare code frequency distributions across groups (chi-squared test, KL divergence)
   c. Compare transition matrices across groups
   d. Report which codes are most differentially used between groups

2. If no metadata: skip this task, note in output that it requires metadata.

**Output:** Statistical test results, comparison plots.

### Task 3.5: Compositionality Tests

**File:** `usv_language/analysis/compositionality.py`

**Purpose:** Test whether code combinations follow compositional rules.

**Logic:**

1. **Bigram productivity:** How many unique bigrams (c_i, c_j) are observed vs. theoretically possible (K²)?
2. **Held-out bigram test:** Can the VQ-VAE decode bigram combinations NOT seen during training into meaningful spectrograms?
3. **Positional independence:** Do codes maintain their identity regardless of position in sequence? (Measure by comparing exemplars of the same code at different sequence positions)

---

## Project Structure

```
usv_language/
├── configs/
│   ├── default_config.yaml          # all hyperparameters in one place
│   └── experiment_configs/          # per-experiment overrides
├── data/
│   ├── bout_extractor.py           # Task 0.1
│   ├── spectrogram.py              # Task 0.2
│   ├── normalization.py            # Task 0.3
│   ├── dataset.py                  # Task 0.4
│   └── prepare_data.py             # Task 0.5
├── models/
│   ├── transformer.py              # Task 1.1
│   └── vqvae.py                    # Task 2.1
├── training/
│   ├── train_transformer.py        # Task 1.2
│   ├── extract_hidden_states.py    # Task 1.3
│   ├── train_vqvae.py              # Task 2.2
│   └── compare_layers.py           # Task 2.3
├── analysis/
│   ├── codebook_viz.py             # Task 3.1
│   ├── sequence_analysis.py        # Task 3.2
│   ├── concept_manipulation.py     # Task 3.3
│   ├── context_analysis.py         # Task 3.4
│   └── compositionality.py         # Task 3.5
└── utils/
    ├── logging.py                  # tensorboard/wandb integration
    ├── checkpointing.py            # save/load model checkpoints
    └── visualization.py            # shared plotting utilities
```

## Configuration

**File:** `usv_language/configs/default_config.yaml`

All hyperparameters should live in a single YAML config file loaded at the start of each script. This ensures reproducibility and makes it easy to run experiments with different settings.

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

## Implementation Order

Execute tasks in this order, verifying each before moving on:

1. **Task 0.2** — Spectrogram extraction (reuse existing code where possible)
2. **Task 0.1** — Bout extraction
3. **Task 0.3** — Normalization
4. **Task 0.4** — Dataset and DataLoader
5. **Task 0.5** — Data preparation integration script
6. **Verify:** Run prepare_data.py, inspect output spectrograms visually, confirm shapes and normalization
7. **Task 1.1** — Transformer model
8. **Verify:** Check parameter count (~25-30M), run a forward pass on dummy data, verify output shapes
9. **Task 1.2** — Transformer training loop
10. **Verify:** Train for ~5 epochs, confirm loss decreases, inspect predicted vs. actual spectrograms
11. **Task 1.3** — Hidden state extraction
12. **Task 2.1** — VQ-VAE model
13. **Verify:** Run forward pass on dummy data, verify codebook assignment, check gradient flow through STE
14. **Task 2.2** — VQ-VAE training loop
15. **Task 2.3** — Multi-layer comparison
16. **Task 3.1** — Codebook visualization
17. **Task 3.2** — Sequence analysis
18. **Task 3.3** — Concept manipulation
19. **Task 3.4** — Context analysis (if metadata available)
20. **Task 3.5** — Compositionality tests

## Dependencies

Add to existing project requirements:
```
torch >= 2.0
torchaudio
librosa
scikit-learn       # for k-means initialization
matplotlib
seaborn
plotly             # optional, for interactive visualizations
umap-learn         # for UMAP projections
tensorboard        # or wandb
pyyaml             # for config loading
```

## Testing Strategy

Each module should have basic tests:
- **Data pipeline:** Test bout extraction on a small synthetic dataset with known USV positions. Verify spectrogram shapes. Test chunking logic (overlap, padding, attention masks).
- **Transformer:** Test forward pass shapes. Test that causal mask prevents attending to future. Test gradient flow. Test that loss decreases on overfitting to a single batch.
- **VQ-VAE:** Test forward pass shapes. Test that codebook indices are valid (0 to K-1). Test straight-through estimator gradient flow. Test dead code reinitialization. Test that reconstruction loss decreases.
- **Analysis:** Test on synthetic code sequences with known properties (e.g., a sequence following Zipf's law should be detected as such).
