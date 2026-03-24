# Architecture Decision Records

This document captures the key technical decisions made for the USV Spectrogram project.
Each ADR records the context, decision, and rationale so future contributors (and future sessions)
can understand *why* things are the way they are.

**Format:** ADR-NNN: Title | Status | Date

---

## ADR-001: Sample Rate — 300 kHz

**Status:** Accepted
**Date:** 2025 (confirmed 2026-02)

**Context:**
Mouse ultrasonic vocalizations (USVs) range up to ~120 kHz. Nyquist requires at least 240 kHz
to capture this range. Our recording hardware operates at 300 kHz, giving comfortable headroom
up to 150 kHz.

**Decision:**
The canonical sample rate for this project is **300,000 Hz (300 kHz)**.

- `DetectionConfig.sample_rate = 300_000`
- `usv_language` STFTConfig uses `sample_rate = 300_000`

**Legacy note:** `SpectrogramConfig.expected_sample_rate_hz = 250_000` is **outdated** and needs
updating. This was an early default from before the recording setup was finalized.
CLAUDE.md still references 250 kHz in several places — these should be updated to 300 kHz.

**Consequences:**
- All DSP code must use sr=300000 (or read from WAV via auto_sample_rate)
- Never rely on librosa's default sample rate
- Frequency resolution at n_fft=512: 300,000 / 512 = 585.9 Hz/bin

---

## ADR-002: STFT Parameters

**Status:** Accepted

**Context:**
USVs are short (10-500 ms) and narrow-band. We need good temporal resolution to capture
onset/offset precisely, while maintaining enough frequency resolution to distinguish USV
subtypes.

**Decision:**
Detection and VQ-VAE pipelines share core STFT parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_fft | 512 | ~1.7 ms window at 300 kHz; good time resolution |
| hop_length | 128 | 75% overlap; smooth temporal coverage |
| window | Hann | Standard for spectral analysis; good sidelobe suppression |
| freq_min_hz | 20,000-25,000 | Below mouse USV range (detection uses 25k, VQ-VAE uses 20k) |
| freq_max_hz | 110,000-120,000 | Upper bound of mouse USV range (detection 110k, VQ-VAE 120k) |

**Derived values:**
- Frame duration: 512 / 300,000 = 1.707 ms
- Hop duration: 128 / 300,000 = 0.427 ms
- Frequency resolution: 300,000 / 512 = 585.9 Hz/bin
- Frequency bins in 20-120 kHz: ~171 bins

**Note:** `SpectrogramConfig` (visualization) uses different parameters optimized for display:
n_fft=2048, zero_padding to 4096, giving 61 Hz/bin resolution. This is intentionally different
from the detection/analysis STFT.

---

## ADR-003: Detection Thresholds

**Status:** Accepted

**Context:**
The detection pipeline is two-stage: energy detector (high recall) followed by CNN classifier
(precision filter). The energy detector should be permissive to avoid missing USVs; the CNN
then rejects false positives.

**Decision:**

| Stage | Parameter | Value | Rationale |
|-------|-----------|-------|-----------|
| Energy | energy_threshold_db | -60.0 dB | Deliberately low for high recall |
| Energy | energy_mode | "peak" | Max energy in band per frame; better for narrow-band USVs |
| Energy | max_bandwidth_hz | 20,000 | Reject broadband noise |
| CNN | cnn_threshold (optimal) | 0.05 | Very sensitive default |
| CNN | high_threshold (app) | 0.40 | High-confidence detection in desktop app |
| CNN | low_threshold (app) | 0.28 | Low-confidence, requires extension check |

**Consequences:**
- Energy stage produces many candidates (high recall, low precision)
- CNN stage filters to final detections
- Threshold tuning requires baseline comparison (see Red Flags in CLAUDE.md)

---

## ADR-004: Dataset Splitting — By Recording

**Status:** Accepted

**Context:**
USVs from the same recording are temporally correlated. If chunks from the same recording
appear in both train and test sets, the model can "cheat" by memorizing recording-specific
noise patterns.

**Decision:**
All dataset splits are performed **by recording file stem**, not by individual candidate/chunk.

- `usv_language`: splits by file stem with seed=42 (80/10/10 train/val/test)
- CNN training: splits by recording to prevent data leakage

**Consequences:**
- Smaller effective training set (can't mix chunks from same file)
- More honest evaluation metrics
- Requires enough distinct recordings for meaningful splits

---

## ADR-005: Class Weighting — 3.0x USV Boost

**Status:** Accepted

**Context:**
USV candidates are heavily imbalanced — noise/non-USV samples outnumber true USVs. Without
class weighting, the model learns to always predict "not USV."

**Decision:**
Apply a **3.0x boost** to the positive (USV) class weight during CNN training.

- Base pos_weight is computed from class ratios (~11.8 raw)
- Additional 3.0x multiplier applied on top
- Final effective pos_weight ~ 35.4

**Consequences:**
- Model biased toward recall (catching USVs) over precision
- Acceptable because the two-stage pipeline (energy + CNN) can tolerate some false positives
- Users can adjust threshold post-hoc for their precision/recall needs

---

## ADR-006: CNN Architecture — 3 Conv Blocks + GlobalAvgPool

**Status:** Accepted

**Context:**
The CNN classifier needs to be small enough to train on limited labeled data (~hundreds to
low thousands of examples) while being expressive enough to distinguish USV spectrograms
from noise.

**Decision:**
**USVClassifierCNN (Small Model):**
- 3 convolutional blocks: [32, 64, 128] filters
- Each block: Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2)
- Global Average Pooling (handles variable input sizes)
- Dense head: Linear(128->64) -> ReLU -> Dropout(0.5) -> Linear(64->1)
- Output: logits for BCEWithLogitsLoss
- **Total parameters: ~101,889** (93,568 conv+bn + 8,321 classifier)

A larger variant exists (USVClassifierCNNLarge, 5 blocks, [32,64,128,256,512]) but is
not recommended for datasets under 5,000 samples.

**Consequences:**
- Fast training on CPU
- Low risk of overfitting on small datasets
- Variable-size input via GlobalAvgPool (no fixed spectrogram dimensions required)

---

## ADR-007: Transformer + VQ-VAE Two-Phase Architecture (v2)

**Status:** Accepted (supersedes v1)

**Context:**
To discover language-like structure in USV repertoires, we need a model that learns rich
contextual representations of the acoustic stream and a discrete codebook that reveals
interpretable "concepts" within those representations.

The v1 approach trained an end-to-end VQ-VAE + Transformer (~437K params, d_model=64, K=512)
that jointly learned discretization and sequence prediction. While functional, this forces
discretization before the model knows what matters, potentially preventing it from discovering
subtle patterns. The v2 architecture separates these concerns into two phases.

**Decision:**

**Phase 1 — Autoregressive Transformer (self-supervised, no bottleneck):**

| Component | Parameter | Value |
|-----------|-----------|-------|
| Input projection | dims | Linear(170 → 512) → GELU → LayerNorm |
| Positional embeddings | type | Learned, max_seq_len × d_model |
| Transformer | d_model | 512 |
| Transformer | n_heads | 8 |
| Transformer | n_layers | 8 |
| Transformer | d_ff | 2048 |
| Transformer | dropout | 0.1 |
| Transformer | architecture | Pre-norm (LayerNorm before attention/FFN) |
| Transformer | attention | Causal (GPT-style) |
| Transformer | max_seq_len | 512 |
| Output head | dims | LayerNorm → Linear(512 → 170) |
| Loss | type | MSE (predicted vs actual next column) |

**Total parameters:** ~25-30M

The transformer receives raw spectrogram columns (170-dim) and predicts the next column
autoregressively. It develops internal representations freely, without discretization.
Deeper layers encode increasingly abstract patterns.

**Phase 2 — VQ-VAE on hidden states (post-hoc interpretability tool):**

| Component | Parameter | Value |
|-----------|-----------|-------|
| Encoder | architecture | Conv1d(512→256, k=5) → GELU → Linear(256→64) → L2-norm |
| Codebook | K (codebook size) | 64 (start; explore 32, 128, 256) |
| Codebook | D (codebook dim) | 64 |
| Codebook | EMA decay | 0.99 |
| Codebook | commitment weight (β) | 0.25 |
| Codebook | dead code threshold | 2.0 |
| Codebook | initialization | K-means on encoder outputs |
| Decoder | architecture | Linear(64→256) → GELU → Linear(256→512) |
| Hidden state source | default layer | 4 of 8 (compare 2, 4, 6, 8) |

After the transformer is frozen, hidden states from a chosen middle layer are extracted.
The VQ-VAE compresses these continuous representations into a small discrete codebook.
Each codebook entry becomes an interpretable "concept."

**Why this order (not end-to-end, not VQ-VAE first):**
- End-to-end: Forces discretization before the model knows what matters, constraining
  what the model can represent.
- VQ-VAE first (DALL-E style): Would only capture local spectral patterns, not the
  contextual, predictive representations where "concepts" live.
- Transformer first: Freely learns whatever representations are most useful for
  prediction. VQ-VAE then discovers structure within those learned representations.

See `docs/plans/theoretical_guide.md` for full scientific rationale and prior work.

**Consequences:**
- Much larger model (~25-30M vs ~437K) — requires HPC/cloud GPU for training
- Two-phase training: transformer must fully converge before VQ-VAE begins
- K=64 gives interpretable codebook (vs K=512 in v1); traditional USV taxonomy has ~10-15 types, K=64 gives headroom for subtypes
- Middle layer extraction captures mid-level concepts (not raw spectral features from early layers, not highly prediction-specialized features from late layers)
- Codebook collapse prevention is critical: EMA updates + dead code reset + k-means init + L2-norm all required simultaneously
- FSQ (Finite Scalar Quantization) available as fallback if collapse persists

---

## ADR-008: Negative Sample Strategy — 3-Source Mix

**Status:** Accepted

**Context:**
The CNN was originally trained only on energy-detector candidates, causing it to classify
everything as USV (0.997 mean probability on random audio chunks). It had never seen
"normal" audio that wasn't already flagged by the energy detector.

**Decision:**
Generate negative training samples from **3 distinct sources**:

1. **Random chunks** (`random`): Random time slices from recordings, no energy filtering
2. **Inter-USV gaps** (`inter_usv_gap`): Audio between known USV detections
3. **Low-energy regions** (`low_energy`): Deliberately quiet segments

**Consequences:**
- Model learns the full spectrum of "not USV" audio
- Random chunks teach it about normal background noise
- Gap samples teach it about near-USV silence
- Low-energy samples prevent false triggers on quiet artifacts

---

## ADR-009: Model Artifacts — PyTorch .pt Files

**Status:** Accepted

**Context:**
We need a standard format for saving trained model weights that integrates with our
PyTorch-based training pipeline.

**Decision:**
All trained models are saved as PyTorch `.pt` files using `torch.save()`.

- CNN classifier: single `.pt` file with state_dict
- VQ-VAE: `.pt` checkpoint with model state_dict, optimizer state, and training metadata
- Checkpoints saved every N epochs during training

**Consequences:**
- Native PyTorch format, no extra dependencies
- Easy to load with `torch.load()` + `model.load_state_dict()`
- Not portable to non-PyTorch frameworks (acceptable for this project)

---

## ADR-010: Label Storage Format — JSON

**Status:** Accepted

**Context:**
The desktop labeling app needs to persist user labels, detection boundaries, and metadata
in a human-readable, version-controllable format.

**Decision:**
Labels are stored as **JSON files**, one per WAV file, with structure:

```json
{
  "metadata": {
    "wav_file": "path",
    "model_file": "path",
    "created_at": "ISO datetime",
    "duration_s": "float",
    "sample_rate": "int",
    "n_detections": "int",
    "file_label": "string|null"
  },
  "detection_params": {
    "high_threshold": "float",
    "low_threshold": "float"
  },
  "detections": [
    {
      "start_time_s": "float",
      "end_time_s": "float",
      "duration_s": "float",
      "max_probability": "float",
      "mean_probability": "float",
      "user_adjusted": "bool",
      "user_action": "string|null"
    }
  ],
  "probability_curve": {
    "times": ["float"],
    "probabilities": ["float"]
  }
}
```

**Consequences:**
- Human-readable, easy to inspect and debug
- Git-friendly for version control
- Slightly verbose compared to binary formats, but file sizes are small

---

## ADR-011: Auto Sample Rate — Read from WAV

**Status:** Accepted

**Context:**
While the canonical sample rate is 300 kHz, WAV files from different recording setups
might have different rates. Hardcoding would silently produce wrong results.

**Decision:**
`DetectionConfig.auto_sample_rate = True` by default. When enabled, the detection pipeline
reads the actual sample rate from the WAV file header and uses that instead of the
configured default.

**Consequences:**
- Robust to varying recording setups
- STFT frequency bins are always correct for the actual data
- The `sample_rate` field in config serves as fallback only

---

## ADR-012: Energy Detection Mode — Peak (Not Mean)

**Status:** Accepted

**Context:**
USVs are narrow-band signals. Using mean energy across the entire frequency band would
dilute the USV signal with noise from non-USV frequencies. Peak energy captures the
strongest frequency component per frame.

**Decision:**
`DetectionConfig.energy_mode = "peak"` — use the maximum energy value within the
frequency band for each time frame.

**Consequences:**
- Better sensitivity to narrow-band USVs
- Slightly more susceptible to narrow-band noise (mitigated by bandwidth filter)
- Mean mode available as alternative for broadband signal detection

---

## ADR-013: Segment Continuity — Enabled by Default

**Status:** Accepted

**Context:**
USVs can have brief amplitude dips that cause the energy detector to split a single
vocalization into multiple fragments. Segment continuity analysis bridges these gaps
by examining frequency and energy patterns in the gap region.

**Decision:**
Segment continuity is **enabled by default** with these key parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_gap_ms | 5.0 | Bridge dips shorter than 5 ms |
| freq_tolerance_hz | 1,500 | Adjacent segments must be within 1.5 kHz |
| energy_tolerance_db | 15.0 | Gap energy within 15 dB of segment energy |
| gap_match_fraction | 0.6 | At least 60% of gap frames must match criteria |

**Consequences:**
- Reduces over-segmentation of single USVs
- May occasionally merge genuinely separate USVs if gap < 5 ms
- Configurable per-run via DetectionConfig fields

---

## ADR-014: Bout-Level Data for Transformer Training

**Status:** Accepted

**Context:**
The transformer (ADR-007 Phase 1) needs input data that preserves inter-USV timing and
context. Three options:
1. **Isolated USV crops** (~40ms windows around each detection) — discards inter-call context
2. **Full WAV files** (seconds to minutes) — mostly silence, wasteful
3. **Bouts** — continuous segments containing clusters of USV activity with padding

**Decision:**
Use **bout-level spectrograms** extracted by grouping nearby USV detections:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| bout_gap_threshold | 500 ms | USVs within 500ms grouped into same bout |
| context_padding | 200 ms | Padding before first / after last USV in bout |
| min_bout_duration | 50 ms | Discard very short bouts (likely noise) |
| max_bout_duration | 10,000 ms | Split very long bouts to bound sequence length |

Bout extraction requires CNN detection results (start/end times per USV per file), which
come from the batch detection pipeline (Phase 13).

**Consequences:**
- Preserves inter-USV timing, silence gaps, and transition patterns
- Transformer sees continuous acoustic stream at sub-USV resolution (~0.427ms per frame)
- Model can discover structure at whatever granularity is informative (within-call, across boundaries, in silence patterns)
- Depends on detection pipeline output — bout quality is bounded by detection quality
- 200ms padding is tunable; provides pre/post context without long dead stretches
