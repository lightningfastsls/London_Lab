# Mickey London Lab — Project State Report

**Generated:** 2026-02-20
**Repository:** mickey_london_lab
**Author:** Claude Code (Opus 4.6) with human researcher

---

## Part 1: Executive Summary

### What This Project Is

This is a research pipeline for analyzing **mouse ultrasonic vocalizations (USVs)** — high-frequency calls (20–120 kHz) that mice produce during social interactions, particularly courtship. The project serves two purposes:

1. **An operational detection and classification system** that processes 300 kHz WAV recordings to find, classify, and label USV calls using a two-stage energy detector + CNN pipeline.
2. **A research platform for testing whether mouse USVs have language-like sequential structure**, using a large autoregressive transformer (~25.6M params) followed by a VQ-VAE codebook discovery phase.

The project is developed by a solo researcher at the Mickey London Lab, with Claude Code as the primary development partner. It includes a fully integrated knowledge management system (arscontexta) that serves as the project's external research memory.

### Current Capability Snapshot

| Capability | Status | Key Metric |
|-----------|--------|------------|
| USV energy detection | Operational | 42 tests, high-recall design |
| CNN binary classification | Operational | F1 91.7%, precision 89.7%, recall 93.8% |
| Desktop detection app (PyQt6) | Operational | Interactive labeling, threshold adjustment |
| Streamlit labeling tool | Operational | ~840 labels collected |
| Spectrogram extraction | Operational | PNG output from candidates |
| Dataset preparation | Operational | Recording-based splits, 7 quality checks |
| Clustering exploration | Operational | k-means, HDBSCAN, GMM on CNN features |
| Bout data pipeline (transformer) | Code complete | 56 tests, bucketed batching |
| Autoregressive transformer | Code complete | 25.6M params, 11 tests |
| VQ-VAE on hidden states | Design ready | K=64 codebook, not yet implemented |
| Analysis & interpretation tools | Designed | Zipf, entropy, concept injection |
| Knowledge system (arscontexta) | Operational | 117 notes, 1011 wiki links, 6 topic maps |

### Where the Research Is Heading

The central research question: **"Do mouse USVs have language-like sequential structure?"**

The approach: Train a GPT-style transformer to predict "what comes next" in the acoustic stream. Then apply a VQ-VAE to the transformer's internal representations to discover discrete "concepts." Finally, test those concept sequences for language-like statistical properties — Zipf's law, transition entropy, excess entropy, bigram productivity. If mouse USVs have combinatorial structure, the codebook sequences will show it.

### Key Numbers

| Metric | Value |
|--------|-------|
| Total tests | 351+ (all passing) |
| Atomic research notes | 117 |
| Wiki links | 1,011 (avg 8.6/note) |
| Architecture Decision Records | 14 |
| Human labels collected | ~840 (458 USV, 374 Not USV, 8 Uncertain) |
| WAV files available | ~6,500 (300 kHz, from LMT system) |
| CNN parameters | ~101K (small model) |
| Transformer parameters | ~25.6M |
| ROADMAP phases defined | 13 (7 done, 2 code-complete, 4 planned) |

---

## Part 2: The USV Detection Pipeline

### Data Flow Overview

```
                  ┌──────────────┐
                  │  WAV Files   │  ~6,500 recordings at 300 kHz
                  │  (300 kHz)   │  from Live Mouse Tracker (LMT)
                  └──────┬───────┘
                         │
                    Phase 1: Energy Detection
                         │  STFT → energy threshold → candidate segments
                         │  Parameters: n_fft=512, hop=128, threshold=-60dB
                         │  Mode: peak energy, 20-120 kHz band
                         │  High recall, many false positives
                         │
                  ┌──────┴───────┐
                  │  Candidates  │  Time segments with USV-like energy
                  └──────┬───────┘
                         │
                    Phase 2: Spectrogram Extraction
                         │  Candidate → PNG spectrogram image
                         │
                    Phase 3: Human Labeling (Streamlit or PyQt6)
                         │  USV / Not USV / Uncertain
                         │  ~840 labels collected
                         │
                    Phase 4: Dataset Preparation
                         │  Recording-based splits (ADR-004)
                         │  7 quality checks, 3-source negatives (ADR-008)
                         │
                    Phase 5: CNN Training
                         │  3 conv blocks + GlobalAvgPool (~101K params)
                         │  BCEWithLogitsLoss, 3x class weight boost
                         │  F1: 91.7% at threshold 0.05
                         │
                    Phase 6: Desktop App (PyQt6)
                         │  Interactive detection, boundary editing
                         │  Sliding CNN inference, hysteresis thresholding
                         │  Progressive labeling presets
                         │
                    Phase 7: Clustering
                         │  CNN as feature extractor
                         │  t-SNE/UMAP, k-means/HDBSCAN/GMM
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    Phase 8:        Phase 9-10:     Phase 13:
    Transformer     Active          Batch
    + VQ-VAE        Learning        Detection
    (language        Cycle          (~6,500 files)
     hypothesis)    (scaling)
```

### Phase-by-Phase Detail

#### Phase 1: Energy Detection (DONE)

The first stage of detection. A permissive energy-based detector finds all candidate USV segments, prioritizing recall over precision.

**How it works:**
1. Compute STFT (n_fft=512, hop=128, Hann window) at sr=300,000
2. Extract energy in the 20–120 kHz frequency band
3. Use peak energy mode (max per frame, not mean) — better for narrow-band USVs
4. Apply threshold at -60 dB (deliberately low for high recall)
5. Reject candidates with bandwidth > 20 kHz (broadband noise filter)
6. Apply segment continuity to bridge brief (<5 ms) amplitude dips within single USVs

**Key decisions:**
- ADR-001: 300 kHz sample rate (Nyquist headroom to 150 kHz)
- ADR-002: STFT params — 1.7 ms window, 586 Hz/bin frequency resolution, 0.427 ms hop
- ADR-003: -60 dB threshold, peak mode, 20 kHz max bandwidth
- ADR-012: Peak energy mode over mean (avoids signal dilution for narrow-band USVs)
- ADR-013: Segment continuity with 5 ms gap bridging, 1.5 kHz frequency tolerance

**Key files:** `src/usv_spectrogram/detection/energy_detector.py`, `detection/config.py`, `detection/candidate.py`
**Tests:** 42 (test_energy_detector.py)

#### Phase 2: Spectrogram Extraction (DONE)

Extracts PNG spectrogram images from candidate segments for visual review and CNN training.

**Key files:** `src/usv_spectrogram/detection/spectrogram_extractor.py`, `detection/extraction_config.py`

#### Phase 3: Labeling Tool (DONE)

Two labeling interfaces:
- **Streamlit app** (`labeling/labeling_app.py`): Web-based labeling with keyboard shortcuts, progress tracking, session management
- **PyQt6 desktop app** (Phase 6): Full-featured detection + labeling in one tool

**Dataset:** ~458 USV, ~374 Not USV, ~8 Uncertain (~840 total labels)

#### Phase 4: Dataset Preparation (DONE)

Splits data by recording (not by candidate) to prevent data leakage. Includes 7 pre-training quality checks.

**Key decisions:**
- ADR-004: Recording-level splits (honest evaluation, smaller effective training set)
- ADR-008: 3-source negative sampling (random chunks 50%, inter-USV gaps 30%, low-energy regions 20%)

**Key files:** `src/usv_spectrogram/dataset/splits.py`, `dataset/quality_checks.py`, `dataset/metadata.py`
**Tests:** 30 (splits) + 41 (quality checks) = 71

#### Phase 5: CNN Classifier (DONE)

Binary USV/Not-USV classifier.

**Architecture (USVClassifierCNN, ~101K params):**
- 3 convolutional blocks: [32, 64, 128] filters
- Each block: Conv2d(3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
- Global Average Pooling (handles variable input sizes)
- Dense head: Linear(128→64) → ReLU → Dropout(0.5) → Linear(64→1)
- Loss: BCEWithLogitsLoss with 3.0× class weight boost (effective pos_weight ~35.4)

**Performance baseline:**

| Metric | Value |
|--------|-------|
| Precision | 89.7% |
| Recall | 93.8% |
| F1 Score | 91.7% |
| Optimal threshold | 0.05 |

**Key decisions:**
- ADR-005: 3.0× USV class weight boost (biases toward recall)
- ADR-006: Small 3-block architecture for small datasets
- ADR-009: Model artifacts as PyTorch .pt files

**Key files:** `src/usv_spectrogram/models/cnn_classifier.py`, `models/trainer.py`, `models/evaluate.py`
**Tests:** 38 (test_cnn_model.py)

**Scaling plan:** Small (101K, <5K labels) → Medium (~400K, 5K–15K) → Large (~1.6M, 15K+)

#### Phase 6: Desktop Detection App (DONE)

PyQt6 desktop application for interactive USV detection and labeling.

**Features:**
- Full-file STFT computation and spectrogram visualization
- CNN sliding window inference with probability curve display
- Hysteresis thresholding (high: 0.40, low: 0.28) for detection
- Interactive boundary handles for adjusting detection boundaries
- Progressive labeling presets for workflow efficiency
- Session tracking and label persistence (JSON format, ADR-010)
- Auto-move of reviewed files to `_reviewed` directories

**Key files:** `src/usv_spectrogram/app/main_window.py` (orchestration), `app/widgets/spectrogram_view.py` (display), `app/core/sliding_inference.py` (CNN inference), `app/core/detection_logic.py` (hysteresis), `app/core/label_storage.py` (persistence)

#### Phase 7: Clustering Exploration (DONE)

Uses the trained CNN as a feature extractor, then clusters and visualizes the resulting feature space.

**Pipeline:** CNN features → t-SNE/UMAP projection → k-means / HDBSCAN / GMM clustering → cluster analysis with exemplars

**Key files:** `src/usv_spectrogram/clustering/feature_extractor.py`, `clustering/visualizer.py`, `clustering/clusterer.py`, `clustering/analyzer.py`

---

## Part 3: The VQ-VAE / Language Hypothesis

### The Research Question

**"Do mouse ultrasonic vocalizations have language-like sequential structure?"**

Mice produce complex sequences of USVs during social interactions. Chabout et al. (2015) showed male mice change syllable syntax with social context. Hertz et al. (2020) demonstrated that USV sequence statistics carry predictive information. But no one has applied modern representation learning to ask whether these sequences have the statistical hallmarks of language — power-law frequency distributions (Zipf's law), sequential predictability (entropy rate), long-range structure (excess entropy), and compositionality (bigram productivity).

### Why v1 Was Abandoned

The original approach (v1) trained an end-to-end VQ-VAE + Transformer jointly (~437K params, d_model=64, K=512 codebook, 4 layers, 4 heads). While functional (63 tests passing, code complete), this architecture forces discretization *before* the model knows what matters. The VQ-VAE bottleneck constrains what the transformer can represent, potentially preventing discovery of subtle patterns.

### The v2 Two-Phase Architecture (Current)

**Phase 1 — Train an autoregressive transformer (no bottleneck):**
The transformer receives raw spectrogram columns (170-dim vectors at 0.427 ms intervals) and predicts the next column autoregressively — "given what came before, what comes next?" It develops internal representations freely, without any discretization constraint. Deeper layers encode increasingly abstract patterns.

**Phase 2 — Apply VQ-VAE to frozen hidden states:**
After the transformer converges, extract hidden states from a middle layer (default: layer 4 of 8). Train a VQ-VAE to compress these 512-dim continuous vectors into a small discrete codebook (K=64 entries). Each codebook entry becomes an interpretable "concept" — a recurring pattern the transformer learned to recognize.

**Why this order:**
- **End-to-end** (v1): Forces discretization before the model knows what matters
- **VQ-VAE first** (DALL-E style): Would only capture local spectral patterns, not contextual representations
- **Transformer first** (v2, chosen): Freely learns whatever is most useful for prediction; VQ-VAE discovers structure within those learned representations

### Phase 8.1: Bout Data Pipeline (DONE)

Extracts bout-level spectrograms from raw WAV files using CNN detection results, normalizes per-frequency-bin, and creates PyTorch datasets with length-bucketed batching.

**Key concept — Bouts:** Continuous recording segments containing clusters of USV activity with surrounding context. USVs within 500 ms are grouped into the same bout, with 200 ms padding before/after. This preserves inter-USV timing, silence gaps, and transitions — the raw material for discovering sequential structure.

**Pipeline:** WAV + detection results → bout extraction (500 ms gap threshold, ±200 ms padding) → STFT (sr=300k, n_fft=512, hop=128, 20–120 kHz → 170 freq bins) → per-frequency-bin normalization (Welford's online algorithm) → chunking (512 frames, 50% overlap) → bucketed batching (6 length buckets)

**Data augmentation (training only, p=0.5):**
- Gaussian noise (SNR ~15–20 dB)
- Gain perturbation (±3 dB)
- Frequency masking (SpecAugment-style, 1–2 bands of ~20–30 bins)
- Time masking (1–2 spans of ~10% sequence length)

**Key files:** `usv_language/data/bout_extractor.py`, `data/spectrogram.py`, `data/normalization.py`, `data/dataset.py`, `data/prepare_data.py`
**Tests:** 56 (15 bout + 12 spectrogram + 8 normalization + 21 dataset)
**Key decision:** ADR-014 (bout-level data over isolated crops or full files)

### Phase 8.2: Autoregressive Transformer (DONE — code complete)

GPT-style causal transformer for next-spectrogram-column prediction.

**Architecture (SpectrogramTransformer, ~25.6M params):**

| Component | Detail |
|-----------|--------|
| Input projection | Linear(170 → 512) → GELU → LayerNorm |
| Positional embeddings | Learned, max 512 positions |
| Transformer blocks | 8× pre-norm (LayerNorm before attention/FFN) |
| Attention | 8 heads, d_model=512, causal mask |
| FFN | Linear(512→2048) → GELU → Dropout → Linear(2048→512) |
| Output head | LayerNorm → Linear(512 → 170) |
| Dropout | 0.1 throughout |

**Training configuration:**
- Loss: MSE between predicted and actual next columns, masked for padding
- Optimizer: AdamW (weight_decay=0.01 for weights only, 0.0 for biases/norms)
- LR schedule: Linear warmup (2000 steps) → cosine decay to 1e-6, peak lr=1e-4
- Gradient clipping: max norm 1.0
- Early stopping: 20 epochs patience
- Multi-GPU support: DataParallel/DistributedDataParallel via CLI flags

**Hidden state extraction:** After training, extract hidden states from layers 2, 4, 6, 8 as memory-mapped numpy arrays with metadata JSON mapping frame indices to bout/chunk/timestamp.

**Key files:** `usv_language/models/transformer.py`, `training/train_transformer.py`, `training/extract_hidden_states.py`
**Tests:** 11 (8 spec + 3 config validation)

**Critical note:** The transformer is ~25.6M params. Training on the full dataset requires HPC/cloud GPU (AMD RX 5700 is insufficient). Code is testable locally on dummy data.

### Phase 8.3: VQ-VAE on Hidden States (BLOCKED — design ready)

VQ-VAE that discovers discrete "concepts" in the transformer's continuous internal representations.

**Architecture (HiddenStateVQVAE):**
- Encoder: Conv1d(512→256, k=5) → GELU → Linear(256→64) → L2-normalize
- Vector Quantizer: K=64 entries, D=64 dimensions
  - Nearest-neighbor lookup with straight-through estimator
  - EMA codebook updates (γ=0.99)
  - Dead code reinitialization (threshold=2.0)
  - K-means initialization from encoder outputs
  - Commitment loss (β=0.25)
- Decoder: Linear(64→256) → GELU → Linear(256→512)

**Codebook collapse prevention (all used simultaneously):**
1. EMA codebook updates (γ=0.99) — gradient-free codebook learning
2. Dead code reinitialization — reinitialize unused entries from encoder outputs
3. K-means initialization — data-driven starting point
4. L2-normalization — encoder outputs and codebook vectors on the unit hypersphere
5. (Fallback) FSQ — Finite Scalar Quantization achieves 100% utilization by construction

**Why K=64:** Traditional USV taxonomy defines ~10–15 discrete types (Holy & Guo, 2005), but Goffinet et al. (2021) showed USVs form a continuum. K=64 provides headroom for finer subtypes while remaining interpretable.

**Layer selection:** Default: layer 4 of 8 (middle = mid-level concepts). Compare layers 2, 4, 6, 8 by training separate VQ-VAEs and comparing perplexity, utilization, and reconstruction loss.

### Phase 8.4: Analysis & Interpretation Tools (BLOCKED — design ready)

The tools that answer the core research question.

**Codebook visualization:**
- Decode each codebook entry back through the full pipeline to spectrogram space
- Exemplar galleries: N=10 nearest encoder outputs with ±50 frame context
- t-SNE/UMAP projection of codebook vectors colored by mean frequency

**Sequential structure analysis:**
- **Zipf's law:** Rank-frequency plot of code usage. α ≈ 1.0 for natural language. What does the USV codebook show?
- **Transition entropy:** Bigram transition matrix P(c_{t+1}|c_t). High entropy = random; low = predictable.
- **Entropy rate:** H(C_n | C_{n-1},...,C_1) with increasing context (1-gram through 8-gram). Should decrease and plateau for structured sequences.
- **Excess entropy:** Mutual information between past and future halves. Higher = more complex long-range structure.
- **Bigram productivity:** Unique observed bigrams / K² possible. High ratio = combinatorial freedom (compositionality).

**Concept manipulation:**
- Inject individual codebook entries at arbitrary positions, generate predictions autoregressively
- Concept scanning: for fixed context, inject all K entries and compare predictions
- Top-k analysis: track competing concepts at each timestep

### What "Success" Looks Like

| Outcome | Interpretation |
|---------|---------------|
| Zipf α ≈ 1.0 | Code frequency follows power law similar to natural language |
| Entropy rate decreases with context | Sequential predictability — not just random symbol emission |
| Excess entropy > 0 (significantly) | Long-range structure beyond simple Markov transitions |
| Bigram productivity > chance | Combinatorial use of codes (compositionality) |
| Codebook entries decode to recognizable USV features | VQ-VAE learned acoustically meaningful categories |
| Wild vs. lab code distributions differ | Domestication affected vocal repertoire structure |

### Open Questions and Risks

1. **Codebook collapse:** Despite 4 prevention mechanisms, collapse remains the #1 failure mode. FSQ fallback available.
2. **Layer selection:** The "right" layer is unknown. Layer 4 is a hypothesis; layers 2, 6, 8 might reveal different structure.
3. **Zipf exponent interpretation:** α ≈ 1 would be striking, but other values are also informative. α > 1 suggests overclustering; α < 1 suggests underclustering.
4. **HPC access:** The transformer (~25.6M params) requires an A100-class GPU for full training. Code is testable locally, but real results need HPC.
5. **MSE blurriness:** MSE loss averages multimodal futures. If predicted spectrograms are too blurry, a GMM output head (K=5–10 mixture components) is the planned fallback.
6. **Bout quality depends on detection quality:** The transformer only sees what the detection pipeline finds. Missed USVs create gaps in bout structure.

### Key ADRs for This Research Line

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-007 | Transformer-first, VQ-VAE second | Avoids premature discretization |
| ADR-014 | Bout-level data (500 ms gap, ±200 ms padding) | Preserves inter-USV timing context |
| ADR-004 | Recording-level splits | Prevents temporal correlation leakage |
| ADR-001 | 300 kHz sample rate | Nyquist headroom to 150 kHz |
| ADR-002 | n_fft=512, hop=128 | 1.7 ms window, 586 Hz/bin, 0.427 ms hop |

---

## Part 4: The Knowledge System (arscontexta)

### What arscontexta Is

arscontexta is a structured knowledge management system integrated directly into the Claude Code development environment. It serves as the project's **external research memory** — every insight, decision, finding, and open question is captured as an atomic note in a wiki-linked knowledge graph.

This is not a documentation system. It's a **thinking tool**. Notes are claims, not categories. Links are relationships, not navigation. Topic maps are attention managers, not folder structures.

### Why It Exists

When a solo researcher uses an AI assistant with limited context windows, knowledge decays between sessions. Critical insights get rediscovered. Architectural decisions lose their rationale. Research threads fragment.

arscontexta solves this by:
1. **Persisting discoveries** as atomic notes that survive session boundaries
2. **Connecting ideas** through wiki links that surface relationships
3. **Enabling semantic discovery** through MCP-integrated search (qmd)
4. **Automating maintenance** through hooks, skills, and condition-based triggers
5. **Integrating with the development workflow** — reviewer agents consult the vault before making recommendations

### Architecture

**Storage:** Plain markdown files with YAML frontmatter. Human-readable, git-versioned, portable.

**Structure:**
```
notes/          117 atomic notes (claims, not categories)
inbox/          Raw material awaiting processing
ops/            Operational coordination
  goals.md      Active research threads
  reminders.md  Time-bound commitments
  tasks.md      Task stack
  config.yaml   System configuration
  methodology/  How the system works and why
  observations/ Friction signals and surprises
  tensions/     Contradictions to resolve
  sessions/     Session logs
  queue/        Processing queue
templates/      Note templates (single source of truth for schema)
archive/        Processed source material
manual/         User-facing documentation
```

**Key design choices (from the Research preset):**
- **Granularity:** Atomic — one claim per note
- **Organization:** Flat — no nested folders, just topic maps
- **Linking:** Explicit (wiki links) + implicit (semantic search)
- **Processing:** Heavy — full pipeline with verification gates
- **Navigation:** 3-tier — index → topic maps → individual notes
- **Maintenance:** Condition-based, not scheduled
- **Automation:** Full — hooks, skills, automated processing

### The Knowledge Graph

**Current stats (as of 2026-02-20):**

| Metric | Value |
|--------|-------|
| Total atomic notes | 117 |
| Topic maps (MOCs) | 6 |
| Wiki links | 1,011 |
| Average links per note | 8.6 |
| Schema compliance | 100% |
| Orphan notes | 0 |
| Pending observations | 0 |
| Pending tensions | 0 |

**Topic maps (Maps of Content):**

| Topic Map | Focus | ~Notes |
|-----------|-------|--------|
| detection | Energy detection, candidate generation, bout extraction | ~17 |
| classification | CNN pipeline, labeling, training, performance baselines | ~20 |
| representation-learning | VQ-VAE, transformer, codebook discovery, information theory | ~24 |
| signal-processing | STFT, frequency analysis, spectrogram parameters | ~12 |
| experimental-methods | Splits, augmentation, evaluation, research hypotheses | ~28+ |
| index | Root navigation pointing to all topic maps | 5 links |

**Note types in the vault:**

| Type | Example |
|------|---------|
| finding | "CNN baseline of 89.7% precision and 93.8% recall at threshold 0.05 validates the two-stage detection approach" |
| decision | "bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes" |
| method | "per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input" |
| hypothesis | "inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence" |
| baseline | "Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space" |
| open-question | "whether attention patterns in the trained transformer attend beyond the immediately preceding frame" |
| pattern | "two-stage coarse-to-fine filtering is effective for imbalanced detection tasks" |

### The Processing Pipeline

Notes are never written directly to `notes/`. All content flows through a quality-gated pipeline:

```
Source Material → inbox/
    │
    ├── /seed          Queue source for processing
    │
    ├── /reduce        Extract atomic claims from source
    │                  Each claim → one note with full provenance
    │
    ├── /reflect       Find connections between notes
    │                  Update topic maps, add wiki links
    │
    ├── /reweave       Backward pass — update OLD notes with NEW connections
    │                  The connection pass that /reflect doesn't do
    │
    └── /verify        Quality gate — schema + description + health check
```

**Full pipeline in one command:** `/pipeline [file]` runs seed → reduce → reflect → reweave → verify end-to-end.

**Batch processing:** `/ralph N` processes N tasks from the queue, spawning isolated subagents per phase to prevent context contamination.

### The 16 Skills

Skills are vocabulary-transformed commands that operate within the knowledge system:

| Skill | Purpose |
|-------|---------|
| /seed | Add source to processing queue |
| /reduce | Extract structured claims from sources |
| /reflect | Find connections, update topic maps |
| /reweave | Backward pass — update old notes with new connections |
| /verify | Schema + description + health quality gate |
| /validate | Schema validation only |
| /pipeline | End-to-end source processing |
| /ralph | Batch queue processing with fresh context per phase |
| /learn | Research a topic via web search, deposit source with provenance |
| /remember | Capture friction/methodology observations |
| /rethink | Challenge assumptions against accumulated evidence |
| /refactor | Plan vault restructuring from config changes |
| /graph | Interactive knowledge graph analysis |
| /stats | Vault statistics and health metrics |
| /tasks | View and manage task stack and queue |
| /next | Surface the most valuable next action |
| /note-history | Show how a note evolved over time (git-based) |

### Session Continuity

Every session follows the **Orient → Work → Persist** rhythm, enforced by hooks:

**SessionStart hook (`session-orient.ps1`):**
- Reads `ops/goals.md` for active threads
- Checks `ops/reminders.md` for overdue items
- Reads `ops/last-session.md` for context bridge
- Reports vault health (note count, inbox, pending observations/tensions)
- Detects queue thresholds and lifecycle archival needs

**SessionStop hook (`session-capture.ps1`):**
- Writes `ops/last-session.md` with session summary
- Enforces the State Update Rule (goals.md, tracking files, MEMORY.md)
- Archives session data to `ops/sessions/`

**Other hooks:**
- `validate-note.cmd` (PostToolUse:Write) — schema validation on note creation
- `auto-commit.cmd` (PostToolUse:Write) — auto-commit note changes
- `check_agents_tag.cmd` (Stop) — enforces `**Agents:** [list]` tag requirement
- `check_plan_mode.cmd` (PreToolUse) — ensures plan mode for non-trivial tasks

### Operational Learning Loop

The system learns about itself through structured observation:

1. **Observations** (`ops/observations/`): Friction signals, surprises, process gaps. When 10+ accumulate → trigger `/rethink`.
2. **Tensions** (`ops/tensions/`): Contradictions between notes, or between implementation and methodology. When 5+ accumulate → trigger `/rethink`.
3. **/rethink**: Triages evidence, detects patterns, generates proposals for system evolution.

**Example cycle:** The classification topic map grew to 49 notes → observation flagged → `/rethink` proposed splitting → classification split into `classification` (CNN operations, ~20 notes) and `representation-learning` (VQ-VAE research, ~24 notes).

### Semantic Search Integration

**qmd v1.0.6** provides MCP-integrated semantic search over the vault:

- **keyword search** (~30ms): Exact phrase matching
- **vector search** (~2s): Meaning-based discovery (finds concepts even with different vocabulary)
- **deep search** (~10s): Auto-expands query into variations, searches each way, reranks

**Infrastructure:** Vulkan GPU acceleration (AMD Radeon RX 5700). Config in `.mcp.json`. After adding notes: `qmd update && qmd embed`.

**Two discovery layers:**
- Wiki links = curated, intentional connections (1,011 total)
- Semantic search = emergent, content-based discovery (104 documents indexed)

### How arscontexta Integrates with USV Research

The knowledge system is not separate from the research — it *is* the research memory:

1. **Reviewer agents consult the vault:** The master-reviewer, dsp-reviewer, detection-validator, and pr-reviewer all read relevant topic maps and grep `notes/` before issuing recommendations.
2. **Research provenance is preserved:** Every claim traces back through `/reduce` to its source. Every source is archived with metadata.
3. **Open questions surface automatically:** Notes typed as `open-question` appear in topic maps, visible to every session's orient phase.
4. **Literature findings inform design:** Notes from `/learn` sessions (VQ-VAE in bioacoustics, FSQ, SSL transfer learning) directly shaped ADR-007 and the v2 architecture.

---

## Part 5: Infrastructure & Tooling

### Tech Stack

| Layer | Technology | Version/Notes |
|-------|-----------|---------------|
| Language | Python | 3.x with type hints |
| Deep learning | PyTorch | >= 2.0 (CNN + transformer) |
| Audio processing | librosa | STFT, spectrogram computation |
| Signal processing | scipy | Additional DSP utilities |
| Desktop app | PyQt6 | Detection app with spectrogram visualization |
| Web app | Streamlit | Parameter Lab, labeling tool |
| Clustering | scikit-learn | k-means, HDBSCAN, GMM |
| Visualization | matplotlib, seaborn | Spectrograms, training curves |
| Dimensionality reduction | umap-learn | UMAP projections |
| Config management | PyYAML | `default_config.yaml` |
| AI assistant | Claude Code (Opus 4.6) | Primary development partner |
| Knowledge system | arscontexta | Plugin for Claude Code |
| Semantic search | qmd v1.0.6 | MCP server, Vulkan GPU |

### Test Suite

**Total: 351+ tests, all passing** (as of the most recent full suite run)

| Test File | Count | Covers |
|-----------|-------|--------|
| test_energy_detector.py | 42 | Energy detection, thresholds, merging |
| test_dataset_splits.py | 30 | Recording-based splitting, leakage prevention |
| test_dataset_quality.py | 41 | 7 quality checks, metadata extraction |
| test_cnn_model.py | 38 | CNN architecture, training, evaluation |
| test_bout_extractor.py | 15 | Bout extraction, gap threshold, padding |
| test_spectrogram.py | 12 | STFT computation, frequency cropping |
| test_normalization.py | 8 | Per-frequency normalization, Welford's algorithm |
| test_dataset.py | 21 | PyTorch dataset, bucketed batching, augmentation |
| test_transformer.py | 11 | Transformer forward pass, causal mask, param count |
| usv_language (v1) | 63 | Legacy VQ-VAE (to be archived) |
| Others | ~70+ | Spectrogram extraction, labeling, app components |

**Running tests:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v           # Main suite
.\.venv\Scripts\python.exe -m pytest usv_language/ -v     # Language model suite
.\.venv\Scripts\python.exe -m py_compile <file.py>        # Syntax check
```

### Claude Code Configuration

**Custom agents (6):**

| Agent | Purpose |
|-------|---------|
| master-reviewer | Checks implementations against ROADMAP, DECISIONS.md, established patterns |
| dsp-reviewer | Reviews DSP/signal processing code for mathematical correctness |
| detection-validator | Validates detection algorithm changes against baselines |
| pr-reviewer | Final quality review before commit/PR |
| streamlit-expert | Implements and reviews Streamlit UI |
| test-architect | Writes failing tests from ROADMAP specs BEFORE implementation |
| test-hardener | Adversarial coverage hardening AFTER implementation |
| test-writer | *(Deprecated)* — redirects to test-architect / test-hardener |

All reviewer agents have knowledge graph awareness — they read relevant topic maps and grep `notes/` for findings before making recommendations.

**Custom commands (7):** /verify, /verify-quick, /commit-push-pr, /run-app, /simplify, /web-handoff, /implement, /review-all

**Hooks (6 active):**

| Hook | Trigger | Purpose |
|------|---------|---------|
| session-orient.ps1 | SessionStart | Orient: goals, reminders, last session, vault health |
| session-capture.ps1 | Stop | Persist: session summary, State Update Rule |
| check_agents_tag.cmd | Stop | Enforce `**Agents:** [list]` tag |
| check_plan_mode.cmd | PreToolUse | Require plan mode for non-trivial tasks |
| validate-note.cmd | PostToolUse:Write | Schema validation on note creation |
| auto-commit.cmd | PostToolUse:Write | Auto-commit vault changes |

### Entry Points

| Script | Purpose | Command |
|--------|---------|---------|
| scripts/run_detection.py | CLI energy detection | `.venv/Scripts/python.exe scripts/run_detection.py` |
| scripts/extract_spectrograms.py | Spectrogram extraction | `.venv/Scripts/python.exe scripts/extract_spectrograms.py` |
| scripts/usv_labeling_tool.py | Streamlit labeling | `.venv/Scripts/python.exe scripts/usv_labeling_tool.py` |
| scripts/run_app.py | PyQt6 desktop app | `.venv/Scripts/python.exe scripts/run_app.py` |
| scripts/train_cnn.py | CNN training | `.venv/Scripts/python.exe scripts/train_cnn.py` |
| scripts/evaluate_experiment.py | Model evaluation | `.venv/Scripts/python.exe scripts/evaluate_experiment.py` |
| scripts/optimize_threshold.py | Threshold sweep | `.venv/Scripts/python.exe scripts/optimize_threshold.py` |
| scripts/prepare_dataset.py | Dataset assembly | `.venv/Scripts/python.exe scripts/prepare_dataset.py` |
| usv_language/data/prepare_data.py | Bout pipeline | `.venv/Scripts/python.exe usv_language/data/prepare_data.py` |
| usv_language/training/train_transformer.py | Transformer training | `.venv/Scripts/python.exe usv_language/training/train_transformer.py` |
| usv_language/training/extract_hidden_states.py | Hidden state extraction | `.venv/Scripts/python.exe usv_language/training/extract_hidden_states.py` |

### Project Directory Structure

```
mickey_london_lab/
├── src/usv_spectrogram/           # Core library
│   ├── config.py                  # SpectrogramConfig
│   ├── io_wav.py                  # WAV loading
│   ├── spectrogram.py             # STFT computation
│   ├── detection/                 # Detection pipeline
│   │   ├── config.py              # DetectionConfig
│   │   ├── candidate.py           # Candidate dataclass
│   │   ├── energy_detector.py     # EnergyDetector
│   │   ├── extraction_config.py   # ExtractionConfig
│   │   └── spectrogram_extractor.py
│   ├── models/                    # CNN classifier
│   │   ├── cnn_classifier.py      # USVClassifier (~101K params)
│   │   ├── config.py              # TrainingConfig
│   │   ├── trainer.py             # USVTrainer
│   │   ├── data_loader.py         # USVDataset
│   │   └── evaluate.py            # Evaluation metrics
│   ├── dataset/                   # Dataset preparation
│   │   ├── splits.py              # Recording-based splitting
│   │   ├── quality_checks.py      # 7 pre-training checks
│   │   └── metadata.py            # Metadata extraction
│   ├── clustering/                # Clustering exploration
│   ├── app/                       # PyQt6 desktop app
│   │   ├── main_window.py         # Main orchestration
│   │   ├── widgets/               # UI components
│   │   └── core/                  # Business logic
│   ├── labeling/                  # Streamlit labeling tool
│   └── param_lab/                 # Streamlit parameter lab
├── usv_language/                  # Transformer + VQ-VAE research
│   ├── configs/                   # YAML configuration
│   ├── data/                      # Bout pipeline (Phase 8.1)
│   ├── models/                    # Transformer (Phase 8.2), VQ-VAE (8.3)
│   ├── training/                  # Training + extraction scripts
│   └── analysis/                  # Interpretation tools (Phase 8.4)
├── tests/                         # Main test suite
├── scripts/                       # CLI entry points
├── notes/                         # arscontexta knowledge vault
├── ops/                           # Operational coordination
├── templates/                     # Note templates
├── docs/                          # Documentation
│   ├── plans/                     # Implementation plans
│   ├── modules/                   # Module documentation
│   ├── architecture/              # Established patterns
│   ├── reviews/                   # Handoffs and reviews
│   ├── workflow/                  # Process documentation
│   └── reference/                 # Signal processing reference
├── CLAUDE.md                      # Claude Code operating instructions
├── DECISIONS.md                   # 14 ADRs
├── ROADMAP.md                     # Master implementation plan
├── IMPLEMENTATION_PROGRESS.md     # Chronological progress log
└── 5970 USV/                      # WAV files (~6,500 recordings)
```

---

## Part 6: Known Gaps & Improvement Opportunities

### Infrastructure Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| **No CI/CD pipeline** | Tests only run manually; no automated regression detection | Not started |
| **No requirements.txt / pyproject.toml** | Dependencies not formally tracked; reproducibility risk | Partial (some manual tracking) |
| **No model versioning** | Models saved as .pt files without systematic versioning or registry | Not started |
| **No experiment tracking** | Training runs not logged to W&B, MLflow, or similar | Not started |
| **No Docker/container setup** | Environment not portable; relies on manual .venv setup | Not started |
| **AMD GPU only** | Vulkan-patched qmd; no CUDA path tested for training | Hardware constraint |

### Research Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| **HPC access needed** | Transformer training (~25.6M params) requires A100-class GPU | Blocked |
| **Batch detection not built** | Cannot process ~6,500 WAVs headlessly; bout extraction blocked | Phase 13 (designed) |
| **No population metadata** | Wild vs. lab comparison requires population labels per recording | Unknown availability |
| **Small labeled dataset** | ~840 labels; scaling plan defines 5 milestones to 30K | Active labeling ongoing |
| **v1 code not archived** | 63 tests in legacy `usv_language/` code that should be archived before v2 proceeds | Noted, not done |

### Knowledge System Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| **qmd index slightly stale** | 104/117 notes indexed; 13 notes added after last `qmd update && qmd embed` | Needs re-sync |
| **Biological-context topic map deferred** | ~8-10 biology notes exist without a dedicated map; below split threshold | Deferred (Phase 3.3) |
| **Split ratio inconsistency** | DECISIONS.md says 80/10/10; ROADMAP Phase 9 says 70/15/15 | Needs resolution |
| **Vulkan patch fragile** | qmd's `llm.js:253` must be re-patched after updates | Manual maintenance |

### Workflow Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| **No automated training pipelines** | Each training cycle requires manual script chaining | Phase 9-10 (designed) |
| **Manual model management** | No systematic way to compare models across training cycles | Not started |
| **No batch labeling workflow** | Desktop app processes one file at a time | Phase 13 designed |
| **Hook errors on Windows** | SessionStart/Stop hooks show cosmetic errors (upstream bug #12671) | Known, cosmetic only |

### What's Ready to Build vs. Blocked vs. Needs Research

| Category | Items |
|----------|-------|
| **Ready to build (no blockers)** | Phase 9 (Dataset Assembly), Phase 10 (Active Learning Cycle Runner), Phase 8.3 (VQ-VAE code — testable on dummy data), Phase 8.4 (Analysis tools — testable on synthetic data), Phase 13 (Batch Detection) |
| **Blocked on infrastructure** | Phase 11.2-11.4 (Transformer training + VQ-VAE + analysis on real data — needs HPC), Phase 11.1 (Bout preprocessing — needs batch detection results) |
| **Needs more data/research** | Phase 12 (Population comparison — needs population metadata + sufficient detection results) |

---

## Part 7: Discussion Prompts for Web Claude

These questions are designed to spark strategic thinking across three dimensions: research strategy, engineering quality, and workflow/productivity.

### Research Strategy

1. **Architecture alternatives:**
   "Given the two-phase transformer→VQ-VAE architecture (ADR-007), where the transformer learns freely and VQ-VAE discretizes post-hoc — what other analysis frameworks could complement the Zipf/entropy/excess-entropy approach for testing language-like structure? Specifically: are there information-theoretic measures beyond what's planned (Zipf's law, transition entropy, entropy rate, excess entropy, bigram productivity) that could distinguish language-like structure from simpler generative processes like Markov chains or renewal processes?"

2. **Research novelty positioning:**
   "No published work has applied end-to-end VQ-VAE to animal vocalizations as of February 2026 (confirmed by systematic literature search). The closest work is Sarkar & Magimai-Doss 2025, who applied post-hoc VQ to frozen HuBERT embeddings for marmoset/dog vocalizations — but their approach dramatically underperformed continuous representations (35% vs 49% UAR). How should the project position its v2 architecture relative to these findings? Is the transformer-first approach actually closer to the post-hoc strategy that underperformed, or is the key difference that our transformer is trained on the target domain?"

3. **Wild vs. lab comparison:**
   "The underlying biological hypothesis is that captive breeding degraded courtship vocal competence in lab mice (through inbreeding and absence of sexual selection pressure). With ~6,500 WAV files and F1 91.7% detection, what's the strongest experimental design for testing this hypothesis? Should we prioritize within-recording sequence analysis (VQ-VAE codes) or between-population repertoire comparison (CNN clustering), or are they complementary?"

4. **FSQ vs. VQ-VAE decision:**
   "FSQ (Finite Scalar Quantization, ICLR 2024) eliminates codebook collapse by construction. VQ-VAE with 4 collapse prevention mechanisms is more complex but more established. Given that codebook collapse is the #1 risk for Phase 8.3, should the project implement FSQ as the primary approach rather than as a fallback? What are the interpretability tradeoffs?"

### Engineering Quality

5. **Most impactful next investment:**
   "With ~840 labels and F1 91.7%, what's the most impactful next investment: more labels (the scaling plan targets 30K), better models (e.g., moving from 101K to 400K params), or better infrastructure (CI/CD, experiment tracking, batch processing)? The scaling roadmap defines 5 milestones from 2K to 30K labels — is this the right granularity, or should the project focus on infrastructure automation first?"

6. **CI/CD for solo research:**
   "What CI/CD and reproducibility practices would most benefit a solo-researcher ML project with 351+ tests, no formal dependency management, and no experiment tracking? The project uses Claude Code as the primary development tool with extensive hooks and custom agents — how does this change the usual CI/CD recommendations?"

7. **Test strategy evolution:**
   "The test suite has 351+ tests covering energy detection, dataset preparation, CNN training, and the transformer/VQ-VAE pipeline. But there's no integration testing between phases (e.g., end-to-end from WAV to detection to bout extraction to transformer input). What integration test strategy would provide the most value as the pipeline lengthens?"

### Workflow & Productivity

8. **Knowledge system evolution:**
   "The arscontexta knowledge system has 117 notes with 8.6 links/note and 100% schema compliance. It uses a flat organization with 6 topic maps and condition-based maintenance. What patterns from knowledge management research suggest the next evolution? Should the system grow more topic maps, increase link density, add note types, or focus on cross-domain integration?"

9. **Knowledge-ML integration:**
   "How could the arscontexta knowledge system feed back into the ML pipeline? For example: using research notes to guide hyperparameter choices, using literature findings to constrain architecture search, or using the knowledge graph structure itself as metadata for experiment tracking. What concrete integration points would create the most value?"

10. **Session optimization:**
    "Each Claude Code session follows Orient→Work→Persist with automated hooks. The knowledge system captures friction observations, tensions, and methodology learnings. What additional instrumentation or feedback loops would improve session-over-session productivity? Are there patterns from software engineering retrospectives or research lab management that apply?"

### Cross-Cutting

11. **Priority call:**
    "Looking at the full landscape — 7 completed phases, 2 code-complete phases, active labeling, a blocked HPC dependency, a knowledge system with 117 notes, and an unresolved split-ratio inconsistency — if you could only recommend THREE actions for the next month of work, what would they be and why? Optimize for research impact, not engineering completeness."

12. **Risk assessment:**
    "What are the top 3 risks to the overall research program, considering both technical risks (codebook collapse, MSE blurriness, insufficient labels) and operational risks (HPC access, solo-researcher bus factor, knowledge system maintenance overhead)? For each risk, what's the mitigation strategy?"

---

## Appendix A: Architecture Decision Records Summary

| ADR | Title | Status |
|-----|-------|--------|
| ADR-001 | Sample Rate — 300 kHz | Accepted |
| ADR-002 | STFT Parameters (n_fft=512, hop=128) | Accepted |
| ADR-003 | Detection Thresholds (-60 dB, peak mode) | Accepted |
| ADR-004 | Dataset Splitting — By Recording | Accepted |
| ADR-005 | Class Weighting — 3.0× USV Boost | Accepted |
| ADR-006 | CNN Architecture — 3 Conv Blocks + GlobalAvgPool | Accepted |
| ADR-007 | Transformer + VQ-VAE Two-Phase Architecture (v2) | Accepted (supersedes v1) |
| ADR-008 | Negative Sample Strategy — 3-Source Mix | Accepted |
| ADR-009 | Model Artifacts — PyTorch .pt Files | Accepted |
| ADR-010 | Label Storage Format — JSON | Accepted |
| ADR-011 | Auto Sample Rate — Read from WAV | Accepted |
| ADR-012 | Energy Detection Mode — Peak (Not Mean) | Accepted |
| ADR-013 | Segment Continuity — Enabled by Default | Accepted |
| ADR-014 | Bout-Level Data for Transformer Training | Accepted |

## Appendix B: Timeline

| Date | Milestone |
|------|-----------|
| 2026-01-16 | Project started, Phase 1 energy detection |
| 2026-02-06 | Scaling plan initiated (boundary adjustment, progressive labeling) |
| 2026-02-07 | Phases 2-3 complete (progressive labeling, constrained jittering) |
| 2026-02-08 | Auto-move reviewed files feature |
| 2026-02-09 | VQ-VAE v1 complete (63 tests, ~437K params) |
| 2026-02-14 | Bug fixes (duration filter, label persistence), new tests |
| 2026-02-16 | Workflow migration (4 sessions): DECISIONS.md, CLAUDE.md, ROADMAP.md, patterns |
| 2026-02-18 | arscontexta vault generated, Phase 8.1 bout pipeline complete (351 tests) |
| 2026-02-18 | VQ-VAE architecture redesign (v1 → v2), ADR-007 updated, ADR-014 added |
| 2026-02-18 | Knowledge graph Phase 3: 100+ notes extracted from docs + brain dumps |
| 2026-02-19 | /reflect passes, /learn VQ-VAE bioacoustics, topic map split, reviewer integration |
| 2026-02-20 | Phase 8.2 transformer code complete (25.6M params, 11 tests) |
| 2026-02-20 | Session continuity hooks, weekly maintenance routine established |
| 2026-02-20 | Vault baseline: 117 notes, 1011 links, 100% compliance, 0 orphans |

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **USV** | Ultrasonic vocalization — high-frequency (20-120 kHz) calls produced by mice |
| **Bout** | Continuous segment of recording containing a cluster of USV activity with context padding |
| **STFT** | Short-Time Fourier Transform — converts audio into time-frequency representation |
| **VQ-VAE** | Vector Quantized Variational Autoencoder — learns a discrete codebook from continuous data |
| **Codebook** | Set of K learned prototype vectors; each input maps to its nearest prototype |
| **Codebook collapse** | Failure mode where only a few codebook entries are used, wasting capacity |
| **FSQ** | Finite Scalar Quantization — alternative to VQ-VAE that eliminates collapse by design |
| **Zipf's law** | Statistical law: frequency ∝ rank^(-α). Natural language has α ≈ 1.0 |
| **Entropy rate** | Rate at which new information appears in a sequence; decreases for structured sequences |
| **Excess entropy** | Mutual information between past and future halves of a sequence; measures long-range structure |
| **ADR** | Architecture Decision Record — documents a technical decision with context and rationale |
| **MOC** | Map of Content — a topic map that curates links to related notes |
| **arscontexta** | The knowledge management system (Claude Code plugin) used for research memory |
| **qmd** | Local semantic search engine over markdown documents (MCP server) |
| **LMT** | Live Mouse Tracker — behavioral tracking system from Institut Pasteur |
| **EMA** | Exponential Moving Average — used for gradient-free codebook updates in VQ-VAE |
| **Hysteresis thresholding** | Two-threshold detection: high to start, low to extend — reduces flickering |
