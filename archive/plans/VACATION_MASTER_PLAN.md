# Vacation Master Plan — Mickey London Lab

**Created:** 2026-02-20
**Purpose:** Structured plan for autonomous + human-guided work while Shachar is on vacation
**Scope:** 4 workstreams, organizational structure, Ralph/arscontexta compatibility

---

## Scope Assessment: Is This Too Much?

**Honest answer: Yes, for a single Ralph loop. No, if structured correctly.**

The key insight is that these 4 workstreams have different autonomy profiles:

| Workstream | Can Ralph Do It? | Why/Why Not |
|-----------|-----------------|-------------|
| Null model implementations | **YES — ideal** | Pure math, synthetic data, verifiable with tests |
| Probing experiment code | **YES — good** | Clear architecture, testable on dummy data |
| LMT integration | **PARTIALLY** | Needs research + design decisions you should review |
| Ralph/arscontexta compat | **NO — do before leaving** | One-time config change, 30 min of work |

**Recommended sequencing:**

1. **Before vacation (30 min):** Fix Ralph/arscontexta compatibility (Section 1)
2. **Ralph Loop 1:** Null models (Section 2) — ~15-20 iterations
3. **Ralph Loop 2:** Probing experiment scaffolding (Section 3) — ~15-20 iterations
4. **After vacation (human-guided):** LMT integration (Section 4) — needs your input on data formats

Run loops 1 and 2 sequentially (not in parallel) to avoid git conflicts and stay within API rate limits. Total estimated API cost: ~$30-60 depending on model.

---

## Section 0: Organizational Structure

### How This All Fits Together

```
VACATION_MASTER_PLAN.md          ← You are here (this file)
│
├── docs/plans/
│   ├── RALPH_ARSCONTEXTA.md     ← Section 1: compatibility fix (do before leaving)
│   ├── NULL_MODELS_PLAN.md      ← Section 2: Ralph prompt + implementation spec
│   ├── PROBING_PLAN.md          ← Section 3: Ralph prompt + implementation spec
│   └── LMT_INTEGRATION_PLAN.md  ← Section 4: research plan (post-vacation)
│
├── prompts/                      ← Ralph prompt files (created from plans)
│   ├── RALPH_NULL_MODELS.md
│   └── RALPH_PROBING.md
│
├── usv_language/
│   ├── analysis/                 ← Null models + probing code lands here
│   │   ├── null_models.py
│   │   ├── probing.py
│   │   └── information_theory.py
│   └── ...
│
└── src/usv_spectrogram/
    └── lmt/                      ← LMT integration lands here (post-vacation)
        ├── event_loader.py
        ├── synchronizer.py
        └── coupling_analysis.py
```

### Tracking Progress

Each Ralph loop should maintain a progress file:

```
docs/plans/NULL_MODELS_PROGRESS.md    ← Ralph updates this each iteration
docs/plans/PROBING_PROGRESS.md        ← Ralph updates this each iteration
```

Format:
```markdown
# Progress: [Task Name]
## Iteration 1 — [timestamp]
- What was done: ...
- Tests added: ...
- Tests passing: yes/no
- Next: ...

## Iteration 2 — [timestamp]
...
```

### Git Strategy

- **Branch per workstream:** `feature/null-models`, `feature/probing`
- Ralph commits to the branch, never to main
- You review and merge after vacation
- This protects your 351+ passing tests on main

---

## Section 1: Ralph + arscontexta Compatibility

### The Problem

Ralph and arscontexta both use Claude Code hooks, and they will conflict:

| Hook | Trigger | arscontexta Purpose | Ralph Conflict |
|------|---------|--------------------|----|
| `session-orient.ps1` | SessionStart | Load goals, reminders, vault health | **Noise** — fires every bash-loop iteration, wastes context on orientation |
| `session-capture.ps1` | Stop | Write last-session.md, enforce state updates | **BLOCKS RALPH** — Stop hook runs before Ralph's hook, may prevent re-feed |
| `check_agents_tag.cmd` | Stop | Enforce `**Agents:** [list]` tag | **BLOCKS RALPH** — may reject exit code, confuse loop |
| `check_plan_mode.cmd` | PreToolUse | Require plan mode for non-trivial tasks | **SLOWS RALPH** — forces plan mode on every tool call |
| `validate-note.cmd` | PostToolUse:Write | Schema validation on note writes | **Safe** — Ralph won't write notes |
| `auto-commit.cmd` | PostToolUse:Write | Auto-commit vault changes | **Safe** — Ralph won't modify vault |

### The Solution: Ralph-Mode Hook Configuration

**Option A: Separate `.claude/` config for Ralph (recommended)**

Create a `.claude/settings.ralph.json` that disables conflicting hooks. Before starting Ralph:

```powershell
# Save current settings
Copy-Item .claude/settings.json .claude/settings.backup.json

# Swap to Ralph-compatible settings
Copy-Item .claude/settings.ralph.json .claude/settings.json
```

After vacation, restore:
```powershell
Copy-Item .claude/settings.backup.json .claude/settings.json
```

**What `.claude/settings.ralph.json` should contain:**
- KEEP: `validate-note.cmd`, `auto-commit.cmd` (won't fire, harmless)
- REMOVE: `session-orient.ps1`, `session-capture.ps1`, `check_agents_tag.cmd`, `check_plan_mode.cmd`
- ADD: Ralph's stop hook

**Option B: Environment variable guard in hooks**

Add to the top of each conflicting hook:

```powershell
# session-orient.ps1
if ($env:RALPH_MODE -eq "1") { exit 0 }
# ... rest of hook
```

```cmd
:: check_agents_tag.cmd
if "%RALPH_MODE%"=="1" exit /b 0
:: ... rest of hook
```

Then launch Ralph with: `$env:RALPH_MODE = "1"` before starting the loop.

**Option B is cleaner** because it doesn't require swapping files, and arscontexta hooks remain installed for normal sessions.

### What About arscontexta Knowledge During Ralph?

Ralph loops should be **read-only** with respect to the knowledge vault:

- Ralph CAN and SHOULD read `notes/`, `DECISIONS.md`, `ROADMAP.md` for context
- Ralph should NOT run `/seed`, `/reduce`, `/reflect`, `/pipeline`, or any vault-write commands
- Ralph prompts should include: "Do NOT modify anything under `notes/`, `ops/`, or `templates/`. These are read-only."

Any discoveries or decisions made during Ralph should be captured in the progress file, and you can feed them into arscontexta after vacation via `/seed` + `/pipeline`.

### Implementation: Give This to Claude Code

```
# Task: Set up Ralph/arscontexta compatibility

1. Read this plan: docs/plans/RALPH_ARSCONTEXTA.md
2. Add RALPH_MODE environment variable guards to:
   - .claude/hooks/session-orient.ps1 (exit 0 if RALPH_MODE=1)
   - .claude/hooks/session-capture.ps1 (exit 0 if RALPH_MODE=1)
   - .claude/hooks/check_agents_tag.cmd (exit /b 0 if RALPH_MODE=1)
   - .claude/hooks/check_plan_mode.cmd (exit /b 0 if RALPH_MODE=1)
3. Install Ralph plugin: /plugin install ralph-wiggum@claude-plugins-official
4. Create prompts/ directory
5. Test: Set RALPH_MODE=1, start Claude Code, verify hooks don't fire
6. Test: Unset RALPH_MODE, start Claude Code, verify hooks DO fire
```

---

## Section 2: Null Models for Information-Theoretic Analysis

### Why This Matters

Your analysis plan (Phase 8.4) will measure Zipf exponent, entropy rate, excess entropy, and bigram productivity on VQ-VAE codebook sequences. But these numbers are meaningless without null models. If mouse USV codes show α ≈ 0.9, is that language-like or expected from any structured process? You need baselines to compare against.

### What to Build

#### 2.1 Core Information Theory Module (`usv_language/analysis/information_theory.py`)

Functions that operate on integer sequences (codebook indices):

```python
def zipf_exponent(sequence: list[int]) -> ZipfResult:
    """Rank-frequency analysis. Fit power law via MLE (Clauset et al. 2009).
    Returns: alpha, xmin, p_value (KS goodness-of-fit)"""

def entropy_rate(sequence: list[int], max_order: int = 8) -> list[float]:
    """H(X_n | X_{n-1},...,X_{n-k}) for k=0..max_order.
    Uses plugin estimator with Miller-Madow correction for small samples."""

def excess_entropy(sequence: list[int], max_half_length: int = 50) -> float:
    """I(past; future) = H(past) + H(future) - H(past, future).
    Estimate via block entropy extrapolation (Crutchfield & Feldman 2003)."""

def bigram_productivity(sequence: list[int], K: int) -> float:
    """Unique observed bigrams / K^2 possible. Bootstrap CI."""

def transition_matrix(sequence: list[int], K: int) -> np.ndarray:
    """K×K bigram transition probability matrix."""

def mutual_information_rate(sequence: list[int], max_lag: int = 20) -> list[float]:
    """I(X_t; X_{t+lag}) for lag=1..max_lag. Detects long-range dependencies."""
```

#### 2.2 Null Model Generators (`usv_language/analysis/null_models.py`)

Each generator produces sequences with known statistical properties:

```python
class NullModelGenerator:
    """Generate null model sequences for comparison with real codebook sequences."""

    def shuffled(self, sequence: list[int], n_surrogates: int = 100) -> list[list[int]]:
        """Random permutation. Preserves marginal frequencies, destroys all structure.
        Expected: Zipf same, entropy rate flat (= marginal entropy), excess entropy ≈ 0."""

    def markov_order_k(self, sequence: list[int], k: int = 1, n_surrogates: int = 100) -> list[list[int]]:
        """Fit k-th order Markov model to data, generate surrogates.
        Expected: Entropy rate = conditional entropy at order k, no long-range structure."""

    def renewal_process(self, sequence: list[int], n_surrogates: int = 100) -> list[list[int]]:
        """Fit inter-event interval distribution, generate surrogates.
        Preserves timing statistics but not sequential dependencies.
        Models the hypothesis: 'USVs are just randomly timed with fixed vocabulary.'"""

    def hmm_surrogate(self, sequence: list[int], n_states: int = 8, n_surrogates: int = 100) -> list[list[int]]:
        """Fit Hidden Markov Model, generate surrogates.
        Models the hypothesis: 'USV sequences are driven by hidden behavioral states,
        not by compositional rules.'"""

    def phase_randomized(self, sequence: list[int], n_surrogates: int = 100) -> list[list[int]]:
        """Randomize phase of Fourier transform of symbol sequence.
        Preserves autocorrelation/power spectrum, destroys higher-order structure.
        Tests: 'Is the structure just pairwise temporal correlation?'"""
```

#### 2.3 Statistical Comparison Framework (`usv_language/analysis/statistical_tests.py`)

```python
class NullModelComparison:
    """Compare real codebook sequences against null model baselines."""

    def compare(self, real_sequence: list[int], null_sequences: list[list[int]],
                metrics: list[str]) -> ComparisonResult:
        """For each metric, compute on real data + all surrogates.
        Returns z-scores, p-values (rank-based), effect sizes."""

    def full_analysis(self, real_sequence: list[int], K: int) -> FullAnalysisResult:
        """Run all null models × all metrics. Produce summary table.
        This is the main entry point for the language hypothesis test."""
```

#### 2.4 Analytical Validation Tests

Every null model has known analytical properties. Tests should verify:

| Null Model | Test | Expected |
|-----------|------|----------|
| Shuffled (uniform K=64) | Entropy rate | = log2(64) = 6.0 bits at all orders |
| Shuffled (non-uniform) | Zipf exponent | Same as input |
| Markov-1 | Entropy rate at order 1 | = conditional entropy of transition matrix |
| Markov-1 | Entropy rate at order 2+ | Same as order 1 (no additional info) |
| Renewal | Excess entropy | ≈ 0 (no past-future MI beyond timing) |
| Phase-randomized | Autocorrelation | Same as input |

These tests are **analytically verifiable** — perfect for Ralph because success/failure is unambiguous.

### Ralph Prompt for This Workstream

See `prompts/RALPH_NULL_MODELS.md` (to be created from this spec).

Key elements:
- Read `DECISIONS.md` and `usv_language/` structure first
- Implement in `usv_language/analysis/` following existing patterns
- Tests in `usv_language/tests/test_analysis/`
- Run `pytest usv_language/ -v` after every change
- Do NOT modify anything outside `usv_language/analysis/` and its tests
- Use numpy, scipy only (already in environment)
- For powerlaw fitting: implement directly (Clauset et al. MLE) — don't add new pip dependencies without checking
- Commit to `feature/null-models` branch
- Update `docs/plans/NULL_MODELS_PROGRESS.md` each iteration

**Estimated iterations:** 15-20
**Estimated API cost:** ~$15-30

---

## Section 3: Transformer Probing Experiments

### Why This Matters

Once the transformer (Phase 8.2) is trained on real data (needs HPC), you'll want to immediately understand what it learned — before investing in VQ-VAE training. Probing classifiers are cheap, fast, and informative. But the **code infrastructure** for probing can be built now and tested on synthetic data.

The idea: train simple linear classifiers on frozen hidden states to predict known acoustic properties. If layer 4 can predict frequency contour but layer 2 cannot, that tells you where abstract representations emerge.

### What to Build

#### 3.1 Probing Framework (`usv_language/analysis/probing.py`)

```python
class ProbingExperiment:
    """Train lightweight probes on frozen transformer hidden states."""

    def __init__(self, hidden_states_path: str, metadata_path: str):
        """Load memory-mapped hidden states + metadata JSON from extract_hidden_states.py output."""

    def extract_features(self, layer: int, pooling: str = "mean") -> np.ndarray:
        """Extract features from a specific layer.
        pooling: 'mean' (average over time), 'max', 'first', 'last'
        Returns: (n_samples, d_model) array"""

    def train_probe(self, features: np.ndarray, labels: np.ndarray,
                    probe_type: str = "linear") -> ProbeResult:
        """Train a probe classifier/regressor.
        probe_type: 'linear' (LogisticRegression/Ridge), 'mlp_1layer' (1 hidden layer)
        Returns: accuracy/R², selectivity (probe acc - majority baseline)"""

    def layer_comparison(self, labels: np.ndarray, label_name: str,
                         layers: list[int] = [2, 4, 6, 8]) -> LayerComparisonResult:
        """Train probes across multiple layers for the same target.
        Returns: per-layer results + plot data"""
```

#### 3.2 Acoustic Property Extractors (`usv_language/analysis/acoustic_properties.py`)

These compute ground-truth labels from the raw spectrogram data for probing:

```python
class AcousticPropertyExtractor:
    """Extract acoustic properties from spectrogram frames for probing targets."""

    def peak_frequency(self, spectrogram_column: np.ndarray) -> float:
        """Dominant frequency bin (argmax of energy). Continuous target."""

    def spectral_centroid(self, spectrogram_column: np.ndarray) -> float:
        """Energy-weighted mean frequency. Continuous target."""

    def energy(self, spectrogram_column: np.ndarray) -> float:
        """Total energy in frame. Continuous target."""

    def is_voiced(self, spectrogram_column: np.ndarray, threshold: float) -> bool:
        """Binary: above energy threshold = USV present. Classification target."""

    def frequency_direction(self, col_prev: np.ndarray, col_curr: np.ndarray) -> str:
        """'rising', 'falling', 'flat'. Classification target."""

    def bout_position(self, frame_index: int, bout_length: int) -> float:
        """Normalized position within bout [0, 1]. Continuous target."""

    def time_since_last_usv(self, frame_index: int, usv_onsets: list[int]) -> float:
        """Temporal distance to preceding USV onset. Continuous target."""
```

#### 3.3 Probing Analysis Pipeline

```python
class ProbingAnalysisPipeline:
    """Run the full probing analysis across layers and properties."""

    TARGETS = {
        # Classification targets
        "is_voiced": {"type": "classification", "extractor": "is_voiced"},
        "frequency_direction": {"type": "classification", "extractor": "frequency_direction"},

        # Regression targets
        "peak_frequency": {"type": "regression", "extractor": "peak_frequency"},
        "spectral_centroid": {"type": "regression", "extractor": "spectral_centroid"},
        "energy": {"type": "regression", "extractor": "energy"},
        "bout_position": {"type": "regression", "extractor": "bout_position"},
        "time_since_last_usv": {"type": "regression", "extractor": "time_since_last_usv"},
    }

    def run_full_analysis(self, layers: list[int] = [2, 4, 6, 8]) -> FullProbingResult:
        """For each target × each layer: train probe, evaluate, compare.
        Key output: heatmap of (layer × property) showing where information lives.
        This directly guides VQ-VAE layer selection."""
```

#### 3.4 Synthetic Data Tests

Since no real hidden states exist yet, all tests should use synthetic data:

```python
# Test: probing a layer that perfectly encodes frequency → should get R² ≈ 1.0
# Test: probing a random layer → should get R² ≈ 0 (selectivity ≈ 0)
# Test: increasing probe complexity shouldn't help if info isn't there
# Test: layer_comparison returns monotonically for synthetic "deeper = better" scenario
```

### Ralph Prompt for This Workstream

Key elements:
- Read `usv_language/training/extract_hidden_states.py` to understand the hidden state format
- Read `usv_language/models/transformer.py` for architecture details
- Implement in `usv_language/analysis/`
- Use sklearn for probes (LogisticRegression, Ridge, MLPClassifier)
- Tests on synthetic hidden states (random + structured)
- Commit to `feature/probing` branch
- Update `docs/plans/PROBING_PROGRESS.md` each iteration
- Do NOT modify anything outside `usv_language/analysis/`

**Estimated iterations:** 12-15
**Estimated API cost:** ~$12-25

---

## Section 4: LMT (micecraft) Integration Plan

### Why This Is Different

This workstream should **NOT** be Ralph'd. Here's why:

1. **Data format discovery** — You need to examine actual LMT output files to understand their structure. Ralph can't make judgment calls about ambiguous data formats.
2. **Synchronization design** — Aligning video-frame behavioral annotations with 300 kHz audio timestamps requires careful engineering decisions about interpolation, temporal resolution, and tolerance windows.
3. **Research design** — Deciding which behavioral events to analyze (approach, mount, sniff, chase, etc.) requires domain knowledge about mouse courtship.

However, I can give you a detailed research + implementation plan for when you're back.

### Background: What LMT Provides

LMT (Live Mouse Tracker, Institut Pasteur) produces:
- **SQLite database** with frame-by-frame animal positions, identity tracking, and behavioral event annotations
- **Behavioral events:** approach, contact, oral-oral contact, oral-genital contact, side-by-side, follow, social approach, etc.
- **Temporal resolution:** Video frame rate (typically 30 fps)
- **Animal identity:** Which mouse is doing what

Your audio recordings are synchronized with LMT sessions (same experimental sessions), meaning each WAV file corresponds to a specific time window of LMT behavioral data.

### What the Integration Enables

Three increasingly ambitious analyses, each publishable:

#### Analysis 1: Event-Triggered Vocal Analysis (simplest, highest confidence)

**Question:** Do mice vocalize more during specific behavioral events?

**Method:**
- Extract behavioral event timestamps from LMT
- For each event type, compute USV rate (detections/second) in a ±2s window
- Compare to baseline USV rate
- Statistical test: permutation test (shuffle event times, recompute rates)

**What this tells you:** Which behaviors co-occur with vocalization. This is the "sanity check" — if USVs don't correlate with social behavior at all, something is wrong with your data alignment.

**Code structure:**
```
src/usv_spectrogram/lmt/
├── event_loader.py       # Parse LMT SQLite → behavioral events with timestamps
├── synchronizer.py       # Align LMT timestamps with WAV file timestamps
├── event_triggered.py    # Peri-event time histograms for USV rate
└── tests/
```

#### Analysis 2: Behavioral State → Vocal Repertoire (medium complexity)

**Question:** Does the type of USV change with behavioral context?

**Method:**
- Assign each USV a behavioral context (what the mouse was doing when it vocalized)
- Use CNN features (or later VQ-VAE codes) as USV representations
- Test whether USV feature distributions differ across behavioral contexts
- Statistical test: MANOVA on CNN features, or clustering + chi-squared on cluster assignments

**What this tells you:** Whether mice use different "words" in different contexts — a prerequisite for language-like communication.

#### Analysis 3: Vocal → Behavioral Prediction (most ambitious)

**Question:** Can you predict what the mouse will do next from its vocal sequence?

**Method:**
- For each USV sequence, predict the next behavioral event (classification)
- Compare to baseline: predict from behavioral history alone
- Measure mutual information: I(vocal_sequence; next_behavior)

**What this tells you:** Whether vocal sequences carry information about behavioral intent — the strongest evidence for communicative function.

### Pre-Vacation Research Tasks

Before implementing any of this, you need to answer:

1. **What format are your LMT files in?** SQLite? CSV export? Which tables/columns?
2. **How are audio and video synchronized?** Same start time? Clock offset? Trigger signal?
3. **Which behavioral events does your LMT setup annotate?** The full LMT event list is long — which are relevant to courtship?
4. **Do you have population labels (wild vs. lab) per recording?** This is needed for Analysis 2-3 to be comparative.

### Suggested Post-Vacation Workflow

```
Day 1: Explore LMT data format
  - Open a few LMT databases, catalog tables and columns
  - Identify timestamp format and synchronization method
  - Pick 2-3 recordings where you know USVs exist

Day 2: Build event_loader.py + synchronizer.py
  - Parse LMT → structured events
  - Align with WAV timestamps
  - Manual verification on known recordings

Day 3: Analysis 1 — Event-triggered vocal analysis
  - Implement peri-event time histograms
  - Run on available data
  - This gives you a figure for the lab meeting / thesis

Day 4+: Analysis 2-3 as time permits
```

### Connection to Null Models (Section 2)

The null model framework directly supports Analysis 3. When you test whether vocal sequences predict behavior, you need null models for "what prediction accuracy would you expect by chance?" The shuffled and Markov null models from Section 2 provide exactly this baseline.

### Connection to Probing (Section 3)

The probing framework from Section 3 can be extended with LMT-derived targets:
- "behavioral_state" as a probing target → does the transformer encode what the mouse is doing?
- "time_to_next_event" as a probing target → does the transformer predict behavioral transitions?

This is where the workstreams converge: probing tells you what the transformer knows, LMT integration tells you what's biologically meaningful, and null models tell you whether it's statistically significant.

---

## Section 5: Putting It All Together — The Vacation Checklist

### Before You Leave (~1-2 hours)

- [ ] **Add RALPH_MODE guards to hooks** (Section 1, Option B)
  - Give Claude Code the task from Section 1
  - Verify: hooks disabled when RALPH_MODE=1, enabled otherwise
- [ ] **Install Ralph plugin** or set up bash loop scripts
- [ ] **Create git branches:** `feature/null-models`, `feature/probing`
- [ ] **Create prompt files** from Sections 2-3 specs
  - `prompts/RALPH_NULL_MODELS.md`
  - `prompts/RALPH_PROBING.md`
- [ ] **Create progress files:**
  - `docs/plans/NULL_MODELS_PROGRESS.md`
  - `docs/plans/PROBING_PROGRESS.md`
- [ ] **Test Ralph:** Run 1-2 iterations of null models, verify it's working correctly
- [ ] **Start Ralph Loop 1:** Null models, max 20 iterations

### While Away

- Ralph Loop 1 runs: null models (several hours)
- When Loop 1 completes → start Ralph Loop 2: probing (several hours)
- Or: set up a sequential script that runs both

### When You Return

- [ ] **Review git diffs** on `feature/null-models` and `feature/probing`
- [ ] **Run full test suite** on each branch: `pytest tests/ -v && pytest usv_language/ -v`
- [ ] **Merge good branches** into main
- [ ] **Feed discoveries into arscontexta:** take progress files → `/seed` → `/pipeline`
- [ ] **Restore normal hooks:** unset RALPH_MODE (or it's already fine since it's per-session)
- [ ] **Begin LMT integration** (Section 4) — start by exploring data format
- [ ] **Resolve split ratio inconsistency** (80/10/10 vs 70/15/15) while you're in decision-making mode

### Sequential Script for Unattended Execution

```powershell
# vacation_ralph.ps1
$env:RALPH_MODE = "1"

# Loop 1: Null Models
Write-Host "Starting null models workstream..."
git checkout -b feature/null-models
# Use bash loop approach for fresh context each iteration
for ($i = 0; $i -lt 20; $i++) {
    Get-Content prompts/RALPH_NULL_MODELS.md | claude --dangerously-skip-permissions
    Write-Host "Null models iteration $($i+1) complete"
    Start-Sleep -Seconds 10
}

# Loop 2: Probing
Write-Host "Starting probing workstream..."
git checkout -b feature/probing
for ($i = 0; $i -lt 15; $i++) {
    Get-Content prompts/RALPH_PROBING.md | claude --dangerously-skip-permissions
    Write-Host "Probing iteration $($i+1) complete"
    Start-Sleep -Seconds 10
}

Write-Host "All vacation work complete. Review branches when back."
```

---

## Section 6: What NOT to Do While Away

To protect the project:

1. **Do NOT Ralph the labeling** — only human eyes can label USVs
2. **Do NOT Ralph arscontexta maintenance** — vault writes need quality gates
3. **Do NOT Ralph LMT integration** — too many design decisions
4. **Do NOT let Ralph modify `src/usv_spectrogram/`** — that code is stable, 351+ tests
5. **Do NOT let Ralph modify `notes/`, `ops/`, `templates/`** — knowledge vault is sacred
6. **Do NOT run parallel Ralph loops** — git conflicts + rate limits
7. **Do NOT skip the first-iteration review** — watch the first 1-2 iterations before walking away
