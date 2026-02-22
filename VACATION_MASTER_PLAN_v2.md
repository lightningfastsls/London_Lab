# Vacation Master Plan v2

**Created:** 2026-02-22
**Goal:** Keep the project moving with ~1 hour/day via RustDesk remote access
**Replaces:** VACATION_MASTER_PLAN.md (v1, which used Ralph — dropped)

---

## How Your Days Look

```
Morning (laptop, 15 min):
  Open 2-3 Windows Terminal tabs
  Paste task spec into each Claude Code session
  Tell each: "enter plan mode"
  Close lid, go enjoy vacation

Mid-morning (phone via RustDesk, 5 min):
  Check plans → "looks good, implement it"

Afternoon (phone, 10 min):
  Check implementations → "run master-reviewer on this"

Late afternoon (phone, 10 min):
  Read reviews → "fix the issues, then run tests"

Evening (laptop, optional 30 min):
  Review, commit, /seed findings into arscontexta
```

**Remote access:** RustDesk (already set up) + Tailscale as network backbone.

**arscontexta:** Stays 100% intact. All hooks run normally since Claude Code runs locally on your laptop. Knowledge vault updates happen when you're at the laptop.

---

## The Three Workstreams

| # | Workstream | Vacation-friendly? | Estimated sessions |
|---|-----------|-------------------|-------------------|
| 1 | Null models + information theory | **Yes — ideal** | 2-3 Claude Code sessions |
| 2 | Probing experiments | **Yes — good** | 1-2 sessions |
| 3 | LMT integration | **Partially** — research first, then code | 2-3 sessions |

---

## Workstream 1: Information Theory + Null Models

### Gemini vs. My Plan — Honest Comparison

Gemini's suggestions and my original plan are complementary, not competing. Here's what each brings:

| Metric | My Plan | Gemini Adds | Verdict |
|--------|---------|-------------|---------|
| **Zipf estimation** | Clauset et al. 2009 MLE (best current method for power law fitting) | Shannon entropy equivalence to PLC — avoids log-log bias on small datasets | **Merge: implement BOTH.** MLE for the direct estimate, entropy-based for validation. Gemini is right that log-log OLS is terrible for small N, but Clauset MLE already addresses this. The entropy equivalence is a valuable cross-check. |
| **Sequential structure** | Entropy rate H(X_n\|X_{n-1},...,X_{n-k}) with Miller-Madow correction | Conditional entropy S_α(L\|K) — same thing, different notation | **Same concept.** My plan already covers this. |
| **Long-range structure** | Excess entropy I(past;future) + mutual information rate at varying lags | Not mentioned by Gemini | **Keep mine.** Excess entropy is critical — it's what distinguishes language from Markov chains. |
| **Burstiness** | NOT in my plan | CV of inter-burst intervals, connection to stimulus valence | **Add from Gemini.** This is a real gap. Burstiness connects temporal emission patterns to behavioral states — directly relevant to the LMT integration. |
| **Compositionality** | Bigram productivity (unique bigrams / K²) | N-gram analysis, "idiom" detection (multi-token sequences exceeding chance) | **Merge.** My bigram productivity is the simple metric. Gemini's framing of detecting statistical "idioms" via expected vs. observed n-gram frequencies is more rigorous. Implement both. |
| **Null models** | 5 generators (shuffled, Markov-k, renewal, HMM, phase-randomized) + statistical comparison framework | Not mentioned at all | **Critical gap in Gemini's plan.** Without null models, none of the metrics mean anything. This is the most important part of the whole framework. |

### What to Implement — Merged Spec

#### Module 1: `usv_language/analysis/information_theory.py`

```python
# --- Zipf's Law (dual approach) ---
def zipf_exponent_mle(sequence: list[int]) -> ZipfResult:
    """Clauset et al. 2009 MLE power law fit.
    Returns: alpha, xmin, p_value (KS goodness-of-fit).
    More principled than OLS on log-log plot."""

def zipf_via_shannon_entropy(sequence: list[int]) -> ZipfEntropyResult:
    """Estimate PLC through Shannon entropy equivalence.
    More robust for small datasets (< 10K tokens).
    Cross-validates the MLE estimate."""

# --- Sequential Structure ---
def entropy_rate(sequence: list[int], max_order: int = 8) -> list[float]:
    """H(X_n | X_{n-1},...,X_{n-k}) for k=0..max_order.
    Plugin estimator + Miller-Madow bias correction.
    Decreasing curve = sequential structure exists."""

def excess_entropy(sequence: list[int], max_half_length: int = 50) -> float:
    """I(past; future) via block entropy extrapolation.
    The key metric: > 0 means long-range structure beyond Markov."""

def mutual_information_rate(sequence: list[int], max_lag: int = 20) -> list[float]:
    """I(X_t; X_{t+lag}) for lag=1..max_lag.
    Reveals how far contextual influence extends."""

# --- Transition Structure ---
def transition_matrix(sequence: list[int], K: int) -> np.ndarray:
    """K×K bigram transition probability matrix."""

def conditional_entropy_by_lag(sequence: list[int], max_lag: int = 10) -> list[float]:
    """H(X_t | X_{t-lag}) for varying lag.
    Reveals the temporal decay of predictive information.
    (Gemini's suggestion — complementary to entropy_rate which uses
    contiguous history, this uses single-token at varying distances.)"""

# --- Compositionality ---
def bigram_productivity(sequence: list[int], K: int) -> ProductivityResult:
    """Unique observed bigrams / K² possible. Bootstrap CI."""

def ngram_idioms(sequence: list[int], K: int, max_n: int = 5,
                 n_shuffles: int = 1000) -> list[IdiomResult]:
    """Detect 'idioms': n-grams occurring significantly above chance.
    For each n from 2 to max_n:
      - Count all n-gram occurrences
      - Compare to expected frequency under independence
      - Flag n-grams with z-score > 3 (or FDR-corrected p < 0.01)
    These are candidates for compositional 'phrases'."""

# --- Temporal Dynamics (NEW — from Gemini) ---
def burstiness_coefficient(event_times: list[float]) -> BurstinessResult:
    """CV (coefficient of variation) of inter-event intervals.
    CV = 1: Poisson (random timing)
    CV > 1: Bursty (clustered emissions)
    CV < 1: Regular/periodic
    Also computes: burst detection via Kleinberg's algorithm,
    mean burst duration, inter-burst interval distribution."""

def burstiness_by_context(event_times: list[float],
                          context_labels: list[str]) -> dict:
    """Burstiness broken down by behavioral context.
    Links temporal emission patterns to behavioral states.
    (This is where information theory meets LMT integration.)"""
```

#### Module 2: `usv_language/analysis/null_models.py`

(Unchanged from v1 — Gemini doesn't have this and it's the most important part)

```python
class NullModelGenerator:
    def shuffled(sequence, n_surrogates=100) -> list[list[int]]:
        """Preserves frequencies, destroys all structure."""

    def markov_order_k(sequence, k=1, n_surrogates=100) -> list[list[int]]:
        """Fit k-th order Markov, generate surrogates."""

    def renewal_process(sequence, n_surrogates=100) -> list[list[int]]:
        """Fit inter-event intervals, destroy sequential dependencies."""

    def hmm_surrogate(sequence, n_states=8, n_surrogates=100) -> list[list[int]]:
        """Fit HMM — tests 'hidden behavioral states' hypothesis."""

    def phase_randomized(sequence, n_surrogates=100) -> list[list[int]]:
        """Preserve autocorrelation, destroy higher-order structure."""
```

#### Module 3: `usv_language/analysis/statistical_tests.py`

```python
class NullModelComparison:
    def compare(real, nulls, metrics) -> ComparisonResult:
        """z-scores, rank-based p-values, effect sizes."""

    def full_analysis(real_sequence, K) -> FullAnalysisResult:
        """All null models × all metrics. The main publishable table."""
```

#### Tests — Analytically Verifiable

| Test | Expected |
|------|----------|
| Shuffled uniform K=64, entropy rate | = log2(64) = 6.0 at all orders |
| Shuffled non-uniform, Zipf MLE | Same alpha as input |
| Zipf MLE vs Shannon entropy estimate | Should agree within CI |
| Markov-1, entropy rate at order ≥ 2 | Same as order 1 |
| Perfectly periodic sequence, burstiness CV | < 1 (regular) |
| Poisson-timed events, burstiness CV | ≈ 1 |
| Known bursty process (gamma with shape < 1) | CV > 1 |
| Sequence with planted idiom (e.g., [5,12,33] at 10x expected rate) | Detected by ngram_idioms |

### Session Plan for This Workstream

**Session 1:** Information theory module — Zipf (both methods), entropy rate, excess entropy, transition matrix
**Session 2:** Null models + statistical comparison framework
**Session 3:** Burstiness, compositionality (bigram productivity + idiom detection), conditional entropy by lag

---

## Workstream 2: Probing Experiments

(Unchanged from v1 — still a clean, well-scoped task)

### What to Build

#### `usv_language/analysis/acoustic_properties.py`

Extracts ground-truth labels from spectrogram data for probing:

- `peak_frequency(column)` → dominant frequency bin (continuous)
- `spectral_centroid(column)` → energy-weighted mean frequency (continuous)
- `energy(column)` → total energy (continuous)
- `is_voiced(column, threshold)` → USV present? (classification)
- `frequency_direction(col_prev, col_curr)` → rising/falling/flat (classification)
- `bout_position(frame_idx, bout_length)` → normalized [0,1] (continuous)
- `time_since_last_usv(frame_idx, usv_onsets)` → temporal distance (continuous)

#### `usv_language/analysis/probing.py`

- `ProbingExperiment` — load hidden states, extract features per layer, train linear/MLP probes with 5-fold CV
- `ProbingAnalysisPipeline` — run all properties × all layers → heatmap showing where information lives
- **Key output:** Which layers encode which acoustic properties → directly guides VQ-VAE layer selection

#### Tests — All Synthetic

- Perfect encoding → R² ≈ 1.0
- Random noise → R² ≈ 0
- Layer comparison with "deeper = better" synthetic data → monotonic

### Session Plan

**Session 1:** Acoustic properties + probing framework
**Session 2:** Analysis pipeline + full test suite

---

## Workstream 3: LMT Integration

### Key Discovery: LMT Already Has a USV Toolbox

The LMT project provides `LMT-USV-Toolbox` (https://github.com/fdechaumont/LMT-USV-Toolbox) which already handles:

- **Synchronization** of WAV files with LMT behavioral data (`LMT.USV.importer`)
- **USV detection** (their own method, available at https://usv.pasteur.cloud)
- **Behavioral context analysis** — scripts for acoustic analyses per behavioral context
- **Burst analysis** — USV burst timing in relation to behavioral events
- **Speed/duration/events with USV** — linking movement data to vocalizations

**Data format:** SQLite databases containing:
- Frame-by-frame animal positions and identity (RFID-based)
- Automatic behavioral event annotations (approach, contact, oral-oral, oral-genital, side-by-side, follow, etc.)
- WAV files at 300 kHz (matches your setup exactly)

**Critically:** LMT already synchronizes USV timestamps with behavioral events. You don't need to build synchronization from scratch — you can use or adapt their `LMT.USV.importer`.

### What to Investigate First

Before writing any code, you need to understand:

1. **Which LMT version generated your data?** Your lab's WAV files in `5970 USV/` — were they recorded through LMT's AviSoft integration, or independently? This determines whether synchronization metadata already exists in the SQLite DB.

2. **Do you have the SQLite databases?** The WAV files alone aren't enough. You need the corresponding `.sqlite` files from LMT to get behavioral annotations.

3. **Which behavioral events matter for courtship?** LMT detects many events. For the courtship degradation hypothesis, the most relevant are probably: approach, oral-genital contact, side-by-side contact, follow. Ask Prof. London which events are in your experimental protocol.

4. **Population labels:** Do the SQLite databases or your lab records indicate which recordings are wild-derived vs. lab mice?

### Three Analysis Tiers

#### Tier 1: Event-Triggered USV Rate (simplest, do first)

**Question:** Do mice vocalize more during specific behavioral events?

**Method:**
- Load LMT events from SQLite (use `lmt-analysis` Python library or direct SQL)
- For each event type, compute USV rate in ±2s window using your CNN detections
- Compare to baseline rate via permutation test
- This is basically a peri-event time histogram (PETH)

**Why it matters:** Sanity check. If USVs don't correlate with social behavior, something is wrong with the data alignment. Also: if wild mice show stronger event-USV coupling than lab mice, that's already evidence for courtship degradation.

#### Tier 2: Behavioral Context × Vocal Repertoire

**Question:** Do USV types change with behavioral context?

**Method:**
- Assign each USV a behavioral context (what was happening when it vocalized)
- Compare CNN feature distributions or VQ-VAE codes across contexts
- Statistical test: MANOVA on features, or chi-squared on cluster/code assignments
- Wild vs. lab comparison: do lab mice show less context-dependent variation?

**Connection to burstiness:** The `burstiness_by_context()` function from Workstream 1 feeds directly into this analysis.

#### Tier 3: Vocal Sequence → Behavioral Prediction

**Question:** Can you predict the next behavioral event from the vocal sequence?

**Method:**
- Use VQ-VAE code sequences as features
- Predict next behavioral event (classification)
- Compare to baseline: predict from behavioral history alone
- Measure mutual information: I(vocal_sequence; next_behavior)

**Connection to probing (Workstream 2):** Add LMT-derived probing targets:
- `behavioral_state` → does the transformer encode what the mouse is doing?
- `time_to_next_event` → does it predict behavioral transitions?

### Code Structure

```
src/usv_spectrogram/lmt/
├── __init__.py
├── db_loader.py          # Load LMT SQLite → structured events
├── synchronizer.py       # Align LMT timestamps with WAV timestamps
│                         # (adapt from LMT-USV-Toolbox if possible)
├── event_triggered.py    # Peri-event time histograms for USV rate
├── context_analysis.py   # Behavioral context × vocal repertoire
└── tests/
```

### Session Plan

**Session 1 (research):** Clone LMT-USV-Toolbox and lmt-analysis. Explore the SQLite schema. Understand synchronization. Produce a design doc.

**Session 2 (code):** Implement `db_loader.py` and `synchronizer.py` — the data access layer.

**Session 3 (analysis):** Implement Tier 1 — event-triggered USV rate analysis. This gives you your first cross-modal result.

### Pre-Vacation Questions to Answer

Before starting this workstream, ask yourself (or Prof. London):

- [ ] Where are the LMT SQLite files for your recordings?
- [ ] Were WAVs recorded via LMT's AviSoft integration?
- [ ] Which mice are wild-derived vs. lab in your dataset?
- [ ] Which LMT version was used?

If you don't have the SQLite files on the laptop, the LMT workstream should wait until you're back. Workstreams 1 and 2 don't need them at all.

---

## Where the Workstreams Converge

This is the research story:

```
Workstream 1                Workstream 2              Workstream 3
(Information Theory)        (Probing)                 (LMT)
        │                       │                         │
   Null models              Layer analysis           Behavioral events
   Zipf, entropy            What transformer         Event-triggered
   Burstiness               learned                  USV rates
   Compositionality              │                         │
        │                       │                         │
        └───────────┬───────────┘                         │
                    │                                     │
          "Do USV sequences                    "Do USVs correlate
           have language-like                   with behavior?"
           structure?"                                │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                    "Does the transformer encode
                     behaviorally meaningful vocal
                     categories that differ between
                     wild and lab populations?"
                                   │
                            THE PAPER
```

- **Null models** prove the structure is real (not statistical artifact)
- **Probing** shows what the transformer learned (and guides VQ-VAE layer choice)
- **Burstiness by context** bridges information theory and behavioral analysis
- **LMT integration** grounds everything in biology

---

## Suggested Vacation Schedule

| Day | Morning (laptop 15 min) | Phone check-ins | Evening (laptop, optional) |
|-----|------------------------|-----------------|--------------------------|
| 1 | Spawn Session 1: info theory (Zipf, entropy, excess entropy) | Review plan → approve → review implementation → run reviewer | Commit, /seed |
| 2 | Spawn Session 2: null models + statistical tests | Same pattern | Commit, /seed |
| 3 | Spawn Session 3: burstiness + compositionality | Same | Commit, /seed |
| 4 | Spawn Session 4: probing framework | Same | Commit, /seed |
| 5 | Spawn Session 5: LMT research (explore SQLite schema, design doc) | Review design doc | Read doc, decide on Tier 1 approach |
| Skip days | Nothing breaks | Nothing breaks | Nothing breaks |

**Total laptop time per day:** 15-45 min
**Total phone time per day:** 15-25 min
**arscontexta updates:** Only when at the laptop, at your pace

---

## Files to Have Ready Before Leaving

- [ ] This plan, accessible on the laptop
- [ ] RustDesk permanent password set
- [ ] Laptop power: lid close = do nothing, sleep = never (plugged in)
- [ ] Repo pulled and tests passing on laptop
- [ ] LMT SQLite files on laptop (if available — ask Prof. London)
- [ ] LMT-USV-Toolbox cloned locally for reference: `git clone https://github.com/fdechaumont/LMT-USV-Toolbox.git`
- [ ] `lmt-analysis` cloned: `git clone https://github.com/fdechaumont/lmt-analysis.git`
