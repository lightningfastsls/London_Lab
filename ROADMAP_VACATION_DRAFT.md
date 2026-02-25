# Vacation Workstreams — ROADMAP Draft

> **Source:** `VACATION_MASTER_PLAN_v2.md`
> **To integrate:** After review, append these phases to `ROADMAP.md` (after existing Phase 13).
> **Numbering:** Continues from Phase 13 in the main ROADMAP.

---

## Phase 14: Information Theory & Null Models

> **Scientific motivation:** The core question is whether mouse USV sequences contain language-like structure or are just statistical artifacts. Information-theoretic metrics quantify the structure; null models prove it's real. Without null models, none of the metrics mean anything — a shuffled sequence can have a Zipf exponent too.
>
> **Relationship to Phase 8.4:** The existing `sequence_analysis.py` (Phase 8.4) provides basic versions of Zipf (OLS log-log), entropy rate, excess entropy, and n-gram counting. This phase adds *rigorous* versions: MLE Zipf fitting (Clauset et al. 2009), Shannon entropy cross-validation, Miller-Madow bias correction, statistical idiom detection, burstiness analysis, and — critically — the null model framework that gives all these metrics scientific meaning.
>
> **Existing code to reuse/extend:**
> - `usv_language/analysis/sequence_analysis.py` — transition_matrix, extract_code_sequences, plotting utilities
> - `usv_language/analysis/compositionality.py` — bigram_productivity (basic version)
> - `usv_language/analysis/context_analysis.py` — group_codes_by_metadata, chi_squared_test

### 14.1 Information Theory Metrics

**What:** Rigorous information-theoretic analysis of discrete code sequences. Dual-approach Zipf estimation (MLE + Shannon entropy), bias-corrected entropy rate, conditional entropy at varying lags, n-gram idiom detection with statistical significance, and burstiness analysis linking temporal emission patterns to behavioral states.
**Status:** READY
**Review Tier:** 3 (complex statistical/mathematical algorithms)
**Depends on:** Phase 8.4 (code sequences from VQ-VAE pipeline)

/implement Information Theory Metrics

Build a comprehensive information theory analysis module for discrete code sequences. This extends the existing `sequence_analysis.py` with statistically rigorous methods and adds entirely new analyses (burstiness, idiom detection). The module operates on integer code sequences produced by the VQ-VAE pipeline.

**Context:** The existing `sequence_analysis.py` (Phase 8.4) uses OLS log-log regression for Zipf fitting, which is known to be biased for small datasets. This module adds two superior approaches: Clauset et al. 2009 MLE and Shannon entropy equivalence for power-law-with-cutoff (PLC) estimation. It also adds Miller-Madow bias correction to entropy estimation, statistical idiom detection (not just n-gram counting), and burstiness analysis that bridges information theory with behavioral analysis (LMT integration, Phase 16).

**Key references:**
- Clauset, Shalizi & Newman (2009) "Power-law distributions in empirical data" — gold standard for power law fitting
- Miller (1955) — bias correction for entropy estimation from finite samples
- Kleinberg (2003) "Bursty and Hierarchical Structure in Streams" — burst detection algorithm

**Files to create:**

1. `usv_language/analysis/information_theory.py` (NEW) — Core information theory metrics

```python
from dataclasses import dataclass
import numpy as np

# --- Result dataclasses ---

@dataclass(frozen=True)
class ZipfResult:
    """Result of MLE power law fitting (Clauset et al. 2009)."""
    alpha: float           # Power law exponent
    xmin: int              # Lower bound of power law behavior
    p_value: float         # KS goodness-of-fit p-value (> 0.1 = plausible)
    n_tail: int            # Number of observations in the tail (x >= xmin)
    log_likelihood_ratio: float  # vs exponential alternative

@dataclass(frozen=True)
class ZipfEntropyResult:
    """Zipf estimation via Shannon entropy equivalence."""
    alpha_estimate: float  # PLC exponent estimated from entropy
    entropy_observed: float  # H(X) of the empirical distribution
    entropy_ci: tuple[float, float]  # 95% bootstrap CI
    method: str = "shannon_entropy_equivalence"

@dataclass(frozen=True)
class EntropyRateResult:
    """Entropy rate at multiple context orders with bias correction."""
    orders: list[int]          # Context orders [0, 1, ..., max_order]
    rates_plugin: list[float]  # Plugin estimates (bits)
    rates_corrected: list[float]  # Miller-Madow corrected estimates (bits)
    convergence_order: int | None  # Order where rate stabilizes (if detected)

@dataclass(frozen=True)
class IdiomResult:
    """A statistically significant n-gram ('idiom')."""
    ngram: tuple[int, ...]   # The n-gram sequence
    observed_count: int      # How many times it appears
    expected_count: float    # Expected under independence assumption
    z_score: float           # Standard deviations above expected
    p_value: float           # After FDR correction
    n: int                   # n-gram length

@dataclass(frozen=True)
class BurstinessResult:
    """Burstiness analysis of temporal emission patterns."""
    cv: float                # Coefficient of variation of inter-event intervals
    # CV = 1: Poisson (random), CV > 1: bursty, CV < 1: regular
    mean_iei: float          # Mean inter-event interval
    std_iei: float           # Std of inter-event intervals
    n_bursts: int            # Number of detected bursts (Kleinberg)
    mean_burst_duration: float
    mean_inter_burst_interval: float
    interpretation: str      # "poisson", "bursty", or "regular"

@dataclass(frozen=True)
class ProductivityResult:
    """Extended bigram/n-gram productivity with bootstrap CI."""
    observed: int            # Unique n-grams observed
    possible: int            # K^n possible n-grams
    productivity_ratio: float  # observed / possible
    ci_lower: float          # 95% bootstrap CI lower bound
    ci_upper: float          # 95% bootstrap CI upper bound
    n: int                   # n-gram length


# --- Zipf's Law (dual approach) ---

def zipf_exponent_mle(sequence: list[int]) -> ZipfResult:
    """Clauset et al. 2009 MLE power law fit.

    Uses maximum likelihood estimation to fit a discrete power law
    distribution to code frequency data. More principled than OLS
    on log-log plot (which the existing sequence_analysis.py uses).

    The method:
    1. Estimate xmin via KS-distance minimization
    2. Estimate alpha via MLE for x >= xmin
    3. Compute KS goodness-of-fit p-value via semi-parametric bootstrap
    4. Compare to exponential alternative via likelihood ratio test

    Args:
        sequence: Integer code sequence (e.g., VQ-VAE code indices)

    Returns:
        ZipfResult with alpha, xmin, p_value, n_tail, log_likelihood_ratio
    """
    ...

def zipf_via_shannon_entropy(sequence: list[int]) -> ZipfEntropyResult:
    """Estimate power-law-with-cutoff (PLC) exponent through Shannon entropy.

    More robust than MLE for small datasets (< 10K tokens).
    For a power law with exponent alpha, Shannon entropy H relates to
    alpha through the Hurwitz zeta function. This inverts that relationship.

    Cross-validates the MLE estimate from zipf_exponent_mle().

    Args:
        sequence: Integer code sequence

    Returns:
        ZipfEntropyResult with alpha_estimate, entropy, CI
    """
    ...


# --- Sequential Structure ---

def entropy_rate(sequence: list[int], max_order: int = 8,
                 bias_correction: str = "miller_madow") -> EntropyRateResult:
    """H(X_n | X_{n-1},...,X_{n-k}) for k=0..max_order.

    Plugin estimator + Miller-Madow bias correction.
    Miller-Madow correction: H_corrected = H_plugin + (m - 1) / (2 * N * ln(2))
    where m = number of bins with nonzero probability, N = sample size.

    A decreasing curve indicates sequential structure.
    Convergence order indicates the effective Markov order.

    NOTE: The existing sequence_analysis.entropy_rate() does NOT apply
    bias correction. This version adds it and also detects convergence.

    Args:
        sequence: Integer code sequence
        max_order: Maximum context length to test
        bias_correction: "miller_madow" or "none"

    Returns:
        EntropyRateResult with plugin and corrected rates per order
    """
    ...

def conditional_entropy_by_lag(sequence: list[int], K: int,
                                max_lag: int = 10) -> list[float]:
    """H(X_t | X_{t-lag}) for varying lag.

    Unlike entropy_rate (which uses contiguous history X_{t-1},...,X_{t-k}),
    this uses a SINGLE token at varying temporal distances.

    Reveals the temporal decay of predictive information from one
    specific past position. Complementary to entropy_rate.

    (Suggested by Gemini analysis — not in original plan.)

    Args:
        sequence: Integer code sequence
        K: Codebook size (number of unique codes)
        max_lag: Maximum lag to compute

    Returns:
        List of H(X_t | X_{t-lag}) for lag=1..max_lag
    """
    ...

def mutual_information_rate(sequence: list[int], K: int,
                            max_lag: int = 20) -> list[float]:
    """I(X_t; X_{t+lag}) for lag=1..max_lag.

    Reveals how far contextual influence extends in the sequence.
    Should decay toward 0 for finite-memory processes.
    Slow decay suggests long-range dependencies (language-like).

    NOTE: Extends existing sequence_analysis.mutual_information_at_lag()
    by computing the full decay curve and fitting a decay model.

    Args:
        sequence: Integer code sequence
        K: Codebook size
        max_lag: Maximum lag

    Returns:
        List of MI values for lag=1..max_lag
    """
    ...


# --- Compositionality ---

def ngram_productivity(sequence: list[int], K: int, n: int = 2,
                       n_bootstrap: int = 1000) -> ProductivityResult:
    """Unique observed n-grams / K^n possible, with bootstrap CI.

    Extends existing compositionality.bigram_productivity() to:
    - Handle arbitrary n (not just bigrams)
    - Provide bootstrap confidence intervals

    Args:
        sequence: Integer code sequence
        K: Codebook size
        n: N-gram length (2 = bigrams, 3 = trigrams, etc.)
        n_bootstrap: Number of bootstrap resamples for CI

    Returns:
        ProductivityResult with ratio and 95% CI
    """
    ...

def ngram_idioms(sequence: list[int], K: int, max_n: int = 5,
                 n_shuffles: int = 1000,
                 fdr_alpha: float = 0.01) -> list[IdiomResult]:
    """Detect 'idioms': n-grams occurring significantly above chance.

    For each n from 2 to max_n:
      1. Count all n-gram occurrences in the real sequence
      2. Generate n_shuffles shuffled surrogates (preserving unigram frequencies)
      3. For each n-gram, compute z-score: (observed - mean_null) / std_null
      4. Apply Benjamini-Hochberg FDR correction at alpha level
      5. Flag n-grams with corrected p < fdr_alpha

    These are candidates for compositional 'phrases' — recurring multi-code
    patterns that exceed what unigram frequencies alone would predict.

    NOTE: This goes beyond existing sequence_analysis.top_ngrams() which
    just counts frequencies without statistical significance testing.

    Args:
        sequence: Integer code sequence
        K: Codebook size
        max_n: Maximum n-gram length to test
        n_shuffles: Number of shuffled surrogates for null distribution
        fdr_alpha: FDR significance threshold

    Returns:
        List of IdiomResult for all significant n-grams, sorted by z-score
    """
    ...


# --- Temporal Dynamics ---

def burstiness_coefficient(event_times: list[float]) -> BurstinessResult:
    """CV (coefficient of variation) of inter-event intervals.

    Characterizes temporal emission patterns:
    - CV = 1: Poisson process (random timing)
    - CV > 1: Bursty (clustered emissions)
    - CV < 1: Regular/periodic

    Also computes burst detection via Kleinberg's (2003) algorithm,
    mean burst duration, and inter-burst interval distribution.

    Args:
        event_times: Sorted list of event timestamps (seconds)

    Returns:
        BurstinessResult with CV, burst statistics, interpretation
    """
    ...

def burstiness_by_context(event_times: list[float],
                          context_labels: list[str]) -> dict[str, BurstinessResult]:
    """Burstiness broken down by behavioral context.

    Links temporal emission patterns to behavioral states.
    Each context label corresponds to the behavioral state at
    the time of each event (e.g., "approach", "contact", "idle").

    This is where information theory meets LMT integration (Phase 16):
    different behavioral contexts may produce fundamentally different
    temporal emission patterns.

    Args:
        event_times: Sorted list of event timestamps (seconds)
        context_labels: Behavioral context for each event (same length)

    Returns:
        Dict mapping context label -> BurstinessResult
    """
    ...
```

**Integration with existing code:**
- Import `transition_matrix` from `sequence_analysis.py` (don't reimplement)
- Import `extract_code_sequences` / `extract_bout_code_sequences` from `sequence_analysis.py`
- The new Zipf functions should be callable alongside the existing `zipf_analysis()` for comparison
- Plotting functions should follow the same matplotlib patterns as existing `sequence_analysis.py`

**Dependencies:**
```
numpy
scipy.stats          # KS test, chi-squared, etc.
powerlaw             # Optional: Clauset et al. MLE (pip install powerlaw)
                     # If unavailable, implement discrete MLE directly
```

**Test plan:**
```
1. Shuffled uniform sequence (K=64): entropy rate = log2(64) = 6.0 at all orders
2. Shuffled non-uniform sequence: Zipf MLE alpha matches input distribution alpha
3. Zipf MLE vs Shannon entropy estimate: agree within CI on synthetic power law data
4. Markov-1 sequence: entropy rate at order >= 2 equals order 1 rate
5. Perfectly periodic sequence: burstiness CV < 1 (regular)
6. Poisson-timed events: burstiness CV approximately 1.0
7. Known bursty process (gamma with shape < 1): burstiness CV > 1
8. Sequence with planted idiom ([5,12,33] at 10x expected rate): detected by ngram_idioms
9. Miller-Madow correction > 0 for all finite sequences
10. conditional_entropy_by_lag at lag=1 matches conditional_entropy from sequence_analysis.py
11. ngram_productivity with n=2 matches existing bigram_productivity for same input
```

**Exit criteria:**
- [ ] All 11 analytically verifiable tests pass
- [ ] Zipf MLE recovers known alpha from synthetic power law (error < 0.1)
- [ ] Entropy rate with Miller-Madow correction >= plugin estimate
- [ ] Burstiness correctly classifies Poisson / bursty / periodic synthetic data
- [ ] ngram_idioms detects planted idiom with z-score > 3
- [ ] All functions handle edge cases: empty sequence, single-element, all-same-code
- [ ] py_compile passes on all new files

---

### 14.2 Null Model Generators

**What:** Five surrogate sequence generators forming a hierarchy from "no structure" to "rich hidden structure." Each null model preserves specific statistical properties while destroying others, enabling targeted hypothesis testing about what kind of structure USV sequences contain.
**Status:** READY
**Review Tier:** 3 (statistical modeling, HMM fitting)
**Depends on:** None (operates on any integer sequence)

/implement Null Model Generators

Build five surrogate sequence generators that form a null model hierarchy. Each generator preserves specific statistical properties while destroying others. This is the most important module in the entire information theory framework — without null models, none of the metrics from 14.1 are scientifically meaningful.

**Context:** The null model hierarchy tests increasingly complex hypotheses:
1. **Shuffled** — preserves only frequency distribution. If real data ≈ shuffled, there's no sequential structure at all.
2. **Markov order k** — preserves k-th order transitions. If real data ≈ Markov-k, structure is fully captured by local context.
3. **Renewal process** — preserves inter-event interval distribution but destroys sequential dependencies. Tests whether temporal spacing alone explains patterns.
4. **HMM** — preserves hidden state dynamics. If real data ≈ HMM, "hidden behavioral states" hypothesis is sufficient.
5. **Phase-randomized** — preserves autocorrelation function but destroys higher-order structure. Tests whether linear temporal correlations explain the patterns.

If real USV sequences significantly exceed ALL null models on metrics like excess entropy and entropy rate, that's evidence for language-like structure beyond simple statistical patterns.

**Files to create:**

1. `usv_language/analysis/null_models.py` (NEW) — Null model generators

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NullModelConfig:
    """Configuration for null model generation."""
    n_surrogates: int = 100      # Number of surrogate sequences per model
    random_seed: int = 42
    # Markov-specific
    markov_orders: tuple[int, ...] = (1, 2, 3)  # Orders to test
    # HMM-specific
    hmm_n_states: int = 8        # Hidden states for HMM surrogate
    hmm_n_iter: int = 100        # EM iterations for HMM fitting
    # Phase randomization
    phase_rand_method: str = "fft"  # "fft" or "aaft" (amplitude-adjusted)


class NullModelGenerator:
    """Generate surrogate sequences under different null hypotheses.

    Each method fits a model to the input sequence, then generates
    n_surrogates synthetic sequences from that model. The surrogates
    preserve specific statistical properties while destroying others.

    Usage:
        gen = NullModelGenerator(NullModelConfig(n_surrogates=100))
        surrogates = gen.shuffled(real_sequence)
        # surrogates is a list of 100 sequences, each same length as input
    """

    def __init__(self, config: NullModelConfig = NullModelConfig()):
        ...

    def shuffled(self, sequence: list[int]) -> list[list[int]]:
        """Random permutation — preserves unigram frequencies, destroys all structure.

        This is the simplest null: if a metric on real data doesn't significantly
        exceed the shuffled baseline, there's no sequential structure to explain.

        Preserves: code frequency distribution
        Destroys: all temporal/sequential structure
        """
        ...

    def markov_order_k(self, sequence: list[int], k: int = 1) -> list[list[int]]:
        """Fit k-th order Markov chain and generate surrogates.

        Estimates transition probabilities from the data, then generates
        new sequences by sampling from the fitted Markov chain.

        Preserves: k-gram transition probabilities
        Destroys: structure beyond k-step memory

        If real data metrics ≈ Markov-k surrogates, the sequence has
        at most k-th order structure (no long-range dependencies).
        """
        ...

    def renewal_process(self, sequence: list[int]) -> list[list[int]]:
        """Fit inter-event interval distribution, generate surrogates.

        Models the sequence as a renewal process: events occur with
        inter-event intervals drawn from the empirical IEI distribution,
        and event types are drawn from the empirical frequency distribution
        independently.

        Preserves: code frequencies, inter-event interval distribution
        Destroys: sequential dependencies between code identities

        Tests whether temporal spacing patterns alone (without sequential
        code-to-code dependencies) explain the observed structure.
        """
        ...

    def hmm_surrogate(self, sequence: list[int],
                      n_states: int | None = None) -> list[list[int]]:
        """Fit Hidden Markov Model and generate surrogates.

        Uses Baum-Welch (EM) to fit an HMM with n_states hidden states
        and K emission symbols. Then generates surrogates by sampling
        from the fitted HMM.

        Preserves: hidden state dynamics (if real data has hidden states)
        Destroys: structure beyond what HMM can capture

        If real data ≈ HMM surrogates, the "hidden behavioral states"
        hypothesis is sufficient to explain the sequence structure.

        Requires: hmmlearn library (pip install hmmlearn)
        """
        ...

    def phase_randomized(self, sequence: list[int]) -> list[list[int]]:
        """Preserve autocorrelation function, destroy higher-order structure.

        Method:
        1. Compute FFT of the sequence (treating codes as integers)
        2. Randomize phases while preserving magnitudes
        3. Inverse FFT and round to nearest valid code

        Preserves: power spectrum (autocorrelation)
        Destroys: higher-order temporal structure, phase relationships

        Tests whether linear temporal correlations alone explain the patterns.

        NOTE: This is most meaningful for continuous signals. For discrete
        code sequences, the rounding step introduces approximation.
        Consider AAFT (Amplitude-Adjusted Fourier Transform) as alternative.
        """
        ...

    def generate_all(self, sequence: list[int]) -> dict[str, list[list[int]]]:
        """Generate surrogates from ALL null models.

        Returns dict mapping model name -> list of surrogate sequences.
        Markov models are generated for each order in config.markov_orders.

        Keys: "shuffled", "markov_1", "markov_2", "markov_3",
              "renewal", "hmm", "phase_randomized"
        """
        ...
```

**Dependencies:**
```
numpy
scipy                # For FFT-based phase randomization
hmmlearn             # For HMM fitting (pip install hmmlearn)
```

**Test plan:**
```
1. Shuffled surrogates preserve exact code frequency distribution
2. Shuffled surrogates have different sequential structure (verify via transition matrix difference)
3. Markov-1 surrogates have similar bigram transition matrix to input (within sampling noise)
4. Markov-1 surrogates from a known Markov-1 process: entropy rate at order >= 2 matches order 1
5. HMM surrogates from a sequence generated by a known HMM: recovered HMM has similar parameters
6. Phase-randomized surrogates preserve autocorrelation function (within tolerance)
7. All surrogates have same length as input sequence
8. generate_all() returns correct number of models and surrogates per model
9. Reproducibility: same seed -> identical surrogates
10. Edge case: very short sequence (< 20 tokens) handled gracefully
```

**Exit criteria:**
- [ ] All 5 null model generators produce valid surrogates (correct length, valid codes)
- [ ] Shuffled preserves frequencies exactly (verified by histogram comparison)
- [ ] Markov-k surrogates reproduce input's k-gram statistics (KS test p > 0.05)
- [ ] HMM fitting converges (log-likelihood increases monotonically)
- [ ] Phase randomization preserves autocorrelation (correlation coefficient > 0.95)
- [ ] generate_all() completes in < 60s for sequence of length 10,000 with n_surrogates=100
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 14.3 Statistical Comparison Framework

**What:** Framework for comparing real sequence metrics against null model surrogates. Computes z-scores, rank-based p-values, and effect sizes for each metric × null model combination. Produces the main publishable results table.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 14.1 (metrics), Phase 14.2 (null models)

/implement Statistical Comparison Framework

Build the statistical testing framework that ties information theory metrics (14.1) and null models (14.2) together. This produces the core publishable result: a table showing whether real USV sequences have significantly more structure than expected under each null hypothesis.

**Context:** The comparison works by computing each metric on the real sequence AND on all surrogates, then comparing. A z-score > 2 (or rank-based p < 0.05) means the real sequence has significantly more of that property than the null model preserves. The full analysis matrix (all null models × all metrics) reveals exactly what kind of structure exists and what explains it.

**Files to create:**

1. `usv_language/analysis/statistical_tests.py` (NEW) — Comparison framework

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class MetricComparison:
    """Result of comparing one metric between real and null surrogates."""
    metric_name: str
    null_model: str
    real_value: float
    null_mean: float
    null_std: float
    z_score: float           # (real - null_mean) / null_std
    rank_p_value: float      # Fraction of surrogates >= real value
    effect_size: float       # Cohen's d
    n_surrogates: int
    significant: bool        # z_score > 2 or rank_p < 0.05

@dataclass
class FullAnalysisResult:
    """Complete null model × metric comparison table."""
    comparisons: list[MetricComparison]
    metrics_used: list[str]
    null_models_used: list[str]
    sequence_length: int
    codebook_size: int
    summary: str             # Plain-language interpretation

    def to_dataframe(self):
        """Convert to pandas DataFrame (null models as rows, metrics as columns)."""
        ...

    def to_markdown(self) -> str:
        """Format as publishable markdown table."""
        ...


class NullModelComparison:
    """Compare real sequence metrics against null model baselines."""

    def compare(self, real_value: float,
                null_values: list[float],
                metric_name: str,
                null_model_name: str) -> MetricComparison:
        """Compare a single metric between real sequence and null surrogates.

        Computes:
        - z-score: (real - mean(null)) / std(null)
        - Rank-based p-value: fraction of null values >= real value
        - Effect size: Cohen's d

        Args:
            real_value: Metric computed on real sequence
            null_values: Same metric computed on each surrogate
            metric_name: Name of the metric
            null_model_name: Name of the null model

        Returns:
            MetricComparison with all statistics
        """
        ...

    def full_analysis(self, real_sequence: list[int], K: int,
                      event_times: list[float] | None = None,
                      null_config: 'NullModelConfig | None' = None
                      ) -> FullAnalysisResult:
        """All null models x all metrics — the main publishable table.

        Steps:
        1. Compute all metrics on the real sequence
        2. Generate surrogates from all null models
        3. Compute all metrics on all surrogates
        4. Compare real vs null for each metric x model pair
        5. Generate summary interpretation

        Metrics computed:
        - zipf_alpha (MLE)
        - entropy_rate (at convergence order)
        - excess_entropy
        - mutual_information (at lag=1)
        - mi_decay_half_life (lag where MI drops to 50%)
        - bigram_productivity
        - n_significant_idioms (count of detected idioms)
        - burstiness_cv (if event_times provided)

        Null models:
        - shuffled
        - markov_1, markov_2, markov_3
        - renewal (if event_times provided)
        - hmm
        - phase_randomized

        Args:
            real_sequence: The real code sequence
            K: Codebook size
            event_times: Optional event timestamps for burstiness analysis
            null_config: Optional NullModelConfig (defaults to sensible defaults)

        Returns:
            FullAnalysisResult with comparison table and summary
        """
        ...
```

2. `usv_language/scripts/run_null_model_analysis.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe usv_language/scripts/run_null_model_analysis.py \
      --hidden-states data/hidden_states/hidden_states_layer4.npy \
      --vqvae-checkpoint models/vqvae_final/best_model.pt \
      --metadata data/hidden_states/metadata.json \
      --output-dir usv_language/results/null_model_analysis/ \
      --n-surrogates 100 \
      --seed 42
```

Output:
```
usv_language/results/null_model_analysis/
├── comparison_table.csv          # Full metric x null model matrix
├── comparison_table.md           # Markdown version for paper
├── metric_distributions/         # Histograms: real value vs null distribution per metric
│   ├── zipf_alpha_vs_shuffled.png
│   ├── entropy_rate_vs_markov1.png
│   └── ...
├── summary.txt                   # Plain-language interpretation
└── raw_results.json              # All raw metric values for reproducibility
```

**Test plan:**
```
1. Known Markov-1 sequence: real entropy rate NOT significant vs Markov-1 null (p > 0.05)
2. Known Markov-1 sequence: real entropy rate IS significant vs shuffled null (p < 0.05)
3. Uniform random sequence: NO metric significant vs shuffled (all p > 0.05)
4. Sequence with planted long-range structure: excess entropy significant vs all Markov nulls
5. z-score computed correctly: manual calculation matches function output
6. Rank p-value = 0 when real value exceeds all surrogates
7. Effect size (Cohen's d) > 0.8 for large differences
8. to_markdown() produces valid markdown table
9. full_analysis() completes in < 5 min for 10,000-token sequence with 100 surrogates
```

**Exit criteria:**
- [ ] Comparison correctly identifies known Markov-1 sequences (not significant vs Markov-1 null)
- [ ] Comparison correctly identifies sequences with long-range structure (significant vs all Markov nulls)
- [ ] Markdown table is well-formatted and publishable
- [ ] Full analysis completes in reasonable time (< 5 min for 10K tokens, 100 surrogates)
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 14 Gate

Before starting Phase 15:
- [ ] All information theory metrics (14.1) produce correct results on analytically verifiable test cases
- [ ] All 5 null model generators (14.2) produce valid surrogates preserving specified properties
- [ ] Full analysis pipeline (14.3) runs end-to-end on synthetic data
- [ ] Comparison table correctly identifies known structure in synthetic sequences
- [ ] All Phase 14 tests pass
- [ ] py_compile passes on all new files

---

## Phase 15: Probing Experiments

> **Scientific motivation:** The transformer (Phase 8.2) learns internal representations of USV sequences, but we don't know what acoustic/temporal properties those representations encode. Probing experiments use simple classifiers (linear or shallow MLP) to predict ground-truth properties from hidden states. If a linear probe can predict peak frequency from layer 4 with high R², then layer 4 encodes frequency information. This directly guides VQ-VAE layer selection and validates that the transformer has learned meaningful representations.
>
> **Key output:** A layers × properties heatmap showing where information lives in the transformer. This is a standard interpretability technique from NLP (Belinkov 2022 "Probing Classifiers").

### 15.1 Acoustic Property Extractors

**What:** Extract ground-truth acoustic properties from spectrogram data for use as probing targets. Each property is computed directly from the spectrogram (not from the model), providing the "labels" for probing experiments.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 8.1 (spectrogram data)

/implement Acoustic Property Extractors

Build functions that extract ground-truth acoustic properties from spectrogram columns. These serve as probing targets — the "labels" that we'll train simple classifiers to predict from transformer hidden states.

**Context:** Per ADR-002, spectrograms are (170 freq bins, T frames) with frequency range 20-120 kHz and frequency resolution ~586 Hz/bin. Each spectrogram column is a 170-dimensional vector representing the frequency content at one time step.

**Files to create:**

1. `usv_language/analysis/acoustic_properties.py` (NEW) — Property extractors

```python
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class AcousticPropertyConfig:
    """Configuration for acoustic property extraction."""
    freq_min_hz: float = 20_000.0
    freq_max_hz: float = 120_000.0
    n_freq_bins: int = 170
    sample_rate: int = 300_000       # ADR-001
    hop_length: int = 128            # ADR-002
    voiced_threshold_db: float = -50.0  # Energy threshold for "USV present"
    direction_threshold: float = 500.0  # Hz change for "rising"/"falling"


def peak_frequency(column: np.ndarray, config: AcousticPropertyConfig) -> float:
    """Dominant frequency bin -> frequency in Hz.

    Args:
        column: (n_freq,) spectrogram column (log magnitude)
        config: Acoustic property configuration

    Returns:
        Peak frequency in Hz (continuous, 20000-120000 range)
    """
    ...

def spectral_centroid(column: np.ndarray, config: AcousticPropertyConfig) -> float:
    """Energy-weighted mean frequency in Hz.

    Centroid = sum(f_i * E_i) / sum(E_i) where E_i = 10^(S_db/10)

    Args:
        column: (n_freq,) spectrogram column (log magnitude)
        config: Acoustic property configuration

    Returns:
        Spectral centroid in Hz (continuous)
    """
    ...

def energy(column: np.ndarray) -> float:
    """Total energy of the spectrogram column.

    Sum of linear-scale magnitudes: sum(10^(S_db/10))

    Args:
        column: (n_freq,) spectrogram column (log magnitude)

    Returns:
        Total energy (continuous, > 0)
    """
    ...

def is_voiced(column: np.ndarray, config: AcousticPropertyConfig) -> bool:
    """Binary classification: is a USV present in this frame?

    Based on energy threshold in the USV frequency range.

    Args:
        column: (n_freq,) spectrogram column (log magnitude)
        config: Acoustic property configuration

    Returns:
        True if energy exceeds voiced threshold (classification target)
    """
    ...

def frequency_direction(col_prev: np.ndarray, col_curr: np.ndarray,
                         config: AcousticPropertyConfig) -> str:
    """Frequency modulation direction: 'rising', 'falling', or 'flat'.

    Compares peak frequency between consecutive frames.

    Args:
        col_prev: Previous frame spectrogram column
        col_curr: Current frame spectrogram column
        config: Acoustic property configuration

    Returns:
        'rising', 'falling', or 'flat' (3-class classification target)
    """
    ...

def bout_position(frame_idx: int, bout_length: int) -> float:
    """Normalized position within bout: frame_idx / bout_length.

    Args:
        frame_idx: Frame index within the bout
        bout_length: Total frames in the bout

    Returns:
        Normalized position in [0, 1] (continuous, regression target)
    """
    ...

def time_since_last_usv(frame_idx: int, usv_onsets: list[int],
                         config: AcousticPropertyConfig) -> float:
    """Temporal distance to nearest preceding USV onset, in milliseconds.

    Args:
        frame_idx: Current frame index
        usv_onsets: Sorted list of frame indices where USVs start
        config: Acoustic property configuration

    Returns:
        Time in ms since last USV onset (continuous, regression target).
        Returns -1.0 if no preceding USV exists.
    """
    ...

def extract_all_properties(spectrogram: np.ndarray,
                            config: AcousticPropertyConfig,
                            usv_onsets: list[int] | None = None
                            ) -> dict[str, np.ndarray]:
    """Extract all properties for every frame in a spectrogram.

    Args:
        spectrogram: (n_freq, T) spectrogram
        config: Acoustic property configuration
        usv_onsets: Optional list of USV onset frame indices

    Returns:
        Dict mapping property name -> (T,) array of values.
        Keys: 'peak_frequency', 'spectral_centroid', 'energy',
              'is_voiced', 'frequency_direction', 'bout_position',
              'time_since_last_usv' (if usv_onsets provided)
    """
    ...
```

**Test plan:**
```
1. Peak frequency of a single-peak column (energy at bin 85 = 70 kHz): returns ~70000 Hz
2. Spectral centroid of uniform energy column: returns center frequency (~70 kHz)
3. Energy of all-zero column: returns ~0 (numerical floor)
4. is_voiced on high-energy column: True; on silence column: False
5. frequency_direction with rising peak: 'rising'; falling: 'falling'; same: 'flat'
6. bout_position at frame 0: 0.0; at last frame: ~1.0
7. time_since_last_usv with no preceding onset: returns -1.0
8. extract_all_properties produces correct number of properties per frame
9. All continuous properties return float; all classification properties return correct types
```

**Exit criteria:**
- [ ] All property extractors produce correct values on synthetic spectrogram data
- [ ] Peak frequency and spectral centroid values are in valid Hz range (20k-120k)
- [ ] extract_all_properties handles edge cases (single-frame spectrogram, empty usv_onsets)
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 15.2 Probing Framework & Analysis Pipeline

**What:** Train linear and MLP probing classifiers to predict acoustic properties from transformer hidden states. Run all properties x all layers to produce a heatmap showing where information lives in the transformer. Directly guides VQ-VAE layer selection.
**Status:** READY
**Review Tier:** 3 (ML training, cross-validation, statistical interpretation)
**Depends on:** Phase 15.1 (acoustic properties), Phase 8.2 (transformer hidden states)

/implement Probing Framework & Analysis Pipeline

Build the probing experiment framework: load transformer hidden states per layer, extract acoustic property labels per frame, train linear/MLP probes with 5-fold CV, and generate a layers x properties heatmap showing where information lives.

**Context:** Probing is a standard NLP interpretability technique (Belinkov 2022). A linear probe tests whether information is *linearly accessible* in the representation; an MLP probe tests whether it's accessible at all. If linear R² is high, the representation explicitly encodes that property. If MLP R² >> linear R², the information is there but requires nonlinear extraction.

**Key insight for this project:** The probing heatmap directly answers "which transformer layer should the VQ-VAE operate on?" The layer with the richest acoustic information (highest average R² across properties) is the best candidate.

**Files to create:**

1. `usv_language/analysis/probing.py` (NEW) — Probing framework

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ProbingConfig:
    """Configuration for probing experiments."""
    probe_types: tuple[str, ...] = ("linear", "mlp")
    mlp_hidden_size: int = 128
    n_folds: int = 5             # Cross-validation folds
    max_samples: int = 50_000    # Subsample if dataset too large
    random_seed: int = 42
    # Regression properties (predict continuous values)
    regression_properties: tuple[str, ...] = (
        "peak_frequency", "spectral_centroid", "energy",
        "bout_position", "time_since_last_usv"
    )
    # Classification properties (predict discrete labels)
    classification_properties: tuple[str, ...] = (
        "is_voiced", "frequency_direction"
    )

@dataclass
class ProbingResult:
    """Result of one probing experiment (one property, one layer, one probe type)."""
    property_name: str
    layer: int
    probe_type: str           # "linear" or "mlp"
    task_type: str            # "regression" or "classification"
    # Regression metrics
    r_squared: float | None   # Mean R² across folds
    r_squared_std: float | None
    mse: float | None
    # Classification metrics
    accuracy: float | None    # Mean accuracy across folds
    accuracy_std: float | None
    f1: float | None
    # Common
    n_samples: int
    n_folds: int

@dataclass
class ProbingAnalysisResult:
    """Full probing analysis: all properties x all layers x all probe types."""
    results: list[ProbingResult]
    layers_tested: list[int]
    properties_tested: list[str]
    best_layer_by_property: dict[str, int]   # property -> best layer
    best_overall_layer: int                   # Highest average R²/accuracy
    summary: str

    def heatmap_data(self, probe_type: str = "linear") -> np.ndarray:
        """Return (n_layers, n_properties) matrix of R²/accuracy values."""
        ...


class ProbingExperiment:
    """Train probing classifiers on transformer hidden states."""

    def __init__(self, config: ProbingConfig = ProbingConfig()):
        ...

    def run_probe(self, hidden_states: np.ndarray,
                  labels: np.ndarray,
                  property_name: str,
                  layer: int,
                  probe_type: str = "linear") -> ProbingResult:
        """Train and evaluate one probing classifier.

        For regression properties: sklearn Ridge (linear) or MLPRegressor (mlp)
        For classification properties: sklearn LogisticRegression (linear) or MLPClassifier (mlp)

        Uses 5-fold stratified CV (classification) or KFold (regression).

        Args:
            hidden_states: (N, d_model) hidden states from one layer
            labels: (N,) ground-truth property values
            property_name: Name of the property
            layer: Which transformer layer
            probe_type: "linear" or "mlp"

        Returns:
            ProbingResult with R²/accuracy and standard deviations
        """
        ...


class ProbingAnalysisPipeline:
    """Run all probing experiments and generate analysis report."""

    def __init__(self, config: ProbingConfig = ProbingConfig()):
        ...

    def run_full_analysis(self,
                          hidden_states_by_layer: dict[int, np.ndarray],
                          properties: dict[str, np.ndarray]
                          ) -> ProbingAnalysisResult:
        """Run all properties x all layers x all probe types.

        Args:
            hidden_states_by_layer: {layer_num: (N, d_model) array}
            properties: {property_name: (N,) array}

        Returns:
            ProbingAnalysisResult with full heatmap and recommendations
        """
        ...

    def plot_heatmap(self, result: ProbingAnalysisResult,
                     probe_type: str = "linear",
                     output_path: str | None = None):
        """Generate layers x properties heatmap (matplotlib)."""
        ...

    def plot_layer_comparison(self, result: ProbingAnalysisResult,
                               output_path: str | None = None):
        """Bar chart: average R²/accuracy per layer across all properties."""
        ...
```

2. `usv_language/scripts/run_probing.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe usv_language/scripts/run_probing.py \
      --hidden-states-dir data/hidden_states/ \
      --spectrogram-dir data/processed/spectrograms/ \
      --metadata data/hidden_states/metadata.json \
      --layers 2 4 6 8 \
      --output-dir usv_language/results/probing/
```

Output:
```
usv_language/results/probing/
├── probing_heatmap_linear.png     # Layers x properties R²/accuracy
├── probing_heatmap_mlp.png        # Same for MLP probes
├── layer_comparison.png           # Average score per layer
├── per_property/                  # Detailed results per property
│   ├── peak_frequency.png
│   └── ...
├── results.json                   # All numerical results
└── probing_report.md              # Summary with layer recommendation
```

**Dependencies:**
```
numpy
scikit-learn         # Ridge, LogisticRegression, MLPRegressor, MLPClassifier
matplotlib
seaborn              # For heatmaps
```

**Test plan:**
```
1. Perfect encoding (hidden state = scaled property value): linear R² ≈ 1.0
2. Random noise hidden states: R² ≈ 0.0
3. Layer comparison with synthetic "deeper = better" data: scores increase monotonically with layer
4. Classification probe on perfectly separable data: accuracy ≈ 1.0
5. Classification probe on random labels: accuracy ≈ chance (1/n_classes)
6. Cross-validation produces n_folds results (5 by default)
7. Heatmap has correct shape (n_layers x n_properties)
8. best_overall_layer selects the layer with highest average score
```

**Exit criteria:**
- [ ] Probing recovers known relationships in synthetic data (R² > 0.95 for perfect encoding)
- [ ] Random baseline produces R² ≈ 0 and accuracy ≈ chance
- [ ] Heatmap visualization is readable with correct axis labels
- [ ] Layer recommendation printed in summary matches highest-scoring layer
- [ ] Full pipeline runs in < 5 min on 50K samples × 4 layers × 7 properties
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 15 Gate

Before starting Phase 16:
- [ ] Acoustic property extractors (15.1) produce correct values on synthetic spectrograms
- [ ] Probing framework (15.2) produces heatmap with correct dimensions
- [ ] Probing correctly distinguishes perfect/random encodings in synthetic tests
- [ ] All Phase 15 tests pass
- [ ] py_compile passes on all new files

---

## Phase 16: LMT Integration

> **Scientific motivation:** LMT (Live Mouse Tracker) provides behavioral annotations from video tracking — which mouse is doing what, at every frame. Combining USV detections with behavioral events answers: "Do mice vocalize more during specific behaviors?" and "Does vocal repertoire change with behavioral context?" This grounds the abstract information theory (Phase 14) in biology.
>
> **CRITICAL PREREQUISITE:** This workstream requires LMT SQLite database files containing behavioral annotations. If the SQLite files are not available on the laptop, Phase 16 should wait.
>
> **External tools:**
> - LMT-USV-Toolbox: https://github.com/fdechaumont/LMT-USV-Toolbox
> - lmt-analysis: https://github.com/fdechaumont/lmt-analysis
>
> **Data format:** SQLite databases with frame-by-frame animal positions (RFID-based), automatic behavioral event annotations (approach, contact, oral-oral, oral-genital, side-by-side, follow, etc.), synchronized with 300 kHz WAV recordings.

### 16.1 LMT Data Access Layer

**What:** Load LMT behavioral data from SQLite databases and synchronize timestamps with WAV recordings. This is the foundation for all behavioral-acoustic analysis.
**Status:** BLOCKED (requires LMT SQLite files on laptop — ask Prof. London)
**Review Tier:** 2
**Depends on:** External data (LMT SQLite files), Phase 1 (detection results)

/implement LMT Data Access Layer

Build the data access layer for LMT behavioral data. Load events from SQLite, synchronize timestamps with WAV recordings, and provide a clean Python API for downstream analysis.

**Context:** LMT (Live Mouse Tracker) uses SQLite databases to store frame-by-frame animal tracking data and automatic behavioral event annotations. The LMT-USV-Toolbox already handles WAV-to-LMT synchronization — adapt their approach rather than building from scratch. WAV files are at 300 kHz (ADR-001).

**Pre-implementation research required:**
Before writing code, clone and explore:
1. `git clone https://github.com/fdechaumont/LMT-USV-Toolbox.git`
2. `git clone https://github.com/fdechaumont/lmt-analysis.git`
3. Examine the SQLite schema (tables, columns, relationships)
4. Understand the timestamp synchronization method

**Files to create:**

1. `src/usv_spectrogram/lmt/__init__.py` (NEW)
2. `src/usv_spectrogram/lmt/db_loader.py` (NEW) — SQLite loading

```python
from dataclasses import dataclass
from pathlib import Path
import sqlite3

@dataclass(frozen=True)
class BehavioralEvent:
    """A single behavioral event from LMT."""
    event_type: str          # "approach", "contact", "oral-oral", etc.
    start_frame: int         # LMT frame number
    end_frame: int           # LMT frame number
    start_time_s: float      # Converted to seconds
    end_time_s: float        # Converted to seconds
    animal_id: int           # Which mouse
    partner_id: int | None   # Other mouse involved (if social event)

@dataclass(frozen=True)
class AnimalInfo:
    """Metadata about one animal in the experiment."""
    animal_id: int
    rfid: str
    genotype: str | None     # If available
    sex: str | None          # "M" or "F"
    strain: str | None       # "wild" or "lab" (if available)

class LMTDatabaseLoader:
    """Load behavioral data from LMT SQLite databases."""

    def __init__(self, db_path: Path):
        ...

    def get_animals(self) -> list[AnimalInfo]:
        """List all animals in the database."""
        ...

    def get_events(self, event_types: list[str] | None = None,
                   animal_id: int | None = None,
                   time_range: tuple[float, float] | None = None
                   ) -> list[BehavioralEvent]:
        """Query behavioral events with optional filters."""
        ...

    def get_event_types(self) -> list[str]:
        """List all available event types in this database."""
        ...

    def get_timeline(self, animal_id: int) -> list[BehavioralEvent]:
        """Get all events for one animal, sorted by time."""
        ...
```

3. `src/usv_spectrogram/lmt/synchronizer.py` (NEW) — Timestamp alignment

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SyncConfig:
    """Configuration for LMT-WAV synchronization."""
    lmt_frame_rate: float = 30.0     # LMT video frame rate (Hz)
    wav_sample_rate: int = 300_000   # ADR-001
    time_offset_s: float = 0.0      # Manual offset if needed

class LMTSynchronizer:
    """Align LMT behavioral timestamps with WAV recording timestamps."""

    def __init__(self, config: SyncConfig = SyncConfig()):
        ...

    def lmt_frame_to_seconds(self, frame: int) -> float:
        """Convert LMT frame number to seconds."""
        ...

    def seconds_to_wav_sample(self, time_s: float) -> int:
        """Convert seconds to WAV sample index."""
        ...

    def seconds_to_spectrogram_frame(self, time_s: float,
                                       hop_length: int = 128) -> int:
        """Convert seconds to spectrogram frame index (ADR-002)."""
        ...

    def align_events_with_detections(
        self,
        events: list['BehavioralEvent'],
        detections: list[dict],       # USV detections with start_time, end_time
    ) -> list[dict]:
        """For each USV detection, find the behavioral context.

        Returns list of dicts with:
        - detection info (start_time, end_time, etc.)
        - behavioral_context: list of events occurring during the USV
        - dominant_event: the most specific event type during the USV
        """
        ...
```

**Test plan:**
```
1. LMTDatabaseLoader opens a real SQLite file without error (integration test, skipped if no DB)
2. get_events with time_range filter returns only events within range
3. get_events with event_type filter returns only matching types
4. lmt_frame_to_seconds: frame 30 at 30 fps = 1.0 second
5. seconds_to_wav_sample: 1.0 second at 300 kHz = sample 300000
6. seconds_to_spectrogram_frame: 1.0 second at hop=128, sr=300k = frame 2343
7. align_events_with_detections correctly pairs overlapping events and detections
8. align_events handles USVs with no concurrent behavioral event (returns empty context)
```

**Exit criteria:**
- [ ] db_loader reads real LMT SQLite file (or: all tests pass on mock/synthetic data if no SQLite available)
- [ ] Timestamp conversions are mathematically correct (verified analytically)
- [ ] Event-detection alignment correctly handles overlapping, non-overlapping, and edge cases
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 16.2 Event-Triggered USV Rate Analysis

**What:** Compute peri-event time histograms (PETH) showing USV rate around behavioral events. The simplest and most important cross-modal analysis: do mice vocalize more during specific behaviors?
**Status:** BLOCKED (requires Phase 16.1 + LMT SQLite files)
**Review Tier:** 2
**Depends on:** Phase 16.1 (data access layer)

/implement Event-Triggered USV Rate Analysis

Build the event-triggered USV rate analysis (Tier 1 from the vacation plan). This computes peri-event time histograms (PETH) showing how USV emission rate changes around specific behavioral events. This is the first cross-modal result and serves as a sanity check: if USVs don't correlate with social behavior, something is wrong with data alignment.

**Context:** Peri-event time histograms (PETH) are standard in neuroscience. For each behavioral event (e.g., "approach"), compute the USV rate in a ±2s window centered on event onset. Compare to baseline rate (whole-recording average) via permutation test. If wild mice show stronger event-USV coupling than lab mice, that's evidence for courtship degradation.

**Connection to Workstream 1:** The `burstiness_by_context()` function from Phase 14.1 feeds directly into this analysis — it provides the temporal emission statistics per behavioral context that complement the PETH visualization.

**Files to create:**

1. `src/usv_spectrogram/lmt/event_triggered.py` (NEW) — PETH computation

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PETHConfig:
    """Configuration for peri-event time histogram."""
    window_before_s: float = 2.0     # Seconds before event onset
    window_after_s: float = 2.0      # Seconds after event onset
    bin_size_s: float = 0.1          # Time bin width (100 ms)
    n_permutations: int = 1000       # For significance testing
    min_events: int = 5              # Minimum events for reliable PETH
    baseline_method: str = "whole_recording"  # or "pre_event"

@dataclass
class PETHResult:
    """Result of peri-event time histogram analysis."""
    event_type: str
    time_bins: np.ndarray        # (n_bins,) bin centers in seconds relative to event onset
    rate: np.ndarray             # (n_bins,) USV rate per bin (Hz)
    rate_ci_lower: np.ndarray    # (n_bins,) 95% CI lower bound
    rate_ci_upper: np.ndarray    # (n_bins,) 95% CI upper bound
    baseline_rate: float         # Baseline USV rate (Hz)
    n_events: int                # Number of events analyzed
    p_value: float               # Permutation test: rate in window vs baseline
    significant: bool            # p < 0.05
    peak_time_s: float           # Time of maximum rate relative to event onset
    peak_rate: float             # Maximum rate


class EventTriggeredAnalysis:
    """Compute peri-event time histograms for USV rate."""

    def __init__(self, config: PETHConfig = PETHConfig()):
        ...

    def compute_peth(self,
                     usv_times: list[float],
                     event_times: list[float],
                     event_type: str,
                     recording_duration: float) -> PETHResult:
        """Compute PETH for one event type.

        Args:
            usv_times: Sorted list of USV onset times (seconds)
            event_times: Sorted list of behavioral event onset times (seconds)
            event_type: Name of the event type
            recording_duration: Total recording duration (seconds) for baseline

        Returns:
            PETHResult with rate histogram, CI, and significance test
        """
        ...

    def compute_all_peths(self,
                          usv_times: list[float],
                          events_by_type: dict[str, list[float]],
                          recording_duration: float
                          ) -> dict[str, PETHResult]:
        """Compute PETH for all event types.

        Returns dict mapping event_type -> PETHResult.
        Skips event types with fewer than config.min_events events.
        """
        ...

    def compare_populations(self,
                            peths_group_a: dict[str, PETHResult],
                            peths_group_b: dict[str, PETHResult],
                            group_names: tuple[str, str] = ("wild", "lab")
                            ) -> dict[str, dict]:
        """Compare PETH profiles between two populations.

        For each event type present in both groups:
        - Compare peak rates (Mann-Whitney U)
        - Compare rate profiles (pointwise permutation test)
        - Report which group shows stronger event-USV coupling

        Returns dict mapping event_type -> comparison results.
        """
        ...

    def plot_peth(self, result: PETHResult, output_path: str | None = None):
        """Plot PETH with confidence interval and baseline."""
        ...

    def plot_all_peths(self, results: dict[str, PETHResult],
                       output_path: str | None = None):
        """Grid of PETH plots for all event types."""
        ...
```

2. `scripts/run_event_triggered_analysis.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/run_event_triggered_analysis.py \
      --lmt-db path/to/experiment.sqlite \
      --detections analysis/batch_detections/results/ \
      --output analysis/event_triggered/ \
      --window 2.0 \
      --event-types approach contact oral-oral oral-genital side-by-side follow
```

Output:
```
analysis/event_triggered/
├── peth_approach.png
├── peth_contact.png
├── peth_all_events.png          # Grid of all PETHs
├── population_comparison.png    # Wild vs lab (if population labels available)
├── results.json                 # All numerical results
└── event_triggered_report.md    # Summary with significant events highlighted
```

**Test plan:**
```
1. Synthetic USVs clustered around event times: PETH shows peak at t=0
2. Uniform random USV times: PETH is flat (no peak, p > 0.05)
3. USVs only BEFORE events: peak at negative time, not at/after onset
4. Baseline rate matches whole-recording average
5. Permutation test produces p-values in [0, 1]
6. compare_populations detects when one group has stronger coupling
7. Empty event_times list returns PETH with n_events=0 and is skipped
8. Very short recording (< window_before + window_after): handled gracefully
```

**Exit criteria:**
- [ ] PETH correctly identifies temporal coupling in synthetic data
- [ ] Permutation test correctly identifies significant vs non-significant coupling
- [ ] Population comparison detects known differences in synthetic data
- [ ] Visualization shows clear PETH structure with baseline and CI
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 16 Gate

Before proceeding to Tier 2/3 behavioral analysis:
- [ ] LMT database loader reads SQLite files and returns structured events
- [ ] Timestamp synchronization is correct (verified analytically)
- [ ] PETH analysis produces correct results on synthetic data
- [ ] At least one real PETH computed on actual LMT + USV data (sanity check)
- [ ] All Phase 16 tests pass
- [ ] py_compile passes on all new files

---

## Dependency Graph (Phases 14-16)

```
Phase 8.4 (DONE) ──→ Phase 14.1 (Info Theory Metrics)
                            │
                            ├──→ Phase 14.3 (Statistical Comparison)
                            │         ↑
Phase 14.2 (Null Models) ──┘

Phase 8.1 (DONE) ──→ Phase 15.1 (Acoustic Properties)
Phase 8.2 (DONE) ──→ Phase 15.2 (Probing Framework)
                            ↑
                      Phase 15.1

External (SQLite) ──→ Phase 16.1 (LMT Data Access)
Phase 1 (DONE) ────→ Phase 16.1
                            │
                            ↓
                      Phase 16.2 (Event-Triggered Analysis)
                            ↑
                      Phase 14.1 (burstiness_by_context bridges WS1 ↔ WS3)
```

**Workstreams 1, 2, and 3 are independent** — they can be implemented in parallel.

---

## Vacation Session Plan (for reference)

| Session | Module | Duration |
|---------|--------|----------|
| 1 | 14.1 — Information Theory Metrics (Zipf, entropy, excess entropy, transitions) | ~1 hour |
| 2 | 14.2 + 14.3 — Null Models + Statistical Comparison | ~1 hour |
| 3 | 14.1 continued — Burstiness, compositionality, conditional entropy by lag | ~45 min |
| 4 | 15.1 + 15.2 — Probing Framework | ~1 hour |
| 5 | 16.1 — LMT research + data access layer (if SQLite available) | ~1 hour |

**Total: ~5 sessions, ~1 hour each, vacation-friendly with RustDesk.**
