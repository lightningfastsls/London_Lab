# Information Theory Metrics

**Phase:** Extension of Phase 8.4 (user-supplied spec)
**ADRs:** None (no DSP/STFT changes)
**Tests:** `usv_language/tests/test_information_theory.py` -- 16 tests across 8 test classes

## Purpose

Provides statistically rigorous information theory metrics for answering "do USVs contain language-like structure?" Extends the basic tools in `sequence_analysis.py` with proper MLE Zipf fitting, bias-corrected entropy rates, shuffle-surrogate idiom detection, bootstrap productivity CIs, and burstiness analysis.

All functions are computation-only (no matplotlib dependency). Inputs are 1D integer arrays (VQ-VAE code indices) or float arrays (event timestamps).

## Public Interface

### Dataclasses (6 frozen)

| Dataclass | Key Fields | Used By |
|-----------|-----------|---------|
| `ZipfResult` | alpha, xmin, p_value, n_tail, log_likelihood_ratio | `zipf_exponent_mle` |
| `ZipfEntropyResult` | alpha_estimate, entropy_observed, entropy_ci, method | `zipf_via_shannon_entropy` |
| `EntropyRateResult` | orders, rates_plugin, rates_corrected, convergence_order | `entropy_rate` |
| `IdiomResult` | ngram, observed_count, expected_count, z_score, p_value, n | `ngram_idioms` |
| `BurstinessResult` | cv, mean_iei, std_iei, n_bursts, interpretation | `burstiness_coefficient` |
| `ProductivityResult` | observed, possible, productivity_ratio, null_ci_lower, null_ci_upper, n | `ngram_productivity` |

### Core Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `zipf_exponent_mle` | `(ndarray) -> ZipfResult` | Discrete MLE Zipf (Clauset et al. 2009) with scipy fallback |
| `zipf_via_shannon_entropy` | `(ndarray, K?) -> ZipfEntropyResult` | PLC entropy inversion via brentq |
| `entropy_rate` | `(ndarray, max_order?, bias_correction?) -> EntropyRateResult` | Miller-Madow corrected h_n + convergence detection |
| `conditional_entropy_by_lag` | `(ndarray, K, max_lag?) -> list[float]` | H(X_t \| X_{t-lag}) via MI identity |
| `mutual_information_rate` | `(ndarray, K, max_lag?) -> list[float]` | MI decay curve wrapper |
| `ngram_productivity` | `(ndarray, K, n?, n_bootstrap?) -> ProductivityResult` | Bootstrap CI on observed/possible ratio |
| `ngram_idioms` | `(ndarray, K, max_n?, n_shuffles?, fdr_alpha?) -> list[IdiomResult]` | Shuffle surrogate + BH FDR |
| `burstiness_coefficient` | `(ndarray) -> BurstinessResult` | CV of IEIs + simplified burst detection |
| `burstiness_by_context` | `(ndarray, ndarray) -> dict[str, BurstinessResult]` | Per-context burstiness |

### Helper Functions (internal)

| Function | Purpose |
|----------|---------|
| `_generalized_harmonic` | Z(alpha, K) = sum(k^{-alpha}) via direct summation |
| `_plc_entropy` | Shannon entropy of truncated power-law |
| `_benjamini_hochberg` | FDR correction (manual, avoids scipy version dependency) |

## Key Decisions

1. **Dual Zipf estimation**: MLE directly estimates the exponent from the frequency distribution; entropy inversion provides a complementary estimate via information content. Agreement between the two is a useful consistency check.

2. **Miller-Madow over other corrections**: Jackknife and Bayesian corrections exist but Miller-Madow is the simplest first-order correction and adequate for our typical sample sizes (1000-50,000 codes).

3. **Manual BH FDR**: Implemented in ~10 lines to avoid depending on `scipy.stats.false_discovery_control` (requires scipy >= 1.11, which may not be installed).

4. **Simplified burst detection**: Uses threshold = mean_iei/2 rather than Kleinberg 2003 dynamic programming. Sufficient for identifying burst clusters without hierarchical structure modelling.

5. **`powerlaw` optional**: The `powerlaw` package (Clauset reference implementation) is import-guarded. The scipy fallback implements discrete MLE directly.

## Integration Points

- **Imports from:** `sequence_analysis.extract_ngrams`, `sequence_analysis.mutual_information_at_lag`
- **Called by:** `run_analysis.py` Section 6
- **Feeds into:** Summary JSON (`analysis_summary.json`)
- **Dependencies:** numpy, scipy (already in requirements). `powerlaw` optional.

## Edge Cases

| Function | Edge Case | Handling |
|----------|-----------|---------|
| All | Empty sequence | Return zero/empty results |
| `zipf_exponent_mle` | < 10 unique values | Return alpha=0, p_value=1.0 |
| `zipf_via_shannon_entropy` | H_obs outside PLC range | Clamp alpha, fall back to grid search |
| `entropy_rate` | max_order > len(sequence) | Clamp to sequence length |
| `ngram_idioms` | std_shuffle = 0 | Set z_score=0, p_value=1.0 |
| `burstiness_coefficient` | < 2 events | Return "insufficient_data" |
| `ngram_productivity` | K^n overflows | Cap at 2^53 |
