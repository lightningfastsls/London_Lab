# Statistical Comparison Framework

**Phase:** Extension of Phase 8.4 (user-supplied spec)
**ADRs:** None (no DSP/STFT changes)
**Tests:** `usv_language/tests/test_statistical_tests.py` -- 14 tests across 9 test classes

## Purpose

Bridges `information_theory.py` (7 metric functions) and `null_models.py` (5 surrogate generators) to produce the core publishable result: a table showing whether real VQ-VAE code sequences have significantly more structure than expected under each null hypothesis.

For each (metric, null_model) pair:
- Compute the metric on the real sequence
- Compute it on all surrogates from that null model
- Compare via z-score (parametric) and rank-based p-value (nonparametric)

The output is the statistical backbone for answering: "Is this USV sequence more structured than chance?"

## Public Interface

### Dataclasses (2 frozen)

| Dataclass | Key Fields | Purpose |
|-----------|-----------|---------|
| `MetricComparison` | real_value, null_mean, null_std, z_score, rank_p_value, effect_size, significant | Result of one (metric, null_model) comparison |
| `FullAnalysisResult` | comparisons, metrics_used, null_models_used, summary | Complete comparison table with output formatters |

### NullModelComparison Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `compare` (static) | `(real_value, null_values, metric_name, null_model_name) -> MetricComparison` | Single comparison |
| `_compute_metric` (static) | `(metric_name, sequence, K, event_times?) -> float` | Maps metric name to function call |
| `full_analysis` | `(real_sequence, K, event_times?, null_config?) -> FullAnalysisResult` | Run all metrics x all null models |

### FullAnalysisResult Methods

| Method | Dependency | Description |
|--------|-----------|-------------|
| `to_markdown()` | None | Publication-ready markdown table (z-scores with * for significant) |
| `to_dataframe()` | pandas (optional) | Pivot DataFrame (rows=null models, cols=metrics, cells=z-scores) |

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `CORE_METRICS` | 7-tuple | The metrics always compared: zipf_alpha, entropy_rate, excess_entropy, mutual_information, mi_decay_half_life, bigram_productivity, n_significant_idioms |
| `_NULL_MODEL_ORDER` | 7-list | Scientific hierarchy for table row ordering: shuffled < markov_1..3 < renewal < hmm < phase_randomized |

## Metric-to-Function Mapping

| Metric | Source Module | Function | Extract |
|--------|--------------|----------|---------|
| `zipf_alpha` | information_theory | `zipf_exponent_mle(seq)` | `.rank_alpha` |
| `entropy_rate` | information_theory | `entropy_rate(seq)` | `.rates_corrected[-1]` |
| `excess_entropy` | sequence_analysis | `excess_entropy(seq, K)` | direct float |
| `mutual_information` | sequence_analysis | `mutual_information_at_lag(seq, K, 1)` | direct float |
| `mi_decay_half_life` | information_theory | `mutual_information_rate(seq, K, 20)` | first lag where MI <= 0.5*MI[0] |
| `bigram_productivity` | information_theory | `ngram_productivity(seq, K, n=2)` | `.productivity_ratio` |
| `n_significant_idioms` | information_theory | `ngram_idioms(seq, K)` | `len(result)` |

## Key Decisions

1. **burstiness_cv excluded from comparison matrix.** Null models operate on code sequences, not timestamps. Burstiness is reported standalone in the summary when event_times are provided.

2. **Significance criterion: `|z| > 2 OR rank_p < 0.05`.** The OR handles the degenerate case where null_std=0 (z-score meaningless but rank p-value still valid).

3. **Bessel's correction (ddof=1).** Sample std, not population std, because surrogates are a sample from the null distribution.

4. **One-sided rank p-value (conservative).** Includes equals: `sum(null >= real) / n`. This is conservative -- it won't falsely claim significance when the real value equals null values.

5. **Null model table ordered by scientific hierarchy**, not alphabetically. Shuffled (weakest) first, phase-randomized (strongest) last. This makes the table interpretable: significance should propagate from top (easy to beat) to bottom (hard to beat).

6. **n_bootstrap=1 for productivity on surrogates.** We only need `.productivity_ratio`, not CI bounds. The CI comes from the outer null-model comparison, not the inner bootstrap.

7. **n_shuffles=50 for idioms on surrogates.** Reduced from default 100 because idiom detection itself uses shuffle surrogates internally (multiplicative cost).

## Known Limitations

1. **mi_decay_half_life degenerate at large K.** Plug-in MI has a bias floor of ~K^2/(2N*ln2) bits. For K=64, N=10000, this is ~0.29 bits, which can make the half-life threshold unreachable. When this happens, the summary includes a diagnostic note.

2. **Histogram visualization uses Normal approximation.** `MetricComparison` stores summary statistics (mean, std) not raw values. CLI histograms regenerate from N(mean, std), which is qualitatively wrong for discrete metrics (n_significant_idioms, mi_decay_half_life). Each histogram includes a text annotation warning about this.

3. **n_significant_idioms is expensive.** Each surrogate runs its own internal shuffle test (50 shuffles x n_surrogates outer). For 100 surrogates on a 10k sequence, this metric dominates runtime.

## Integration Points

- **Imports from:** `information_theory` (7 metric functions), `null_models` (NullModelConfig, NullModelGenerator), `sequence_analysis` (excess_entropy, mutual_information_at_lag)
- **Called by:** `usv_language/scripts/run_null_model_analysis.py` (CLI)
- **Exported from:** `usv_language/analysis/__init__.py` (MetricComparison, FullAnalysisResult, NullModelComparison)

## Edge Cases

| Case | Handling |
|------|----------|
| `null_std == 0` | z_score=0.0, effect_size=0.0, significance uses rank_p only |
| All surrogates produce NaN | n_surrogates=0, all stats NaN, significant=False, warning issued |
| NaN real_value | z_score=NaN, significant=False |
| Empty sequence | ValueError raised early |
| K <= 0 | ValueError raised early |
| hmmlearn not installed | `generate_all()` skips "hmm" key |
| pandas not installed | `to_dataframe()` raises ImportError with message |
