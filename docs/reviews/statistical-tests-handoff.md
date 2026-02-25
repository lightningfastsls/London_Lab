# Implementation Handoff: Statistical Comparison Framework

**Module:** Statistical Comparison Framework (`usv_language/analysis/statistical_tests.py`)
**Review Tier:** 2 (Standard -- cross-domain: statistics + information theory)
**Date:** 2026-02-24
**Branch:** main

## What Changed

- New analysis module with 2 frozen dataclasses (`MetricComparison`, `FullAnalysisResult`) and 1 class (`NullModelComparison`) implementing the null model x metric comparison framework
- `NullModelComparison.compare()`: z-score + rank p-value for a single (metric, null_model) pair
- `NullModelComparison._compute_metric()`: maps 7 metric names to information_theory/sequence_analysis function calls
- `NullModelComparison.full_analysis()`: orchestrates all metrics x all null models, builds plain-language summary
- `FullAnalysisResult.to_markdown()`: publication-ready table (no pandas)
- `FullAnalysisResult.to_dataframe()`: pivot DataFrame (optional pandas)
- CLI script following established `run_analysis.py` pattern with histogram output
- 14 tests (9 plan-specified + 5 edge cases), all passing
- Module documentation, handoff, and review documents

## Files Changed

- `usv_language/analysis/statistical_tests.py` (NEW) -- 2 dataclasses, 1 class, ~430 lines
- `usv_language/scripts/run_null_model_analysis.py` (NEW) -- CLI entry point, ~195 lines
- `usv_language/tests/test_statistical_tests.py` (NEW) -- 9 test classes, 14 test items, 5 fixtures
- `usv_language/analysis/__init__.py` (MODIFIED) -- Added exports and docstring entry
- `docs/modules/statistical-tests.md` (NEW) -- Module documentation
- `docs/reviews/statistical-tests-handoff.md` (NEW) -- This file
- `docs/reviews/statistical-tests-review.md` (NEW) -- Master review + fixes applied

## Key Decisions Made

1. **burstiness_cv excluded from comparison matrix.** Null models operate on code sequences, not timestamps. Reported standalone in summary.

2. **Significance OR criterion.** `|z| > 2 OR rank_p < 0.05` handles the degenerate null_std=0 case where z-score is uninformative but rank p-value is still valid.

3. **Scientific hierarchy ordering for table rows.** Null models ordered shuffled -> markov_1..3 -> renewal -> hmm -> phase_randomized, not alphabetically.

4. **Normal approximation for histograms.** Raw surrogate values not stored in MetricComparison to avoid memory bloat. Histograms annotated with caveat.

5. **Reduced inner computation for surrogates.** `n_bootstrap=1` for productivity (only need ratio), `n_shuffles=50` for idioms (multiplicative cost).

## Dependencies

- **Upstream:** `information_theory.py`, `null_models.py`, `sequence_analysis.py`
- **Downstream:** CLI output (CSV, markdown, JSON, PNGs)
- **External:** numpy (computation). Optional: pandas (to_dataframe), matplotlib (CLI histograms)

## Known Limitations

1. **mi_decay_half_life**: Plug-in MI bias floor can make this metric degenerate at large K. Summary includes diagnostic note.
2. **Histogram approximation**: CLI histograms use N(mean, std) not raw values. Annotated.
3. **n_significant_idioms runtime**: Dominates full_analysis for large sequences due to internal shuffle surrogates.

## Test Results

```
usv_language/tests/test_statistical_tests.py: 14 passed in 17.05s
Full suite: 232 passed, 1 skipped, 0 failures in 21.32s
py_compile: all modified source files pass
```

## Review Status

Master-reviewed (Tier 2). Review at `docs/reviews/statistical-tests-review.md`.
All blockers and warnings addressed. See "Fixes Applied" section in review file.
