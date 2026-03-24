# Implementation Handoff: Information Theory Metrics

**Module:** Information Theory Metrics (`usv_language/analysis/information_theory.py`)
**Review Tier:** 2 (Standard)
**Date:** 2026-02-24
**Branch:** main

## What Changed

- New analysis module with 6 frozen dataclasses and 9 public functions for statistically rigorous information theory metrics on VQ-VAE code sequences
- MLE-based Zipf exponent estimation (Clauset et al. 2009 discrete MLE) with scipy fallback and optional `powerlaw` package support
- Shannon entropy inversion for alternative Zipf alpha estimation via PLC distribution
- Bias-corrected (Miller-Madow) entropy rates with convergence detection
- Shuffle-surrogate idiom detection with Benjamini-Hochberg FDR correction
- Bootstrap confidence intervals for n-gram productivity
- Burstiness coefficient (CV of inter-event intervals) with simplified burst detection
- Integration into `run_analysis.py` as Section 6 with summary JSON fields
- 16 tests covering all planned test cases plus supplementary coverage

## Files Changed

- `usv_language/analysis/information_theory.py` (NEW) -- 6 dataclasses, 3 helpers, 9 public functions (~480 lines)
- `usv_language/tests/test_information_theory.py` (NEW) -- 8 test classes, 16 test items, 6 fixtures (~280 lines)
- `usv_language/analysis/__init__.py` (MODIFIED) -- Added information_theory to module docstring
- `usv_language/analysis/run_analysis.py` (MODIFIED) -- Added Section 6 info theory calls + 6 summary fields

## Key Decisions Made

1. **Scipy-based MLE as primary path.** The `powerlaw` package is not installed; the scipy fallback implements discrete MLE (alpha = 1 + n/sum(ln(x/(xmin-0.5)))) with KS-minimized xmin selection and bootstrap p-values (100 samples).

2. **Manual Benjamini-Hochberg.** 10-line implementation avoids requiring scipy >= 1.11 for `false_discovery_control`. Standard BH procedure: sort p-values, compare against (rank/m)*alpha thresholds.

3. **Entropy rate bias: correct code, adjusted tests.** Higher-order n-gram entropy rate estimates are systematically biased downward with finite data (curse of dimensionality). Tests were adjusted to only assert tight bounds where sampling is adequate (order 1 for K=64, orders 1-2 for K=8).

4. **No changes to existing modules.** `sequence_analysis.py` and `compositionality.py` are untouched. New module imports `extract_ngrams` and `mutual_information_at_lag` as read-only dependencies.

## Dependencies

- **Upstream:** Phase 8.4 (`sequence_analysis.py` -- `extract_ngrams`, `mutual_information_at_lag`)
- **Downstream:** Analysis summary JSON (consumed by interpretation workflows)
- **External:** numpy, scipy (brentq, stats.norm). Optional: `powerlaw`

## Known Limitations

1. **Zipf MLE bootstrap uses fixed seed** (rng=42) for reproducibility, but this means p-values are deterministic rather than truly random.
2. **Idiom detection with n_shuffles=100** trades statistical power for speed. For publication-quality results, increase to 1000.
3. **Burstiness burst detection is simplified** (threshold-based, not Kleinberg hierarchical). Adequate for identifying clusters but doesn't model nested burst structure.
4. **Entropy rate undersampling**: The Miller-Madow correction is first-order and insufficient for very sparse distributions (K^n >> N). This is documented in test comments.

## Test Results

```
usv_language/tests/test_information_theory.py: 16 passed in 3.97s
py_compile: all 3 modified/created source files pass
```

## Review Status

Pending master-reviewer.
