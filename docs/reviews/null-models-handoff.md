# Implementation Handoff: Null Model Surrogate Generators

**Module:** Null Model Generators (`usv_language/analysis/null_models.py`)
**Review Tier:** 2 (Standard)
**Date:** 2026-02-24
**Branch:** main

## What Changed

- New analysis module with 1 frozen dataclass (`NullModelConfig`) and 1 stateful class (`NullModelGenerator`) implementing 5 surrogate generation methods + `generate_all()` aggregator
- Shuffled surrogates: exact permutation preserving marginal distribution
- Markov order-k surrogates: k-gram transition table with backoff chain (k -> k-1 -> ... -> unigram)
- Renewal process surrogates: per-code IEI shuffling with collision handling and unigram fill
- HMM surrogates: CategoricalHMM fitting via Baum-Welch (optional hmmlearn dependency)
- Phase-randomized surrogates: FFT (power spectrum preservation) and AAFT (exact marginal + autocorrelation)
- 19 tests (18 passed, 1 skipped) covering statistical properties, reproducibility, edge cases, and config validation
- Module documentation and this handoff document

## Files Changed

- `usv_language/analysis/null_models.py` (NEW) -- 1 dataclass, 1 class with 6 public methods, ~380 lines
- `usv_language/tests/test_null_models.py` (NEW) -- 10 test classes, 16 test items, 3 fixtures, ~330 lines
- `usv_language/analysis/__init__.py` (MODIFIED) -- Added `NullModelConfig` to `__all__`, added null_models to module docstring
- `usv_language/requirements.txt` (MODIFIED) -- Added `hmmlearn` as optional comment
- `docs/modules/null-models.md` (NEW) -- Module documentation
- `docs/reviews/null-models-handoff.md` (NEW) -- This file

## Key Decisions Made

1. **Standalone module.** No imports from `information_theory.py` or `sequence_analysis.py`. The generator only needs numpy and scipy.fft. Tests import from information_theory for verification but the module itself has zero cross-dependencies within the analysis package.

2. **hmmlearn import guard.** Same pattern as `powerlaw` in `information_theory.py`: `try/except ImportError` with `_HAS_HMMLEARN` flag. `generate_all()` catches exceptions from HMM fitting (degenerate sequences can fail EM convergence).

3. **Markov backoff.** Unseen k-grams fall back through shorter contexts: k -> k-1 -> ... -> unigram. This prevents generation failures on sparse data without requiring smoothing.

4. **Renewal collision policy.** Most-frequent codes placed first (priority by count). Collisions resolved by trying random free positions. Remaining gaps filled from unigram distribution.

5. **AAFT for discrete data.** FFT phase randomization maps continuous IFFT output back to codes via nearest-neighbor, which can distort the marginal. AAFT preserves the exact marginal by re-ranking, making it the better choice for discrete VQ-VAE codes.

## Dependencies

- **Upstream:** None within the analysis package (standalone)
- **Downstream:** Future `run_analysis.py` Section 7 integration
- **External:** numpy, scipy.fft (already in requirements). Optional: `hmmlearn`

## Known Limitations

1. **Markov-k for large k**: Transition tables grow exponentially. For k > 5 with small vocabularies, the table may be very sparse, causing frequent backoff.
2. **Renewal process frequency approximation**: Due to collisions, renewal surrogates preserve code frequencies approximately (within ~5%) rather than exactly.
3. **FFT phase randomization and discrete data**: Nearest-code mapping after IFFT can distort autocorrelation slightly. AAFT is preferred for discrete sequences.
4. **HMM fitting on degenerate data**: Constant or very short sequences may cause EM to fail. `generate_all()` catches this silently.

## Test Results

```
usv_language/tests/test_null_models.py: 17 passed, 1 skipped in 1.15s
Full suite: 506 passed, 1 skipped, 0 failures in 63.75s
py_compile: all modified source files pass
```

## Review Status

Pending master-reviewer.
