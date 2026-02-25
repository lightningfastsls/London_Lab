# Null Models Module Review

**Module:** `usv_language/analysis/null_models.py`
**Review Tier:** 2 (Standard)
**Reviewer:** master-reviewer
**Date:** 2026-02-24
**Branch:** main
**Test baseline:** 504 passed, 1 skipped (hmmlearn), 0 failures (full suite)
**Module tests:** 16 passed, 1 skipped — all collected items pass

---

## Summary

The null models module is well-structured: one frozen dataclass, one stateful generator, correct import guard pattern, good edge case handling for empty sequences and short sequences. Five of the five surrogate generators are conceptually correct and follow the vault's documented null model hierarchy. Four of them have no algorithmic bugs. One — the AAFT path of `phase_randomized` — has an inverted rank-matching assignment that causes the Gaussian proxy to have the wrong rank structure relative to the input sequence.

The bug is classified BLOCKER because (a) the module explicitly advertises AAFT as "better for discrete data" and recommends it over FFT for VQ-VAE codes, (b) the defective rank-matching makes AAFT's autocorrelation preservation degrade toward FFT-quality without warning, and (c) the test suite does not catch it because `test_aaft_marginal` only checks the marginal distribution, which is still preserved by step 3 regardless of the bug.

Remaining issues are warnings and suggestions. No DSP parameters are involved (pure sequence statistics module; no STFT, no sample rate). No data leakage issues exist (no ML splitting). Pattern compliance is excellent.

---

## Findings

### BLOCKER

#### B1 — AAFT rank-matching is inverted, degrading autocorrelation preservation

**What:** `_phase_randomized_aaft` at line 511 uses `y[original_rank_idx] = np.sort(gaussian)` but the correct AAFT assignment is `y = np.sort(gaussian)[original_rank_idx]`.

**Fix:** Change line 511, remove dead code at line 506, add AAFT autocorrelation test.

### WARNINGS

- **W1** — Dead code: `gaussian_sorted_idx` variable never used (line 506)
- **W2** — Markov `start_kgrams` includes terminal k-gram with no outgoing transitions
- **W3** — `generate_all` catches `Exception` too broadly around HMM fitting
- **W4** — Missing test: AAFT autocorrelation preservation
- **W5** — `random_seed` not validated in `NullModelConfig.__post_init__`
- **W6** — `NullModelGenerator` not exported from `__init__.py`

### SUGGESTIONS

- **S1** — Replace deprecated `np.random.RandomState` with `np.random.default_rng` (deferred)
- **S3** — Handoff test count discrepancy (17 items collected, not 16)

---

## Verdict

**CHANGES NEEDED** — One blocker (B1) + six warnings to fix in same pass.

---

## Fixes Applied

### B1 — AAFT rank-matching fixed
- **File:** `null_models.py:506-511`
- **Change:** Removed dead `gaussian_sorted_idx` variable; changed `y[original_rank_idx] = np.sort(gaussian)` to `y = np.sort(gaussian)[original_rank_idx]`
- **Why:** The inverted assignment gave the Gaussian proxy the wrong rank structure, degrading autocorrelation preservation

### W1 — Dead code removed
- Addressed as part of B1 fix

### W2 — Terminal k-gram excluded from start seeds
- **File:** `null_models.py:203`
- **Change:** `range(n - k_eff + 1)` → `range(max(1, n - k_eff))`
- **Why:** Terminal k-gram has no outgoing transitions, causing immediate backoff

### W3 — Broad exception catch replaced with warning
- **File:** `null_models.py:578-583`
- **Change:** Added `import warnings` and replaced bare `except Exception: pass` with `warnings.warn(...)` + `RuntimeWarning`
- **Why:** Silent exception swallowing hides real bugs

### W4 — AAFT autocorrelation test added
- **File:** `test_null_models.py`, new `test_aaft_autocorrelation` method
- **Change:** Added test verifying AAFT surrogates preserve autocorrelation (corr > 0.8)
- **Why:** Without this, B1-class bugs go undetected

### W5 — random_seed validation added
- **File:** `null_models.py`, `NullModelConfig.__post_init__`
- **Change:** Added bounds check `0 <= random_seed <= 2**32 - 1`
- **Why:** Pattern 1 requires all parameters validated at config creation

### W6 — NullModelGenerator exported from __init__.py
- **File:** `__init__.py`
- **Change:** Added `NullModelGenerator` to import and `__all__`
- **Why:** Config is useless without the generator class

### S3 — Handoff test count corrected
- **File:** `docs/reviews/null-models-handoff.md`
- **Change:** Updated count to "17 items collected (16 passed, 1 skipped)"

---

## Post-Fix Verification

```
usv_language/tests/test_null_models.py: 17 passed, 1 skipped in 1.15s
Full suite: 506 passed, 1 skipped, 0 failures in 63.75s
py_compile: all modified source files pass
```

New test `test_aaft_autocorrelation` PASSES with the fixed rank-matching code (B1 fix verified).
