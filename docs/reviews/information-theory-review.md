# Information Theory Metrics Module Review

**Reviewed by:** Master Reviewer (claude-sonnet-4-6)
**Date:** 2026-02-24
**Module:** `usv_language/analysis/information_theory.py`
**Tier:** 2 (Standard)
**Verdict:** CHANGES NEEDED

---

## Summary

Reviewed against: handoff document, DECISIONS.md, ROADMAP.md (no `/implement` block for this module — user-supplied spec), `docs/architecture/patterns.md`, and the Phase 8.4 `sequence_analysis.py` upstream interface.

The module is architecturally sound. All six frozen dataclasses follow Pattern 1. DSP/STFT concerns do not apply (this is pure information-theoretic computation on code indices). The math in the PLC entropy, BH FDR, Miller-Madow correction, and Clauset MLE is correct and verified. Convergence detection is also correct — an earlier impression of an off-by-one was traced and confirmed as correct by manual tracing.

Two blockers found: the test suite cannot actually run as claimed (conftest h5py issue prevents normal invocation), and the `ngram_productivity` bootstrap CI does not contain the observed ratio for structured sequences — which is exactly the type of data this module will be used on.

---

## Test Results

**Actual run command required:** `pytest usv_language/tests/test_information_theory.py --noconftest`

**Result:** 16 passed in 4.05s

**Normal invocation** (`pytest usv_language/tests/test_information_theory.py`) **fails** with `ModuleNotFoundError: No module named 'h5py'` due to a top-level import in `usv_language/tests/conftest.py:9`. The handoff states "16 passed in 3.97s" without disclosing this dependency. This is a false completion claim.

---

## BLOCKER

### B1. conftest.py imports h5py at module level — all usv_language tests fail in this environment

**File:** `usv_language/tests/conftest.py:9`

**Problem:** `import h5py` is at the top level of the shared conftest. `h5py` is not installed in the project venv (`pip list` confirms absence). Pytest loads conftest before collecting any tests, so `pytest usv_language/tests/test_information_theory.py` fails immediately with `ModuleNotFoundError` before a single test runs.

**Fix:** Use lazy import inside the fixture that actually uses it (Option A).

---

### B2. ngram_productivity bootstrap CI does not bracket observed ratio for structured sequences

**File:** `usv_language/analysis/information_theory.py`

**Problem:** The bootstrap resamples sequence elements independently (i.i.d.) to construct the CI on productivity ratio. For structured sequences, independent resampling destroys sequential structure and produces far higher n-gram diversity than the original. This means `ci_lower > productivity_ratio` for any structured sequence.

**Fix:** Rename CI fields to `null_ci_lower`/`null_ci_upper` to make the null-hypothesis semantics explicit, and fix the test assertion.

---

## WARNING

### W1. ZipfResult.log_likelihood_ratio uses wrong normalization constant

**File:** `usv_language/analysis/information_theory.py`

**Problem:** `_generalized_harmonic(alpha, int(tail.max()))` computes `sum(k^{-alpha}, k=1..x_max)` but should compute `sum(k^{-alpha}, k=xmin..x_max)`.

**Fix:** Use `np.arange(int(best_xmin), int(tail.max()) + 1)` for the normalization range.

---

### W2. burstiness_coefficient not called in run_analysis.py Section 6

**File:** `usv_language/analysis/run_analysis.py`

**Problem:** 4 of 9 public functions are not invoked. Burstiness is the most notable omission since it's unique to this module.

**Fix:** Add burstiness call with frame-based event time approximation.

---

### W3. IMPLEMENTATION_PROGRESS.md not updated

**Fix:** Add dated entry for information theory module.

---

## SUGGESTION

### S1. zipf_via_shannon_entropy K=None uses max(sequence)+1

Documented: K should be passed explicitly from call sites that know the codebook size.

### S2. Idiom detection one-sided test direction not documented

Add note to docstring about only testing over-represented patterns.

---

## Fixes Applied

### B1 Fix — conftest.py lazy h5py import
- **File:** `usv_language/tests/conftest.py:9`
- **Change:** Replaced top-level `import h5py` with `pytest.importorskip("h5py")` inside the `synthetic_hdf5` fixture
- **Why:** Allows tests that don't need h5py (like test_information_theory.py) to run normally

### B2 Fix — Rename CI fields to null-hypothesis semantics
- **Files:** `usv_language/analysis/information_theory.py` (ProductivityResult dataclass + ngram_productivity function), `usv_language/tests/test_information_theory.py` (TestProductivity), `docs/modules/information-theory.md`
- **Change:** Renamed `ci_lower`/`ci_upper` to `null_ci_lower`/`null_ci_upper`, updated docstrings, fixed test assertion
- **Why:** i.i.d. bootstrap computes the null distribution (token independence), not a CI around the observed ratio

### W1 Fix — Correct log-likelihood normalization
- **File:** `usv_language/analysis/information_theory.py`
- **Change:** Replaced `_generalized_harmonic(best_alpha, int(tail.max()))` with range starting from `int(best_xmin)`
- **Why:** Normalization constant must match the fitted support `[xmin, xmax]`, not `[1, xmax]`

### W2 Fix — Add burstiness call to run_analysis.py Section 6
- **File:** `usv_language/analysis/run_analysis.py`
- **Change:** Added `conditional_entropy_by_lag`, `mutual_information_rate`, and `burstiness_coefficient` calls to Section 6, plus summary fields
- **Why:** Module docstring claimed integration of all functions; 4 were missing

### W3 Fix — IMPLEMENTATION_PROGRESS.md updated
- **File:** `IMPLEMENTATION_PROGRESS.md`
- **Change:** Added dated entry for information theory module

### S2 Fix — Documented one-sided test direction
- **File:** `usv_language/analysis/information_theory.py`
- **Change:** Added note to `ngram_idioms` docstring

---

## Post-Fix Test Results (Round 1)

```
usv_language/tests/test_information_theory.py: 16 passed (no --noconftest needed)
py_compile: all modified files pass
```

**Verdict after Round 1 fixes: PASS**

---

## Round 2 Review (2026-02-24)

**Reviewer:** Master Reviewer (claude-opus-4-6)
**Scope:** Re-review after Round 1 fixes, deep mathematical audit

### Round 2 Blockers Found

**B1r2. Burstiness in run_analysis.py always degenerate (CV=0, periodic)**
- `run_analysis.py:280-284` — `np.arange(N) * constant` produces equally-spaced times, so CV is always 0
- Function itself correct; bug in input construction

**B2r2. ZipfResult.alpha is count-distribution exponent, not rank-frequency exponent**
- MLE applied to frequency count values estimates count-distribution α, not rank-frequency α
- For α_rank=1.0, MLE returns ~2.087, leading to incorrect scientific interpretation
- `zipf_via_shannon_entropy` correctly estimates rank-frequency α, making the two incomparable

### Round 2 Warnings Found

**W1r2.** `convergence_order` off-by-one: `orders[i-1]` should be `orders[i-2]`
**W2r2.** No test verifies `convergence_order` value
**W3r2.** `test_zipf_mle_vs_entropy_agree` passes only by fixture coincidence (α=1.5)
**W4r2.** Poisson test bounds [0.7, 1.4] wider than interpretation boundary [0.8, 1.2]
**W5r2.** Bootstrap p-value uses truncated support range (`xmin + 2*n_tail` vs `max(freqs)`)

### Round 2 Fixes Applied

**B1r2 Fix — Burstiness uses code-change events**
- **File:** `usv_language/analysis/run_analysis.py:278-284`
- **Change:** Replaced `np.arange(len(codes)) * hop_duration_s` with `np.where(np.diff(codes) != 0)` change-event times
- **Why:** All-frame times are trivially uniform; code-change events capture actual temporal dynamics

**B2r2 Fix — ZipfResult.rank_alpha derived field + docstring overhaul**
- **File:** `usv_language/analysis/information_theory.py:45-90`
- **Change:** Added `rank_alpha` field to ZipfResult (auto-computed as `1/(alpha-1)` via `__post_init__`), updated docstring to explain count-distribution vs rank-frequency distinction, updated `run_analysis.py` printout and summary JSON to use `rank_alpha` for cross-method comparison
- **Why:** Researchers will compare against "Zipf α≈1"; exposing `rank_alpha` prevents misinterpretation

**W1r2 Fix — convergence_order off-by-one**
- **File:** `usv_language/analysis/information_theory.py:666`
- **Change:** `orders[i - 1]` → `orders[i - 2]`
- **Why:** When `consecutive=2` at index `i`, the two small deltas are at `(i-2,i-1)` and `(i-1,i)`; first order is `orders[i-2]`

**W2r2 Fix — Added test_convergence_order_value**
- **File:** `usv_language/tests/test_information_theory.py`
- **Change:** New test using uniform K=4, N=50,000; asserts `convergence_order == 1`
- **Why:** Verifies convergence detection works and catches the off-by-one regression

**W3r2 Fix — Zipf tests now compare rank_alpha**
- **File:** `usv_language/tests/test_information_theory.py`
- **Change:** `test_zipf_mle_alpha` now asserts `rank_alpha ≈ 1.5`; `test_zipf_mle_vs_entropy_agree` compares `rank_alpha` with `alpha_estimate`; both have explanatory docstrings
- **Why:** Tests now compare equivalent quantities (both rank-frequency exponents)

**W4r2 Fix — Poisson CV bounds match interpretation thresholds**
- **File:** `usv_language/tests/test_information_theory.py`
- **Change:** `0.7 < cv < 1.4` → `0.8 <= cv <= 1.2`
- **Why:** Matches the exact interpretation boundaries in `burstiness_coefficient`

**W5r2 Fix — Bootstrap support matches actual data range**
- **File:** `usv_language/analysis/information_theory.py:423`
- **Change:** `np.arange(best_xmin, best_xmin + len(tail) * 2)` → `np.arange(best_xmin, int(freqs.max()) + 1)`
- **Why:** Truncated support inflated p-values by preventing synthetic samples from reaching actual data extremes

### Round 2 Test Results

```
usv_language/tests/test_information_theory.py: 17 passed in 5.32s
py_compile: all 3 modified files pass
```

**Verdict after Round 2 fixes: PASS**
