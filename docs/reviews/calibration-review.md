# Temperature Scaling Calibration Module Review

**Module:** Temperature Scaling Calibration (ROADMAP_POST_PROCESSING.md §15.3)
**Review Tier:** 2 (standard)
**Date:** 2026-03-27
**Reviewer:** Senior Technical Reviewer (independent session)

---

## Pre-Review Expectations

Before reading the implementation, based on ROADMAP §15.3 and patterns.md:

- **Required files:** `calibration.py`, `__init__.py` (modified), `sliding_inference.py` (modified), `calibrate_temperature.py` (CLI), `tests/test_calibration.py`, `docs/modules/calibration.md`
- **Pattern adherence:** TemperatureScaler should follow the Config Dataclass Pattern (Pattern 1), but ROADMAP explicitly shows it as non-frozen (it mutates during `fit()`). This deviation must be documented.
- **Expected invariants:** Calibration must not touch the test split. Val split (29 recordings) is the only source for fitting T. L-BFGS-B with bounds `[0.01, 50.0]` prevents division-by-zero.
- **Likely failure modes:** (1) Energy-skipped windows producing inconsistent logit/probability values; (2) Normalization mismatch between calibration CLI and live inference; (3) ROADMAP test plan coverage gaps; (4) Missing IMPLEMENTATION_PROGRESS.md entry.

---

## Test Results

```
pytest tests/test_calibration.py tests/test_hysteresis.py tests/test_dataset_assembler.py -v
40 passed, 0 failed
```

The handoff claims "37 passed" — actual count is 40. The 3-count discrepancy is accounted for by tests added to `test_hysteresis.py` and `test_dataset_assembler.py` after the handoff was written. The calibration module's 9 tests all pass, and no regressions were introduced.

Full suite (excluding pre-existing `anthropic`-import collection errors): 479 passed, 5 failed. The 5 failures are in `test_deepsqueak_import.py` due to a missing `openpyxl` dependency — pre-existing, unrelated.

---

## Math Trace

### NLL Formula Verification

The `_binary_nll` function implements:
```
nll = max(z, 0) + log1p(exp(-|z|)) - y * z
```

This is the numerically stable form of `log(1 + exp(z)) - y*z`, provably equivalent to binary cross-entropy:
- If `y=1`: `log(1+exp(z)) - z = log(1+exp(−z)) = −log σ(z)` ✓
- If `y=0`: `log(1+exp(z)) − 0 = −log(1−σ(z))` ✓

### ECE Formula Verification

Equal-width bins with first-bin special case for `prob == 0.0`. No double-counting — first bin covers `[0.0, 1/15]`, second covers `(1/15, 2/15]`.

`test_ece_poorly_calibrated` analytically correct: all probs in one bin, `ECE = |0.5 − 0.9| = 0.4 > 0.3` ✓

### CNN `forward()` Output Shape

`USVClassifierCNN.forward()` returns `(N, 1)`. The `squeeze(dim=1)` correctly collapses to `(N,)`, consistent with `evaluate.py:56`.

---

## BLOCKER

None.

---

## WARNINGS

### W-1. Missing ROADMAP test: `return_logits=True` shape verification

**File:** `tests/test_calibration.py`
**Problem:** ROADMAP §15.3 test plan item 7 — "InferenceResult.logits has correct shape when return_logits=True" — is not implemented. The `squeeze(dim=1)` path is a latent shape regression risk.
**Fix:** Add a test that creates a mock CNN, calls `SlidingInference.infer(..., return_logits=True)`, and asserts `result.logits.shape == result.probabilities.shape`.

### W-2. Skipped windows: `logits=0.0` vs `probabilities=0.0` inconsistency

**File:** `src/usv_spectrogram/app/core/sliding_inference.py:177-178`
**Problem:** Energy-skipped windows have `probabilities=0.0` but `logits=0.0`, which maps to `sigmoid(0)=0.5` after calibration. Inconsistent for any caller that calibrates logits directly.
**Fix:** Initialize `all_logits` with a large negative sentinel (e.g., `-20.0`) so `sigmoid(-20/T) ≈ 0.0`, matching the probability convention.

### W-3. Missing IMPLEMENTATION_PROGRESS.md entry

**File:** `IMPLEMENTATION_PROGRESS.md`
**Problem:** No entry for Phase 15.3. The append-only log is the project's implementation archive.
**Fix:** Append a dated entry for Phase 15.3 before marking implementation complete.

### W-4. Default temperature=1.0 deviates from ROADMAP spec (1.5)

**File:** `src/usv_spectrogram/postprocessing/calibration.py:28`
**Problem:** ROADMAP specifies `temperature: float = 1.5` as default. Implementation uses `1.0`. The `x0=[1.0]` in `fit()` is hardcoded, ignoring the field.
**Fix:** Either update default to 1.5 to match spec, or update `x0=[self.temperature]` so the field controls the starting point. Document the choice.

### W-5. No shape validation in `fit()`

**File:** `src/usv_spectrogram/postprocessing/calibration.py:39`
**Problem:** `fit(logits, labels)` with mismatched shapes silently broadcasts. `fit(np.array([1.0, 2.0]), np.array([1.0]))` returns `T=0.01` without error.
**Fix:** Add `if logits.shape != labels.shape: raise ValueError(...)` at top of `fit()`.

---

## SUGGESTIONS

| # | Issue | File | Fix |
|---|-------|------|-----|
| S-1 | `calibrate()` on unfitted scaler silently returns raw sigmoid | `calibration.py:63` | Add `warnings.warn()` if `not self.fitted` |
| S-2 | `compute_ece` doesn't validate `n_bins > 0` | `calibration.py:98` | Add `if n_bins < 1: raise ValueError(...)` |
| S-3 | CLI fits on saved PNGs, inference uses MAD-normalized spectrograms — different normalization paths | `calibrate_temperature.py` vs `sliding_inference.py` | Validate ECE on live inference logits after fitting |

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | 0 | — |
| WARNING | 5 | W-1, W-2, W-3, W-4, W-5 |
| SUGGESTION | 3 | S-1, S-2, S-3 |

---

## Verdict

**CHANGES NEEDED**

Five warnings require resolution. No blockers — self-verification of fixes is acceptable for this Tier 2 module.

Priority order:
1. W-5 (shape validation — silent wrong results)
2. W-2 (logit sentinel — blocks §15.6)
3. W-1 (missing test)
4. W-4 (spec alignment)
5. W-3 (progress log)

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| `docs/modules/calibration.md` | EXISTS — accurate | Matches public API. NLL formula documented. |
| `docs/architecture/patterns.md` | UP TO DATE | No new pattern. Non-frozen mutable dataclass acknowledged in handoff. |
| `IMPLEMENTATION_PROGRESS.md` | APPENDED | W-3 resolved. |

---

## Fix Log

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| W-1 | FIXED | `tests/test_calibration.py` | 2026-03-28 | Added `test_logits_shape_matches_probabilities` |
| W-2 | FIXED | `sliding_inference.py:178` | 2026-03-28 | Changed `np.zeros` to `np.full(..., -20.0)` for skipped-window logits |
| W-3 | FIXED | `IMPLEMENTATION_PROGRESS.md` | 2026-03-28 | Appended Phase 15.3 entry |
| W-4 | FIXED | `calibration.py:28,53` | 2026-03-28 | Default temperature=1.5, x0 uses self.temperature |
| W-5 | FIXED | `calibration.py:49-53` | 2026-03-28 | Added shape validation + `test_fit_rejects_shape_mismatch` test |
