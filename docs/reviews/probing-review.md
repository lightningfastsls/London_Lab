# Probing Framework Module Review

**Reviewed by:** Master Reviewer
**Date:** 2026-02-24
**Module:** `usv_language/analysis/probing.py`
**Tier:** 2 (ML correctness -- data leakage prevention, proper CV)
**Verdict:** APPROVED (all warnings fixed)

---

## Findings Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | 0 | -- |
| WARNING | 4 | W1-W4 (all fixed) |
| SUGGESTION | 7 | S1 (deferred), S2-S7 (fixed) |

## Warnings (all resolved)

- **W1**: No test exercised the `-1.0` sentinel filter in `_filter_labels` -- greenwashing risk.
  Fixed: Added `test_sentinel_filtered_before_regression` and `test_too_few_samples_graceful_return`.
- **W2**: `plot_probing_heatmap` and `plot_layer_comparison` not exported from `__init__.py`.
  Fixed: Added both to imports and `__all__`.
- **W3**: Module doc missing (`docs/modules/probing.md`).
  Fixed: Created `docs/modules/probing.md`.
- **W4**: Orientation heuristic in `run_probing.py` misclassified short bouts where T < n_freq.
  Fixed: Replaced `shape[0] < shape[1]` heuristic with `model.config.n_freq` matching.

## Suggestions

- **S1** (deferred): Probe selectivity (accuracy minus majority baseline) not implemented. Only affects interpretation on imbalanced real data; will implement when running on real recordings.
- **S2** (fixed): Frame-level probing strategy documented in `ProbingAnalysisPipeline.run()` docstring.
- **S3** (fixed): Late matplotlib imports moved to module top in `test_probing.py`.
- **S4** (fixed): Duplicate `import matplotlib.pyplot as plt` removed from `run_probing.py` main().
- **S5** (fixed): Dead `ConvergenceWarning` message-based filter removed (`probing.py:353-355`). Only the class-based filter actually suppresses the warning; the message filter matched the class name, not the warning text.
- **S6** (fixed): MLP probe end-to-end test added (`test_mlp_probe_regression`). Verifies MLP regression on perfectly linear data achieves R² > 0.8.
- **S7** (fixed): Three-class classification test added (`test_three_class_classification`). Verifies StratifiedKFold with 3 imbalanced classes (45/45/10 split) does not crash and achieves accuracy > 0.8.

## ML Correctness

- **StandardScaler inside Pipeline**: CONFIRMED. Scaler fits only on training folds (`probing.py:436-439`).
- **KFold/StratifiedKFold**: CONFIRMED. Regression uses KFold; classification uses StratifiedKFold (`probing.py:334-348`).
- **Sentinel filter**: CONFIRMED. `-1.0` and NaN/inf filtered before regression; classification labels bypass the filter (`probing.py:378-390`). Now covered by dedicated test.
- **Subsampling**: CONFIRMED. Uses seeded `RandomState` for deterministic behavior.
- **Label encoding**: CONFIRMED. `np.unique` produces deterministic integer encoding for string labels.

## Test Counts

- `test_probing.py`: 18 passed
- Full `usv_language/tests/`: 272 passed, 1 skipped, 0 failed

## Files Modified/Created

| File | Action |
|------|--------|
| `usv_language/analysis/probing.py` | CREATED (~465 lines) |
| `usv_language/tests/test_probing.py` | CREATED (18 tests) |
| `usv_language/scripts/run_probing.py` | CREATED (~250 lines) |
| `usv_language/analysis/__init__.py` | MODIFIED (+7 exports) |
| `docs/modules/probing.md` | CREATED |
| `docs/reviews/probing-review.md` | CREATED |
