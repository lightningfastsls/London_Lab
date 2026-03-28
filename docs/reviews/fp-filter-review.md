# False Positive Filter Module Review (15.5)

**Date:** 2026-03-28
**Review Tier:** Tier 2 (Standard)
**Reviewer:** master-reviewer (fresh session)
**Handoff:** `docs/reviews/fp-filter-handoff.md`

## Verdict: CHANGES NEEDED

Two warnings require fixes before module is complete.

## Findings

### WARNING 1: `predict([])` crashes on a fitted (non-constant) filter

`_features_to_array([])` produces shape `(0,)` instead of `(0, 11)`, causing sklearn's `StandardScaler.transform` to fail. Realistic scenario — some recordings may have zero detected events.

**Fix:** Add empty-list guard in `_features_to_array`.

### WARNING 2: `_label_individual_events` uses onset-only matching

Training script labels events by onset proximity only, but `match_events_collar` uses onset OR offset OR overlap. Events matching by offset/overlap are incorrectly labeled as FP in training data.

**Fix:** Mirror the three-condition check from `match_events_collar`.

### SUGGESTION 1: `predict_proba` column ordering comment

Add `# sklearn orders classes_ = [False, True] for bool labels` to clarify the implicit convention.

### SUGGESTION 2: CV fold count guard

Add a guard for when `len(stems) < n_folds` in the training script.

## Fixes Applied

1. **WARNING 1 — empty-list guard**: Added `if not features: return np.empty((0, len(_FEATURE_NAMES)))` in `_features_to_array()` (`fp_filter.py:34-35`). `predict([])` and `predict_proba([])` now return empty list / empty (0,2) array.

2. **WARNING 2 — three-condition matching**: Replaced onset-only check in `_label_individual_events()` with onset collar OR offset collar OR temporal overlap (`train_fp_filter.py:311-315`). Mirrors `match_events_collar` logic.

3. **SUGGESTION 1 — column ordering comment**: Added `sklearn orders classes_ = [False, True] for bool labels` to `predict_proba` docstring (`fp_filter.py:116`).

4. **SUGGESTION 2 — CV fold guard**: Added `if len(stems) < n_folds` check that auto-reduces fold count with a warning (`train_fp_filter.py:340-345`).

**Post-fix tests:** 16/16 pass. Both files compile clean.
