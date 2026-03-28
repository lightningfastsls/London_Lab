# FP Filter — Implementation Handoff

**Module:** 15.5 — Second-Stage False Positive Filter
**Date:** 2026-03-28
**Review Tier:** Tier 2 (standard)

## Summary

Implemented `FalsePositiveFilter` — a logistic regression classifier that takes 11 `EventFeatures` per hysteresis-detected event and predicts USV (True) vs false positive (False). Uses a StandardScaler + LogisticRegression pipeline with balanced class weights.

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `src/usv_spectrogram/postprocessing/fp_filter.py` | **NEW** | Core module: `FalsePositiveFilter` class |
| `scripts/train_fp_filter.py` | **NEW** | CLI training script with CV evaluation |
| `src/usv_spectrogram/postprocessing/__init__.py` | Modified | Added `FalsePositiveFilter` to exports |
| `tests/test_fp_filter.py` | Modified | Fixed spec error: added deterministic shuffle to CV test (data ordering bug) |

## Architecture Decisions

1. **Constant-label fallback**: When all training labels are the same class, sklearn's LogisticRegression raises ValueError. The filter detects this and falls back to a constant predictor with zero coefficients, satisfying the edge-case tests without crashing.

2. **Feature-to-array conversion**: Uses `dataclasses.astuple()` for O(1) field extraction per instance — field order matches `_FEATURE_NAMES` derived from `dataclasses.fields(EventFeatures)`, ensuring the feature importance dict always has the correct keys.

3. **Pickle serialization**: The entire `FalsePositiveFilter` instance (including pipeline + metadata) is pickled, so `load()` returns the correct type rather than a bare sklearn object.

4. **Recording-level CV in training script**: Cross-validation splits by recording stem (not by event) to avoid data leakage — events from the same recording share CNN inference context.

## Test Spec Error Fix

`test_cross_validated_f2_above_threshold` had an ordering bug: `_make_balanced_dataset()` creates 30 True then 30 False labels, and the sequential 5-fold split produced 2 folds with only negative validation samples (F2=0.0 by definition). Max achievable mean F2 was 0.6, below the 0.80 threshold. Fixed by adding a deterministic shuffle (`rng = np.random.default_rng(0)`) before splitting. Assertion threshold unchanged.

## Test Results

- Pre-existing tests from test-architect: 16
- Test spec fixes: 1 (CV data ordering)
- Tests after hardening: 39 total (16 pre-existing + 0 implementation + 23 hardener)
- Bugs found by hardener: 2 (empty-list predict/predict_proba on fitted filter — fixed)
- **All 39 pass**
- Full suite: 769 passed, 24 failed (pre-existing triage module failures), 5 skipped

## Dependencies

- Consumes: `EventFeatures` from `event_features.py` (module 15.4)
- Uses: `match_events_collar`, `compute_f_beta` from `event_scoring.py` (module 15.2) — in training script
- Uses: `hysteresis_detect`, `HysteresisConfig` from `hysteresis.py` (module 15.1) — in training script
- External: `sklearn` (LogisticRegression, StandardScaler, Pipeline)

## Known Limitations

- No regularization parameter tuning — uses `C=1.0` as default. Upgrade path: grid search over C if >1000 labeled events.
- Feature importance uses absolute logistic regression coefficients, not permutation importance. Sufficient for interpretability at this scale.
- Training script duplicates `RecordingInfo` and `load_recording_info` from `optimize_hysteresis.py` (scripts are self-contained by project convention).
