# Event-Triggered Analysis (PETH) — Review Handoff

**Review Tier:** 2 (Standard — new analysis module)
**Date:** 2026-02-24

## What Changed

New module: peri-event time histogram (PETH) computation bridging LMT
behavioral events with USV detections. First cross-modal analysis in the
pipeline.

## Files

| File | Action | Lines |
|------|--------|-------|
| `src/usv_spectrogram/lmt/event_triggered.py` | CREATE | ~350 |
| `src/usv_spectrogram/lmt/__init__.py` | MODIFY | +3 exports |
| `scripts/run_event_triggered_analysis.py` | CREATE | ~270 |
| `tests/test_event_triggered.py` | CREATE | ~310 |
| `docs/modules/event-triggered-analysis.md` | CREATE | Module doc |

## Review Focus Areas

### 1. Statistical Correctness
- [ ] Circular-shift permutation preserves USV autocorrelation
- [ ] Conservative p-value formula `(n_exceed + 1) / (n_perm + 1)`
- [ ] Bootstrap CI resamples events (not USVs) — captures between-event variability
- [ ] Baseline computation: `whole_recording` and `pre_event` methods

### 2. Numerical Edge Cases
- [ ] Empty USV list → all-zero rate
- [ ] Too few events → returns None
- [ ] Short recording (< window) → no crash
- [ ] Division by zero guarded in baseline and comparison

### 3. Integration
- [ ] Reads existing `start_ms` CSV column correctly
- [ ] Uses `BehavioralEvent.start_time_s` for onset times
- [ ] No new dependencies beyond numpy + matplotlib (lazy)

### 4. Pattern Compliance
- [ ] Frozen dataclass with `__post_init__` validation
- [ ] Tuples for immutable array fields
- [ ] Script follows Pattern 4 (CLI with path bootstrap)
- [ ] Tests use synthetic data, fixed seeds

## Test Results

```
pytest tests/test_event_triggered.py -v
# [paste results here after running]
```

## Key Design Decisions

1. **Tuples over ndarray** in PETHResult — JSON-serializable, frozen-compatible
2. **Binary search** (`np.searchsorted`) per event — O(E log U) complexity
3. **200 bootstrap resamples** — standard, balances accuracy vs speed
4. **Descriptive-only population comparison** — formal MWU needs multiple
   recordings per group (future extension documented in docstring)
