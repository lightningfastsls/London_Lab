# Phase 10.1: Active Learning Cycle Runner — Module Review

**Reviewed by:** Master Reviewer (Sonnet 4.6)
**Date:** 2026-02-21
**Module:** `src/usv_spectrogram/training/` + `scripts/run_training_cycle.py`
**Tier:** 2 (Standard Implementation Review)
**Handoff:** `docs/reviews/training-cycle-handoff.md`

---

## Summary

The Active Learning Cycle Runner is an orchestration module that chains 7 pipeline steps (assemble, train, evaluate, optimize threshold, mine hard negatives, compare, report) into a single reproducible CLI. The implementation is clean, well-structured, and the test suite covers the unit-testable components thoroughly.

All 27 module tests pass. Full suite: 461 passed, 0 regressions.

No DSP correctness issues (this module orchestrates existing DSP code, not DSP itself). One blocker was found related to architecture mismatch between trained models and the detection app for non-default model sizes.

---

## BLOCKER

### B1. Medium/large model checkpoints incompatible with detection app

**What:** `sliding_inference.py:104` instantiates `USVClassifierCNN()` with no arguments, hardcoding default filters `[32, 64, 128]` and `dense_units=64`. The training cycle now supports `--model-size medium` (filters `[64, 128, 256]`) and `--model-size large` (filters `[128, 256, 512]`). If a medium or large model is trained, the detection app will crash at load time with a `RuntimeError: size mismatch` from `model.load_state_dict()`.

**Where:** `src/usv_spectrogram/app/core/sliding_inference.py:104`

**Why it matters:** The ROADMAP Phase 10.1 exit criterion explicitly states "Output model loadable by the detection app (run_app.py)." This criterion only passes for `--model-size small`. At milestone 2 (5K labels), the spec says to switch to medium. The detection app would silently break at that milestone unless fixed now.

**Fix:** Save architecture metadata in the checkpoint and restore it on load (Option A preferred, implemented below).

---

## WARNINGS

### W1. Path bootstrap deviates from Pattern 8

**Where:** `scripts/run_training_cycle.py:44`

**Fix:** Replace with Pattern 8 form: guarded insert with `.resolve()`.

### W2. Comparison evaluates previous model on current-cycle test data

**Where:** `scripts/run_training_cycle.py` step_compare

**Fix:** Add caveat note in generated report's Model Comparison section.

### W3. `step_optimize_threshold` inserts into `sys.path` unconditionally

**Where:** `scripts/run_training_cycle.py` step_optimize_threshold

**Fix:** Apply guard pattern.

### W4. ROADMAP test plan item 1 is entirely unimplemented

**Where:** `tests/test_training_cycle.py` — absent integration test

**Fix:** Add integration test with mocked step functions.

### W5. `CycleMetrics` is not a frozen dataclass (deviates from Pattern 1)

**Where:** `src/usv_spectrogram/training/cycle_report.py:17`

**Mitigation:** Documented and justified. Acceptable for orchestration state.

---

## SUGGESTIONS

- S1: Surface `label_count` approximation in report table ("~" prefix)
- S2: Remove unused return value from `step_train`
- S3: Create module doc `docs/modules/training-cycle.md`
- S4: Add "Recommended Next Steps" section to report

---

## Fixes Applied

### B1 Fix: Architecture metadata in checkpoints

**Files changed:**
- `src/usv_spectrogram/models/trainer.py:225-232` — Added `num_filters` and `dense_units` to checkpoint dict
- `scripts/run_training_cycle.py:191-200` — `load_model_with_architecture` reads metadata from checkpoint with fallback

**Approach:** Option A from review — save architecture in checkpoint at training time. Detection app and other loaders can read it with fallback to defaults for backward compatibility.

### W1 Fix: Pattern 8 path bootstrap

**File:** `scripts/run_training_cycle.py:43-46` — Replaced with guarded `.resolve()` form.

### W2 Fix: Comparison caveat in report

**File:** `src/usv_spectrogram/training/cycle_report.py` — Added note in Model Comparison section.

### W3 Fix: Guarded sys.path insert for optimize_threshold

**File:** `scripts/run_training_cycle.py` step_optimize_threshold — Added guard.

### W4 Fix: Integration test for step ordering

**File:** `tests/test_training_cycle.py` — Added `TestMainStepOrder` with mocked step functions.

### S1 Fix: Label count approximation surfaced

**File:** `src/usv_spectrogram/training/cycle_report.py` — Changed to "Unique labels (approx)" with ~ prefix.

### S2 Fix: step_train return removed

**File:** `scripts/run_training_cycle.py` — `step_train` now returns None.

### S4 Fix: Recommended next steps section

**File:** `src/usv_spectrogram/training/cycle_report.py` — Added heuristic next steps section.

---

## Verification After Fixes

- [x] `py_compile` on all modified files
- [x] `pytest tests/test_training_cycle.py -v` — all pass
- [x] `pytest tests/ -q` — full suite passes, 0 regressions
