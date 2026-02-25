# Event-Triggered Analysis (PETH) Module Review

**Reviewed by:** Master Reviewer (claude-sonnet-4-6)
**Date:** 2026-02-25
**Module:** event-triggered analysis (PETH)
**Tier:** 2 (Standard -- new cross-modal analysis module)
**Handoff:** `docs/reviews/event-triggered-handoff.md`

---

## Summary

The PETH module is well-structured and statistically sound for its primary use case. The circular-shift permutation test is implemented correctly. The conservative p-value formula is correct. The bootstrap CI resamples events (not USVs), which is the right choice. All 23 tests pass and all formulas were verified analytically.

No blockers were found. Four warnings and three suggestions were identified and all resolved.

**Test results:** 23/23 passed (module). 354 passed, 1 skipped (full suite excluding pre-existing notion_notes import errors).

---

## DSP Correctness

Not applicable in the traditional sense. This module operates in the time-domain (seconds, not samples), and contains no STFT, dB scaling, or frequency band math.

Statistical math verified analytically:

| Formula | Implementation | Verification |
|---------|---------------|--------------|
| Rate = counts / (n_events * actual_bin_width) | event_triggered.py:253 | CORRECT |
| p = (n_exceed + 1) / (n_perm + 1) | event_triggered.py:307 | CORRECT, avoids zero p-values |
| Circular shift: mod(times + shift, duration) | event_triggered.py:293 | CORRECT, preserves autocorrelation |
| Bootstrap CI: 2.5th/97.5th percentile | event_triggered.py:282 | CORRECT, standard percentile method |
| Baseline: n_usv / recording_duration | event_triggered.py:258 | CORRECT (when duration is accurate) |

---

## Spec Compliance

All files listed in the handoff exist and are consistent with the stated intent:

| File | Present | Matches Handoff |
|------|---------|-----------------|
| `src/usv_spectrogram/lmt/event_triggered.py` | YES | ~350 lines |
| `src/usv_spectrogram/lmt/__init__.py` | YES | +5 exports (PETHConfig, PETHResult, compute_peth, compute_all_peths, compare_populations) |
| `scripts/run_event_triggered_analysis.py` | YES | ~270 lines |
| `tests/test_event_triggered.py` | YES | 23 tests |
| `docs/modules/event-triggered-analysis.md` | YES | Updated with correct test count |

---

## Integration Correctness

**Pattern 1 (Config Dataclass):** `PETHConfig` is `frozen=True`, has `__post_init__` with full validation, and uses `_s` and `_hz` unit suffixes throughout. Fully compliant.

**Pattern 3 (Test Fixtures):** All tests use purely synthetic data (`rng.uniform`, arithmetic sequences). No real files, no real recordings. Fully compliant.

**Pattern 4 (Script CLI):** `scripts/run_event_triggered_analysis.py` uses `REPO_ROOT = Path(__file__).resolve().parents[1]` and `SRC_ROOT = REPO_ROOT / "src"` with the guard `if str(SRC_ROOT) not in sys.path`. Fully compliant.

**LMT integration:** The `group_events_by_type` function correctly reads `ev.start_time_s`, matching `BehavioralEvent` from the LMT Data Access Layer. CSV reader correctly reads `start_ms` column and converts to seconds.

---

## Findings & Resolution

### W1: CLI recording duration auto-detect (RESOLVED)
Added prominent warning when auto-detecting duration, explaining the risk to baseline rate and permutation test accuracy.

### W2: Missing exports from `__init__.py` (RESOLVED)
Added `compute_all_peths` and `compare_populations` to `__init__.py`.

### W3: Module doc reports 12 tests; actual is 23 (RESOLVED)
Updated module doc with correct count and comprehensive test list.

### W4: `plot_all_peths` return type annotation (RESOLVED)
Changed from `-> None` to `-> Optional[object]` with updated docstring.

### S1: Rate formula bin_size consistency (RESOLVED)
Rate computation now uses `actual_bin_width = (window_before + window_after) / n_bins` instead of `config.bin_size_s`, guaranteeing correctness regardless of bin_size divisibility.

### S2: `test_uniform_not_significant` fragility (RESOLVED)
Increased `n_permutations` from 99 to 499 (min p = 0.002) for robust non-significance testing.

### S3: CI coverage test strength (RESOLVED)
Strengthened from 80%-of-bins check to `np.all(ci_lo <= rate + eps)` with tolerance.

---

## Verdict

**APPROVED** (all warnings and suggestions resolved)

---

## Fix Log

| Item | Status | Fixed in | Date |
|------|--------|----------|------|
| W1 | RESOLVED | `scripts/run_event_triggered_analysis.py` | 2026-02-25 |
| W2 | RESOLVED | `src/usv_spectrogram/lmt/__init__.py` | 2026-02-25 |
| W3 | RESOLVED | `docs/modules/event-triggered-analysis.md` | 2026-02-25 |
| W4 | RESOLVED | `src/usv_spectrogram/lmt/event_triggered.py` | 2026-02-25 |
| S1 | RESOLVED | `src/usv_spectrogram/lmt/event_triggered.py` | 2026-02-25 |
| S2 | RESOLVED | `tests/test_event_triggered.py` | 2026-02-25 |
| S3 | RESOLVED | `tests/test_event_triggered.py` | 2026-02-25 |
