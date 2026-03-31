# Event Scoring & Hysteresis Parameter Optimization Module Review

**Date:** 2026-03-28
**Reviewer:** Master Reviewer (Claude Opus 4.6)
**Handoff:** `docs/reviews/event-scoring-hysteresis-optimization-handoff.md`
**Review Tier:** 3 (algorithmic module: greedy matching, F-beta scoring, cross-validated grid search, 1SE rule)

---

## Pre-Review Expectations

Before reading implementation, based on ROADMAP Phase 15.2 and established patterns:

- **Config Dataclass Pattern (Pattern 1):** `EventScoringConfig` must be a frozen dataclass with `__post_init__` validation. Spec includes `min_iou: float = 0.0`.
- **Script CLI Pattern (Pattern 4):** `optimize_hysteresis.py` must have path bootstrap, separate `parse_args()`, and `main()` returning int exit code.
- **ROADMAP grid ranges:** onset=[0.60..0.95], sustain=[0.20..0.50], gap=[0..5], min_dur=[3..10].
- **1SE rule:** Standard form is `threshold = best_mean - std/sqrt(k)`.
- **Data leakage risk:** Splits must be by recording. Stratified CV on `has_usvs`.

---

## Test Run Results

```
tests/test_event_scoring.py  -- 14 passed in 0.03s
tests/test_hysteresis.py     -- 21 passed (pre-existing)
tests/test_dataset_assembler.py -- 10 passed (pre-existing)
Total: 45 passed, 0 failed
```

Note: handoff claims 15 for test_hysteresis and 13 for test_dataset_assembler (total 42). Actual total is 45.

---

## Findings

### BLOCKER

**B-1: `_find_one_se_params` simplicity ordering inverts the sustain parameter direction**

- **What:** The simplicity key uses `(onset, -sustain, -gap, min_dur)` sorted descending. The negative sign on sustain means the function prefers *lower* sustain thresholds. But lower sustain = more permissive (events continue at lower probabilities). The comment claims "most conservative" but returns the most permissive sustain.
- **Where:** `scripts/optimize_hysteresis.py` ~lines 408-416
- **Why it matters:** When multiple param combos are within the 1SE band, tie-breaking selects the most permissive sustain. This silently corrupts the parameter selection result.
- **Math trace:** Given candidates A (sustain=0.20) and B (sustain=0.50) with equal onset/gap/min_dur: simplicity_A = (x, -0.20, ...) vs simplicity_B = (x, -0.50, ...). Sorted descending, A wins because -0.20 > -0.50. But A is more permissive. Bug confirmed.
- **Fix:** Change `-params["sustain_threshold"]` to `+params["sustain_threshold"]`:

```python
simplicity = (
    params["onset_threshold"],        # higher = more conservative (keep +)
    params["sustain_threshold"],       # higher = more conservative (fix: remove -)
    -params["gap_fill_windows"],       # lower = more conservative (keep -)
    params["min_duration_windows"],    # higher = more conservative (keep +)
)
```

---

### WARNING

**W-1: Grid ranges deviate from ROADMAP spec**

- **What:** Onset starts at 0.50 (spec: 0.60). `min_duration_windows` includes 1 and 2 (spec: 3..10), drops 7 and 9.
- **Where:** `scripts/optimize_hysteresis.py` ~lines 227-231 (`build_grid()`)
- **Why it matters:** Onset=0.50 seeds events from any window with >50% probability -- more aggressive than designed. min_dur=1 allows single-window events outside the 5-350ms target range. Handoff mentions onset shift but not min_dur changes.
- **Fix:** Either restore spec values or add `# DECISION:` comments explaining the deviation and update the plan file.

**W-2: Missing test -- "Grid search finds known-optimal params on synthetic data" (plan item 8)**

- **What:** No integration test exercises the grid search + 1SE selection end-to-end. A synthetic test would have caught B-1.
- **Where:** `tests/test_event_scoring.py` (absent)
- **Fix:** Add a test with known-optimal params that verifies `_find_one_se_params` returns the most conservative valid option.

**W-3: Missing test -- "One detection spanning two GT events" (plan item 5)**

- **What:** Plan specifies: "One detection spanning two GT events -> 1 TP + 1 FN." Existing `test_one_detection_two_gts` tests temporally separated GTs, not a detection physically spanning both.
- **Where:** `tests/test_event_scoring.py` ~line 99
- **Fix:** Add `test_one_detection_spans_two_adjacent_gts` with e.g. `GT=[(1.0, 1.5), (1.6, 2.0)]`, `det=[(1.0, 2.0)]`, asserting `(1, 0, 1)`.

**W-4: `main()` violates Pattern 4 (return int exit code)**

- **What:** `main()` returns `None`, entry point lacks `sys.exit()`.
- **Where:** `scripts/optimize_hysteresis.py` ~lines 473, 529
- **Fix:** `def main() -> int:`, add `return 0`, change entry to `sys.exit(main())`.

**W-5: `EventScoringConfig` missing `min_iou` field from spec**

- **What:** Spec includes `min_iou: float = 0.0`. Implementation omits it entirely.
- **Where:** `src/usv_spectrogram/postprocessing/event_scoring.py` ~line 24
- **Fix:** Add the field with a docstring note that collar matching ignores it, or document the deliberate exclusion with a `# DECISION:` comment.

**W-6: Handoff test counts are stale**

- **What:** Claims test_hysteresis=15 and test_dataset_assembler=13 (total 42). Actual: 21 and 10 (total 45).
- **Where:** `docs/reviews/event-scoring-hysteresis-optimization-handoff.md` lines 33-37
- **Fix:** Update the counts.

**W-7: No `docs/modules/event-scoring.md` created**

- **What:** New public module with exported API but no module doc. Handoff defers to "after optimization run" but the API is stable now.
- **Fix:** Create `docs/modules/event-scoring.md` documenting `EventScoringConfig`, `match_events_collar`, `compute_f_beta`. Performance results section can be TBD.

**W-8: Uses raw std but calls it "1SE rule" -- misleading terminology**

- **What:** Standard 1SE = std/sqrt(k). Implementation uses raw std (~2.24x wider for k=5). Code comment says "1SE rule" without noting the deviation.
- **Where:** `scripts/optimize_hysteresis.py` ~lines 342-345
- **Fix:** Rename to "1SD rule" or add inline comment: `# Intentionally uses raw std (wider than SEM) -- see design rationale`.

---

## Math/Logic Verification (Tier 3)

**F-beta formula:** TP=8, FP=2, FN=1, beta=2.0 -> beta_sq=4.0, numerator=5*8=40, denominator=40+4*1+2=46, result=40/46~=0.8696. Matches test. **Correct.**

**Greedy matching score:** Onset closeness used as tie-breaker when overlap is zero. Offset dimension not included (documented limitation in handoff). **Correct.**

**Micro-averaging:** Accumulates TP/FP/FN across all val recordings before computing F2. Noise recordings contribute (0,0,0) = no penalty. **Correct.**

**Cross-validation:** Only `val_idx` recordings evaluated. No train-time leakage. **No leakage detected.**

---

## Spec Compliance Summary

| Requirement | Status |
|-------------|--------|
| EventScoringConfig dataclass (frozen) | PASS |
| match_events_collar with collar/onset/offset/overlap | PASS |
| compute_f_beta (beta=2.0 default) | PASS |
| Grid: onset=[0.60..0.95] | FAIL (W-1) |
| Grid: min_dur=[3..10] | FAIL (W-1) |
| 5-fold stratified CV on has_usvs | PASS |
| Micro-averaged F2 | PASS |
| Inference caching with .npz | PASS |
| 1SE rule for conservative selection | FAIL -- sustain inverted (B-1), raw std not SEM (W-8) |
| Output JSON with fold-level scores | PASS |
| min_iou field in EventScoringConfig | MISSING (W-5) |
| All 8 test plan items | PARTIAL -- items 5 and 8 missing (W-2, W-3) |
| main() returns int + sys.exit() | FAIL (W-4) |

---

## Verdict

**CHANGES NEEDED**

**Blocker (1):** B-1 must be fixed before merge. The sustain direction inversion silently selects the most permissive threshold when it should select the most conservative.

**Warnings (8):** W-1 through W-8. Highest priority: W-2 (integration test that would catch B-1), W-1 (grid spec deviation), W-7 (missing module doc).

Core algorithmic logic (matching, F-beta, micro-averaging, caching, CV) is correct and well-tested. B-1 is a targeted fix. After resolving B-1 and creating the integration test (W-2), this module is close to approvable.

**Re-review rule:** B-1 fix must be verified by master reviewer, not self-reported.

---

## Fixes Applied (2026-03-28)

### B-1: Sustain direction inversion (BLOCKER → FIXED)
- **File:** `scripts/optimize_hysteresis.py` `_find_one_se_params()`
- **Change:** `−params["sustain_threshold"]` → `+params["sustain_threshold"]` in simplicity key
- **Why:** Higher sustain = harder to extend events = more conservative. Negating it selected the most permissive threshold.
- **Verification:** New test `test_one_sd_conservative_selection` confirms correct direction and that old direction would pick wrong answer.

### W-1: Grid ranges restored to ROADMAP spec (WARNING → FIXED)
- **File:** `scripts/optimize_hysteresis.py` `build_grid()`
- **Change:** onset=[0.60..0.95] (was 0.50..0.85), min_dur=[3..10] (was [1,2,3,4,5,6,8,10])
- **Why:** Onset=0.50 too aggressive; min_dur=1 allows single-window events outside plausible USV range.

### W-2: Missing integration test for 1SD selection (WARNING → FIXED)
- **File:** `tests/test_event_scoring.py` — added `test_one_sd_conservative_selection`
- **Why:** Exercises the conservative selection logic end-to-end. Also confirms the buggy direction would produce wrong answer.

### W-3: Missing test for spanning detection (WARNING → FIXED)
- **File:** `tests/test_event_scoring.py` — added `test_one_detection_spans_two_adjacent_gts`
- **Why:** Plan item 5 required: one detection spanning two GTs → 1 TP + 1 FN.

### W-4: main() return type + sys.exit() (WARNING → FIXED)
- **File:** `scripts/optimize_hysteresis.py`
- **Change:** `def main() -> int:`, added `return 0`, changed entry to `sys.exit(main())`

### W-5: Missing min_iou field (WARNING → FIXED)
- **File:** `src/usv_spectrogram/postprocessing/event_scoring.py`
- **Change:** Added `min_iou: float = 0.0` with docstring noting it's reserved for future IoU mode.

### W-6: Stale handoff test counts (WARNING → FIXED)
- **File:** `docs/reviews/event-scoring-hysteresis-optimization-handoff.md`
- **Change:** Updated to 16 + 21 + 10 = 47 (was 14 + 15 + 13 = 42).

### W-7: Missing module doc (WARNING → FIXED)
- **File:** `docs/modules/event-scoring.md` (NEW)
- **Content:** API reference, design decisions, usage.

### W-8: Misleading "1SE" terminology (WARNING → FIXED)
- **File:** `scripts/optimize_hysteresis.py`
- **Change:** Renamed comments/prints to "1SD rule" with inline explanation that raw std (not SEM) is intentional.

### Post-fix test results
```
tests/test_event_scoring.py: 16 passed
tests/test_hysteresis.py: 21 passed
tests/test_dataset_assembler.py: 10 passed
Total: 47 passed, 0 failed
py_compile: all files pass
```
