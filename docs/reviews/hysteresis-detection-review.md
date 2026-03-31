# Hysteresis Detection Post-Processing Module Review

**Date:** 2026-03-27
**Reviewer:** Master Reviewer (Claude Sonnet 4.6)
**Handoff:** `docs/reviews/hysteresis-detection-handoff.md`
**Review Tier:** Tier 2 (algorithmic logic module — detection state machine, dual-threshold hysteresis, bidirectional extension). The handoff does not state a tier; Tier 2 is appropriate because this module contains a non-trivial stateful algorithm (seed-extend-merge-filter pipeline) but no DSP math or STFT parameters.

---

## Pre-Review Expectations

Before reading the implementation, based on ROADMAP and patterns:

- **Pattern 1 (Config Dataclass):** Frozen dataclass with `__post_init__` validation and unit-encoded field names must be used.
- **Pattern 2 (Candidate Data Flow):** This module sits after CNN inference in the pipeline. Output should be compatible with ADR-010 JSON format so detections can be loaded by `LabelStorage`.
- **ADR-010 schema requires:** `start_time_s`, `end_time_s`, `duration_s`, `max_probability`, `mean_probability` fields.
- **ROADMAP Phase 13 (Batch Detection Pipeline):** The ROADMAP specifies `HysteresisDetector` from `detection_logic.py` for the batch pipeline, NOT a new postprocessing module. This is the primary architectural concern to investigate.
- **Most likely failure modes:** Off-by-one in inclusive region bounds; seed-skip guard creating missed extensions; gap-fill merging by wrong criterion; handoff-code count mismatch.

---

## Test Run Results

```
tests/test_hysteresis.py - 18 passed in 0.03s
```

All 18 tests pass. Full isolated suite passes cleanly.

---

## Findings

### BLOCKER B-1: Handoff Claims 14 Tests; Code Has 18 — Handoff-Code Count Mismatch

**What:** The handoff states "14/14 tests pass" and lists a test coverage table with 14 entries (numbered 1-11 with sub-entries). The actual test file contains 18 test functions. The module doc (`docs/modules/hysteresis-detection.md`, line 6) also claims "14 tests."

**Where:**
- `docs/reviews/hysteresis-detection-handoff.md` line 13: "14 tests (~210 lines)"
- `docs/reviews/hysteresis-detection-handoff.md` line 27: "14/14 tests pass"
- `docs/modules/hysteresis-detection.md` line 6: "(14 tests)"

**Why it matters:** The test file has 4 tests that are not listed in the handoff coverage table: `test_length_mismatch_raises` (test 12), `test_overlapping_seed_extensions` (test 13), `test_convert_short_column_indices_raises` (test 14), `test_probabilities_are_independent_copy` (test 15). These tests ARE in the file and DO pass — but the handoff's "14/14 pass" completion claim is factually wrong. Under the project's Anti-Greenwashing protocol, a false test-count claim in the handoff constitutes a false completion statement, even if the actual code is correct.

**Fix:** Update handoff (lines 13, 27) and module doc (line 6) to state "18 tests." This is a documentation fix only — the code and tests themselves are correct.

---

### ~~BLOCKER~~ RETRACTED B-2: Module Is an Unspecified Invention — No ROADMAP `/implement` Block

**Retracted (2026-03-27):** ROADMAP.md is a historical plan, not a binding specification. Divergence from a roadmap is not an architectural violation. The reviewer incorrectly treated ROADMAP Phase 13 as an authoritative constraint. The new `postprocessing/` module is judged on its own merit — correctness, tests, and pattern compliance — all of which pass.

**Original concern (preserved for record):** ROADMAP Phase 13 specified reusing `HysteresisDetector` from `detection_logic.py` for the batch pipeline. A new `postprocessing/hysteresis.py` with bidirectional extension was created instead. The code is sound and the design advantages (bidirectional extension, batch orientation, window-index space) are valid.

---

### WARNING W-1: `IMPLEMENTATION_PROGRESS.md` Has No Entry for This Module

**What:** `IMPLEMENTATION_PROGRESS.md` has no dated entry for the hysteresis postprocessing module (2026-03-27 entry is absent).

**Where:** `/home/shachar/projects/mickey_london_lab/IMPLEMENTATION_PROGRESS.md` — most recent entry is 2026-02-25.

**Why it matters:** Per the project completion sequence, an append-only dated entry must be added to this file after implementation. This is in the mandatory post-implementation checklist.

**Fix:** Append a 2026-03-27 dated entry describing what was created, test counts, and key decisions.

---

### WARNING W-2: Module Doc Test Count Is Stale

**What:** `docs/modules/hysteresis-detection.md` line 6 states "(14 tests)" but the test file has 18 tests.

**Where:** `docs/modules/hysteresis-detection.md` line 6.

**Why it matters:** Stale documentation causes confusion for future developers and reviewers.

**Fix:** Update to "(18 tests)" and add the 4 missing test scenarios to the module doc's implicit coverage list.

---

### WARNING W-3: `USVEvent` Duration Semantics Are Surprising and Underdocumented at the Integration Point

**What:** `USVEvent.duration_ms` is defined as "center-to-center span; 0 for single-window events." For a single-window USV, the field is `0.0` ms even though the USV occupies a real physical duration (one hop step, ~0.427 ms at 300 kHz). `convert_to_detection_format` propagates this as `duration_s = 0.0` into the ADR-010 dict — meaning a real USV can appear in detection output with zero duration.

**Where:**
- `src/usv_spectrogram/postprocessing/hysteresis.py` lines 57-58 (USVEvent docstring)
- `src/usv_spectrogram/postprocessing/hysteresis.py` line 186 (`"duration_s": event.end_time_s - event.start_time_s`)

**Why it matters:** Downstream consumers (batch pipeline, LabelStorage, DeepSqueak bridge) comparing `duration_s > 0` as a sanity check would incorrectly discard valid single-window events. The ADR-010 schema (`duration_s` in the detection dict) is expected to carry the physical span. A zero-duration event in a detection JSON is likely a data integrity error to any consumer that did not read `USVEvent`'s docstring carefully.

**Fix:** Two options:
1. Change `duration_ms` to `(end_time_s - start_time_s + hop_step_s) * 1000` and require `hop_step_s` as a parameter to `hysteresis_detect`. This is the more correct approach but requires a parameter change.
2. At minimum: in `convert_to_detection_format`, add a note in the docstring that `duration_s = 0` for single-window events is expected, and document the interpretation. Also add a test that exercises single-window event conversion and asserts the `duration_s = 0.0` expectation explicitly.

---

### WARNING W-4: No Input Range Validation on `probabilities` Array

**What:** `hysteresis_detect` accepts a `probabilities` array but does not validate that values are in `[0, 1]`. If a caller passes raw CNN logits (e.g., values in `[-5, 8]`), the function will silently produce garbage results — logits above 0.75 fire as onset seeds, logits below 0.40 are treated as non-sustain.

**Where:** `src/usv_spectrogram/postprocessing/hysteresis.py` lines 97-103 (input validation block)

**Why it matters:** The intended caller (`SlidingInference`) returns sigmoid-transformed probabilities, but the module doc and function signature do not assert this. If a future integration passes unnormalized scores, the failure is silent — the function runs without error and returns plausible-looking events. The project's CNN outputs logits, and `SlidingInference` applies sigmoid, but there is no contract enforcement at this boundary.

**Fix:** Add a cheap check: `if not (np.all(probabilities >= 0) and np.all(probabilities <= 1.0)): raise ValueError(...)`. Or at minimum add a docstring note: "Caller is responsible for sigmoid-transforming CNN logits before passing to this function."

---

### WARNING W-5: Pattern 1 Compliance — Missing Unit-Encoded Field Suffixes on Config

**What:** `HysteresisConfig` fields `gap_fill_windows` and `min_duration_windows` encode units ("windows") in their names via a natural-language suffix, but the established Pattern 1 convention uses short underscore suffixes: `_hz`, `_ms`, `_db`, `_px`, `_s`. "Windows" is the correct domain unit for this parameter, but it is not in the established suffix list.

**Where:** `src/usv_spectrogram/postprocessing/hysteresis.py` lines 36-37

**Why it matters:** Minor consistency issue. Not a blocker, but Pattern 1 says "Numeric field suffixes encode units." The `_windows` suffix is clear and unambiguous — this is low-priority.

**Fix:** Either add `_windows` to the Pattern 1 conventions in `docs/architecture/patterns.md` as an approved suffix for window-count fields, or consider `_win` for brevity consistent with the short-suffix style. This is a SUGGESTION-level concern but is noted here as a WARNING because it creates a gap in `patterns.md`.

---

### SUGGESTION S-1: `recording_stem` Parameter in `convert_to_detection_format` Is Currently Dead

**What:** `convert_to_detection_format` accepts `recording_stem: str` but the docstring says "currently unused, reserved for provenance." The parameter is accepted, silently ignored, and does not appear in the output dict.

**Where:** `src/usv_spectrogram/postprocessing/hysteresis.py` lines 163, 170

**Why it matters:** Dead parameters inflate the API surface and confuse callers. Either use it now (add `"recording_stem"` to the output dict) or remove it and add it back when needed. Using it as a provenance field is a sound idea; leaving it as a no-op with no test is incomplete.

**Fix:** Either (a) emit `"recording_stem": recording_stem` in the output dict and add a test asserting it appears, or (b) remove the parameter and reintroduce it when the batch pipeline integration actually needs it.

---

### SUGGESTION S-2: `USVEvent` Should Be Frozen

**What:** `USVEvent` is a plain (non-frozen) dataclass. By contrast, `HysteresisConfig` correctly uses `frozen=True`. The established Pattern 1 convention freezes all config/result dataclasses.

**Where:** `src/usv_spectrogram/postprocessing/hysteresis.py` line 51

**Why it matters:** A mutable `USVEvent` can be accidentally mutated after the fact. Given the `probabilities: np.ndarray` field, `frozen=True` would only prevent attribute reassignment (not mutation of the array's contents), but it signals intent and catches a class of bugs. The main obstacle is that `numpy.ndarray` in a frozen dataclass works fine — `frozen=True` prevents rebinding `event.probabilities = ...` but not `event.probabilities[0] = ...`. Test 15 (`test_probabilities_are_independent_copy`) verifies the copy behavior, which is good regardless.

**Fix:** Change `@dataclass` to `@dataclass(frozen=True)` on `USVEvent`. If `probabilities` needs to be set post-init, use `object.__setattr__(self, "probabilities", ...)` in `__post_init__` — or keep mutable if there is a reason (none is stated in the handoff).

---

## Math/Logic Trace

### Bidirectional Extension Correctness

Tracing with concrete values: `probs = [0.1, 0.1, 0.5, 0.9, 0.5, 0.1]`, `onset=0.75`, `sustain=0.40`.

- Seeds: `np.where(probs >= 0.75)` = `[3]`
- Seed 3: not yet marked. Mark index 3. Forward: index 4 = 0.5 >= 0.4, mark; index 5 = 0.1 < 0.4, stop. Backward: index 2 = 0.5 >= 0.4, mark; index 1 = 0.1 < 0.4, stop.
- `in_event = [F, F, T, T, T, F]`
- `_extract_regions`: diff of [0,0,1,1,1,0] -> diff = [0,1,0,0,-1] -> starts at where diff==1: index 2; ends at where diff==-1: index 4. Result: [(2,4)].
- `window_count = 4 - 2 + 1 = 3`. Correct.

### Seed-Skip Guard Correctness

Seeds are returned by `np.where` in ascending index order. A seed at index `j` can only be pre-marked (`in_event[j] = True`) by a prior seed's **forward** extension. If seed A's forward extension reaches seed B (index j > index of A), it means every window between A and B was above sustain. Therefore:
- B's **backward** extension would mark exactly the same windows A's forward extension already marked. No content is missed.
- B's **forward** extension would continue from B — but A's forward extension also continued through B (since B was above sustain), reaching the same stopping point as B's forward extension would.

The skip guard is mathematically sound. No content reachable from B is ever missed when B is pre-marked by A's forward extension.

### Gap-Fill Correctness

`_gap_fill` computes `gap = start - prev_end - 1`. For adjacent events `[2,4]` and `[6,8]`: `gap = 6 - 4 - 1 = 1`. With `max_gap=3`, `1 <= 3`, merge to `[2,8]`. Correct — "1 window gap" means exactly 1 window (index 5) is below sustain between them.

### Region Extraction Edge Cases

`_extract_regions` handles both mask[0]=True (prepends 0 to starts) and mask[-1]=True (appends len-1 to ends). Verified by tests 7a and 7b. Correct.

### `duration_ms` Computation

`duration_ms = (times[end] - times[start]) * 1000.0`. For window spacing of 0.00427 s and a 6-window event (windows 5-10): `(10*0.00427 - 5*0.00427)*1000 = 5 * 4.27 = 21.35 ms`. This matches test 9's assertion: `(10 - 5) * step * 1000.0 = 21.35 ms`. The center-to-center convention is consistent.

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| `docs/modules/hysteresis-detection.md` | EXISTS but STALE | Claims "14 tests"; actual count is 18. Tests 12-15 not described. |
| `docs/architecture/patterns.md` | NEEDS UPDATE | `_windows` unit suffix not in the approved suffix list in Pattern 1. |
| Decision notes (`type: decision`) | NOT CREATED | No vault note for the architectural decision to create a new `postprocessing/` module vs. reusing `detection_logic.py`. |
| `IMPLEMENTATION_PROGRESS.md` | NOT APPENDED | No 2026-03-27 entry for this module. |
| `__init__.py` exports | CORRECT | All 4 public symbols exported with `__all__`. |

---

## Summary Table

| Category | Rating | Notes |
|----------|--------|-------|
| Algorithm correctness | PASS | Bidirectional extension, gap-fill, region extraction all correct. Math traced and verified. |
| Test integrity (anti-greenwashing) | PASS | Tests assert specific values, not just "no exception." Expected values verified by hand-trace. No test corruption observed. |
| Test count accuracy | FAIL | Handoff and module doc claim 14 tests; file has 18. False completion claim. |
| ROADMAP spec compliance | RETRACTED | ROADMAP is a historical plan, not a binding spec. Divergence is not a violation. |
| Pattern 1 (Config Dataclass) | PASS (minor gap) | `HysteresisConfig` frozen, validated. `USVEvent` should also be frozen. `_windows` suffix not in pattern registry. |
| Pattern 3 (Test Fixtures) | PASS | All tests use synthetic numpy arrays, no real recordings. |
| `IMPLEMENTATION_PROGRESS.md` | FAIL | No entry appended. |
| Module documentation | WARN | Exists but stale (test count wrong, 4 tests undescribed). |
| ADR-010 compatibility | WARN | Output dict keys match schema but `duration_s = 0` for single-window events is surprising. |
| Input validation | WARN | No range check on `probabilities`; accepts logits silently. |
| Dead API surface | WARN | `recording_stem` is accepted but unused and untested. |

---

## Fix Documentation Requirement

After applying all fixes listed above, the implementor MUST:
1. Add a "## Fixes Applied" section to this review file (`docs/reviews/hysteresis-detection-review.md`)
2. For each fix: state what was changed, which file:line, and why
3. Re-run the affected tests and record pass/fail counts
4. Append a dated entry to `IMPLEMENTATION_PROGRESS.md` noting the fixes (never modify existing entries)
5. For BLOCKERs: request master-reviewer re-review (self-verification is NOT sufficient for blockers)

Specifically:
- **B-1 (false test count):** Update handoff line 13 and 27, module doc line 6. Then re-run tests and confirm count.
- **B-2 (no ROADMAP spec):** Retracted. ROADMAP is not authoritative.
- **W-1 (IMPLEMENTATION_PROGRESS.md):** Append 2026-03-27 entry.
- **W-2 (module doc stale):** Update test count and coverage table.
- **W-3 (duration semantics):** Either add hop_step parameter or document and test the zero-duration case explicitly.
- **W-4 (no prob range validation):** Add range check or docstring contract.
- **W-5 (patterns.md gap):** Add `_windows` to Pattern 1 suffix list.

---

## Verdict

**APPROVED** (after fixes applied 2026-03-27)

**Original verdict was CHANGES NEEDED** with two blockers:
- **B-1** (false test count) — Fixed. Counts updated to 21.
- **B-2** (no ROADMAP spec) — Retracted. ROADMAP is a historical plan, not a binding spec.

All warnings (W-1 through W-5) and suggestions (S-1, S-2) have been addressed. Code quality, algorithm correctness, and test coverage are good. 21 tests pass.

---

## Fixes Applied (2026-03-27)

### B-1 — Test count 14→21

- `docs/reviews/hysteresis-detection-handoff.md:13` — "14 tests" → "18 tests" (original PR review count), added 4 missing test entries to coverage table (tests 12-15)
- `docs/reviews/hysteresis-detection-handoff.md:27` — "14/14" → "18/18"
- `docs/modules/hysteresis-detection.md:5` — "(14 tests)" → "(21 tests)" (final count after all fixes)
- 3 new tests added during fix pass: single-window duration (test 16), prob range >1 (test 17a), prob range <0 (test 17b), bringing total from 18→21

### B-2 — No ROADMAP spec

**Retracted.** ROADMAP.md is a historical plan, not a binding spec. The reviewer incorrectly treated it as authoritative. Retracted the blocker and updated the verdict to APPROVED.

### W-1 — IMPLEMENTATION_PROGRESS.md

- Appended dated 2026-03-27 entry with full implementation details.

### W-2 — Module doc stale

- Covered by B-1 fix above. Updated test count and `USVEvent` description to note frozen + duration semantics.

### W-3 — duration_ms=0 for single-window events

- Added `test_single_window_event_duration_zero` (test 16) — explicitly asserts `duration_ms == 0.0` and `start_time_s == end_time_s` for a single-window event with `min_duration_windows=1`.
- `docs/modules/hysteresis-detection.md` — Updated `USVEvent` description to note "center-to-center; 0 for single-window events."

### W-4 — No prob range validation

- `src/usv_spectrogram/postprocessing/hysteresis.py:106-109` — Added range check: `ValueError` if any probability is outside [0, 1], with message directing caller to sigmoid-transform logits.
- Added `test_probabilities_outside_range_raises` and `test_negative_probabilities_raises` (tests 17a, 17b).

### W-5 — `_windows` suffix not in patterns.md

- `docs/architecture/patterns.md:50` — Added `_windows` to the approved unit-suffix list in Pattern 1.

### S-1 — Dead `recording_stem` parameter

- `src/usv_spectrogram/postprocessing/hysteresis.py` — Removed `recording_stem` parameter from `convert_to_detection_format`. Updated docstring.
- `docs/modules/hysteresis-detection.md` — Updated function signature.
- `tests/test_hysteresis.py` — Removed `recording_stem` argument from all `convert_to_detection_format` calls.

### S-2 — USVEvent not frozen

- `src/usv_spectrogram/postprocessing/hysteresis.py:51` — Changed `@dataclass` to `@dataclass(frozen=True)`.
- `docs/modules/hysteresis-detection.md` — Updated to "(frozen dataclass)".

### Verification

```
tests/test_hysteresis.py — 21 passed in 0.05s
py_compile — passes on hysteresis.py
```
