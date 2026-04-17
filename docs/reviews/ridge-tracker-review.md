# DP-Based Ridge Tracker (17.3) Module Review

**Reviewer:** master-reviewer (Tier 3)
**Date:** 2026-04-17
**Module:** `src/usv_spectrogram/features/ridge_tracker.py`
**Review Tier:** 3 (critical — upstream of 17.4 iMSA and 17.5 Oren vectorization)
**Handoff:** `docs/reviews/ridge-tracker-handoff.md`
**Spec:** `ROADMAP_SIS_BENCHMARK.md` §17.3 (lines 208–296)

---

## Pre-Review Expectations

Before reading the code, based on the ROADMAP and ADR constraints:

- **DSP parameters:** This module receives `freqs_hz` from the caller; ADR-001/002 constants (sr=300000, n_fft=512, hop=128) must appear only in callers and in the test fixture, not in this module. The test file should build `freqs_hz` from `np.fft.rfftfreq(512, 1/300_000)` giving 257 bins and ~585.9 Hz/bin.
- **DP objective:** Must implement `score = Σ magnitude[f_t, t] − λ·Σ|f_t − f_{t-1}|` with a hard `|Δf| ≤ W` constraint. Back-trace must recover the MAP sequence, not an approximation.
- **Silent runs:** Each contiguous non-silent interval must be solved independently. NaN output must be consistent between `fm_hz` and `am`.
- **Pattern 1:** `RidgeConfig` must be a frozen dataclass with `__post_init__` validation.
- **Likely failure modes:** Off-by-one in back-trace indexing; wrong direction of shift (source vs destination); invalid slice bounds at boundary bins; docstring-code contradictions on zero-penalty semantics.

---

## Test Results

```
tests/test_ridge_tracker.py — 14/14 PASSED in 0.14s
```

Handoff claims 14/14. Confirmed. No test modifications were made (zero modification markers found, no test anti-greenwashing). Adjacent module tests also pass: 143/143 across `test_ridge_tracker`, `test_spectrogram_filter`, `test_sis_baselines`, `test_energy_detector`, `test_config`.

---

## DP Correctness Analysis (Hand Trace)

### Forward pass shift direction

The key DP recurrence is: for bin `f` at column `t`, choose the source bin `g = f + shift` at column `t-1` that maximizes `cur_cost[g] − λ·|shift|`.

The code computes, for each `shift` in `[-W, +W]`:

```python
f_lo = max(0, -shift)
f_hi = min(n_bins, n_bins - shift)
candidate = cur_cost[f_lo + shift : f_hi + shift] - penalty * abs(shift)
```

Traced for all shifts:

- `shift = +2`: `f_lo=0, f_hi=n-2`; slice `cur_cost[2:n]`; destination bins `[0, n-2)` receive costs from source bins `[2, n)`. For bin `f`, source `g = f+2`. Correct.
- `shift = -2`: `f_lo=2, f_hi=n`; slice `cur_cost[0:n-2]`; destination bins `[2, n)` receive costs from source bins `[0, n-2)`. For bin `f`, source `g = f-2`. Correct.
- `shift = 0`: full range, source = destination. Correct.

The "stay-put" transition (`shift=0`) covers every bin in `[0, n_bins)` on every column. This means `best[f]` can never remain at `-inf` for any bin — every bin is reachable from itself — so there are no unreachable-bin pathological states.

**Verdict: DP forward pass is correct.** The slice reads from source bins `g = f + shift`, the cost is `cur_cost[g] − λ·|shift|`, and `best_src[f] = f + shift = g` is the correct source to record.

### Bounds verification

For each shift:
- `f_lo = max(0, -shift)`: the smallest `f` for which `g = f+shift ≥ 0`.
- `f_hi = min(n_bins, n_bins - shift)`: the largest `f` (exclusive) for which `g = f+shift < n_bins`.

Algebraically verified for both positive and negative shifts — all in-bounds. When `max_jump_bins > n_bins`, large-magnitude shifts produce `f_lo ≥ f_hi` and are skipped via the `continue` guard. No array out-of-bounds possible.

### Back-trace correctness

Allocation: `backtrace = np.zeros((n_bins, run_len))`.

The forward loop stores `backtrace[:, local_t] = best_src` for `local_t in [1, run_len)`. Convention: `backtrace[f, local_t]` = the source bin `g` at column `r_start + local_t − 1` from which the MAP path arrived at bin `f` at column `r_start + local_t`.

Reverse walk traced for `run_len=5` (`r_start=0`, `r_end=5`):

| Step | local_t | Reads | Writes |
|------|---------|-------|--------|
| Init | — | `argmax(cur_cost)` → `end_bin` | `ridge_idx[4]` ← end_bin (column 4) |
| 1 | 4 | `backtrace[end_bin, 4]` = source at col 3 | `ridge_idx[3]` ← new end_bin |
| 2 | 3 | `backtrace[end_bin, 3]` = source at col 2 | `ridge_idx[2]` ← new end_bin |
| 3 | 2 | `backtrace[end_bin, 2]` = source at col 1 | `ridge_idx[1]` ← new end_bin |
| 4 | 1 | `backtrace[end_bin, 1]` = source at col 0 | `ridge_idx[0]` ← new end_bin |

Index formula: `ridge_idx[r_start + local_t − 1]`. At `local_t=1`, writes `ridge_idx[0]`, which is column `r_start + 0 = r_start`. Complete MAP path recovered, no off-by-one.

**Verified by concrete 3-column hand-computation:** a 5-bin spectrogram with deliberately chosen magnitudes at bins 2, 4, and 3 across 3 columns produces MAP path `[2, 4, 3]` exactly matching hand-calculated scores. The code returned `[2.0, 4.0, 3.0]`.

**Verdict: Back-trace is correct.**

### In-place mutation via numpy view

Lines 175–184 use `slice_best = best[f_lo:f_hi]` as a view, then mutate via `slice_best[improved] = candidate[improved]`, then redundantly reassign `best[f_lo:f_hi] = slice_best`. The redundant reassignment is harmless — numpy contiguous slices are always views, so the mutation propagates regardless. No correctness issue.

---

## Silent-Run Segmentation

`_non_silent_runs` returns half-open `[start, end)` intervals. All boundary cases verified:

| Pattern | Expected | Actual |
|---|---|---|
| Leading silent `[T,T,F,F,F,T]` | `[(2,5)]` | `[(2,5)]` |
| Trailing silent `[F,F,F,T,T]` | `[(0,3)]` | `[(0,3)]` |
| All silent | `[]` | `[]` |
| Single non-silent `[T,F,T]` | `[(1,2)]` | `[(1,2)]` |
| Alternating `[F,T,F,T,F]` | `[(0,1),(2,3),(4,5)]` | `[(0,1),(2,3),(4,5)]` |
| No silent | `[(0,3)]` | `[(0,3)]` |
| Trailing active `[T,F,F]` | `[(1,3)]` | `[(1,3)]` |

All edge cases pass. The trailing-run guard (`if start is not None: runs.append((start, n))`) correctly handles runs that reach end-of-array without a trailing silent.

---

## DSP Correctness vs ADRs

This module correctly receives `freqs_hz` from the caller and contains no hardcoded sample-rate-dependent constants. The test file constructs `freqs_hz = np.fft.rfftfreq(512, d=1.0/300_000)` (257 bins, ~585.9 Hz/bin), matching ADR-001 and ADR-002 exactly. The module docstring usage example uses `scipy.signal.stft(..., fs=300_000, nperseg=512, noverlap=384)` — `noverlap = 512 − 128 = 384` is correct, and `scipy.signal.stft` defaults to `window='hann'` (matches ADR-002).

No ADR violations.

---

## Pattern Conformance (Pattern 1: Frozen Dataclass)

`RidgeConfig` uses `@dataclass(frozen=True)`, has three fields with typed defaults, validates in `__post_init__`, and uses unit-suffixed field names (`_bins`, threshold without suffix — acceptable since it is a magnitude level, not a unit-conveying quantity). This matches the `FilterConfig` style from 17.2 and the broader Pattern 1 convention.

---

## Edge Case: `transition_penalty = 0`

The code docstring (line 31) states: "A value of 0 reduces the tracker to per-column argmax." This is inaccurate. With `penalty=0` and a finite `max_jump_bins` window, the tracker still enforces the hard window constraint — it is only equivalent to per-column argmax when `max_jump_bins ≥ n_bins`. The handoff (line 74) correctly qualifies this: "per-column argmax with a hard window." The module doc table (line 49) repeats the imprecise language from the source docstring.

This is a documentation-only issue (the DP itself is correct), but it is a concrete wrong claim in the source code docstring and the module doc that could mislead downstream developers using `penalty=0` for debugging.

---

## Findings

### WARNING W1 — Misleading docstring: `transition_penalty = 0` claim

**What:** The `RidgeConfig.transition_penalty` docstring (line 31 of `ridge_tracker.py`) says "A value of 0 reduces the tracker to per-column argmax." This is false when `max_jump_bins < n_bins` (i.e., in all practical use cases with the default `max_jump_bins=10`). With `penalty=0` and `max_jump_bins=10`, the tracker is still window-constrained: bins more than 10 bins away from the current position at `t-1` are structurally unreachable at `t`. The same incorrect claim appears in the module doc table (`docs/modules/ridge-tracker.md`, line 49).

**Where:** `src/usv_spectrogram/features/ridge_tracker.py` line 31; `docs/modules/ridge-tracker.md` line 49.

**Why it matters:** The `penalty=0` mode is explicitly described as a "useful degenerate mode for debugging comparisons against the naive baseline." If a developer uses `penalty=0` expecting it to match per-column argmax (as stated) they will see different results and either suspect a bug where none exists, or trust a comparison baseline that is not what they intended.

**Fix:** Change both occurrences from "reduces the tracker to per-column argmax" to "reduces the tracker to windowed-argmax (per-column argmax subject to the `max_jump_bins` hard constraint; set `max_jump_bins = n_bins` to get true per-column argmax)."

---

### WARNING W2 — Test file header miscounts ROADMAP test functions

**What:** The `test_ridge_tracker.py` module docstring (line 36) states "Total: 13 tests (9 from ROADMAP, 4 additional)." The actual count is 14 test functions (confirmed by `grep`). ROADMAP item 7 maps to two functions (`test_ridgeconfig_rejects_negative_transition_penalty` and `test_ridgeconfig_rejects_zero_max_jump_bins`), giving 10 ROADMAP-spec functions, not 9. Total: 10 + 4 = 14. The `IMPLEMENTATION_PROGRESS.md` entry correctly states "14 (10 ROADMAP + 4 additional)." The `docs/modules/ridge-tracker.md` exit criteria also miscounts: "13 from ROADMAP spec + 1 additional = 14."

**Where:** `tests/test_ridge_tracker.py` line 36; `docs/modules/ridge-tracker.md` lines 180–181.

**Why it matters:** Low severity (tests all pass, counts are cosmetic), but the inconsistency between the test file header, the module doc, and the progress tracker could cause confusion when test-hardener is adding tests and tracking coverage.

**Fix:** Update `tests/test_ridge_tracker.py` line 36 to "Total: 14 tests (10 from ROADMAP, 4 additional)." Update `docs/modules/ridge-tracker.md` exit criterion to "14 tests pass (10 from ROADMAP spec + 4 additional from test-architect)."

---

### SUGGESTION S1 — Redundant `best[f_lo:f_hi] = slice_best` assignment in hot loop

**What:** In `_track_run`, `slice_best = best[f_lo:f_hi]` is a numpy view. Mutating `slice_best[improved] = ...` already propagates to `best` in-place. The subsequent `best[f_lo:f_hi] = slice_best` (line 179) is a no-op.

**Where:** `src/usv_spectrogram/features/ridge_tracker.py` lines 175–179.

**Why it matters:** No correctness impact. Very minor: the redundant write could mislead a reader into thinking a copy was needed, and it adds a small amount of work in the hot loop (though numpy likely optimizes self-assignment).

**Fix:** Remove line 179 (`best[f_lo:f_hi] = slice_best`) and add a comment: `# mutation via view propagates to best directly`.

---

### SUGGESTION S2 — Usage example omits explicit `window='hann'` in `scipy.signal.stft` call

**What:** The usage example in `docs/modules/ridge-tracker.md` calls `stft(audio, fs=sample_rate, nperseg=512, noverlap=384)` without specifying `window='hann'`. `scipy.signal.stft` defaults to `window='hann'`, so the result is correct, but making the ADR-002 Hann window explicit would make the example more educational.

**Where:** `docs/modules/ridge-tracker.md` line 119.

**Why it matters:** Cosmetic. Documentation example should be self-evident about the window choice per ADR-002.

**Fix:** Change to `stft(audio, fs=sample_rate, window='hann', nperseg=512, noverlap=384)`.

---

## Test-Hardener Recommendations

The following cases are not covered by the current 14 tests and represent plausible failure modes for a buggy implementation. These are non-blocking recommendations for test-hardener:

1. **Output dtype is float64 regardless of input dtype.** A `float32` input spectrogram should still produce `float64` `fm_hz` and `am` (guaranteed by `np.full(n_cols, np.nan, dtype=float)` which defaults to float64). One-line test: `assert fm_hz.dtype == np.float64`.

2. **Two or more consecutive silent columns in the interior.** The current test (`test_silent_column_produces_nan_neighbors_intact`) exercises one silent column. A test with columns `[active, silent, silent, active]` verifies that run segmentation correctly produces two independent runs of length 1 each, and that both single-column runs use the `run_len == 1` argmax seed path.

3. **`max_jump_bins = 1` (minimum allowed).** Tests the tightest legal constraint. With `max_jump_bins=1`, a 2-bin shift is impossible — test that a sweep changing by 1 bin per column is tracked, while a sweep changing by 2+ bins per column is not followed.

4. **`transition_penalty = 0` with `max_jump_bins = n_bins`.** Verifies the documented degenerate mode truly equals per-column argmax when the window is unconstrained. This directly addresses the W1 docstring fix and provides a regression anchor for the corrected behavior description.

5. **`freqs_hz` is non-monotonic or has repeated values.** The module docstring does not specify monotonicity requirements on `freqs_hz`. While the DP logic is bin-index based and doesn't rely on frequency order, a test documenting the behavior (fm_hz values would be scrambled but no crash) would clarify the interface.

6. **All-active run of length exactly 2.** The `run_len == 1` special case is exercised by `test_single_column_spectrogram_shape_and_value`. The `run_len == 2` case exercises the forward pass at `local_t=1` then immediately back-traces. Already confirmed working manually; an explicit test would serve as regression anchor.

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| `docs/modules/ridge-tracker.md` | EXISTS | W2: test count incorrect (says "13 from ROADMAP spec + 1 additional"); S2: usage example omits `window='hann'` |
| `docs/architecture/patterns.md` | UP TO DATE | No new patterns established; Pattern 1 already covers frozen dataclass |
| `IMPLEMENTATION_PROGRESS.md` | APPENDED | Dated entry for 17.3 present and accurate (correctly states 14 = 10 ROADMAP + 4 additional) |
| `features/__init__.py` exports | PRESENT AND CORRECT | `RidgeConfig`, `track_ridge` in both imports and `__all__` |
| Decision notes (`type: decision`) | NO NEW NOTE NEEDED | Key decisions documented in module doc and handoff; no architectural novelty requiring a vault note beyond the existing ridge-extraction note |

---

## Verdict

**APPROVED**

The DP-Based Ridge Tracker passes all checks for a Tier 3 critical module:

- **DP correctness:** The forward pass correctly implements `best[f] = max_{shift} (cur_cost[f+shift] − λ·|shift|)` with correct bounds. Hand-traced with concrete values and verified by running the code on a constructed 3-column case with known MAP solution.
- **Back-trace correctness:** The reverse walk recovers the full MAP path with no off-by-one errors. Traced for `run_len=5` explicitly.
- **Bounds:** All `f_lo`/`f_hi` calculations are mathematically verified to keep both destination bin `f` and source bin `g = f+shift` within `[0, n_bins)`. The `w > n_bins` edge case is gracefully skipped via `f_lo >= f_hi` guard.
- **Silent-run segmentation:** All seven boundary patterns produce correct `[start, end)` intervals.
- **ADR compliance:** No hardcoded DSP constants; test file uses correct ADR-001/002 values.
- **Pattern 1:** `RidgeConfig` is a correctly formed frozen dataclass.
- **Tests:** 14/14 pass; no test expectations were modified; handoff test count claim is accurate.

There are two WARNINGs (a misleading docstring about `penalty=0` semantics and an off-by-one in test count documentation) and two SUGGESTIONs. None affect the runtime behavior of the module. Both WARNINGs are documentation-only issues that should be fixed before test-hardener runs to avoid test-hardener writing tests against the incorrect description.

---

## Fixes Applied (2026-04-17)

All four findings (W1, W2, S1, S2) addressed before test-hardener. No re-review
required — WARNINGs are documentation-only, none affect runtime behavior.

- **W1 (WARNING)** — `RidgeConfig.transition_penalty` docstring
  (`src/usv_spectrogram/features/ridge_tracker.py` ~line 31) rewritten to
  say "reduces the tracker to windowed-argmax (per-column argmax subject
  to the `max_jump_bins` hard constraint). True per-column argmax requires
  `penalty = 0` AND `max_jump_bins >= n_bins`." Same correction applied to
  the module doc table (`docs/modules/ridge-tracker.md` line 49).
- **W2 (WARNING)** — Test file header docstring
  (`tests/test_ridge_tracker.py` line 36) updated: "Total: 14 tests (10
  from ROADMAP, 4 additional)". Module doc exit criterion
  (`docs/modules/ridge-tracker.md`) updated: "14 tests pass (10 from
  ROADMAP spec + 4 additional from test-architect)". This is a docstring-
  only edit, no test assertions touched — does not violate the anti-
  greenwashing protocol (test expectations are unchanged).
- **S1 (SUGGESTION)** — Removed the redundant `best[f_lo:f_hi] = slice_best`
  write in `_track_run`. Added a `# view into best — mutations propagate`
  comment on the `slice_best = best[f_lo:f_hi]` line to document intent.
- **S2 (SUGGESTION)** — Usage example in `docs/modules/ridge-tracker.md`
  updated to pass `window="hann"` explicitly to `scipy.signal.stft`,
  matching ADR-002 intent.

**Verification after fixes:**
- `py_compile src/usv_spectrogram/features/ridge_tracker.py` — OK
- `pytest tests/test_ridge_tracker.py` — 14/14 pass in 0.15 s (no
  regressions from S1's dead-write removal)
- No assertion logic modified; no behavioral change to the tracker.

**Cleared for test-hardener.**
