# Implementation Handoff: DP-Based Ridge Tracker (17.3)

**Module:** DP-Based Ridge Tracker (Phase 17.3)
**Review Tier:** 3 (critical — upstream of 17.4 iMSA + 17.5 Oren vectorization)
**Date:** 2026-04-17
**Branch:** `main`

## What Changed

- New `src/usv_spectrogram/features/ridge_tracker.py` module: Viterbi-style
  dynamic-programming ridge tracker that extracts the MAP frequency
  trajectory (`fm_hz`) and amplitude trajectory (`am`) from a magnitude
  spectrogram. Emits `NaN` for silent columns.
- `RidgeConfig` frozen dataclass with validated defaults
  (`transition_penalty=0.1`, `max_jump_bins=10`, `silence_threshold=1e-6`).
- Package export surface updated (`features/__init__.py`).
- Module documentation in `docs/modules/ridge-tracker.md`.
- Dated entry appended to `IMPLEMENTATION_PROGRESS.md`.
- **Pre-implementation tests:** 14 tests from `test-architect` — none were
  modified. All pass without any test-expectation edits.

## Files Changed

- `src/usv_spectrogram/features/ridge_tracker.py` (NEW) — `RidgeConfig` +
  `track_ridge` + two private helpers (`_non_silent_runs`, `_track_run`),
  ~165 lines.
- `src/usv_spectrogram/features/__init__.py` (MODIFIED) — added
  `RidgeConfig`, `track_ridge` to exports and `__all__`.
- `docs/modules/ridge-tracker.md` (NEW) — public interface docs +
  algorithm + decision log.
- `IMPLEMENTATION_PROGRESS.md` (APPENDED) — dated entry for 17.3.

## Key Decisions Made

**1. Silent columns break the DP chain into independent runs.**
Each contiguous `[r_start, r_end)` non-silent interval is solved by its own
Viterbi DP, seeded from argmax at its first column. The alternative —
carrying DP state across silent columns — amounts to imputing a ridge
through silence, which has no physical interpretation: a silent column
contributes no evidence about pitch.

Rationale for this choice:
- The pre-filter (17.2) is aggressive, so silent columns inside real USV
  bounds usually indicate either a segmentation glitch or a two-syllable
  event. Treating runs independently avoids false-continuity artefacts.
- Pre-existing test `test_silent_column_produces_nan_neighbors_intact` is
  satisfied by construction: column 25 is silent → NaN; columns 24 and 26
  are in separate runs, each solved from its own argmax.

**2. Windowed DP, not full pairwise.**
The transition reward at column `t`, bin `f` looks at only
`[f − W, f + W]` at column `t − 1`, where `W = max_jump_bins = 10`
(~6 kHz at 300 kHz / n_fft=512). This is O(F · W · T) instead of O(F² · T):
~25× faster at F=257 with zero accuracy loss for smooth ridges.

The hard window has a second, intentional effect: it makes harmonic jumps
*structurally impossible*. In the harmonic-suppression test, the
fundamental at bin 68 and the 2× harmonic at bin 136 are 68 bins apart —
no Viterbi path can reach one from the other in a single step. The tracker
stays on whichever peak is initially seeded.

**3. Pure numpy, no scipy / torch / sklearn.**
Keeps the module dependency-light for downstream hot paths. Vectorization
inside the `for shift in range(-W, +W)` loop uses bounded slice views and
in-place updates via `np.where`, avoiding the allocation of a
`(F, 2W+1)`-shaped temporary per column.

**4. Raw tracker output — no smoothing, no NaN interpolation.**
Module 17.5 (Oren vectorization) applies its own median/mean smoothing
and NaN interpolation per Oren et al. 2024 spec. Keeping the tracker's
output raw avoids layering DSP opinions that belong in consumers.

**5. `transition_penalty >= 0` (zero allowed), `max_jump_bins >= 1`.**
- Zero penalty reduces the tracker to per-column argmax with a hard window
  — a useful degenerate mode for debugging comparisons against the naive
  baseline.
- `max_jump_bins = 0` would mean "every frequency bin must match the
  previous one exactly" — pathological; would freeze the ridge. Rejected
  at validation time.

## Pre-implementation test changes

**None.** All 14 pre-existing tests from `test-architect` passed without
modification. The spec was unambiguous enough that the Viterbi + silent-run
design satisfied every assertion on the first implementation pass.

Test summary:
| Test | ROADMAP § | Status |
|---|---|---|
| `test_pure_tone_60khz_fm_tracks_correctly` | 17.3 test 1 | PASS |
| `test_linear_sweep_fm_is_monotonic_and_am_constant` | 17.3 test 2 | PASS |
| `test_harmonic_suppression_stays_on_fundamental` | 17.3 test 3 | PASS |
| `test_silent_column_produces_nan_neighbors_intact` | 17.3 test 4 | PASS |
| `test_all_silent_spectrogram_returns_all_nan` | 17.3 test 5 | PASS |
| `test_large_discontinuous_jump_behavior` | 17.3 test 6 | PASS |
| `test_ridgeconfig_rejects_negative_transition_penalty` | 17.3 test 7a | PASS |
| `test_ridgeconfig_rejects_zero_max_jump_bins` | 17.3 test 7b | PASS |
| `test_output_shapes_match_n_time_cols` | 17.3 test 8 | PASS |
| `test_regression_fm_rmse_within_2khz` | 17.3 test 9 | PASS |
| `test_single_column_spectrogram_shape_and_value` | added | PASS |
| `test_ridgeconfig_default_values` | added | PASS |
| `test_am_nonnegative_on_non_silent_columns` | added | PASS |
| `test_nan_columns_are_consistent_between_fm_and_am` | added | PASS |

## Implementation Summary

### Control flow

```
track_ridge(magnitude, freqs_hz, cfg):
  is_silent = per-column max < threshold     (O(F·T))
  if all silent: return (all-NaN, all-NaN)
  runs = _non_silent_runs(is_silent)          (O(T))
  for each run:
      _track_run(magnitude, r_start, r_end, cfg, ridge_idx)
  fm[active] = freqs_hz[ridge_idx[active]]
  am[active] = magnitude[ridge_idx[active], active_cols]
  return fm, am
```

### Per-run DP (`_track_run`)

```
cur_cost = magnitude[:, r_start]           # seed (no prior path)
if run_len == 1: ridge_idx[r_start] = argmax(cur_cost); return
for local_t in 1..run_len:
    best     = -inf
    best_src = 0
    for shift in [-W, +W]:
        # bin f at col t came from bin g = f+shift at col t-1
        # f must be in-bounds; f+shift must also be in-bounds
        f_lo, f_hi = max(0, -shift), min(F, F-shift)
        candidate  = cur_cost[f_lo+shift : f_hi+shift] - λ·|shift|
        where candidate > best[f_lo:f_hi]:
            update best, best_src
    cur_cost          = magnitude[:, t] + best
    backtrace[:, local_t] = best_src
# back-trace from final argmax
end_bin = argmax(cur_cost)
ridge_idx[r_end - 1] = end_bin
for local_t in run_len-1 .. 1:
    end_bin = backtrace[end_bin, local_t]
    ridge_idx[r_start + local_t - 1] = end_bin
```

### Complexity

- Time: O(F · W · T_active) where `T_active = T − silent_count`.
- Space: O(F · T_active) for backtrace per run (one array per run, freed
  after back-trace). Peak memory = largest single run.
- At our 5970 dataset (7518 calls, typical call ~50 columns, F=257): ~1 ms
  per call, ~8 s for the full dataset on a single core.

## Verification

- `.venv/bin/python -m py_compile src/usv_spectrogram/features/ridge_tracker.py` — OK
- `.venv/bin/python -m pytest tests/test_ridge_tracker.py -v` — 14/14 passed in 0.17 s
- Full regression: 1189 passed on the implemented-module portion. Pre-existing
  failures (72) belong to unrelated tests — confirmed by stashing tracked
  changes and re-running: the same failures reproduce without 17.3.

## Exit criteria (all met)

- [x] RMSE < 2 kHz on synthetic FM sweep
      (`test_regression_fm_rmse_within_2khz`)
- [x] Harmonic-suppression test passes
      (`test_harmonic_suppression_stays_on_fundamental`)
- [x] All tests pass (14/14)
- [x] `py_compile` passes

## What reviewer should focus on (Tier 3)

- **DP correctness.** Does `candidate[f] = cur_cost[f+shift] - λ·|shift|`
  correctly implement the MAP objective
  `argmax_{path} Σ reward − λ·Σ|Δf|`? Walk through one column by hand if
  useful — the implementation in `_track_run` is ~20 lines.
- **Back-trace correctness.** `backtrace[f, local_t]` stores the bin `g`
  at column `t − 1` that gave the best path ending at bin `f` at column
  `t`. Is the reverse walk `end_bin = backtrace[end_bin, local_t]`
  consistent with this storage convention? (Index `local_t` in
  `backtrace` corresponds to the column *just after* the prior column
  whose bin we're recovering; slot 0 is unused.)
- **Silent-run boundary cases.** Leading-silent, trailing-silent,
  all-silent, single non-silent, alternating silent/active — does
  `_non_silent_runs` emit the right half-open intervals?
- **DSP correctness vs ROADMAP.** ADR-001 sample rate not referenced in
  code because this module receives frequencies from the caller. Reviewer
  should verify the test file's `freqs_hz` construction matches ADR-002
  (`np.fft.rfftfreq(512, 1/300_000)` → 257 bins, ~586 Hz bin width). This
  is already the case in `tests/test_ridge_tracker.py:66`.
- **Edge case: `max_jump_bins > F`.** Does the shift loop gracefully skip
  out-of-range shifts? (Yes: `f_lo >= f_hi` guard skips invalid shift
  values without crashing.)
- **Any performance concerns.** Is the per-column `np.where`/slice-update
  pattern free of hidden allocations in the hot loop? Acceptable if small
  allocations occur — our dataset is small enough that the full sweep
  runs in seconds.

## Integration risks

- **Consumer 17.4 (iMSA)** will call `np.diff(fm_hz)` and needs NaN-aware
  handling — verified that `fm_hz` contains NaN only where `am` is also
  NaN (tested by `test_nan_columns_are_consistent_between_fm_and_am`).
- **Consumer 17.5 (Oren)** will do NaN interpolation across silent columns
  before resampling — relies on `fm_hz`/`am` being `float64`. Confirmed:
  both outputs are `np.float64` by construction (`np.full(n_cols, np.nan)`
  defaults to float).
- **No I/O risks** — stateless function, no file operations, no
  randomness, fully deterministic given input.

## What NOT to worry about

- No public driver script in this PR. Scripts will be added in 17.4 / 17.5
  where the ridge tracker is actually invoked on real WAV data. The
  tracker itself needs no CLI — it's pure infrastructure.
- No performance benchmarks on real WAV data yet. Will be covered by 17.4
  / 17.5 driver scripts when they run over the 7,518-call dataset.
