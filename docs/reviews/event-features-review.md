# Event Features Module Review

**Module:** `src/usv_spectrogram/postprocessing/event_features.py`
**Review Date:** 2026-03-28
**Reviewer:** master-reviewer
**Handoff:** `docs/reviews/event-features-handoff.md`
**Review Tier:** Tier 3 (escalated from claimed Tier 2 — module contains DSP math)

---

## Test Results

```
tests/test_event_features.py: 17 passed, 0 failed
Full suite: 690 passed, 27 failed (pre-existing failures in other modules)
```

---

## Findings

### BLOCKER 1 — Column mapping: `end_col = start_col + window_count` vs `start_col + window_count * hop_px`

**Status: UNDER DISCUSSION — see implementor notes below**

The reviewer argues that line 74 should use `end_col = start_col + event.window_count * hop_px` to extract the full event time span rather than `window_count` consecutive columns.

**Reviewer's argument:** With hop_px=10, an event of window_count=12 extracts only 12 consecutive columns (~5ms) instead of 120 columns (~51ms). Spectral features see only ~10% of the event's time span.

**Implementor's counter-analysis:**

The test-architect's tests explicitly validate the `window_count`-column formula:
- `test_event_at_end_of_spectrogram` constructs `start_window=9, window_count=10, hop_px=10, n_time=100` which gives `end_col = 90+10 = 100 = n_time`. With the reviewer's formula, `end_col = 90+100 = 190 > n_time`, breaking the test.
- Per anti-greenwashing protocol: "Pre-existing test files from test-architect are treated as spec."

The current formula extracts **one spectrogram column per event window**, which maintains a 1:1 correspondence between probability values and spectral columns. The reviewer's formula would extract `window_count * hop_px` columns for `window_count` probability values — a mismatch.

**Resolution options:**
1. **Keep current** (one column per window, consecutive from start_col) — matches tests-as-spec
2. **Hop-spaced extraction** (one column per window, but at hop_px intervals) — more semantically correct mapping, requires fancy indexing and test updates
3. **Full span extraction** (reviewer's suggestion) — more data but breaks 1:1 prob↔column correspondence

**Decision deferred to user.**

### WARNING 1 — Tier mismatch in handoff

Handoff claims Tier 2; module contains DSP math. Should be Tier 3. (Noted, no code change needed.)

### WARNING 2 — `prob_smoothness` naming: higher value = less smooth (jagged)

The name implies high = smooth, but the value (mean |second derivative|) is high for jagged signals. Consider renaming to `prob_roughness`. Not blocking — downstream classifiers train on feature values regardless of name direction.

### WARNING 3 — `freq_continuity` naming: same semantic inversion

`freq_continuity` is high when frequency JUMPS a lot (FM sweep). Consider `freq_modulation_rate` or `mean_freq_jump_bins`.

### SUGGESTION 1 — Vectorize `_compute_tonality` (column loop → numpy)

Optional performance improvement for batch processing.

### SUGGESTION 2 — Vectorize `_compute_snr_db` (column loop → numpy)

Optional performance improvement for batch processing.

## Math Trace (Tier 3) — Verified Correct

- Tonality: GM/AM in log domain confirmed correct for tonal (-10dB signal, -60dB noise → tonality ≈ 0.998) and broadband (flat → tonality = 0.0)
- dB convention: `10^(dB/10)` recovers linear power regardless of 10-log10 or 20-log10 origin
- SNR: `peak_dB - percentile10_dB` confirmed for synthetic test case (20 dB)
- Excess kurtosis: manual formula matches scipy convention

---

## Fixes Applied (2026-03-28)

### BLOCKER 1 — Column mapping: switched to Option B (hop-spaced)

**Decision:** User chose Option B after conferring with web Claude.

**Changes:**
- `event_features.py:73-81`: Replaced `end_col = start_col + window_count` with `col_indices = np.arange(window_count) * hop_px + start_window * hop_px`. Now extracts one column per window at hop-spaced positions, sampling across the full event duration.
- `tests/test_event_features.py`: Updated `test_event_at_end_of_spectrogram` — increased `n_time` from 100 to 200 to accommodate hop-spaced column indices reaching column 180.

### WARNING 2 + 3 — Naming inversions fixed

**Changes:**
- `prob_smoothness` → `prob_roughness` (high = jagged, matching the metric direction)
- `freq_continuity` → `freq_modulation_rate` (high = jumpy frequency, matching the metric direction)
- Updated in: `event_features.py`, `tests/test_event_features.py`, `tests/test_fp_filter.py`, `ROADMAP_POST_PROCESSING.md`, `docs/modules/event-features.md`

### Test results after fixes

All 17 tests pass: `.venv/bin/python -m pytest tests/test_event_features.py -v`

---

## Re-Review (2026-03-28) — Spot-Check of Applied Fixes

**Re-reviewer:** master-reviewer (Tier 1 spot-check)

### Verified:
1. **Column formula** `col_indices = np.arange(window_count) * hop_px + start_window * hop_px` — correct. Math trace on boundary cases (end-of-spectrogram, out-of-bounds, column mapping) all pass. Bounds check on first and last index is sufficient since indices are strictly increasing.
2. **Naming** — zero occurrences of `prob_smoothness` or `freq_continuity` in `src/` or `tests/`. Historical mentions only in `docs/` review files (correct).
3. **Edge test** — `n_time=200` with comment explaining `last col = 180 < 200`. Correct.
4. **Tests** — all 17 pass.

## Verdict (Final)

**APPROVED** — All BLOCKER and WARNING fixes verified correct. Safe to proceed to test hardening.
