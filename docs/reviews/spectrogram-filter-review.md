# Spectrogram Pre-Filtering Module Review (17.2)

**Reviewed by:** Master Reviewer (Sonnet 4.6)
**Date:** 2026-04-17
**Module:** `src/usv_spectrogram/features/spectrogram_filter.py`
**Tier:** 3 (Critical — shared DSP infrastructure for modules 17.3, 17.5, 17.6)
**Verdict:** CHANGES NEEDED

---

## Pre-Findings Checklist

- [x] `IMPLEMENTATION_PROGRESS.md` — dated entry appended (2026-04-17, §17.2)
- [x] `docs/modules/spectrogram-filter.md` — exists and accurate
- [x] `__init__.py` exports — `FilterConfig` and `prefilter_spectrogram` exported from `usv_spectrogram.features`
- [x] `docs/architecture/patterns.md` — no new patterns established; existing patterns followed correctly

---

## Expected Constraints (Pre-Reading)

Before reading the implementation, the following invariants were established from spec reading:

- **ADR-001**: `sample_rate=300_000` must be explicit. This module operates on pre-computed spectrograms, so sr is only in `FilterConfig` for downstream propagation — acceptable.
- **ADR-002**: n_fft=512, hop_length=128, Hann window at sr=300_000. Frequency resolution = 585.9 Hz/bin. This module doesn't compute STFT; it operates on magnitudes passed by callers.
- **Frequency band**: ROADMAP §17.2 specifies 25–120 kHz, consistent with Oren 2024 mouse USV passband (versus detection's 25–110 kHz per ADR-002 note).
- **Most likely failure modes**: DSP filter parameters wrong; noise floor being pathologically over-aggressive or permissive; test anti-greenwashing (expected values copied from output); edge case crashes; docstring-code mismatch.

---

## BLOCKER (must fix before next module)

### B1. `test_snr_improves_by_10db_on_noisy_tone` passes trivially for the wrong reason

**File:** `tests/test_spectrogram_filter.py:564–612`

**Problem:** This test uses a 1-bin-wide tone ridge (`mag[tone_bin, :] = tone_amplitude`), which the 3×3 median filter correctly destroys — the center bin degrades to background level (numerically verified: `cleaned[tone_bin, :].mean() == 0.0` across all 100 columns). The SNR helper function then computes: `signal_power = mean(0²) = 0`, `noise_power = mean(noise² after masking) = 0` (background noise at 0.01 is also below the 3× noise-floor threshold and gets masked out). With `noise_power == 0`, the function returns `float('inf')`. The test therefore passes because `inf - snr_before >= 10.0`, not because the filter preserved a tone with improved SNR.

This test was NOT in the list of 4 tests revised during implementation (the handoff lists tests 1, 3, 11, 16). All four tone-preservation tests were correctly updated to use Gaussian ridges — but the SNR test, which is also a tone-preservation test and is the primary **exit criterion** validator, was missed.

**Why it matters:** This is the module's exit criterion test: "Filter reduces broadband noise on a synthetic noisy tone by >10 dB SNR improvement." A test that passes by destroying both signal and noise does not validate this criterion. If the filter was broken in a way that zero'd everything, this test would still pass.

**Fix:** Replace the 1-bin tone with a Gaussian-profiled ridge (exactly as done in tests 1, 3, 11, 16) and fix the SNR measurement to use all ridge bins as signal (not just the single center bin).

---

## WARNINGS (fix soon)

### W1. σ=1.2 bins is wider than real Hann STFT leakage — inaccurate physical claim in docstring

**File:** `tests/test_spectrogram_filter.py` (`_add_leakage_ridge` docstring)

Real Hann STFT at n_fft=512: on-bin pure tone produces `center=1.0, ±1=0.50, ±2≈0` → 3-bin FWHM. σ=1.2 Gaussian produces `center=1.0, ±1=0.71, ±2=0.25, ±3=0.044` → 5-bin width. σ=1.2 is ~65% wider than reality. Real ridges still survive the 3×3 median (verified), so this is a docstring-accuracy issue, not a correctness bug. Update docstring to describe σ=1.2 as a conservative synthetic model.

### W2. `test_single_outlier_pixel_removed_by_median_filter` assertion threshold is too weak

**File:** `tests/test_spectrogram_filter.py:246` (`assert cleaned[outlier_bin, outlier_col] < 100.0`)

Assertion `< 100.0` would pass even if the filter only reduced the outlier from 1000 to 99 (a mere 10× reduction). Actual value is 0.0. Tighten to `< 2.0`.

### W3. `freqs_hz` shape mismatch raises `ValueError` without a helpful message

**File:** `src/usv_spectrogram/features/spectrogram_filter.py:81`

Cryptic NumPy broadcast error. Add explicit shape validation.

### W4. No test for `n_freq_bins=1` (degenerate spectrogram)

**File:** `tests/test_spectrogram_filter.py` (no such test)

`n_time_cols=1` is tested, symmetric case missing.

---

## SUGGESTIONS

| # | Issue | Fix |
|---|-------|-----|
| S1 | Negative-magnitude inputs silently pass noise floor | Add docstring warning "magnitude must be non-negative (linear, not dB)" |
| S2 | `FilterConfig.sample_rate` unused internally | Add docstring note that it's a carrier field for downstream consumers |
| S3 | `_make_pure_tone_magnitude` helper is now dead code | Remove or comment |
| S4 | Per-column noise floor uses ALL bins; not documented | Add inline comment |
| S5 | `test_freq_boundary_inclusive` doesn't assert `freqs_hz[10]==freq_min` | Add explicit pre-assertions |

---

## Tier 3 Deep Analysis — Summary

**Spec fidelity:** Both documented deviations (3×3 kept; output uses `filtered * mask`) are defensible. Real Hann-window STFT ridges at n_fft=512 span 3 bins and survive the 3×3 median at 50% amplitude — the spec's filter choice is validated numerically.

**Test integrity:** The 4 rewritten tests preserve the original invariants (tone preservation, silent-column masking, band-inclusive boundary). The boundary-test tightening from OR to AND is more stringent (correct direction).

**DSP correctness:** Noise floor per column is robust (3–5 ridge bins out of 129 cannot shift the median). `mode='reflect'` handles all degenerate shapes. All-zero input produces all-zero output with no NaN.

**Interface:** `(cleaned, mask)` contract is right for all 3 downstream consumers. Boolean mask dtype correct. `FilterConfig` exposes the right knobs.

**Performance:** ~2ms per call × 7518 calls = ~15 s total. Acceptable.

---

## What Passed

| Area | Verdict |
|------|---------|
| DSP correctness — filter logic | PASS |
| Real STFT ridge survival | PASS |
| Spec alignment | PASS |
| Test integrity (4 rewritten tests) | PASS |
| FilterConfig validation | PASS |
| Pattern adherence | PASS |
| ADR-001/ADR-002 alignment | PASS |
| All-zero input correctness | PASS |
| dtype preservation | PASS |
| Documentation | PASS |
| IMPLEMENTATION_PROGRESS.md | PASS |
| Performance | PASS |

---

## Verdict

**CHANGES NEEDED**

B1 must be fixed before proceeding to module 17.3. The SNR exit-criterion test passes trivially by destroying both the tone and the noise. After fixing B1, a Tier 1 spot-check re-review is sufficient.

---

## Fix Log

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| B1 | FIXED | `tests/test_spectrogram_filter.py:584–645` | 2026-04-17 | Gaussian ridge + ridge-aware SNR + both-branches-guarded helper |
| W1 | FIXED | `tests/test_spectrogram_filter.py:81–103` | 2026-04-17 | Docstring rewritten to describe σ=1.2 as conservative synthetic model |
| W2 | FIXED | `tests/test_spectrogram_filter.py:204–213` | 2026-04-17 | Outlier threshold tightened from `< 100.0` to `< 2.0` |
| W3 | FIXED | `src/.../spectrogram_filter.py:69–74` | 2026-04-17 | Explicit shape validation with clear ValueError message |
| W4 | FIXED | `tests/test_spectrogram_filter.py:650–685` | 2026-04-17 | Added `test_single_freq_bin_no_crash` + `test_freqs_hz_shape_mismatch_raises` |
| S1 | FIXED | `src/.../spectrogram_filter.py` docstring | 2026-04-17 | "Must be non-negative; dB-scaled inputs are not supported" |
| S2 | FIXED | `src/.../spectrogram_filter.py` FilterConfig docstring | 2026-04-17 | "carrier field for downstream consumers" note |
| S3 | FIXED | `tests/test_spectrogram_filter.py` | 2026-04-17 | Dead `_make_pure_tone_magnitude` helper removed |
| S4 | FIXED | `src/.../spectrogram_filter.py:76–80` | 2026-04-17 | Inline comment on per-column median robustness |
| S5 | FIXED | `tests/test_spectrogram_filter.py:547–549` | 2026-04-17 | `assert freqs_hz[10] == freq_min` and `[50] == freq_max` added |

## Fixes Applied (2026-04-17)

### B1 — SNR exit-criterion test rewritten (BLOCKER)

**Files:** `tests/test_spectrogram_filter.py` (test_snr_improves_by_10db_on_noisy_tone).

**Root cause:** When rewriting the other 4 tone-preservation tests with Gaussian leakage ridges, this test — which is *also* a tone-preservation test — was not updated. Its 1-bin delta ridge got destroyed by the 3×3 median filter, causing both signal and noise power to drop to zero, which the SNR helper's `if noise_power == 0: return inf` branch converted to a trivially passing result.

**Fix:**
1. Replaced `mag[tone_bin, :] = tone_amplitude` with `_add_leakage_ridge(mag, center_bin=tone_bin, amplitude=tone_amplitude)` — same pattern as the other 4 tests.
2. Redefined the SNR helper to compute signal power over **all ridge bins** (center ± 3σ), not just the center bin. This prevents leakage bins from being miscounted as noise.
3. Added a second guard `if signal_power == 0.0: return float("-inf")` that triggers *before* the noise-power check. A broken filter that zeroes everything would now return snr_after = -inf → improvement = -inf → test fails correctly. Verified with a paranoia check (see handoff).

**Verification:** SNR before = 31.2 dB, SNR after = +inf (noise zeroed, signal preserved at power 0.232). A hypothetical broken filter zeroing everything now correctly fails the test (improvement = -inf).

### W1 — `_add_leakage_ridge` docstring corrected

Updated the docstring to state explicitly that σ=1.2 bins produces a 5-bin profile that is *wider* than real Hann-STFT leakage at n_fft=512 (3 bins for on-bin tones, up to 4 for off-bin). The σ=1.2 value is described as a deliberately conservative synthetic model that guarantees robust test behavior even under minor implementation changes.

### W2 — Outlier-suppression assertion tightened

`cleaned[outlier_bin, outlier_col] < 100.0` → `< 2.0`. The actual post-filter value is 0.0 (noise-floor mask zeroes it after the 3×3 median reduces 1000 → 1.0). The new threshold of 2.0 would fail if the filter only provided a 10× reduction, which would indicate a broken median filter.

### W3 — Explicit `freqs_hz` shape validation

Added at the start of `prefilter_spectrogram`:
```python
if freqs_hz.ndim != 1 or freqs_hz.shape[0] != magnitude.shape[0]:
    raise ValueError(
        f"freqs_hz shape {freqs_hz.shape} does not match magnitude "
        f"frequency axis ({magnitude.shape[0]},). ..."
    )
```
Previously a shape mismatch produced a cryptic NumPy broadcast error from the `freq_mask[:, None]` line.

### W4 — Edge-case tests added

Two new tests at the end of `test_spectrogram_filter.py`:
- `test_single_freq_bin_no_crash`: `n_freq_bins=1` spectrogram must not crash (symmetric to the existing `test_single_time_column_no_crash`).
- `test_freqs_hz_shape_mismatch_raises`: verifies the new W3 validation with both wrong-length and 2-D inputs.

### S1–S5 — Documentation polish

- **S1:** Added "Must be non-negative; dB-scaled inputs are not supported" to the `magnitude` parameter docstring.
- **S2:** Added a `FilterConfig` class docstring clarifying `sample_rate` is a carrier field not used internally.
- **S3:** Removed the now-dead `_make_pure_tone_magnitude` helper from the test file.
- **S4:** Added an inline comment above the `col_median` line explaining why median-over-all-bins is robust to in-band ridges.
- **S5:** Added `assert freqs_hz[10] == freq_min; assert freqs_hz[50] == freq_max` in `test_freq_boundary_inclusive` as explicit preconditions.

### Test counts after fixes

- **Before fixes:** 18 tests passing (included the trivially-passing SNR test).
- **After fixes:** 20 tests passing, 0.08 s wall-clock.
- **Breakdown:** 16 pre-implementation (4 rewritten for leakage, 1 for SNR correctness, rest unchanged) + 2 added in fix-pass (W4) + 2 edge-case expansions that existed in the original file.
- **No regressions:** 114/114 pass across `test_spectrogram_filter.py + test_sis_baselines.py + test_energy_detector.py + test_config.py`.

### Re-review recommendation

Per Tier 3 protocol ("BLOCKERs require re-review — the implementor cannot self-report that blockers are resolved"), a **Tier 1 spot-check re-review** is requested focused on:
1. B1 fix: does the ridge-aware SNR definition + `-inf` signal-power guard adequately prevent the trivial-pass failure mode?
2. Whether the new SNR test correctly fails when the filter zeros everything (paranoia check result included above).

---

## Tier 1 Spot-Check Re-Review (2026-04-17)

**Reviewer:** Master Reviewer (Sonnet 4.6)
**Verdict:** APPROVED

**B1 verified:**
- Ridge is Gaussian-profiled via `_add_leakage_ridge` (7-bin span, σ=1.2).
- SNR helper branch order confirmed: `signal_power == 0 → -inf` fires FIRST, before `noise_power == 0 → +inf`.
- Three-scenario trace completed:
  * Correct filter → signal > 0, noise = 0 → +inf → PASSES.
  * Broken filter zeroing everything → signal = 0 → -inf → FAILS.
  * Broken filter preserving noise but destroying ridge → signal = 0 → -inf → FAILS.
- No remaining trivial-pass path found.

**W1–W4, S1–S5 skim-check:** All 9 items confirmed in place at their claimed file:line locations.

**Note (separate-session recommendation):** Per Tier 3 protocol, re-review ideally happens in a fresh session to avoid anchoring bias. This spot-check was performed in-session as a pragmatic Tier 1 check on a focused blocker fix. For full confidence before consuming modules 17.3/17.5/17.6, a separate-session re-review of the Tier 3 scope is still recommended.

Module 17.2 is cleared to proceed to test-hardener (Phase 4.5).
