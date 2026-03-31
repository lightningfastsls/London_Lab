# Event Features — Implementation Handoff

**Module:** `src/usv_spectrogram/postprocessing/event_features.py`
**Date:** 2026-03-28
**Review tier:** Tier 3 (escalated — contains DSP math)

## Summary

Implemented event-level feature extraction for second-stage USV classification. The module extracts 11 discriminative features from each `USVEvent`: 6 probability-based (peak, mean, std, kurtosis, roughness, duration) and 5 spectral (tonality, peak frequency, frequency range, modulation rate, SNR).

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/usv_spectrogram/postprocessing/event_features.py` | ~170 | Core module: `EventFeatures` dataclass + `extract_event_features()` |
| `docs/modules/event-features.md` | ~65 | Module documentation |

## Files Modified

| File | Change |
|------|--------|
| `src/usv_spectrogram/postprocessing/__init__.py` | Added `EventFeatures`, `extract_event_features` to exports |

## Pre-existing Tests

**Pre-existing tests from test-architect: 17** (test file had grown from documented 14 to actual 17)

All 17 tests pass. No test expectations were modified. No additional tests written — the test-architect suite provides comprehensive coverage including:
- Probability statistics (constant, hand-computed values)
- Tonality discrimination (tonal > 0.5, broadband < 0.2, ordering)
- Frequency features (chirp continuity, stationary range)
- Edge cases (start/end boundary, single-window, zero spectrogram)
- Numeric stability (all finite, SNR positive)
- Structure validation (field existence, numeric types)
- Column mapping verification (hop_px correctness)
- Out-of-bounds guard (raises on invalid windows)

## Key Implementation Decisions

1. **Tonality = 1 - SFM** — Inverted spectral flatness measure so tonal=high. GM computed in log domain for numerical stability.
2. **dB input, power conversion for tonality** — `power = 10^(dB/10)` with eps=1e-10 floor.
3. **SNR in dB space** — `peak_dB - percentile10_dB` avoids unnecessary dB↔linear round-trips.
4. **Population std (ddof=0)** — Event probabilities are the full population, not a sample.
5. **Excess kurtosis** — Manual computation (mean((x-μ)⁴)/σ⁴ - 3) to avoid scipy dependency.

## Test Results

- `tests/test_event_features.py`: **17 passed**
- Full suite: **690 passed, 27 failed** (all failures pre-existing: triage/normalization/deepsqueak import modules)

## Post-Review Changes (2026-03-28)

After master-reviewer flagged BLOCKER + naming warnings, user decided:

1. **Column mapping → Option B (hop-spaced)**: Changed from consecutive columns to hop-spaced extraction (`cols = np.arange(window_count) * hop_px + start_col`). Samples across full event duration while maintaining 1:1 prob↔column correspondence. Decision documented in `docs/handoffs/event-features-column-mapping.md`.
2. **`prob_smoothness` → `prob_roughness`**: Name now matches metric direction (high = jagged).
3. **`freq_continuity` → `freq_modulation_rate`**: Name now matches metric direction (high = FM sweep).
4. **Test updated**: `test_event_at_end_of_spectrogram` uses `n_time=200` (was 100) for hop-spaced bounds.
5. **Downstream refs updated**: `ROADMAP_POST_PROCESSING.md`, `tests/test_fp_filter.py`.

All 17 tests pass after changes.

## Test Hardening (2026-03-28)

**Tests after hardening: 60 total (17 pre-existing + 43 hardener)**

Hardener found 2 bugs (deferred — input validation at system boundary, not logic errors):
1. **NaN propagation**: NaN in spectrogram silently propagates to features. Upstream `AudioLoader` never produces NaN, so this is a defensive guard improvement.
2. **-Inf in SNR**: `-Inf` dB values cause `NaN` in `snr_db` via `percentile(-Inf) - percentile(-Inf)`. Upstream uses eps to prevent -Inf.

Both deferred because `AudioLoader` already guards against these inputs. Can add input validation in a future hardening pass if the module is called with non-AudioLoader spectrograms.

Also noted: `hop_px=0` is unguarded (would extract same column repeatedly); `am < eps` branch in tonality is unreachable dead code after the `np.maximum(power, eps)` clamp.

## Integration Notes

- Consumes `USVEvent` from `hysteresis.py` and `spectrogram_db` from `AudioLoader`
- `hop_px` parameter must match `SlidingInference` stride (default 10)
- Ready for downstream FP filter / second-stage classifier consumption
