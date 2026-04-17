# Implementation Handoff: Spectrogram Pre-Filtering (17.2)

**Module:** Spectrogram Pre-Filtering Module (Phase 17.2)
**Review Tier:** 3 (critical DSP infrastructure — 3 downstream consumers)
**Date:** 2026-04-17
**Branch:** `main`

## What Changed

- New `src/usv_spectrogram/features/` subpackage (first module in a directory
  that will house 17.2, 17.3, 17.5, 17.6 per ROADMAP plan).
- `FilterConfig` frozen dataclass + stateless `prefilter_spectrogram`
  function cleaning a magnitude spectrogram via 3×3 median filter, local
  noise-floor mask, and 25–120 kHz frequency band mask.
- Package export surface (`features/__init__.py`).
- Module documentation in `docs/modules/spectrogram-filter.md`.
- Dated entry appended to `IMPLEMENTATION_PROGRESS.md`.
- **Pre-implementation tests:** 4 of 16 (tests 1, 3, 11, 16) were
  rewritten during implementation after discussion with the user — see
  "Pre-implementation test changes" below. This IS a modification of
  test expectations; explicit user approval was obtained (2026-04-17).

## Files Changed

- `src/usv_spectrogram/features/__init__.py` (NEW) — package init, 12 lines.
- `src/usv_spectrogram/features/spectrogram_filter.py` (NEW) —
  `FilterConfig` + `prefilter_spectrogram`, 82 lines.
- `tests/test_spectrogram_filter.py` (MODIFIED) — added
  `_add_leakage_ridge` helper, updated 4 tests to use Gaussian ridges.
- `docs/modules/spectrogram-filter.md` (NEW) — public interface docs +
  algorithm + decision log.
- `IMPLEMENTATION_PROGRESS.md` (APPENDED) — dated entry for 17.2.

## Key Decisions Made

**1. Kept the spec's 3×3 median filter (not 1-D time-only).**
Reinterpretation as `(1, 3)` was considered after initial test failures
but rejected: real STFT ridges span 3–5 bins due to Hann-window leakage
(ADR-002), so the 3×3 filter preserves them in practice. 1-D time-only
would lose the ability to suppress frequency-isolated outliers (e.g.
single-bin EMI pickup). User approved keeping 3×3.

**2. Output uses `filtered * mask`, not `magnitude * mask`.**
The ROADMAP spec text literally says `cleaned = magnitude * mask`, but
this interpretation fails the spec's own ROADMAP test 2
(`test_single_outlier_pixel_removed_by_median_filter`): an isolated
outlier with amplitude ≫ threshold would propagate unchanged to the
output. Using `filtered * mask` preserves the 3×3 filter's outlier
suppression while still producing a useful cleaned spectrogram for
ridge tracking and autoencoder training.

**3. Noise-floor per column = rolling median of per-column medians.**
The spec text "rolling median over `noise_floor_window_cols` centered on
that column" is ambiguous about what gets rolled. The implementation:
(a) collapses each column to a scalar via `np.median(filtered, axis=0)`,
(b) smooths this with `scipy.ndimage.median_filter(..., mode='reflect')`.
This is `O(T)` cheap, and `mode='reflect'` gracefully handles the
`n_time_cols < window` edge case.

**4. `mode='reflect'` everywhere, not `'constant'`.** Reflection keeps the
rolling median stable near edges; constant-zero padding artificially
lowers the noise floor at boundaries, which would allow noise to pass
through in those columns.

**5. Boolean mask dtype (not float).** Downstream consumers (ridge
tracker, autoencoder) use the mask as an index. A float mask would
require an explicit cast at every callsite.

## Pre-implementation test changes

During implementation, 4 of the 16 pre-existing tests failed. After
analysis, all 4 failures shared a single root cause: the tests used
idealized synthetic inputs (1-bin-wide "tone" ridges, or uniform
magnitude) that the 3×3 median filter correctly treats as pixel noise
(not signal). Real STFT ridges have spectral leakage that makes them
3–5 bins wide, and a filter that preserved 1-bin ridges would also
preserve 1-bin glitches.

**User approval workflow (2026-04-17):**
1. STOPPED before modifying anything per CLAUDE.md test-protocol rule
   "Unknown/Fail → don't assume which is wrong, discuss".
2. Presented three options (strict spec, time-only median, rewrite tests).
3. User approved Option (a): rewrite the tests with leakage-realistic
   Gaussian ridges, keep the 3×3 median filter as specified.

**Tests modified:**
- `test_pure_tone_peak_preserved_after_filtering` — replaced `mag[tone_bin, :] = 10.0`
  with `_add_leakage_ridge(mag, center_bin=tone_bin, amplitude=10.0)`
- `test_silent_column_masked_signal_column_passes` — same pattern
- `test_inband_signal_not_entirely_masked` — same pattern
- `test_freq_boundary_inclusive` — replaced uniform 100 input with
  background 1.0 + leakage ridges at boundary bins 10 and 50. Also
  tightened assertion from `sum(10) > 0 OR sum(50) > 0` to `sum(10) > 0
  AND sum(50) > 0` — the new input lets both be non-zero, which is what
  "boundary inclusive" truly means.

**Helper added:** `_add_leakage_ridge(mag, center_bin, amplitude, cols, sigma_bins=1.2)`
— places a Gaussian profile across ±3σ bins, modeling STFT window
leakage. Documented in the test file's module docstring.

**No expectations were loosened.** The assertions about tone preservation,
silent-column masking, in-band signal survival, and boundary inclusivity
remain exactly as specified. Only the *synthetic inputs* changed to match
the physics the filter is designed for.

## What I'm Unsure About

- **Is the Gaussian σ = 1.2 bins physically realistic?** I chose it so the
  ridge occupies ~5 bins above 10% of peak amplitude, matching rough
  intuition about Hann-window main-lobe width at n_fft=512. A more
  rigorous value would derive from the exact window function's spectral
  leakage. The tests are not sensitive to σ within a wide range, so this
  is unlikely to be a problem, but a DSP reviewer may have a stronger
  opinion.
- **Per-column median aggregation.** Real USV spectrograms have ~60
  frequency bins of signal in the 25–120 kHz band vs. ~70 out-of-band
  bins. The median over all 129 bins is dominated by the out-of-band
  tail, which is usually genuine noise — good. But if a recording has
  high in-band noise AND out-of-band silence, the median would
  underestimate the true in-band noise floor. I considered computing
  the median only over in-band rows but rejected it as spec deviation.
- **Dtype preservation.** `cleaned = filtered * mask.astype(filtered.dtype)`
  preserves dtype across `float32`/`float64` inputs. Integer inputs would
  silently convert — not a problem for magnitude spectrograms (always
  float) but worth flagging.

## Test Coverage

- **Pre-existing tests from test-architect:** 16 (9 ROADMAP + 7 gap-pattern)
- **Additional tests:** 0 (the 16 cover happy path, validation, edge cases,
  shape preservation, and the >10 dB SNR exit criterion)
- **Modified tests:** 4 (discussed above)
- **Total:** 18 tests (the collection count is 18 because the suite
  includes two parametric expansions of validation tests)
- **Result:** 18/18 passing, 0.09 s wall-clock.

Full suite check: 112 tests pass in the scope of implemented modules
(`test_spectrogram_filter.py` + `test_sis_baselines.py` +
`test_energy_detector.py` + `test_config.py`). Other test files
reference modules not yet implemented (17.3, 17.5, 17.6, etc.) or have
unrelated collection errors (`anthropic` package missing), both
pre-existing and out of scope for this module.

## DSP Sanity Checks

- [x] Uses ADR-001 `sample_rate=300_000` as default.
- [x] Frequency band matches Oren 2024 mouse USV passband (25–120 kHz).
- [x] `mode='reflect'` everywhere — no artefacts at spectrogram edges.
- [x] All-zero input → all-zero output (no NaN from 0/0 anywhere).
- [x] >10 dB SNR improvement on synthetic noisy-tone test (exit criterion).
- [x] Boolean mask dtype.
- [x] No library defaults rely on implicit `sr` — STFT is the caller's
      responsibility (this module operates on pre-computed spectrograms).

## Next Steps

- **Master-reviewer pass** (Tier 3 — critical infrastructure with 3 downstream
  consumers).
- **Test-hardener pass** after reviewer approval.
- **Module 17.3** (DP-based ridge tracker) can proceed — it consumes
  `cleaned` and `mask` from this module.

## References

- ROADMAP: `ROADMAP_SIS_BENCHMARK.md` §17.2 (lines 116–206)
- ADR-002: STFT parameters
- Test file: `tests/test_spectrogram_filter.py`
- Module: `src/usv_spectrogram/features/spectrogram_filter.py`
- Module docs: `docs/modules/spectrogram-filter.md`
