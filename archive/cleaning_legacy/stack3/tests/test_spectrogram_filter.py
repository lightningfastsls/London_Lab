"""Tests for spectrogram_filter — written by test-architect BEFORE implementation.

Module under test: src/usv_spectrogram/features/spectrogram_filter.py
Spec source: ROADMAP_SIS_BENCHMARK.md §17.2 (lines 116-206)

ROADMAP test plan coverage:
  1. Pure-tone spectrogram: peak preserved after filtering
     -> test_pure_tone_peak_preserved_after_filtering
  2. Spectrogram with one high-amplitude pixel outlier: median filter removes it
     -> test_single_outlier_pixel_removed_by_median_filter
  3. Silent column + signal column: silent cols get mostly masked, signal col passes
     -> test_silent_column_masked_signal_column_passes
  4. Low-frequency content (<25 kHz): fully masked to zero
     -> test_below_freq_min_fully_masked_to_zero
  5. Frequency mask shape broadcasts correctly on (129, 1000) input
     -> test_frequency_mask_broadcasts_on_large_spectrogram
  6. FilterConfig validation: freq_min >= freq_max raises; even median_filter_size raises
     -> test_filterconfig_validation_freq_min_ge_freq_max_raises
     -> test_filterconfig_validation_even_median_filter_size_raises
     -> test_filterconfig_validation_noise_floor_multiplier_le_one_raises
  7. Input shape (n_freq_bins, n_time_cols) is preserved
     -> test_output_shape_matches_input_shape
  8. Edge case: very short signal (n_time_cols < noise_floor_window_cols) — no crash
     -> test_short_signal_no_crash
  9. All-zero input returns all-zero output without NaN
     -> test_all_zero_input_returns_all_zero_no_nan

Additional coverage (recurring gap patterns):
  - Config defaults are sane/correct values -> test_filterconfig_default_values
  - Mask dtype is boolean -> test_mask_is_boolean_array
  - Above freq_max fully masked -> test_above_freq_max_fully_masked_to_zero
  - In-band region not entirely masked when signal present -> test_inband_signal_not_entirely_masked
  - Single time column (n_time_cols=1) -> test_single_time_column_no_crash
  - Frequency boundary is inclusive (exactly at freq_min and freq_max) -> test_freq_boundary_inclusive
  - SNR improvement on synthetic noisy tone (exit criterion) -> test_snr_improves_by_10db_on_noisy_tone

Revision 2026-04-17: tests 1, 3, 11, and 16 were updated during implementation
to use leakage-realistic Gaussian-profiled ridges (σ ≈ 1.2 bins) instead of
1-bin delta ridges.  Rationale: the ROADMAP §17.2 spec specifies a 3×3 median
filter, which correctly treats a 1-bin-wide synthetic ridge as isolated pixel
noise and erases it.  Real STFT-derived USV ridges span 3–5 bins due to
window-function leakage (ADR-002, Hann window at n_fft=512), so the spec's
filter works in practice.  See ``_add_leakage_ridge`` below for the ridge
construction helper.  Decision approved by user 2026-04-17.

Total: 35 tests (16 original + 2 reviewer fixes + 17 from test-hardener and adversarial pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

# Pattern 8: import bootstrap — tests/ is one level below repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# These imports will raise ImportError until the module is implemented.
# That is the expected initial failure mode.
from usv_spectrogram.features.spectrogram_filter import (  # noqa: E402
    FilterConfig,
    prefilter_spectrogram,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freqs(n_bins: int, sample_rate: int = 300_000) -> np.ndarray:
    """Return linearly spaced frequency array matching rfft bin layout."""
    return np.linspace(0.0, sample_rate / 2.0, n_bins)


def _add_leakage_ridge(
    mag: np.ndarray,
    center_bin: int,
    amplitude: float,
    cols: slice | None = None,
    sigma_bins: float = 1.2,
) -> None:
    """Add a Gaussian-profiled ridge to ``mag`` in-place.

    Real Hann-window STFT ridges at n_fft=512 (ADR-002) span 3 bins for
    on-bin pure tones (center + ±1 at ~50% amplitude) or up to 4 bins for
    off-bin tones. σ=1.2 produces a 5-bin profile above 10% amplitude —
    **wider than real STFT leakage**. This is a deliberately CONSERVATIVE
    synthetic model:

    * Real ridges (≥3 bins) also survive the 3×3 median filter, as verified
      numerically — the 3×3 median at the center of a 3-bin 50% profile
      sees `[0.5]*4 + [1.0] + [0.5]*4` → median = 0.5, which still
      comfortably exceeds the noise-floor threshold at real SNRs.
    * σ=1.2 guarantees the filter sees a clear majority of signal pixels
      in every 3×3 neighbourhood at the ridge center, making the tests
      robust to small implementation changes in filter ordering.

    Profile: amplitude * exp(-0.5 * (offset / sigma)**2) over ±3σ bins.
    """
    if cols is None:
        cols = slice(None)
    offsets = np.arange(-3, 4)  # ±3 σ coverage
    profile = amplitude * np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    n_freq_bins = mag.shape[0]
    for offset, value in zip(offsets, profile):
        row = center_bin + int(offset)
        if 0 <= row < n_freq_bins:
            mag[row, cols] = np.maximum(mag[row, cols], value.astype(mag.dtype))


# ---------------------------------------------------------------------------
# FilterConfig default / validation tests
# ---------------------------------------------------------------------------


def test_filterconfig_default_values() -> None:
    """Verify FilterConfig defaults match the spec exactly (§17.2 dataclass block)."""
    cfg = FilterConfig()
    assert cfg.sample_rate == 300_000
    assert cfg.noise_floor_multiplier == pytest.approx(3.0)
    assert cfg.noise_floor_window_cols == 20
    assert cfg.median_filter_size == 3
    assert cfg.freq_min_hz == pytest.approx(25_000.0)
    assert cfg.freq_max_hz == pytest.approx(120_000.0)


def test_filterconfig_validation_freq_min_ge_freq_max_raises() -> None:
    """FilterConfig.__post_init__ must raise ValueError when freq_min_hz >= freq_max_hz."""
    with pytest.raises(ValueError, match="freq_min_hz"):
        FilterConfig(freq_min_hz=120_000.0, freq_max_hz=25_000.0)

    # Equal values also invalid
    with pytest.raises(ValueError):
        FilterConfig(freq_min_hz=50_000.0, freq_max_hz=50_000.0)


def test_filterconfig_validation_even_median_filter_size_raises() -> None:
    """FilterConfig.__post_init__ must raise ValueError for even median_filter_size."""
    with pytest.raises(ValueError, match="odd"):
        FilterConfig(median_filter_size=4)

    with pytest.raises(ValueError):
        FilterConfig(median_filter_size=2)


def test_filterconfig_validation_noise_floor_multiplier_le_one_raises() -> None:
    """FilterConfig.__post_init__ must raise ValueError when noise_floor_multiplier <= 1."""
    with pytest.raises(ValueError, match="noise_floor_multiplier"):
        FilterConfig(noise_floor_multiplier=1.0)

    with pytest.raises(ValueError):
        FilterConfig(noise_floor_multiplier=0.5)


# ---------------------------------------------------------------------------
# ROADMAP test 1 — pure-tone peak preserved
# ---------------------------------------------------------------------------


def test_pure_tone_peak_preserved_after_filtering() -> None:
    """Pure-tone ridge at 70 kHz (well inside 25–120 kHz band) must survive filtering.

    Verifies ROADMAP test plan item 1: the bin containing the tone should not
    be zeroed out by the noise-floor mask.  The tone amplitude (10.0) is ~1000x
    the background noise (0.01), so it comfortably exceeds 3× local median.

    Ridge is constructed with Gaussian spectral leakage (σ = 1.2 bins) so the
    3×3 median filter does not erase it — this mirrors real STFT windowing
    behaviour (ADR-002).  A 1-bin-wide delta ridge is a synthetic artefact
    that the 3×3 median filter correctly treats as isolated pixel noise.
    """
    n_freq_bins, n_time_cols = 64, 50
    freqs_hz = _make_freqs(n_freq_bins)
    # Pick bin closest to 70 kHz (well within 25–120 kHz band)
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))

    rng = np.random.default_rng(0)
    mag = 0.01 * (1.0 + rng.random((n_freq_bins, n_time_cols)).astype(np.float32))
    _add_leakage_ridge(mag, center_bin=tone_bin, amplitude=10.0)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # The tone bin should NOT be fully masked — at least 80% of columns pass
    tone_row_passes = mask[tone_bin, :].sum()
    assert tone_row_passes >= int(0.8 * n_time_cols), (
        f"Expected >= 80% of tone columns to pass mask, got {tone_row_passes}/{n_time_cols}"
    )

    # Cleaned amplitude at tone bin should be close to original (not zeroed)
    assert cleaned[tone_bin, :].mean() > 1.0, (
        "Cleaned magnitude at tone bin unexpectedly close to zero"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 2 — single outlier pixel removed by median filter
# ---------------------------------------------------------------------------


def test_single_outlier_pixel_removed_by_median_filter() -> None:
    """A single isolated high-amplitude pixel must be suppressed by the 3x3 median filter.

    Verifies ROADMAP test plan item 2.  We place one extreme outlier pixel
    (amplitude 1000) surrounded by a uniform low background (amplitude 1.0).
    After median filtering the outlier position should revert close to the
    background level, preventing it from surviving the noise-floor mask.
    """
    n_freq_bins, n_time_cols = 64, 60
    freqs_hz = _make_freqs(n_freq_bins)

    # Flat background in the 50–80 kHz band (inside filter passband)
    mag = np.ones((n_freq_bins, n_time_cols), dtype=np.float32)

    # Place single outlier at center of the spectrogram, inside passband
    outlier_bin = int(np.argmin(np.abs(freqs_hz - 60_000.0)))
    outlier_col = n_time_cols // 2
    mag[outlier_bin, outlier_col] = 1000.0  # extreme isolated spike

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # The outlier position in the cleaned output must NOT retain its extreme value.
    # After 3x3 median filtering a neighbourhood of all-1.0, the median is 1.0,
    # so the outlier is replaced by 1.0; the noise-floor mask then zeroes it
    # (since 1.0 is not > 3 × column_median ≈ 3.0). The assertion threshold
    # 2.0 is tight enough to fail if the filter only gave, say, a 10× reduction
    # (1000 → 100), which would indicate a broken median filter.
    assert cleaned[outlier_bin, outlier_col] < 2.0, (
        "Outlier pixel was not suppressed: 3×3 median filter should reduce it "
        "to the background level (~1.0) before the noise-floor mask zeroes it"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 3 — silent column masked, signal column passes
# ---------------------------------------------------------------------------


def test_silent_column_masked_signal_column_passes() -> None:
    """Silent columns (all near-zero) must be masked; high-SNR columns must pass.

    Verifies ROADMAP test plan item 3.  Constructs a spectrogram with half the
    columns at near-zero amplitude and half at strong-signal amplitude.
    """
    n_freq_bins, n_time_cols = 64, 60
    freqs_hz = _make_freqs(n_freq_bins)

    mag = np.zeros((n_freq_bins, n_time_cols), dtype=np.float32)

    # Silent region: columns 0..29 with tiny noise
    mag[:, :30] = 1e-6

    # Signal region: columns 30..59 with strong 70 kHz tone (with leakage)
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    _add_leakage_ridge(mag, center_bin=tone_bin, amplitude=5.0, cols=slice(30, None))

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Silent columns: most (>= 90%) of their cells should be masked
    silent_mask_fraction = mask[:, :30].mean()
    assert silent_mask_fraction < 0.10, (
        f"Expected < 10% of silent-column cells to pass, got {silent_mask_fraction:.2%}"
    )

    # Signal column: the tone bin must pass in the signal region
    signal_passes = mask[tone_bin, 30:].mean()
    assert signal_passes > 0.5, (
        f"Expected > 50% of tone-bin cells in signal region to pass, got {signal_passes:.2%}"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 4 — below freq_min fully masked
# ---------------------------------------------------------------------------


def test_below_freq_min_fully_masked_to_zero() -> None:
    """All frequency bins below freq_min_hz (25 kHz) must be zeroed in cleaned output.

    Verifies ROADMAP test plan item 4 and exit criterion:
    'Frequency bins outside [25, 120] kHz are zero after filtering.'
    """
    n_freq_bins = 129
    n_time_cols = 40
    # Nyquist = 150 kHz for sr=300 kHz → linspace(0, 150 kHz, 129)
    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)

    # Uniform strong signal everywhere so masking is purely from freq band
    mag = 10.0 * np.ones((n_freq_bins, n_time_cols), dtype=np.float32)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    below_min_bins = freqs_hz < cfg.freq_min_hz
    assert below_min_bins.sum() > 0, "Test requires at least one below-min bin"

    # Every cell in below-min rows must be zero in cleaned output
    npt.assert_array_equal(
        cleaned[below_min_bins, :],
        np.zeros((below_min_bins.sum(), n_time_cols), dtype=np.float32),
        err_msg="Below-freq_min bins must be fully zeroed in cleaned output",
    )


# ---------------------------------------------------------------------------
# ROADMAP test 5 — broadcast shape on (129, 1000)
# ---------------------------------------------------------------------------


def test_frequency_mask_broadcasts_on_large_spectrogram() -> None:
    """Frequency mask must broadcast correctly to a (129, 1000) spectrogram.

    Verifies ROADMAP test plan item 5: no shape errors when n_time_cols is large.
    Checks that out-of-band bins are zero and in-band bins can be non-zero.
    """
    n_freq_bins, n_time_cols = 129, 1000
    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)

    # Strong uniform signal
    mag = 5.0 * np.ones((n_freq_bins, n_time_cols), dtype=np.float32)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Output shapes must match input
    assert cleaned.shape == (n_freq_bins, n_time_cols)
    assert mask.shape == (n_freq_bins, n_time_cols)

    # Out-of-band rows (below 25 kHz and above 120 kHz) must be all-zero
    out_of_band = (freqs_hz < cfg.freq_min_hz) | (freqs_hz > cfg.freq_max_hz)
    npt.assert_array_equal(
        cleaned[out_of_band, :],
        np.zeros((out_of_band.sum(), n_time_cols), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# ROADMAP test 7 — output shape preserved
# ---------------------------------------------------------------------------


def test_output_shape_matches_input_shape() -> None:
    """prefilter_spectrogram must return (cleaned, mask) with same shape as input.

    Verifies ROADMAP test plan item 7.  Tests several non-square shapes.
    """
    cfg = FilterConfig()

    for n_freq_bins, n_time_cols in [(32, 10), (64, 100), (257, 500)]:
        freqs_hz = _make_freqs(n_freq_bins)
        mag = np.abs(np.random.randn(n_freq_bins, n_time_cols).astype(np.float32))
        cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

        assert cleaned.shape == (n_freq_bins, n_time_cols), (
            f"Shape mismatch for input ({n_freq_bins}, {n_time_cols}): got {cleaned.shape}"
        )
        assert mask.shape == (n_freq_bins, n_time_cols), (
            f"Mask shape mismatch for input ({n_freq_bins}, {n_time_cols}): got {mask.shape}"
        )


# ---------------------------------------------------------------------------
# ROADMAP test 8 — short signal no crash
# ---------------------------------------------------------------------------


def test_short_signal_no_crash() -> None:
    """n_time_cols < noise_floor_window_cols must not raise any exception.

    Verifies ROADMAP test plan item 8: the rolling window logic must handle
    edge mode gracefully when fewer columns than the window size exist.
    The default noise_floor_window_cols=20; we test with n_time_cols=5.
    """
    n_freq_bins = 64
    n_time_cols = 5  # much shorter than noise_floor_window_cols=20
    freqs_hz = _make_freqs(n_freq_bins)

    mag = np.abs(np.random.default_rng(7).random((n_freq_bins, n_time_cols)).astype(np.float32))

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Must return correct shapes with no NaN
    assert cleaned.shape == (n_freq_bins, n_time_cols)
    assert mask.shape == (n_freq_bins, n_time_cols)
    assert not np.any(np.isnan(cleaned)), "NaN found in cleaned output for short signal"


# ---------------------------------------------------------------------------
# ROADMAP test 9 — all-zero input
# ---------------------------------------------------------------------------


def test_all_zero_input_returns_all_zero_no_nan() -> None:
    """All-zero magnitude input must produce all-zero output with no NaN anywhere.

    Verifies ROADMAP test plan item 9: the division implicit in noise-floor
    computation (e.g. 0/0 when local median is zero) must be guarded.
    """
    n_freq_bins, n_time_cols = 64, 30
    freqs_hz = _make_freqs(n_freq_bins)
    mag = np.zeros((n_freq_bins, n_time_cols), dtype=np.float32)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert not np.any(np.isnan(cleaned)), "NaN in cleaned output for all-zero input"
    assert not np.any(np.isnan(mask.astype(np.float32))), "NaN in mask for all-zero input"
    npt.assert_array_equal(
        cleaned,
        np.zeros_like(mag),
        err_msg="All-zero input must produce all-zero cleaned output",
    )


# ---------------------------------------------------------------------------
# Additional gap-pattern tests
# ---------------------------------------------------------------------------


def test_mask_is_boolean_array() -> None:
    """prefilter_spectrogram must return a boolean mask, not a float mask.

    Behavioral contract: callers (ridge tracker, autoencoder) need bool dtype
    to use the mask as an index without accidental float arithmetic.
    """
    n_freq_bins, n_time_cols = 64, 30
    freqs_hz = _make_freqs(n_freq_bins)
    mag = np.abs(np.random.default_rng(3).random((n_freq_bins, n_time_cols)).astype(np.float32))

    cfg = FilterConfig()
    _, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert mask.dtype == bool or np.issubdtype(mask.dtype, np.bool_), (
        f"Expected boolean mask, got dtype={mask.dtype}"
    )


def test_above_freq_max_fully_masked_to_zero() -> None:
    """All frequency bins above freq_max_hz (120 kHz) must be zeroed in cleaned output.

    Complements ROADMAP test 4 (which checks below freq_min).
    Exit criterion: 'Frequency bins outside [25, 120] kHz are zero after filtering.'
    """
    n_freq_bins = 129
    n_time_cols = 40
    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)

    mag = 10.0 * np.ones((n_freq_bins, n_time_cols), dtype=np.float32)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    above_max_bins = freqs_hz > cfg.freq_max_hz
    assert above_max_bins.sum() > 0, "Test requires at least one above-max bin"

    npt.assert_array_equal(
        cleaned[above_max_bins, :],
        np.zeros((above_max_bins.sum(), n_time_cols), dtype=np.float32),
        err_msg="Above-freq_max bins must be fully zeroed in cleaned output",
    )


def test_inband_signal_not_entirely_masked() -> None:
    """A strong in-band signal must not be entirely masked out.

    Guards against an implementation that erroneously masks everything.
    A sine-wave magnitude of 100 at 70 kHz is >3x any reasonable local noise
    floor composed of 0.001 background values.
    """
    n_freq_bins = 64
    n_time_cols = 40
    freqs_hz = _make_freqs(n_freq_bins)

    mag = 0.001 * np.ones((n_freq_bins, n_time_cols), dtype=np.float32)
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    # Leakage-profiled ridge so the 3×3 median filter preserves it
    _add_leakage_ridge(mag, center_bin=tone_bin, amplitude=100.0)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # At least one cell must survive in the cleaned output
    assert cleaned.max() > 0.0, "Expected at least one non-zero value in cleaned output"
    assert mask.any(), "Expected at least one True in mask"


def test_single_time_column_no_crash() -> None:
    """n_time_cols=1 must not raise any exception.

    Single-item edge case: the rolling window for noise floor must handle
    a single column without indexing errors.
    """
    n_freq_bins = 64
    freqs_hz = _make_freqs(n_freq_bins)
    mag = np.abs(np.random.default_rng(9).random((n_freq_bins, 1)).astype(np.float32))

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert cleaned.shape == (n_freq_bins, 1)
    assert mask.shape == (n_freq_bins, 1)
    assert not np.any(np.isnan(cleaned))


def test_freq_boundary_inclusive() -> None:
    """Bins at exactly freq_min_hz and freq_max_hz must be included (not masked out).

    Spec says ``mask &= (freqs_hz >= freq_min_hz) & (freqs_hz <= freq_max_hz)``,
    so boundary bins are inclusive.  Tests the ``>=`` / ``<=`` logic.

    Input construction: low background (amplitude 1.0) with two localized
    leakage-profiled ridges at the boundary bins (bin 10 at freq_min, bin 50
    at freq_max).  The ridges exceed 3× the local noise floor so the
    amplitude mask passes them; the boundary bins remain non-zero only if
    the frequency mask is inclusive on both ends.
    """
    n_freq_bins = 64
    n_time_cols = 40
    freq_min = 25_000.0
    freq_max = 120_000.0

    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)
    # Snap two bins to exact boundary values
    freqs_hz[10] = freq_min
    freqs_hz[50] = freq_max
    # Precondition: boundary bins are exactly at the configured cutoffs.
    assert freqs_hz[10] == freq_min
    assert freqs_hz[50] == freq_max

    # Low uniform background that the threshold lets pass nowhere, plus
    # localized ridges at the boundary bins that WILL pass the amplitude mask.
    mag = np.ones((n_freq_bins, n_time_cols), dtype=np.float32)
    _add_leakage_ridge(mag, center_bin=10, amplitude=100.0)
    _add_leakage_ridge(mag, center_bin=50, amplitude=100.0)

    cfg = FilterConfig(freq_min_hz=freq_min, freq_max_hz=freq_max)
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Both boundary bins must retain signal in the cleaned output.
    assert cleaned[10, :].sum() > 0, (
        "Bin 10 (freq_min boundary) was zeroed — freq_mask should include >= freq_min"
    )
    assert cleaned[50, :].sum() > 0, (
        "Bin 50 (freq_max boundary) was zeroed — freq_mask should include <= freq_max"
    )


def test_snr_improves_by_10db_on_noisy_tone() -> None:
    """Filtering must improve SNR by > 10 dB on a synthetic noisy spectrogram.

    Verifies the exit criterion: 'Filter reduces broadband noise on a synthetic
    noisy tone by >10 dB SNR improvement.'

    Construction:
    - Background broadband noise: 0.01 (+ uniform random) in all bins
    - 70 kHz tone with Gaussian spectral leakage (σ=1.2 bins, amplitude 1.0)
      so the 3×3 median filter preserves it (see ROADMAP §17.2 note above)
    - After filtering, noise outside the ridge should be zeroed, driving SNR
      higher.

    SNR formula: 10 * log10(signal_power / noise_power)
    where signal = ridge bins (center ± 3σ, in-band), noise = all OTHER
    in-band bins (excluding the ridge so leakage is not miscounted as noise).

    The ridge-aware SNR definition ensures the test cannot pass trivially:
    - If the filter destroyed the ridge, signal_power would be 0 and
      log(0/x) = -inf → improvement = -inf → test fails.
    - If the filter destroyed both ridge and noise, noise_power = 0 →
      ridge_bins-based signal_power is also 0 → 0/0 → we return a finite
      sentinel (-inf) so the test still fails rather than spuriously
      passing with +inf.
    """
    n_freq_bins = 129
    n_time_cols = 100
    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)

    rng = np.random.default_rng(42)
    noise_amplitude = 0.01
    tone_amplitude = 1.0

    mag = noise_amplitude * (1.0 + rng.random((n_freq_bins, n_time_cols)).astype(np.float32))
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    _add_leakage_ridge(mag, center_bin=tone_bin, amplitude=tone_amplitude)

    in_band = (freqs_hz >= 25_000.0) & (freqs_hz <= 120_000.0)

    # Ridge bins: center ± 3σ coverage used by _add_leakage_ridge
    ridge_bins = np.array(
        [b for b in range(tone_bin - 3, tone_bin + 4) if 0 <= b < n_freq_bins and in_band[b]]
    )
    non_ridge_inband = np.array(
        [b for b in np.where(in_band)[0] if b not in ridge_bins]
    )

    def snr_db(arr: np.ndarray) -> float:
        signal_power = float(np.mean(arr[ridge_bins, :] ** 2))
        noise_power = float(np.mean(arr[non_ridge_inband, :] ** 2))
        # Guard both branches: an all-zero signal cannot improve SNR, so
        # return a finite sentinel that will fail the downstream
        # improvement >= 10 dB check (unlike +inf which would spuriously pass).
        if signal_power == 0.0:
            return float("-inf")
        if noise_power == 0.0:
            return float("inf")
        return 10.0 * np.log10(signal_power / noise_power)

    snr_before = snr_db(mag)

    cfg = FilterConfig()
    cleaned, _ = prefilter_spectrogram(mag, freqs_hz, cfg)

    snr_after = snr_db(cleaned)

    improvement = snr_after - snr_before
    assert improvement >= 10.0, (
        f"Expected >= 10 dB SNR improvement, got {improvement:.1f} dB "
        f"(before={snr_before:.1f} dB, after={snr_after:.1f} dB)"
    )


# ---------------------------------------------------------------------------
# Degenerate-shape edge cases (review W4)
# ---------------------------------------------------------------------------


def test_single_freq_bin_no_crash() -> None:
    """n_freq_bins=1 must not crash (degenerate frequency axis).

    Symmetric to test_single_time_column_no_crash.  A caller might pass a
    pre-cropped single-bin spectrogram; the 3×3 median filter with
    mode='reflect' handles this by reflecting the single row.
    """
    freqs_hz = np.array([70_000.0])
    mag = np.ones((1, 20), dtype=np.float32)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert cleaned.shape == (1, 20)
    assert mask.shape == (1, 20)
    assert not np.any(np.isnan(cleaned))


def test_freqs_hz_shape_mismatch_raises() -> None:
    """Mismatched ``freqs_hz`` length must raise a clear ValueError.

    Guards against silent shape bugs in downstream consumers (17.3 ridge
    tracker, 17.5 Oren vectorization, 17.6 AMVOC autoencoder) that build
    ``freqs_hz`` separately from the STFT call.
    """
    mag = np.ones((64, 20), dtype=np.float32)
    bad_freqs = np.linspace(0.0, 150_000.0, 32)  # wrong length

    cfg = FilterConfig()
    with pytest.raises(ValueError, match="freqs_hz"):
        prefilter_spectrogram(mag, bad_freqs, cfg)

    # 2-D freqs_hz also rejected
    with pytest.raises(ValueError, match="freqs_hz"):
        prefilter_spectrogram(mag, np.zeros((64, 1)), cfg)


# ---------------------------------------------------------------------------
# Adversarial tests added by test-hardener (2026-04-17)
# ---------------------------------------------------------------------------
# Gap categories addressed:
#   A. dtype / precision (float64, int32)
#   B. Parameter-boundary combinations (multiplier=1+eps, window=1, filter=1, filter=99)
#   C. Physical filter invariants (amplitude scaling, cleaned <= magnitude regression)
#   D. Pathological inputs (NaN, Inf, empty spectrogram, non-contiguous)
#   E. Downstream-consumer invariants (mask.any() skip contract, realistic STFT shape)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A. Dtype / precision
# ---------------------------------------------------------------------------


def test_float64_input_dtype_preserved() -> None:
    """float64 input must produce float64 cleaned output.

    Downstream Zarr storage (17.6 AMVOC autoencoder) writes whatever dtype
    the cleaned array carries. If a float64 spectrogram is inadvertently
    cast to float32, precision is silently lost. The implementation's
    ``filtered * mask.astype(filtered.dtype)`` chain must preserve the
    original dtype end-to-end.
    """
    n_freq_bins, n_time_cols = 64, 40
    freqs_hz = _make_freqs(n_freq_bins)
    rng = np.random.default_rng(0)
    mag = rng.random((n_freq_bins, n_time_cols))  # float64 by default
    assert mag.dtype == np.float64

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert cleaned.dtype == np.float64, (
        f"Expected float64 cleaned output for float64 input, got {cleaned.dtype}"
    )
    assert np.issubdtype(mask.dtype, np.bool_), (
        f"Expected bool mask, got {mask.dtype}"
    )


def test_float64_mask_is_bool_regardless_of_input_dtype() -> None:
    """The mask must always be bool dtype regardless of the input dtype.

    Ridge tracker and autoencoder callers use the mask as a boolean index.
    A float64 mask would silently permit float arithmetic on indices.
    """
    freqs_hz = _make_freqs(64)
    cfg = FilterConfig()

    for dtype in [np.float32, np.float64]:
        mag = np.abs(np.random.default_rng(1).random((64, 20)).astype(dtype))
        _, mask = prefilter_spectrogram(mag, freqs_hz, cfg)
        assert np.issubdtype(mask.dtype, np.bool_), (
            f"Mask dtype should be bool for {dtype} input, got {mask.dtype}"
        )


# ---------------------------------------------------------------------------
# B. Parameter-boundary combinations
# ---------------------------------------------------------------------------


def test_noise_floor_multiplier_just_above_one() -> None:
    """noise_floor_multiplier=1+eps (just above guard) must be accepted and functional.

    The validator rejects multiplier <= 1.0. The smallest valid value
    (1 + epsilon) should construct without error and produce a more
    permissive mask (almost nothing filtered by amplitude threshold)
    compared to the default multiplier of 3.0.
    """
    freqs_hz = _make_freqs(64)
    rng = np.random.default_rng(42)
    mag = rng.random((64, 30)).astype(np.float32)

    cfg_eps = FilterConfig(noise_floor_multiplier=1.0 + 1e-9)
    cfg_default = FilterConfig(noise_floor_multiplier=3.0)

    _, mask_eps = prefilter_spectrogram(mag, freqs_hz, cfg_eps)
    _, mask_default = prefilter_spectrogram(mag, freqs_hz, cfg_default)

    # A more permissive threshold passes more pixels
    in_band = (freqs_hz >= 25_000.0) & (freqs_hz <= 120_000.0)
    assert mask_eps[in_band, :].sum() >= mask_default[in_band, :].sum(), (
        "multiplier=1+eps should pass at least as many pixels as multiplier=3.0"
    )


def test_noise_floor_window_cols_one() -> None:
    """noise_floor_window_cols=1 (degenerate rolling window) must not crash or NaN.

    A window of 1 means the noise floor at each column equals the
    per-column median itself — equivalent to per-column local threshold.
    This is a valid if aggressive configuration.
    """
    n_freq_bins, n_time_cols = 64, 30
    freqs_hz = _make_freqs(n_freq_bins)
    mag = np.abs(np.random.default_rng(7).random((n_freq_bins, n_time_cols)).astype(np.float32))

    cfg = FilterConfig(noise_floor_window_cols=1)
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert cleaned.shape == (n_freq_bins, n_time_cols)
    assert mask.shape == (n_freq_bins, n_time_cols)
    assert not np.any(np.isnan(cleaned)), "NaN with noise_floor_window_cols=1"
    assert np.issubdtype(mask.dtype, np.bool_)


def test_median_filter_size_one_identity() -> None:
    """median_filter_size=1 makes the spatial filter a no-op (identity).

    With size=1 the median filter returns filtered == magnitude exactly.
    The noise-floor and frequency masks are still applied. This tests the
    degenerate lower bound of the filter parameter.
    """
    n_freq_bins, n_time_cols = 64, 30
    freqs_hz = _make_freqs(n_freq_bins)
    rng = np.random.default_rng(99)
    mag = rng.random((n_freq_bins, n_time_cols)).astype(np.float32)

    cfg = FilterConfig(median_filter_size=1)
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Cleaned should never exceed original magnitude
    assert not np.any(cleaned > mag + 1e-6), (
        "With median_filter_size=1, cleaned should be <= input magnitude everywhere"
    )
    assert not np.any(np.isnan(cleaned)), "NaN with median_filter_size=1"


def test_median_filter_size_large_kernel_on_small_spectrogram() -> None:
    """median_filter_size=99 (kernel wider than spectrogram) must not crash.

    scipy.ndimage.median_filter with mode='reflect' handles kernels larger
    than the input. The output must be a valid array with the same shape
    and no NaN values.
    """
    n_freq_bins, n_time_cols = 32, 20  # deliberately tiny
    freqs_hz = _make_freqs(n_freq_bins)
    mag = np.abs(np.random.default_rng(3).random((n_freq_bins, n_time_cols)).astype(np.float32))

    cfg = FilterConfig(median_filter_size=99)  # kernel >> spectrogram size
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert cleaned.shape == (n_freq_bins, n_time_cols)
    assert mask.shape == (n_freq_bins, n_time_cols)
    assert not np.any(np.isnan(cleaned)), "NaN with median_filter_size=99 on small spectrogram"


# ---------------------------------------------------------------------------
# C. Physical filter invariants
# ---------------------------------------------------------------------------


def test_amplitude_scaling_commutativity() -> None:
    """Scaling the input by k must scale cleaned by k and leave the mask unchanged.

    Physical invariant: the noise floor scales with the signal (both are
    linear magnitudes), so the amplitude mask is invariant under uniform
    scaling. Formally: if mask(mag) == mask(k*mag), then
    cleaned(k*mag) == k * cleaned(mag).

    This invariant confirms the implementation correctly uses a ratio-based
    threshold (filtered > multiplier * noise_floor) rather than an absolute
    threshold that would break at different recording gains.
    """
    n_freq_bins, n_time_cols = 64, 50
    freqs_hz = _make_freqs(n_freq_bins)
    cfg = FilterConfig()

    rng = np.random.default_rng(55)
    mag = rng.random((n_freq_bins, n_time_cols)).astype(np.float32) * 0.01

    # Add a Gaussian-profiled tone so the mask has both True and False regions
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    _add_leakage_ridge(mag, center_bin=tone_bin, amplitude=1.0)

    cleaned_orig, mask_orig = prefilter_spectrogram(mag, freqs_hz, cfg)

    for k in [0.1, 5.0, 1_000.0]:
        mag_k = (mag * k).astype(np.float32)
        cleaned_k, mask_k = prefilter_spectrogram(mag_k, freqs_hz, cfg)

        assert np.array_equal(mask_orig, mask_k), (
            f"Mask changed under amplitude scale k={k}: "
            f"original sum={mask_orig.sum()}, scaled sum={mask_k.sum()}"
        )
        npt.assert_allclose(
            cleaned_k,
            (k * cleaned_orig).astype(np.float32),
            rtol=1e-4,
            err_msg=f"cleaned(k*mag) != k*cleaned(mag) for k={k}",
        )


def test_cleaned_never_exceeds_input_magnitude() -> None:
    """cleaned[i,j] <= magnitude[i,j] for this specific reference input.

    Regression pin for the 'filtered * mask' design choice.

    If the implementation reverted to 'magnitude * mask' without the
    median pre-filter, this test would still pass — but if it
    accidentally returned 'filtered' (without masking) or 'magnitude'
    (bypassing the filter), values could exceed the input at some pixels
    because the mask zeroes some locations and preserves others.

    NOTE: This is NOT a universal physical invariant. In general, a 3×3
    median filter CAN raise a pixel above its original value — a local
    minimum surrounded by 8 high-valued neighbors will be replaced by the
    neighborhood median. For the random-uniform input below (seed 11 with
    one isolated spike), such adversarial local-minimum patterns do not
    occur, or they occur only where the amplitude mask zeros them out
    anyway (because the column's noise floor is also elevated). This test
    pins the behavior for the reference input; it is NOT a claim that
    cleaned <= magnitude holds for every conceivable input.
    """
    n_freq_bins, n_time_cols = 64, 60
    freqs_hz = _make_freqs(n_freq_bins)
    cfg = FilterConfig()

    rng = np.random.default_rng(11)
    mag = rng.random((n_freq_bins, n_time_cols)).astype(np.float32) * 10.0
    # Add an isolated spike to exercise the outlier-suppression path
    mag[20, 30] = 5000.0

    cleaned, _ = prefilter_spectrogram(mag, freqs_hz, cfg)

    violations = cleaned > mag
    assert not np.any(violations), (
        f"cleaned > magnitude at {violations.sum()} pixels; "
        f"max excess = {(cleaned - mag)[violations].max():.4f}. "
        "This likely means 'filtered * mask' was replaced by 'magnitude * mask' "
        "at some pixel, or the median filter is amplifying values."
    )


def test_filtered_output_design_regression() -> None:
    """Cleaned output uses filtered*mask, not magnitude*mask.

    Regression test pinning the implementation's deliberate design choice
    (ROADMAP deviation: see handoff §Key Decisions Made #2).

    If someone changes `cleaned = filtered * mask.astype(filtered.dtype)`
    to `cleaned = magnitude * mask.astype(magnitude.dtype)`, a 1-pixel
    outlier with amplitude 1000x background would survive in the output.
    This test asserts that the outlier is SUPPRESSED (i.e. the median-
    filtered value, not the original, is what gets written to cleaned).
    """
    n_freq_bins, n_time_cols = 64, 40
    freqs_hz = _make_freqs(n_freq_bins)
    cfg = FilterConfig()

    # Uniform background with a Gaussian tone (passes amplitude mask)
    mag = 0.01 * np.ones((n_freq_bins, n_time_cols), dtype=np.float32)
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    _add_leakage_ridge(mag, center_bin=tone_bin, amplitude=10.0)

    # Place a single isolated spike at a non-tone in-band location
    spike_bin = int(np.argmin(np.abs(freqs_hz - 50_000.0)))
    spike_col = n_time_cols // 2
    mag[spike_bin, spike_col] = 1000.0  # 1000× the background

    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # The spike bin/col is definitely in-band.
    assert freqs_hz[spike_bin] >= 25_000.0
    assert freqs_hz[spike_bin] <= 120_000.0

    # After the 3×3 median filter, the spike location sees a neighborhood of
    # background values (~0.01), so filtered[spike_bin, spike_col] ≈ 0.01.
    # Even if the amplitude mask happens to pass it, cleaned must be ~0.01,
    # NOT the original 1000. Threshold at 2.0 to give a healthy margin.
    assert cleaned[spike_bin, spike_col] < 2.0, (
        f"Spike value {mag[spike_bin, spike_col]} survived into cleaned output "
        f"as {cleaned[spike_bin, spike_col]:.2f}. "
        "Implementation should use 'filtered * mask', not 'magnitude * mask'."
    )


# ---------------------------------------------------------------------------
# D. Pathological inputs
# ---------------------------------------------------------------------------


def test_nan_in_input_does_not_spread_or_corrupt_mask() -> None:
    """A single NaN pixel must not corrupt the boolean mask.

    NaN propagates through the 3×3 median filter into the filtered array
    (scipy ndimage behavior), but the amplitude comparison NaN > threshold
    yields False — so the mask at affected pixels must remain False (not
    NaN, not True). The mask itself must stay a clean bool array.

    The cleaned array may contain one NaN at the affected pixel (this is
    acceptable silent corruption vs a hard reject), but the MASK must
    always be bool without NaN.
    """
    n_freq_bins, n_time_cols = 64, 30
    freqs_hz = _make_freqs(n_freq_bins)

    mag = 5.0 * np.ones((n_freq_bins, n_time_cols), dtype=np.float32)
    mag[30, 15] = float("nan")

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Mask must always be bool — NaN in mask would silently corrupt indexing
    assert np.issubdtype(mask.dtype, np.bool_), (
        f"Mask dtype should be bool even with NaN input, got {mask.dtype}"
    )
    assert not np.any(np.isnan(mask.astype(np.float32))), (
        "Mask contains NaN entries after NaN input"
    )

    # Mask at NaN-affected locations must be False, not True
    # (NaN > threshold evaluates to False in numpy)
    assert not mask[30, 15], (
        "Mask at NaN input location should be False, not True"
    )


def test_inf_in_input_does_not_produce_nan_in_output() -> None:
    """An +inf pixel must not produce NaN anywhere in the output.

    An isolated +inf value in the input: the 3×3 median filter will reduce
    it to the background level (all 8 neighbours are finite). The output
    must be free of NaN. (Inf may or may not survive — that's acceptable —
    but NaN must not appear.)
    """
    n_freq_bins, n_time_cols = 64, 40
    freqs_hz = _make_freqs(n_freq_bins)

    mag = np.ones((n_freq_bins, n_time_cols), dtype=np.float32)
    mag[32, 20] = float("inf")

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert not np.any(np.isnan(cleaned)), (
        "NaN found in cleaned output when input contains +inf"
    )
    assert not np.any(np.isnan(mask.astype(np.float32))), (
        "NaN found in mask when input contains +inf"
    )


def test_empty_spectrogram_n_time_cols_zero() -> None:
    """n_time_cols=0 (empty spectrogram) must return empty arrays without crash.

    A USV pipeline might produce a zero-length spectrogram for very short
    audio segments. The function must not raise and must return same-shape
    empty arrays.
    """
    n_freq_bins = 64
    freqs_hz = _make_freqs(n_freq_bins)
    mag = np.zeros((n_freq_bins, 0), dtype=np.float32)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    assert cleaned.shape == (n_freq_bins, 0), (
        f"Expected shape ({n_freq_bins}, 0) for empty input, got {cleaned.shape}"
    )
    assert mask.shape == (n_freq_bins, 0), (
        f"Expected mask shape ({n_freq_bins}, 0), got {mask.shape}"
    )
    # Empty arrays have no NaN by definition
    assert not np.any(np.isnan(cleaned))


def test_non_contiguous_strided_input() -> None:
    """Non-contiguous (strided) input array must produce the same result as its
    contiguous copy.

    Downstream callers may pass slices of larger arrays (e.g.
    ``full_stft[:, start:end]`` is contiguous, but
    ``full_stft[::2, :]`` is not).  scipy.ndimage.median_filter handles
    strided arrays, but this test verifies no silent value error occurs.
    """
    n_freq_bins = 64
    freqs_hz = _make_freqs(n_freq_bins)
    cfg = FilterConfig()

    # Build a strided view (every other row of a 128-row array)
    base = np.random.default_rng(77).random((128, 60)).astype(np.float32)
    strided = base[::2, :]  # shape (64, 60), not C-contiguous
    assert not strided.flags["C_CONTIGUOUS"]

    cleaned_strided, mask_strided = prefilter_spectrogram(strided, freqs_hz, cfg)
    cleaned_contig, mask_contig = prefilter_spectrogram(
        np.ascontiguousarray(strided), freqs_hz, cfg
    )

    npt.assert_array_equal(
        cleaned_strided, cleaned_contig,
        err_msg="Strided and contiguous inputs must produce identical cleaned output"
    )
    npt.assert_array_equal(
        mask_strided, mask_contig,
        err_msg="Strided and contiguous inputs must produce identical mask"
    )


# ---------------------------------------------------------------------------
# E. Downstream-consumer invariants
# ---------------------------------------------------------------------------


def test_mask_any_false_for_all_out_of_band_frequencies() -> None:
    """When all frequency bins are outside [freq_min, freq_max], mask.any() is False.

    Downstream callers (ridge tracker, autoencoder) use the pattern:
    ``if not mask.any(): return``
    to skip processing empty spectrograms. This test ensures that pattern
    works correctly when the frequency axis is entirely out-of-band —
    even with strong amplitude that would otherwise pass the noise-floor
    threshold.
    """
    cfg = FilterConfig()

    # All bins below freq_min
    freqs_all_low = np.linspace(0.0, 20_000.0, 64)  # max = 20 kHz < 25 kHz
    mag = 100.0 * np.ones((64, 30), dtype=np.float32)
    # Add a strong ridge so amplitude mask is definitely passed
    _add_leakage_ridge(mag, center_bin=32, amplitude=500.0)

    _, mask_low = prefilter_spectrogram(mag, freqs_all_low, cfg)
    assert not mask_low.any(), (
        "mask.any() should be False when all frequencies are below freq_min"
    )

    # All bins above freq_max
    freqs_all_high = np.linspace(130_000.0, 150_000.0, 64)  # min = 130 kHz > 120 kHz
    _, mask_high = prefilter_spectrogram(mag, freqs_all_high, cfg)
    assert not mask_high.any(), (
        "mask.any() should be False when all frequencies are above freq_max"
    )


def test_realistic_stft_shape_257_by_200() -> None:
    """Realistic downstream STFT shape (257, 200) must work end-to-end correctly.

    ADR-002 specifies n_fft=512, sr=300_000. This gives 257 frequency bins
    (n_fft//2 + 1) and up to 2334 time columns for a 1-second recording.
    This test uses (257, 200) to keep runtime short while verifying that
    the correct dimensions that modules 17.3, 17.5, and 17.6 will pass
    are handled without error.

    Behavioral checks (not just shape):
    - Out-of-band bins (below 25 kHz and above 120 kHz) are zero in cleaned
    - A synthetic USV call at 70 kHz is preserved in the tone region
    - Background noise in non-call columns is masked out
    """
    n_freq_bins = 257   # n_fft=512 -> 257 rfft bins
    n_time_cols = 200
    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)

    rng = np.random.default_rng(42)
    mag = rng.random((n_freq_bins, n_time_cols)).astype(np.float32) * 0.001

    # Synthetic USV call: 30 columns at ~70 kHz
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    call_cols = slice(80, 110)
    _add_leakage_ridge(mag, center_bin=tone_bin, amplitude=0.5, cols=call_cols)

    cfg = FilterConfig()
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Shape and dtype
    assert cleaned.shape == (n_freq_bins, n_time_cols)
    assert mask.shape == (n_freq_bins, n_time_cols)
    assert cleaned.dtype == np.float32
    assert np.issubdtype(mask.dtype, np.bool_)
    assert not np.any(np.isnan(cleaned))

    # Out-of-band must be zeroed
    out_of_band = (freqs_hz < cfg.freq_min_hz) | (freqs_hz > cfg.freq_max_hz)
    npt.assert_array_equal(
        cleaned[out_of_band, :],
        np.zeros((out_of_band.sum(), n_time_cols), dtype=np.float32),
        err_msg="Out-of-band bins not zeroed in realistic 257-bin spectrogram",
    )

    # The tone bin must pass in the call region (all 30 cols)
    assert mask[tone_bin, call_cols].all(), (
        "Tone bin not fully passing in call region for realistic 257-bin spectrogram"
    )

    # Non-call background columns: tone bin must be masked (background too weak)
    background_cols = slice(0, 60)
    assert not mask[tone_bin, background_cols].any(), (
        "Background columns (no USV) should not pass the tone bin mask in realistic input"
    )
