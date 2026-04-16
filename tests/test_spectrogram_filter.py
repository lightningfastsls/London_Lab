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

Total: 16 tests (9 from ROADMAP, 7 additional)
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


def _make_pure_tone_magnitude(
    n_freq_bins: int,
    n_time_cols: int,
    tone_bin: int,
    tone_amplitude: float = 10.0,
    noise_floor: float = 0.01,
    sample_rate: int = 300_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic magnitude spectrogram with a pure-tone ridge.

    Returns (magnitude, freqs_hz).  The tone_bin should be within the
    25–120 kHz band for the ridge to survive filtering.
    """
    rng = np.random.default_rng(42)
    mag = noise_floor * (1.0 + rng.random((n_freq_bins, n_time_cols)).astype(np.float32))
    mag[tone_bin, :] = tone_amplitude
    freqs_hz = _make_freqs(n_freq_bins, sample_rate)
    return mag, freqs_hz


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
    be zeroed out by the noise-floor mask.  The tone amplitude (10.0) is 1000x
    the background noise (0.01), so it comfortably exceeds 3× local median.
    """
    n_freq_bins, n_time_cols = 64, 50
    freqs_hz = _make_freqs(n_freq_bins)
    # Pick bin closest to 70 kHz (well within 25–120 kHz band)
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))

    rng = np.random.default_rng(0)
    mag = 0.01 * (1.0 + rng.random((n_freq_bins, n_time_cols)).astype(np.float32))
    mag[tone_bin, :] = 10.0  # strong pure tone

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
    # After median filtering a 3x3 neighbourhood of all-1.0 around it, the
    # median is 1.0, so the outlier is replaced by 1.0 — the noise-floor test
    # then masks it (1.0 is not > 3 * median_of_column ~ 1.0*3, borderline, but
    # the point is the raw value should not be ~1000 anymore).
    assert cleaned[outlier_bin, outlier_col] < 100.0, (
        "Outlier pixel was not suppressed: median filter did not work"
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

    # Signal region: columns 30..59 with strong 70 kHz tone
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    mag[tone_bin, 30:] = 5.0  # strong signal in passband

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
    mag[tone_bin, :] = 100.0

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

    Spec says mask &= freqs_hz >= freq_min_hz AND freqs_hz <= freq_max_hz,
    so boundary bins are inclusive.  Tests the >= / <= logic.
    """
    # Construct freqs array with exact boundary values
    n_freq_bins = 64
    n_time_cols = 30
    freq_min = 25_000.0
    freq_max = 120_000.0

    # Build freqs so that two bins land exactly on boundaries
    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)
    # Snap two bins to exact boundary values
    freqs_hz[10] = freq_min
    freqs_hz[50] = freq_max

    mag = 100.0 * np.ones((n_freq_bins, n_time_cols), dtype=np.float32)  # strong signal

    cfg = FilterConfig(freq_min_hz=freq_min, freq_max_hz=freq_max)
    cleaned, mask = prefilter_spectrogram(mag, freqs_hz, cfg)

    # Bins 10 and 50 are ON the boundary — they must NOT be zeroed by the freq mask
    # (they could still be masked by noise floor, but the freq band component must be True)
    # We check that at least some columns at these bins are non-zero
    assert cleaned[10, :].sum() > 0 or cleaned[50, :].sum() > 0, (
        "Boundary frequency bins were masked out — boundaries should be inclusive (>=, <=)"
    )


def test_snr_improves_by_10db_on_noisy_tone() -> None:
    """Filtering must improve SNR by > 10 dB on a synthetic noisy spectrogram.

    Verifies the exit criterion: 'Filter reduces broadband noise on a synthetic
    noisy tone by >10 dB SNR improvement.'

    Construction:
    - Background broadband noise: 0.01 in all bins
    - 70 kHz pure tone: amplitude 1.0 (100x noise → ~40 dB SNR before filtering)
    - After filtering, noise outside the tone should be zeroed, driving SNR higher.

    SNR formula: 10 * log10(signal_power / noise_power)
    where signal = tone_bin row, noise = all other in-band rows.
    """
    n_freq_bins = 129
    n_time_cols = 100
    freqs_hz = np.linspace(0.0, 150_000.0, n_freq_bins)

    rng = np.random.default_rng(42)
    noise_amplitude = 0.01
    tone_amplitude = 1.0

    mag = noise_amplitude * (1.0 + rng.random((n_freq_bins, n_time_cols)).astype(np.float32))
    tone_bin = int(np.argmin(np.abs(freqs_hz - 70_000.0)))
    mag[tone_bin, :] = tone_amplitude

    in_band = (freqs_hz >= 25_000.0) & (freqs_hz <= 120_000.0)

    def snr_db(arr: np.ndarray, signal_bin: int) -> float:
        signal_power = float(np.mean(arr[signal_bin, :] ** 2))
        noise_bins = np.where(in_band)[0]
        noise_bins = noise_bins[noise_bins != signal_bin]
        noise_power = float(np.mean(arr[noise_bins, :] ** 2))
        if noise_power == 0.0:
            return float("inf")
        return 10.0 * np.log10(signal_power / noise_power)

    snr_before = snr_db(mag, tone_bin)

    cfg = FilterConfig()
    cleaned, _ = prefilter_spectrogram(mag, freqs_hz, cfg)

    snr_after = snr_db(cleaned, tone_bin)

    improvement = snr_after - snr_before
    assert improvement >= 10.0, (
        f"Expected >= 10 dB SNR improvement, got {improvement:.1f} dB "
        f"(before={snr_before:.1f} dB, after={snr_after:.1f} dB)"
    )
