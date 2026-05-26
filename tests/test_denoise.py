"""Unit tests for app.core.denoise.subtract_temporal_baseline.

The stationary-band-vs-transient test is the central guarantee:

  - A frequency bin with constant amplitude across all time frames
    (equipment harmonic) must collapse to ~epsilon after subtraction.
  - A frequency bin with a single bright frame (a USV burst) must retain
    nearly its full amplitude — only the small temporal-baseline of that
    bin (the noise floor underneath the burst) gets subtracted.

These together verify both halves of the Boll 1979 spectral-subtraction
contract used in the lab pre-CNN preprocessing path.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_spectrogram.app.core.denoise import (
    DEFAULT_BASELINE_PERCENTILE,
    DEFAULT_EPSILON,
    subtract_temporal_baseline,
)


def _make_synthetic_spec(
    n_freq: int = 64,
    n_time: int = 200,
    band_amplitude: float = 1.0,
    transient_amplitude: float = 5.0,
    band_freq_idx: int = 20,
    transient_freq_idx: int = 40,
    transient_time_idx: int = 100,
    background_amplitude: float = 0.05,
    rng_seed: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Build a (n_freq, n_time) magnitude spectrogram with:
      - low broadband background noise everywhere
      - a stationary band at ``band_freq_idx`` (constant amplitude, all frames)
      - a single-frame transient blob at ``(transient_freq_idx, transient_time_idx)``
    """
    rng = np.random.default_rng(rng_seed)
    spec = rng.uniform(0, background_amplitude, size=(n_freq, n_time)).astype(np.float64)
    spec[band_freq_idx, :] = band_amplitude
    spec[transient_freq_idx, transient_time_idx] = transient_amplitude
    return spec, band_freq_idx, transient_freq_idx


class TestStationaryBandRemoval:
    def test_constant_band_is_eliminated(self) -> None:
        spec, band_idx, _ = _make_synthetic_spec()
        cleaned = subtract_temporal_baseline(spec, percentile=10.0)
        # Stationary band: 10th percentile == band amplitude itself, so
        # subtraction floors the entire row at the eps clamp.
        assert np.all(cleaned[band_idx, :] <= 1e-9), (
            "Stationary band should be reduced to ~epsilon, "
            f"got max {cleaned[band_idx, :].max():.2e}"
        )

    def test_floor_at_epsilon_not_zero(self) -> None:
        spec, band_idx, _ = _make_synthetic_spec()
        cleaned = subtract_temporal_baseline(spec, percentile=10.0, epsilon=1e-7)
        # Floor must be epsilon (so downstream log is safe), not literally 0.
        assert cleaned[band_idx, :].min() == pytest.approx(1e-7)


class TestTransientPreservation:
    def test_transient_burst_survives(self) -> None:
        spec, _, transient_idx = _make_synthetic_spec(
            transient_amplitude=5.0,
            background_amplitude=0.05,
        )
        cleaned = subtract_temporal_baseline(spec, percentile=10.0)
        # The transient frame held magnitude=5.0; its 10th-percentile baseline
        # is dominated by the broadband background (≤0.05), so the cleaned
        # transient should be ≥ 4.9.
        original_peak = spec[transient_idx, 100]
        cleaned_peak = cleaned[transient_idx, 100]
        retained = cleaned_peak / original_peak
        assert retained > 0.95, (
            f"Transient should retain >95% of its amplitude, retained {retained:.3f}"
        )

    def test_transient_more_robust_than_band(self) -> None:
        """Side-by-side: same starting amplitude, only one survives."""
        spec, band_idx, transient_idx = _make_synthetic_spec(
            band_amplitude=5.0,
            transient_amplitude=5.0,
        )
        cleaned = subtract_temporal_baseline(spec, percentile=10.0)
        assert cleaned[band_idx, :].max() < 1e-9
        assert cleaned[transient_idx, 100] > 4.9


class TestShapeAndDtype:
    def test_output_shape_matches_input(self) -> None:
        spec = np.random.default_rng(0).uniform(0, 1, size=(40, 150))
        cleaned = subtract_temporal_baseline(spec)
        assert cleaned.shape == spec.shape

    def test_2d_required(self) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            subtract_temporal_baseline(np.zeros(50))
        with pytest.raises(ValueError, match="must be 2D"):
            subtract_temporal_baseline(np.zeros((4, 5, 6)))

    def test_empty_time_axis_returns_copy(self) -> None:
        spec = np.zeros((32, 0))
        cleaned = subtract_temporal_baseline(spec)
        assert cleaned.shape == (32, 0)
        # Should not raise on the np.percentile path; it bails out early.

    def test_default_percentile_matches_constant(self) -> None:
        # Sanity: the public default constant matches the function default.
        assert DEFAULT_BASELINE_PERCENTILE == 10.0
        assert DEFAULT_EPSILON == 1e-10


class TestMedianEnvelopeMethod:
    """The median_envelope method is the amplitude-modulated-band fix.

    Plain percentile subtraction strips a stationary band's *floor* but
    leaves the variance, which is enough to keep the CNN hallucinating a
    horizontal line. Median envelope tracks each bin's slow-varying
    temporal envelope and subtracts it — capturing both the floor and
    the modulation as long as the kernel is wider than any USV burst.
    """

    def test_modulated_band_is_killed_better_than_percentile(self) -> None:
        # Build a band whose amplitude varies sinusoidally across the chunk.
        n_freq, n_time = 32, 800
        rng = np.random.default_rng(0)
        spec = rng.uniform(0, 0.05, size=(n_freq, n_time))
        # Strong amplitude-modulated band at bin 16:
        t = np.arange(n_time)
        spec[16, :] = 1.0 + 0.5 * np.sin(2 * np.pi * t / n_time)  # range ~0.5 to 1.5

        # 0-percentile method: subtract a single floor value
        cleaned_p = subtract_temporal_baseline(spec, method="percentile", percentile=10.0)
        # Envelope method: subtract the per-bin local envelope
        cleaned_e = subtract_temporal_baseline(
            spec, method="median_envelope", envelope_kernel_frames=51
        )

        # Envelope should reduce the band's max residual much more than percentile
        residue_p_max = cleaned_p[16, :].max()
        residue_e_max = cleaned_e[16, :].max()
        assert residue_e_max < 0.5 * residue_p_max, (
            f"envelope residue {residue_e_max:.3f} should be ≤ half of "
            f"percentile residue {residue_p_max:.3f} on a modulated band"
        )

    def test_short_burst_survives_envelope(self) -> None:
        # Single-frame USV-like burst — kernel must be wide enough that
        # the median is unaffected by the burst.
        n_freq, n_time = 32, 800
        rng = np.random.default_rng(42)
        spec = rng.uniform(0, 0.05, size=(n_freq, n_time))
        spec[20, 400] = 5.0  # bright transient

        cleaned = subtract_temporal_baseline(
            spec, method="median_envelope", envelope_kernel_frames=51
        )
        retained = cleaned[20, 400] / 5.0
        assert retained > 0.95, (
            f"USV-like burst should be ≥95% preserved, got {100*retained:.1f}%"
        )

    def test_method_string_validated(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            subtract_temporal_baseline(np.zeros((4, 100)), method="not_a_real_method")

    def test_envelope_kernel_forced_odd(self) -> None:
        # Internally we force the kernel to be odd; even input should still work.
        spec = np.random.default_rng(0).uniform(0, 1, size=(8, 200))
        cleaned = subtract_temporal_baseline(
            spec, method="median_envelope", envelope_kernel_frames=20
        )
        assert cleaned.shape == spec.shape


class TestPercentileAggressiveness:
    def test_higher_percentile_is_more_aggressive(self) -> None:
        """A 25th-percentile baseline removes MORE than 10th-percentile baseline.

        For a bin where USVs occupy ~5% of time and noise the rest, both
        percentiles fall in the noise band — but the 25th is higher, so its
        subtraction is larger and the residual is smaller.
        """
        rng = np.random.default_rng(42)
        n_freq, n_time = 16, 400
        # Pure broadband stationary noise, no transients in this bin.
        spec = rng.uniform(0.5, 1.0, size=(n_freq, n_time))
        cleaned_p10 = subtract_temporal_baseline(spec, percentile=10.0)
        cleaned_p25 = subtract_temporal_baseline(spec, percentile=25.0)
        assert cleaned_p25.mean() < cleaned_p10.mean()
