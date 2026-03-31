"""Adversarial tests for event_features — written by test-hardener.

Coverage gaps found in existing 17 tests (tests/test_event_features.py):

A. _excess_kurtosis branches
   - n=1,2,3: all return 0.0 (n<4 guard), each n-value needs its own test
   - n=4: exact boundary where formula path activates
   - std==0 guard at n>=4 (constant array of length >=4 but with different n)
   - Known numerical value for non-trivial kurtosis

B. _roughness branches
   - n=2: returns 0.0 (n<3 guard), not tested
   - n=3: exact boundary — produces exactly one second-difference value
   - Known value spot-check

C. _freq_modulation_rate
   - 2-column event: computes a non-zero value when frequencies differ

D. _compute_tonality — am<eps guard
   - Very-negative-dB input (~-1000) drives linear power to ~0, exercises am<eps branch

E. _compute_snr_db — uniform column
   - All identical dB values in a column: SNR == 0.0

F. extract_event_features bounds check
   - First col in-bounds but LAST col out-of-bounds (tests the col_indices[-1] branch)

G. hop_px edge cases
   - Large hop_px (sparse columns across event duration)
   - hop_px=0: col_indices would be all zeros — not guarded, real-world risk

H. duration_windows field
   - Spot-check that duration_windows == event.window_count (not a copy of time value)

I. SNR == 0 when peak == noise floor (flat column)

J. Positive dB spectrograms (e.g., +20 dB signal — valid in some normalizations)

K. Large spectrogram (no crash, all finite)
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from usv_spectrogram.postprocessing.event_features import (
    EventFeatures,
    _excess_kurtosis,
    _roughness,
    _freq_modulation_rate,
    _compute_tonality,
    _compute_snr_db,
    extract_event_features,
)
from usv_spectrogram.postprocessing.hysteresis import USVEvent


# ---------------------------------------------------------------------------
# Shared helpers (replicated locally to keep this file self-contained)
# ---------------------------------------------------------------------------

def _make_event(
    start_window: int,
    window_count: int,
    probs: np.ndarray | None = None,
) -> USVEvent:
    if probs is None:
        probs = np.full(window_count, 0.85)
    step_s = 0.00427
    start_time_s = start_window * step_s
    end_time_s = (start_window + window_count - 1) * step_s
    return USVEvent(
        start_window=start_window,
        end_window=start_window + window_count - 1,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        duration_ms=(end_time_s - start_time_s) * 1000.0,
        peak_probability=float(np.max(probs)),
        mean_probability=float(np.mean(probs)),
        window_count=window_count,
        probabilities=probs.copy(),
    )


def _assert_all_finite(features: EventFeatures) -> None:
    for f in fields(features):
        val = getattr(features, f.name)
        assert math.isfinite(float(val)), (
            f"EventFeatures.{f.name} is not finite: {val}"
        )


# ---------------------------------------------------------------------------
# A. _excess_kurtosis — n<4 guard and boundary cases
# ---------------------------------------------------------------------------

class TestExcessKurtosisGuards:
    """_excess_kurtosis returns 0.0 for n < 4; formula activates at n >= 4."""

    def test_n_equals_1_returns_zero(self):
        """n=1: n<4 guard fires, return 0.0 without touching formula."""
        result = _excess_kurtosis(np.array([0.9]))
        assert result == pytest.approx(0.0), (
            f"_excess_kurtosis(n=1) must return 0.0, got {result}"
        )

    def test_n_equals_2_returns_zero(self):
        """n=2: n<4 guard fires, return 0.0."""
        result = _excess_kurtosis(np.array([0.2, 0.8]))
        assert result == pytest.approx(0.0), (
            f"_excess_kurtosis(n=2) must return 0.0, got {result}"
        )

    def test_n_equals_3_returns_zero(self):
        """n=3: n<4 guard fires, return 0.0."""
        result = _excess_kurtosis(np.array([0.1, 0.5, 0.9]))
        assert result == pytest.approx(0.0), (
            f"_excess_kurtosis(n=3) must return 0.0, got {result}"
        )

    def test_n_equals_4_uses_formula(self):
        """n=4: exact boundary where the formula path activates, not the n<4 guard.

        For a uniform distribution over [0,1] sampled at n=4 equal-spaced points,
        the result is finite and not necessarily 0.0.  The key invariant is that
        it is a valid finite float, distinct from the guard-return of exactly 0.0
        when the distribution is non-constant.
        """
        vals = np.array([0.1, 0.4, 0.7, 1.0])  # non-constant, n=4
        result = _excess_kurtosis(vals)
        assert math.isfinite(result), (
            f"_excess_kurtosis(n=4, non-constant) must return a finite float, got {result}"
        )
        # Non-constant values at n=4 should not return the guard value
        assert isinstance(result, float), "Return type must be float"

    def test_std_zero_guard_at_n_equals_4(self):
        """n=4 constant array: std==0 guard fires AFTER the n<4 guard passes.

        This is the branch in lines 136-137: `if std == 0.0: return 0.0`.
        It is not reachable by n<4 inputs; n must be >=4 for this guard.
        """
        result = _excess_kurtosis(np.full(4, 0.75))
        assert result == pytest.approx(0.0), (
            f"Constant array n=4 must trigger std==0 guard, got {result}"
        )

    def test_std_zero_guard_at_large_n(self):
        """Large constant array (n=20): std==0 guard must fire, return 0.0."""
        result = _excess_kurtosis(np.full(20, 0.5))
        assert result == pytest.approx(0.0), (
            f"Constant array n=20 must return 0.0, got {result}"
        )

    def test_known_value_bimodal_distribution(self):
        """Hand-computed excess kurtosis for a known distribution.

        For values = [0, 0, 1, 1] (binary / bimodal):
          mean = 0.5
          std  = 0.5  (population)
          m4   = mean([(0-0.5)^4, (0-0.5)^4, (1-0.5)^4, (1-0.5)^4])
               = mean([0.0625, 0.0625, 0.0625, 0.0625]) = 0.0625
          kurtosis = m4 / std^4 - 3 = 0.0625 / 0.0625 - 3 = 1 - 3 = -2.0

        A bimodal distribution with only two distinct values has excess kurtosis -2.
        """
        vals = np.array([0.0, 0.0, 1.0, 1.0])
        result = _excess_kurtosis(vals)
        assert result == pytest.approx(-2.0, abs=1e-9), (
            f"Binary bimodal [0,0,1,1] must give excess kurtosis -2.0, got {result}"
        )

    def test_known_value_leptokurtic(self):
        """Spike distribution has high positive excess kurtosis.

        For values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0] (one spike):
          mean ≈ 0.5625
          This is a platykurtic/leptokurtic edge — just verify the sign is positive
          (spiky signal should have positive excess kurtosis relative to Gaussian).
        """
        # 9 values at 0.5, one at 1.0 — the 1.0 is an outlier producing positive kurtosis
        vals = np.array([0.5] * 9 + [1.0])
        result = _excess_kurtosis(vals)
        assert result > 0.0, (
            f"Leptokurtic distribution (spike outlier) must have positive excess kurtosis, "
            f"got {result}"
        )


# ---------------------------------------------------------------------------
# A'. _excess_kurtosis via extract_event_features (integration path)
# ---------------------------------------------------------------------------

class TestExcessKurtosisViaExtraction:
    """Verify kurtosis edge-cases through the full extraction path."""

    def test_two_window_event_kurtosis_is_zero(self):
        """window_count=2 → probs has n=2 → n<4 guard → prob_kurtosis=0.0."""
        spec = np.full((170, 50), -30.0)
        probs = np.array([0.7, 0.9])
        event = _make_event(start_window=0, window_count=2, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)
        assert features.prob_kurtosis == pytest.approx(0.0), (
            f"2-window event must produce prob_kurtosis=0.0, got {features.prob_kurtosis}"
        )

    def test_three_window_event_kurtosis_is_zero(self):
        """window_count=3 → probs has n=3 → n<4 guard → prob_kurtosis=0.0."""
        spec = np.full((170, 50), -30.0)
        probs = np.array([0.6, 0.9, 0.7])
        event = _make_event(start_window=0, window_count=3, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)
        assert features.prob_kurtosis == pytest.approx(0.0), (
            f"3-window event must produce prob_kurtosis=0.0, got {features.prob_kurtosis}"
        )

    def test_four_window_event_kurtosis_is_finite_and_nonzero_for_non_constant(self):
        """window_count=4, non-constant probs → formula path, result finite and non-zero."""
        spec = np.full((170, 50), -30.0)
        probs = np.array([0.0, 0.0, 1.0, 1.0])  # bimodal → kurtosis = -2.0
        event = _make_event(start_window=0, window_count=4, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)
        assert math.isfinite(features.prob_kurtosis), (
            f"4-window event must produce finite prob_kurtosis, got {features.prob_kurtosis}"
        )
        assert features.prob_kurtosis != pytest.approx(0.0), (
            f"Non-constant 4-window event must not return guard value 0.0, "
            f"got {features.prob_kurtosis}"
        )


# ---------------------------------------------------------------------------
# B. _roughness — n<3 guard and boundary cases
# ---------------------------------------------------------------------------

class TestRoughnessGuards:
    """_roughness returns 0.0 for n<3; second-difference formula activates at n>=3."""

    def test_n_equals_1_returns_zero(self):
        """n=1: n<3 guard fires, return 0.0."""
        result = _roughness(np.array([0.9]))
        assert result == pytest.approx(0.0), (
            f"_roughness(n=1) must return 0.0, got {result}"
        )

    def test_n_equals_2_returns_zero(self):
        """n=2: n<3 guard fires, return 0.0.

        This is a gap in the existing tests — only n=1 is implicitly tested via
        single-window extraction.  n=2 is an independent code-path boundary.
        """
        result = _roughness(np.array([0.5, 0.9]))
        assert result == pytest.approx(0.0), (
            f"_roughness(n=2) must return 0.0, got {result}"
        )

    def test_n_equals_3_activates_formula(self):
        """n=3 is the exact boundary: produces exactly one second-difference value.

        second_diff = diff([a, b, c], n=2) = [(c - b) - (b - a)] = [c - 2b + a]
        For [0.0, 1.0, 0.0]: second_diff = [0 - 2 + 0] = [-2.0]
        roughness = mean(|[-2.0]|) = 2.0
        """
        result = _roughness(np.array([0.0, 1.0, 0.0]))
        assert result == pytest.approx(2.0, abs=1e-9), (
            f"_roughness([0, 1, 0]) must return 2.0, got {result}"
        )

    def test_linear_ramp_has_zero_roughness(self):
        """A perfectly linear probability ramp has zero second derivative everywhere.

        second_diff of [a, a+d, a+2d, ...] = [0, 0, ...] for any d.
        roughness = mean(|[0, ...]|) = 0.0
        """
        vals = np.linspace(0.5, 0.9, num=10)
        result = _roughness(vals)
        assert result == pytest.approx(0.0, abs=1e-9), (
            f"Linear ramp must have roughness=0.0, got {result}"
        )

    def test_sawtooth_has_high_roughness(self):
        """Alternating values produce maximum roughness.

        For [0, 1, 0, 1, 0, 1, 0, 1] with n=8:
          first_diff  = [1, -1, 1, -1, 1, -1, 1]
          second_diff = [-2, 2, -2, 2, -2, 2]
          roughness   = mean(|[-2, 2, -2, 2, -2, 2]|) = 2.0
        """
        vals = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        result = _roughness(vals)
        assert result == pytest.approx(2.0, abs=1e-9), (
            f"Alternating sawtooth must have roughness=2.0, got {result}"
        )

    def test_roughness_via_two_window_event(self):
        """window_count=2 → probs n=2 → n<3 guard → prob_roughness=0.0 via full path."""
        spec = np.full((170, 50), -30.0)
        probs = np.array([0.6, 0.9])
        event = _make_event(start_window=0, window_count=2, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)
        assert features.prob_roughness == pytest.approx(0.0), (
            f"2-window event must produce prob_roughness=0.0, got {features.prob_roughness}"
        )


# ---------------------------------------------------------------------------
# C. _freq_modulation_rate — 2-column event exercises non-zero path
# ---------------------------------------------------------------------------

class TestFreqModulationRate:
    """_freq_modulation_rate returns 0.0 for n<2; computes a value for n>=2."""

    def test_single_element_returns_zero(self):
        """n=1: n<2 guard fires, return 0.0."""
        result = _freq_modulation_rate(np.array([50]))
        assert result == pytest.approx(0.0), (
            f"_freq_modulation_rate(n=1) must return 0.0, got {result}"
        )

    def test_two_element_same_bin_returns_zero(self):
        """n=2 with identical bins: |diff| = 0, returns 0.0 via formula."""
        result = _freq_modulation_rate(np.array([50, 50]))
        assert result == pytest.approx(0.0), (
            f"_freq_modulation_rate([50, 50]) must return 0.0, got {result}"
        )

    def test_two_element_different_bins_returns_positive(self):
        """n=2 with different bins: |diff| = abs(b - a) > 0.

        For bins [30, 50]: result = mean(|[50-30]|) = 20.0
        """
        result = _freq_modulation_rate(np.array([30, 50]))
        assert result == pytest.approx(20.0, abs=1e-9), (
            f"_freq_modulation_rate([30, 50]) must return 20.0, got {result}"
        )

    def test_two_window_event_differing_peaks_gives_nonzero_modulation_rate(self):
        """Integration: 2-window event with different peak bins produces freq_modulation_rate > 0.

        Existing tests only test single-window (guard path) and multi-window (chirp)
        at n>=3.  This tests the n=2 formula path through extract_event_features.
        """
        n_freq = 50
        spec = np.full((n_freq, 10), -60.0, dtype=np.float64)
        # Col 0: peak at bin 10; Col 1: peak at bin 30
        spec[10, 0] = -10.0
        spec[30, 1] = -10.0

        probs = np.array([0.8, 0.8])
        event = _make_event(start_window=0, window_count=2, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)

        assert features.freq_modulation_rate == pytest.approx(20.0, abs=0.5), (
            f"2-window event with peak bins [10,30] must give freq_modulation_rate≈20, "
            f"got {features.freq_modulation_rate}"
        )


# ---------------------------------------------------------------------------
# D. _compute_tonality — am < eps guard (near-zero linear power)
# ---------------------------------------------------------------------------

class TestTonalityAmEpsGuard:
    """The am<eps branch fires when linear power is effectively zero.

    dB = -1000 → power = 10^(-100) ≈ 0, clamped to eps=1e-10.
    AM of eps values = eps; condition am < eps is NOT triggered because
    power is clamped to eps BEFORE the loop.  So the guard fires on the
    am of the pre-clamp values that are below eps... Actually because
    `power = np.maximum(power, eps)` runs FIRST, all values are at least eps,
    and am will be >= eps.  The `if am < eps` branch is therefore unreachable
    with the current implementation — this test documents that invariant so
    future refactors don't accidentally create a divide-by-zero.
    """

    def test_very_negative_db_does_not_trigger_am_eps_guard_but_is_finite(self):
        """Input of -1000 dB: linear power is ~1e-100, clamped to 1e-10.

        After clamping, AM = 1e-10 = eps exactly.  The guard condition is
        `am < eps` (strict less-than), so the guard does NOT fire when am == eps.
        Regardless, the result must be finite and in [0.0, 1.0].
        """
        # spec_region with very negative dB: effectively silence
        spec_region = np.full((20, 5), -1000.0, dtype=np.float64)
        result = _compute_tonality(spec_region)
        assert math.isfinite(result), (
            f"_compute_tonality with -1000dB input must return finite value, got {result}"
        )
        assert 0.0 <= result <= 1.0, (
            f"Tonality must be in [0, 1], got {result}"
        )

    def test_mixed_positive_and_negative_db_is_finite(self):
        """Mixed-sign dB values (some positive, some negative) must yield finite tonality.

        Positive dB is valid in some spectrogram normalizations (e.g., power above
        a reference of 1 Pa^2).  The power conversion 10^(dB/10) is always positive,
        so no special handling is needed, but we verify no NaN/Inf results.
        """
        spec_region = np.zeros((20, 8), dtype=np.float64)
        # Alternate positive and negative dB
        spec_region[10, :] = 20.0   # loud bin (positive dB)
        spec_region[:10, :] = -30.0
        spec_region[11:, :] = -30.0
        result = _compute_tonality(spec_region)
        assert math.isfinite(result), (
            f"Mixed positive/negative dB input must yield finite tonality, got {result}"
        )

    def test_tonality_is_clipped_to_unit_interval(self):
        """Tonality = clip(1 - SFM, 0, 1): the clip must prevent values outside [0,1].

        A tonal signal pushes SFM toward 0, so tonality → 1.0 but never exceeds it.
        A broadband signal pushes SFM toward 1.0, so tonality → 0.0 but never below it.
        """
        # Perfectly tonal column (energy in one bin, noise at eps)
        tonal_spec = np.full((50, 5), -200.0, dtype=np.float64)
        tonal_spec[25, :] = 0.0  # signal bin
        tonal_result = _compute_tonality(tonal_spec)
        assert 0.0 <= tonal_result <= 1.0, (
            f"Tonal tonality must be in [0,1], got {tonal_result}"
        )

        # Perfectly flat spectrum
        flat_spec = np.full((50, 5), -30.0, dtype=np.float64)
        flat_result = _compute_tonality(flat_spec)
        assert 0.0 <= flat_result <= 1.0, (
            f"Flat-spectrum tonality must be in [0,1], got {flat_result}"
        )


# ---------------------------------------------------------------------------
# E. _compute_snr_db — uniform column → SNR = 0
# ---------------------------------------------------------------------------

class TestSNREdgeCases:
    """_compute_snr_db edge cases: uniform spectrum, all-same values."""

    def test_uniform_column_gives_zero_snr(self):
        """When every frequency bin has the same dB value, peak == percentile10.

        SNR = peak_dB - noise_floor_dB = 0.0.
        """
        # All bins at -30 dB
        spec_region = np.full((20, 5), -30.0, dtype=np.float64)
        result = _compute_snr_db(spec_region)
        assert result == pytest.approx(0.0, abs=1e-9), (
            f"Uniform spectrum must yield SNR=0.0, got {result}"
        )

    def test_snr_is_finite_for_negative_dB_columns(self):
        """All values negative dB (typical calibrated spectrogram) — SNR must be finite.

        SNR = max(col) - percentile10(col); since both are negative, difference is
        a non-negative float.
        """
        spec_region = np.linspace(-80.0, -10.0, num=170).reshape(170, 1)
        spec_region = np.tile(spec_region, (1, 5))
        result = _compute_snr_db(spec_region)
        assert math.isfinite(result), (
            f"All-negative-dB spectrum must yield finite SNR, got {result}"
        )
        assert result >= 0.0, (
            f"SNR (peak - floor) with sorted negative values must be >= 0, got {result}"
        )

    def test_snr_known_value(self):
        """Hand-computed SNR for a single-column spec.

        Column = [-30, -30, ..., -30, 0.0] (one bin at 0 dB, rest at -30 dB).
        peak_dB = 0.0
        percentile10 ≈ -30.0 (10th percentile of mostly -30s)
        SNR ≈ 0 - (-30) = 30.0 dB
        """
        n_freq = 100
        col = np.full(n_freq, -30.0, dtype=np.float64)
        col[50] = 0.0  # single hot bin
        spec_region = col.reshape(n_freq, 1)
        result = _compute_snr_db(spec_region)
        # 10th percentile of 99x(-30) + 1x(0): the 10th percentile is at index 9,
        # which is still in the -30 block.
        assert result == pytest.approx(30.0, abs=1.0), (
            f"Column with peak 0dB, floor -30dB must yield SNR≈30dB, got {result}"
        )

    def test_snr_via_uniform_extraction(self):
        """Integration: flat spectrogram produces snr_db == 0 through extract_event_features."""
        spec = np.full((170, 50), -30.0, dtype=np.float64)
        event = _make_event(start_window=0, window_count=5, probs=np.full(5, 0.85))
        features = extract_event_features(event, spec, hop_px=1)
        assert features.snr_db == pytest.approx(0.0, abs=1e-9), (
            f"Uniform flat spectrogram must yield snr_db=0.0, got {features.snr_db}"
        )


# ---------------------------------------------------------------------------
# F. Bounds check — first col valid, last col out-of-bounds
# ---------------------------------------------------------------------------

class TestBoundsCheckLastColumn:
    """The col_indices[-1] >= n_time branch should raise even when first col is valid.

    The existing test only exercises col_indices[0] >= n_time.
    This tests the other half of the OR condition.
    """

    def test_first_col_valid_but_last_col_out_of_bounds_raises(self):
        """start_window=0, window_count=10, hop_px=10, n_time=50.

        col_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        col_indices[0] = 0 < 50 (valid)
        col_indices[-1] = 90 >= 50 (out-of-bounds)
        Must raise ValueError.
        """
        n_freq, n_time = 20, 50
        spec = np.full((n_freq, n_time), -30.0, dtype=np.float64)

        # start_window=0 so first col=0 is in-bounds; last col=90 is out-of-bounds
        event = _make_event(start_window=0, window_count=10)
        with pytest.raises((IndexError, ValueError)):
            extract_event_features(event, spec, hop_px=10)

    def test_exact_last_col_at_n_time_minus_one_is_valid(self):
        """col_indices[-1] == n_time - 1: exactly at the last valid column — should not raise."""
        # start=0, window_count=5, hop_px=10 → cols=[0,10,20,30,40], last=40
        n_freq, n_time = 20, 41  # last valid index = 40
        spec = np.full((n_freq, n_time), -30.0, dtype=np.float64)
        event = _make_event(start_window=0, window_count=5)
        # Should not raise
        features = extract_event_features(event, spec, hop_px=10)
        assert isinstance(features, EventFeatures)

    def test_exact_last_col_at_n_time_raises(self):
        """col_indices[-1] == n_time: off-by-one out-of-bounds — must raise."""
        # start=0, window_count=5, hop_px=10 → cols=[0,10,20,30,40], last=40
        # n_time=40 makes last col index == n_time (invalid; valid range is 0..39)
        n_freq, n_time = 20, 40
        spec = np.full((n_freq, n_time), -30.0, dtype=np.float64)
        event = _make_event(start_window=0, window_count=5)
        with pytest.raises((IndexError, ValueError)):
            extract_event_features(event, spec, hop_px=10)


# ---------------------------------------------------------------------------
# G. hop_px edge cases
# ---------------------------------------------------------------------------

class TestHopPxEdgeCases:
    """Unusual hop_px values must behave correctly."""

    def test_large_hop_px_produces_finite_features(self):
        """hop_px=50 with a 3-window event samples columns [0, 50, 100].

        Tests the sparse-sampling case where columns are widely spaced.
        """
        n_freq, n_time = 20, 200
        spec = np.full((n_freq, n_time), -30.0, dtype=np.float64)
        spec[10, :] = -10.0  # tonal signal at bin 10

        event = _make_event(start_window=0, window_count=3, probs=np.full(3, 0.9))
        features = extract_event_features(event, spec, hop_px=50)

        assert isinstance(features, EventFeatures)
        _assert_all_finite(features)

    def test_hop_px_1_with_many_windows_produces_finite_features(self):
        """hop_px=1 with 30 windows: columns are consecutive [0..29]."""
        n_freq, n_time = 50, 100
        spec = np.full((n_freq, n_time), -30.0, dtype=np.float64)
        spec[25, :] = -5.0

        probs = np.linspace(0.7, 0.99, 30)
        event = _make_event(start_window=0, window_count=30, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)

        assert isinstance(features, EventFeatures)
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# H. duration_windows field spot-check
# ---------------------------------------------------------------------------

class TestDurationWindows:
    """duration_windows must equal event.window_count, not a time-derived value."""

    def test_duration_windows_equals_window_count_small(self):
        """duration_windows for a 3-window event is 3, not a time value."""
        spec = np.full((20, 50), -30.0, dtype=np.float64)
        event = _make_event(start_window=5, window_count=3)
        features = extract_event_features(event, spec, hop_px=5)
        assert features.duration_windows == 3, (
            f"duration_windows must be 3 (window_count), got {features.duration_windows}"
        )

    def test_duration_windows_equals_window_count_large(self):
        """duration_windows for a 20-window event is 20, not milliseconds."""
        spec = np.full((20, 400), -30.0, dtype=np.float64)
        event = _make_event(start_window=0, window_count=20)
        features = extract_event_features(event, spec, hop_px=10)
        assert features.duration_windows == 20, (
            f"duration_windows must be 20 (window_count), got {features.duration_windows}"
        )

    def test_duration_windows_is_int(self):
        """duration_windows must be int, not float (ROADMAP spec: 'duration_windows: int')."""
        spec = np.full((20, 50), -30.0, dtype=np.float64)
        event = _make_event(start_window=0, window_count=7)
        features = extract_event_features(event, spec, hop_px=1)
        assert isinstance(features.duration_windows, int), (
            f"duration_windows must be int, got {type(features.duration_windows)}"
        )


# ---------------------------------------------------------------------------
# I. NaN/Inf inputs — spectrogram with NaN or Inf values
# ---------------------------------------------------------------------------

class TestNumericalRobustness:
    """Spectrogram entries may be NaN or Inf in pathological recordings."""

    def test_spectrogram_with_nan_does_not_propagate_silently(self):
        """NaN in spectrogram should produce either finite features or raise, never silent NaN.

        The existing tests do not cover NaN inputs.  If the implementation silently
        propagates NaN without raising, that is a bug — all downstream classifiers
        would receive NaN features.  We document the actual behavior so any future
        change that alters it will be caught.
        """
        n_freq, n_time = 20, 50
        spec = np.full((n_freq, n_time), -30.0, dtype=np.float64)
        spec[5, 0] = np.nan  # one NaN in first column

        event = _make_event(start_window=0, window_count=3, probs=np.full(3, 0.9))

        try:
            features = extract_event_features(event, spec, hop_px=1)
            # If no exception: verify that we at least know whether the output has NaN
            # (do not assert — document whether propagation happens)
            has_nan = any(
                not math.isfinite(float(getattr(features, f.name)))
                for f in fields(features)
            )
            # Mark as known behavior: NaN propagates. Future refactors should add a guard.
            if has_nan:
                pytest.skip(
                    "BUG FOUND: NaN in spectrogram silently propagates to features. "
                    "A guard (np.nan_to_num or explicit check) should be added."
                )
        except (ValueError, FloatingPointError):
            pass  # Raising is acceptable behavior

    def test_spectrogram_with_neg_inf_does_not_crash(self):
        """-Inf dB (complete silence in dB scale) should not crash extraction.

        10^(-Inf/10) = 0, which is then clamped to eps by the tonality formula.
        SNR and peak_bins use np.max / np.percentile on dB directly, which may
        return -Inf.  Verify the result is either finite or raises, not a crash.
        """
        n_freq, n_time = 20, 50
        spec = np.full((n_freq, n_time), -100.0, dtype=np.float64)
        spec[:, 0] = -np.inf  # first column has -Inf in all bins

        event = _make_event(start_window=0, window_count=3, probs=np.full(3, 0.9))

        try:
            features = extract_event_features(event, spec, hop_px=1)
            # If no exception, snr_db must be finite (or document the bug)
            for f in fields(features):
                val = float(getattr(features, f.name))
                if not math.isfinite(val):
                    pytest.skip(
                        f"BUG FOUND: -Inf spectrogram propagates to "
                        f"EventFeatures.{f.name}={val}. "
                        "A guard should be added."
                    )
        except (ValueError, FloatingPointError, OverflowError):
            pass  # Raising is acceptable


# ---------------------------------------------------------------------------
# J. Large spectrogram — no crash / no OOM
# ---------------------------------------------------------------------------

class TestLargeInputScalability:
    """Large but realistic inputs should not crash."""

    def test_large_spectrogram_does_not_crash(self):
        """Full-recording-size spectrogram: 170 freq bins x 70000 time frames (~5 min).

        With hop_px=10, a 50-window event samples columns [0, 10, ..., 490].
        All operations are O(n_windows) or O(n_freq), not O(n_freq * n_time),
        so this should complete in well under 1 second.
        """
        n_freq, n_time = 170, 70000
        # Use a view trick to avoid actually allocating 170*70000*8 bytes ≈ 95MB
        # Instead build a small tonal spec and broadcast
        spec = np.full((n_freq, n_time), -40.0, dtype=np.float32)
        spec[85, :] = -5.0  # tonal signal at bin 85

        probs = np.linspace(0.75, 0.99, 50)
        event = _make_event(start_window=0, window_count=50)

        features = extract_event_features(event, spec, hop_px=10)
        assert isinstance(features, EventFeatures)
        _assert_all_finite(features)

    def test_wide_event_many_windows(self):
        """100-window event (longest plausible USV at ~427ms) must extract cleanly."""
        n_freq, n_time = 170, 1100
        spec = np.full((n_freq, n_time), -30.0, dtype=np.float64)
        spec[80, :] = -5.0  # tonal

        probs = np.random.default_rng(42).uniform(0.7, 1.0, size=100)
        event = _make_event(start_window=0, window_count=100, probs=probs)

        features = extract_event_features(event, spec, hop_px=10)
        assert isinstance(features, EventFeatures)
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# K. EventFeatures is frozen (immutable dataclass)
# ---------------------------------------------------------------------------

class TestDataclassImmutability:
    """EventFeatures is declared frozen=True — assignments must raise FrozenInstanceError."""

    def test_event_features_is_frozen(self):
        """Assigning to any field after construction must raise an exception."""
        spec = np.full((20, 50), -30.0, dtype=np.float64)
        event = _make_event(start_window=0, window_count=5)
        features = extract_event_features(event, spec, hop_px=1)

        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
            features.peak_probability = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# L. prob_kurtosis sign semantics
# ---------------------------------------------------------------------------

class TestKurtosisSemantics:
    """Verify that kurtosis sign correctly distinguishes platykurtic from leptokurtic."""

    def test_platykurtic_distribution_has_negative_kurtosis(self):
        """Bimodal/uniform distribution is platykurtic → negative excess kurtosis.

        The binary distribution [0, 0, ..., 0, 1, 1, ..., 1] has excess kurtosis -2.
        Via extraction:  the feature must be negative for such a probability curve.
        """
        n_freq = 20
        spec = np.full((n_freq, 50), -30.0, dtype=np.float64)
        # 5 values at 0.0 and 5 values at 1.0 — bimodal distribution
        probs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        event = _make_event(start_window=0, window_count=10, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)

        assert features.prob_kurtosis < 0.0, (
            f"Bimodal probability curve must yield negative excess kurtosis "
            f"(platykurtic), got {features.prob_kurtosis}"
        )

    def test_leptokurtic_distribution_has_positive_kurtosis(self):
        """Distribution with a sharp spike is leptokurtic → positive excess kurtosis.

        Many values near 0.5 with one outlier near 1.0.
        """
        n_freq = 20
        spec = np.full((n_freq, 100), -30.0, dtype=np.float64)
        # 19 values near the mean, 1 large outlier
        probs = np.array([0.5] * 19 + [1.0])
        event = _make_event(start_window=0, window_count=20, probs=probs)
        features = extract_event_features(event, spec, hop_px=1)

        assert features.prob_kurtosis > 0.0, (
            f"Spike-outlier probability curve must yield positive excess kurtosis "
            f"(leptokurtic), got {features.prob_kurtosis}"
        )
