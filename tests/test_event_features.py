"""Tests for event_features module — written by test-architect BEFORE implementation.

ROADMAP test plan coverage (section 15.4):
  1. Constant-probability event → prob_std=0, prob_roughness=0
     -> test_constant_probability_event_has_zero_std_and_smoothness
  2. Tonal synthetic signal (single freq) → high tonality (>0.5)
     -> test_tonal_spectrogram_produces_high_tonality
  3. Broadband noise → low tonality (<0.2)
     -> test_broadband_spectrogram_produces_low_tonality
  4. Monotonically increasing frequency → freq_modulation_rate > 0, freq_range > 0
     -> test_monotonically_increasing_frequency_has_positive_continuity_and_range
  5. Feature extraction handles edge events (start/end of spectrogram)
     -> test_event_at_start_of_spectrogram_does_not_error
     -> test_event_at_end_of_spectrogram_does_not_error
  6. All features are finite (no NaN/Inf) on real spectrogram data
     -> test_all_features_are_finite_on_realistic_input

Additional coverage (recurring gap patterns):
  - Empty/null: single-column (1-window) event -> test_single_window_event_produces_finite_features
  - Dataclass fields existence and types -> test_event_features_dataclass_has_all_required_fields
  - All-zero spectrogram -> test_zero_spectrogram_does_not_produce_nan_or_inf
  - Column mapping: start_col = start_window * hop_px -> test_hop_px_controls_spectrogram_column_mapping
  - Probability stat spot-check with hand-computed values -> test_prob_stats_spot_check_known_values
  - SNR positive when peak clearly above noise -> test_snr_is_positive_when_signal_above_noise_floor
  - Tonality ordering: tonal > broadband -> test_tonality_is_higher_for_tonal_than_broadband
  - freq_range_bins zero for flat frequency -> test_stationary_frequency_has_zero_range
  - Invalid event window out-of-bounds spectrogram -> test_event_window_beyond_spectrogram_raises

Total: 14 tests (6 from ROADMAP, 8 additional)

Implementation will live at:
  src/usv_spectrogram/postprocessing/event_features.py

All tests will fail with ImportError until the module is created.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

# This import WILL fail until the module is implemented — that is expected.
from usv_spectrogram.postprocessing.event_features import (
    EventFeatures,
    extract_event_features,
)
from usv_spectrogram.postprocessing.hysteresis import USVEvent


# ---------------------------------------------------------------------------
# Helpers — build minimal USVEvent and spectrogram inputs
# ---------------------------------------------------------------------------

def _make_event(
    start_window: int,
    window_count: int,
    probs: np.ndarray | None = None,
) -> USVEvent:
    """Build a USVEvent with plausible values for testing."""
    if probs is None:
        probs = np.full(window_count, 0.85)
    step_s = 0.00427  # ~4.27 ms per window (hop=128/sr=300k * stride=10)
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


def _make_spectrogram(n_freq: int, n_time: int, fill: float = 0.0) -> np.ndarray:
    """Return a (n_freq, n_time) spectrogram array filled with `fill`."""
    return np.full((n_freq, n_time), fill, dtype=np.float32)


def _make_tonal_spectrogram(
    n_freq: int,
    n_time: int,
    active_bin: int,
    signal_db: float = -10.0,
    noise_db: float = -60.0,
) -> np.ndarray:
    """Return a spectrogram where energy is concentrated in a single frequency bin.

    All other bins are at noise_db. Simulates a pure tone USV.
    """
    spec = np.full((n_freq, n_time), noise_db, dtype=np.float64)
    spec[active_bin, :] = signal_db
    return spec


def _make_broadband_spectrogram(
    n_freq: int,
    n_time: int,
    level_db: float = -30.0,
) -> np.ndarray:
    """Return a spectrogram with equal energy across all frequency bins (flat spectrum)."""
    return np.full((n_freq, n_time), level_db, dtype=np.float64)


def _make_chirp_spectrogram(
    n_freq: int,
    n_time: int,
    start_bin: int,
    end_bin: int,
    signal_db: float = -10.0,
    noise_db: float = -60.0,
) -> np.ndarray:
    """Return a spectrogram where the peak frequency linearly sweeps from start_bin to end_bin.

    Each time column has energy concentrated at one bin that increases monotonically.
    """
    spec = np.full((n_freq, n_time), noise_db, dtype=np.float64)
    for col in range(n_time):
        # Linear interpolation of peak bin across time
        frac = col / max(n_time - 1, 1)
        bin_idx = int(round(start_bin + frac * (end_bin - start_bin)))
        bin_idx = max(0, min(n_freq - 1, bin_idx))
        spec[bin_idx, col] = signal_db
    return spec


# ---------------------------------------------------------------------------
# ROADMAP test plan item 1 — constant probability
# ---------------------------------------------------------------------------

class TestConstantProbabilityEvent:
    """Spec requirement: constant-probability event → prob_std=0, prob_roughness=0."""

    def test_constant_probability_event_has_zero_std_and_smoothness(self):
        """Constant prob curve has zero variance and zero second derivative everywhere.

        Hand computation:
          probs = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
          std = 0.0  (all values equal)
          first_diff = [0, 0, 0, 0, 0, 0, 0]
          second_diff = [0, 0, 0, 0, 0, 0]
          smoothness = mean(|second_diff|) = 0.0
        """
        n_windows = 8
        probs = np.full(n_windows, 0.85)
        event = _make_event(start_window=5, window_count=n_windows, probs=probs)

        # Spectrogram columns 5*10=50 through 57*10=570, need at least 580 cols
        n_freq, n_time = 170, 600
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=80)

        features = extract_event_features(event, spec, hop_px=10)

        assert features.prob_std == pytest.approx(0.0, abs=1e-9), (
            "Constant probabilities must yield prob_std == 0.0"
        )
        assert features.prob_roughness == pytest.approx(0.0, abs=1e-9), (
            "Constant probabilities must yield prob_roughness == 0.0 "
            "(mean |second derivative| is zero)"
        )

    def test_prob_stats_spot_check_known_values(self):
        """Spot-check prob_std and peak_probability against hand-computed values.

        probs = [0.6, 0.8, 1.0, 0.8, 0.6]
          mean = (0.6+0.8+1.0+0.8+0.6)/5 = 3.8/5 = 0.76
          peak = 1.0
          deviations^2 = [0.0256, 0.0016, 0.0576, 0.0016, 0.0256]
          variance = 0.112 / 5 = 0.0224  (population std)
          std = sqrt(0.0224) ≈ 0.14967
        """
        probs = np.array([0.6, 0.8, 1.0, 0.8, 0.6])
        event = _make_event(start_window=2, window_count=5, probs=probs)

        n_freq, n_time = 170, 100
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=60)

        features = extract_event_features(event, spec, hop_px=10)

        assert features.peak_probability == pytest.approx(1.0, abs=1e-9), (
            "Peak probability must equal max(probs) = 1.0"
        )
        assert features.mean_probability == pytest.approx(0.76, abs=1e-9), (
            "Mean probability: (0.6+0.8+1.0+0.8+0.6)/5 = 0.76"
        )
        # Accept both population std (ddof=0) and sample std (ddof=1) since the spec
        # does not specify; verify it is in the right ballpark.
        assert 0.13 < features.prob_std < 0.17, (
            f"prob_std for [0.6,0.8,1.0,0.8,0.6] should be ≈0.15, got {features.prob_std}"
        )


# ---------------------------------------------------------------------------
# ROADMAP test plan item 2 — tonal signal → high tonality
# ---------------------------------------------------------------------------

class TestTonalSignal:
    """Spec requirement: tonal synthetic signal → tonality > 0.5."""

    def test_tonal_spectrogram_produces_high_tonality(self):
        """Single-frequency spectrogram must produce tonality > 0.5 per spec.

        The spectrogram has almost all energy in one bin (active_bin=80),
        all others at -60 dB. This is the most tonal signal possible.
        Spec states: 'Values > 0.3 suggest tonal content (DeepSqueak convention)'
        and test plan requires tonality > 0.5 for synthetic single-freq input.
        """
        n_freq, n_time = 170, 50
        active_bin = 80
        spec = _make_tonal_spectrogram(
            n_freq, n_time, active_bin=active_bin, signal_db=-10.0, noise_db=-60.0
        )

        event = _make_event(start_window=0, window_count=5, probs=np.full(5, 0.9))
        features = extract_event_features(event, spec, hop_px=1)

        assert features.tonality > 0.5, (
            f"Single-frequency spectrogram must have tonality > 0.5, got {features.tonality}. "
            "Spec (section 15.4): 'Values > 0.3 suggest tonal content; "
            "test plan requires > 0.5 for synthetic single-freq signal.'"
        )


# ---------------------------------------------------------------------------
# ROADMAP test plan item 3 — broadband noise → low tonality
# ---------------------------------------------------------------------------

class TestBroadbandSignal:
    """Spec requirement: broadband noise → tonality < 0.2."""

    def test_broadband_spectrogram_produces_low_tonality(self):
        """Flat spectrum (equal power all bins) must yield tonality < 0.2 per spec.

        Standard Spectral Flatness Measure (GM/AM) = 1.0 for perfectly flat spectrum.
        But the spec defines tonality such that broadband gives LOW values (< 0.2).
        This confirms the implementer uses tonality = 1 - SFM or equivalent inversion.
        """
        n_freq, n_time = 170, 50
        spec = _make_broadband_spectrogram(n_freq, n_time, level_db=-30.0)

        event = _make_event(start_window=0, window_count=5, probs=np.full(5, 0.9))
        features = extract_event_features(event, spec, hop_px=1)

        assert features.tonality < 0.2, (
            f"Broadband (flat-spectrum) spectrogram must have tonality < 0.2, "
            f"got {features.tonality}. Spec (section 15.4) test plan item 3."
        )

    def test_tonality_is_higher_for_tonal_than_broadband(self):
        """Tonal signal must produce strictly higher tonality than broadband signal.

        This is a monotonicity invariant that any correct tonality implementation
        must satisfy, regardless of the exact formula used.
        """
        n_freq, n_time = 170, 50

        tonal_spec = _make_tonal_spectrogram(
            n_freq, n_time, active_bin=80, signal_db=-10.0, noise_db=-60.0
        )
        broadband_spec = _make_broadband_spectrogram(n_freq, n_time, level_db=-30.0)

        event = _make_event(start_window=0, window_count=5, probs=np.full(5, 0.9))

        tonal_features = extract_event_features(event, tonal_spec, hop_px=1)
        broadband_features = extract_event_features(event, broadband_spec, hop_px=1)

        assert tonal_features.tonality > broadband_features.tonality, (
            f"Tonal tonality ({tonal_features.tonality}) must exceed broadband "
            f"tonality ({broadband_features.tonality})"
        )


# ---------------------------------------------------------------------------
# ROADMAP test plan item 4 — monotonically increasing frequency
# ---------------------------------------------------------------------------

class TestMonotonicallyIncreasingFrequency:
    """Spec requirement: monotonically increasing freq → freq_modulation_rate > 0, freq_range > 0."""

    def test_monotonically_increasing_frequency_has_positive_continuity_and_range(self):
        """Chirp (linearly swept frequency) must yield freq_modulation_rate > 0 and freq_range > 0.

        The chirp sweeps from bin 30 to bin 60 across 10 columns.
        Each column's peak_freq_bin increases by 3 (= 30 bins / 9 steps).
        freq_modulation_rate = mean |delta peak_freq_bin| between adjacent columns > 0.
        freq_range_bins = max_peak_bin - min_peak_bin = 30 > 0.
        """
        n_freq, n_time = 170, 50
        start_bin, end_bin = 30, 60
        spec = _make_chirp_spectrogram(
            n_freq, n_time, start_bin=start_bin, end_bin=end_bin,
            signal_db=-10.0, noise_db=-60.0
        )

        # Use 10 columns: hop_px=5 so 10 windows cover cols 0..45
        n_windows = 10
        event = _make_event(
            start_window=0, window_count=n_windows, probs=np.full(n_windows, 0.9)
        )

        features = extract_event_features(event, spec, hop_px=5)

        assert features.freq_modulation_rate > 0, (
            f"Chirp signal must have freq_modulation_rate > 0, got {features.freq_modulation_rate}. "
            "Spec: mean |delta peak_freq| between columns."
        )
        assert features.freq_range_bins > 0, (
            f"Chirp signal must have freq_range_bins > 0, got {features.freq_range_bins}. "
            "Spec: frequency modulation extent (max - min peak bin)."
        )

    def test_stationary_frequency_has_zero_range(self):
        """Pure-tone (non-sweeping) event must have freq_range_bins = 0.

        When the peak frequency bin is the same in every column,
        max(peak_bins) - min(peak_bins) = 0.
        """
        n_freq, n_time = 170, 50
        fixed_bin = 70
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=fixed_bin)

        n_windows = 8
        event = _make_event(
            start_window=0, window_count=n_windows, probs=np.full(n_windows, 0.9)
        )

        features = extract_event_features(event, spec, hop_px=1)

        assert features.freq_range_bins == pytest.approx(0.0, abs=0.5), (
            f"Stationary-frequency event must have freq_range_bins ≈ 0, "
            f"got {features.freq_range_bins}"
        )


# ---------------------------------------------------------------------------
# ROADMAP test plan item 5 — edge events (boundary handling)
# ---------------------------------------------------------------------------

class TestEdgeEvents:
    """Spec requirement: feature extraction handles events at start/end of spectrogram."""

    def test_event_at_start_of_spectrogram_does_not_error(self):
        """An event starting at window 0 (spectrogram column 0) must extract without error.

        This is the left-boundary case: start_col = 0 * hop_px = 0.
        """
        n_freq, n_time = 170, 100
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=80)

        n_windows = 5
        event = _make_event(start_window=0, window_count=n_windows)

        features = extract_event_features(event, spec, hop_px=10)

        # Must return an EventFeatures with all finite values
        assert isinstance(features, EventFeatures)
        _assert_all_finite(features)

    def test_event_at_end_of_spectrogram_does_not_error(self):
        """An event whose last hop-spaced column is near the end of the spectrogram.

        With hop_px=10, an event at start_window=9, window_count=10 samples
        columns [90, 100, 110, ..., 180]. Need n_time >= 181.
        """
        n_freq, n_time = 170, 200
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=80)

        n_windows = 10
        # col_indices = [90, 100, 110, ..., 180]; last col = 180 < 200
        event = _make_event(start_window=9, window_count=n_windows)

        features = extract_event_features(event, spec, hop_px=10)

        assert isinstance(features, EventFeatures)
        _assert_all_finite(features)


# ---------------------------------------------------------------------------
# ROADMAP test plan item 6 — all features finite on realistic input
# ---------------------------------------------------------------------------

class TestFiniteFeatures:
    """Spec requirement: all features are finite (no NaN/Inf) on real spectrogram data."""

    def test_all_features_are_finite_on_realistic_input(self):
        """Full extraction on a realistic spectrogram (USV-like tonal signal) yields finite features.

        This catches divide-by-zero bugs in tonality, SNR, and smoothness computation.
        The spectrogram simulates 170 freq bins x 200 time frames (as used by the VQ-VAE
        and described in docs/modules/ for the 20-120 kHz band at ~586 Hz/bin).
        """
        n_freq, n_time = 170, 200
        spec = _make_tonal_spectrogram(
            n_freq, n_time, active_bin=85, signal_db=-15.0, noise_db=-55.0
        )

        probs = np.array([0.78, 0.85, 0.92, 0.97, 0.95, 0.88, 0.82, 0.79])
        event = _make_event(start_window=5, window_count=8, probs=probs)

        features = extract_event_features(event, spec, hop_px=10)

        _assert_all_finite(features)

    def test_zero_spectrogram_does_not_produce_nan_or_inf(self):
        """A spectrogram of all zeros (silence) must not produce NaN or Inf features.

        This is the degenerate-input guard: tonality = GM/AM is 0/0 when all values
        are zero. The implementation must handle this (e.g., return 0.0 or clamp).
        Spec section 15.4 exit criteria: 'No NaN or Inf values on any of the 126 recordings.'
        """
        n_freq, n_time = 170, 100
        spec = _make_spectrogram(n_freq, n_time, fill=0.0)

        event = _make_event(start_window=2, window_count=6)

        features = extract_event_features(event, spec, hop_px=10)

        _assert_all_finite(features)

    def test_snr_is_positive_when_signal_above_noise_floor(self):
        """SNR in dB must be positive when peak power clearly exceeds the 10th-percentile floor.

        Spec: SNR = mean(10 * log10(peak_power / noise_floor))
        where noise_floor = 10th percentile per column.

        Setup: columns have one bin at power 100 (linear), all others at power 1.
        10th-percentile ≈ 1 (noise).
        SNR = 10 * log10(100 / 1) = 20 dB > 0.
        """
        # Build spectrogram in linear power units (the spec says 'power spectrum')
        n_freq, n_time = 20, 30
        # All bins at power 1 (noise), one bin at power 100 (signal)
        spec_linear = np.ones((n_freq, n_time), dtype=np.float64)
        spec_linear[10, :] = 100.0

        # Convert to dB as the rest of the pipeline does
        spec_db = 10.0 * np.log10(spec_linear)

        event = _make_event(start_window=0, window_count=5, probs=np.full(5, 0.9))

        features = extract_event_features(event, spec_db, hop_px=1)

        assert features.snr_db > 0, (
            f"SNR must be positive when peak power (100) >> noise floor (1), "
            f"got snr_db={features.snr_db}"
        )


# ---------------------------------------------------------------------------
# Dataclass structure tests
# ---------------------------------------------------------------------------

class TestEventFeaturesDataclass:
    """Verify EventFeatures has all required fields with correct types."""

    def test_event_features_dataclass_has_all_required_fields(self):
        """EventFeatures must expose all 11 fields defined in spec section 15.4.

        This test will fail at import time until event_features.py is created,
        and will fail at runtime if any field is missing.
        """
        required_fields = {
            "peak_probability",
            "mean_probability",
            "prob_std",
            "prob_kurtosis",
            "prob_roughness",
            "duration_windows",
            "tonality",
            "mean_peak_freq_bin",
            "freq_range_bins",
            "freq_modulation_rate",
            "snr_db",
        }
        actual_fields = {f.name for f in fields(EventFeatures)}
        missing = required_fields - actual_fields
        assert not missing, (
            f"EventFeatures is missing required fields: {sorted(missing)}. "
            "See spec section 15.4 for the full list."
        )

    def test_event_features_fields_are_numeric(self):
        """All EventFeatures fields must hold float or int values (not arrays or strings).

        Spec defines: duration_windows: int, all others: float.
        """
        n_freq, n_time = 170, 50
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=80)
        event = _make_event(start_window=0, window_count=5)

        features = extract_event_features(event, spec, hop_px=1)

        for f in fields(EventFeatures):
            val = getattr(features, f.name)
            assert isinstance(val, (int, float, np.floating, np.integer)), (
                f"EventFeatures.{f.name} must be numeric, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Column mapping tests
# ---------------------------------------------------------------------------

class TestColumnMapping:
    """Verify hop_px correctly maps window indices to spectrogram columns."""

    def test_hop_px_controls_spectrogram_column_mapping(self):
        """start_col = start_window * hop_px — changing hop_px must shift the extracted region.

        We construct a spectrogram where the tonal bin differs in two halves:
        - Columns 0-29: active_bin = 50
        - Columns 30-59: active_bin = 100

        With hop_px=10:
          - event at start_window=0 uses cols 0..0+n_windows-1 (first half → bin≈50)
          - event at start_window=3 uses cols 30..30+n_windows-1 (second half → bin≈100)
        mean_peak_freq_bin for the second event must be higher.
        """
        n_freq, n_time = 170, 60
        n_windows = 3

        spec = np.full((n_freq, n_time), -60.0, dtype=np.float64)
        spec[50, :30] = -10.0   # First half: bin 50 is active
        spec[100, 30:] = -10.0  # Second half: bin 100 is active

        event_first_half = _make_event(start_window=0, window_count=n_windows)
        event_second_half = _make_event(start_window=3, window_count=n_windows)

        f_first = extract_event_features(event_first_half, spec, hop_px=10)
        f_second = extract_event_features(event_second_half, spec, hop_px=10)

        assert f_second.mean_peak_freq_bin > f_first.mean_peak_freq_bin, (
            f"Event in second half (cols 30-59, active bin 100) must have higher "
            f"mean_peak_freq_bin than event in first half (cols 0-29, active bin 50). "
            f"Got first={f_first.mean_peak_freq_bin}, second={f_second.mean_peak_freq_bin}. "
            "This verifies start_col = start_window * hop_px."
        )


# ---------------------------------------------------------------------------
# Single-window edge case
# ---------------------------------------------------------------------------

class TestSingleWindowEvent:
    """Spec gap: single-window events (window_count=1) must not crash."""

    def test_single_window_event_produces_finite_features(self):
        """An event spanning exactly one window (one spectrogram column) must work.

        Smoothness and freq_modulation_rate require at least 2 values for differences.
        The implementation must handle the degenerate n=1 case gracefully.
        """
        n_freq, n_time = 170, 50
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=80)

        probs = np.array([0.90])
        event = _make_event(start_window=5, window_count=1, probs=probs)

        features = extract_event_features(event, spec, hop_px=5)

        assert isinstance(features, EventFeatures)
        _assert_all_finite(features)
        # A single-window event has no "continuity" — result should be 0 or well-defined
        assert features.freq_modulation_rate >= 0.0, (
            f"freq_modulation_rate must be >= 0 for single-window event, "
            f"got {features.freq_modulation_rate}"
        )
        assert features.prob_roughness >= 0.0, (
            f"prob_roughness must be >= 0 for single-window event, "
            f"got {features.prob_roughness}"
        )


# ---------------------------------------------------------------------------
# Out-of-bounds guard
# ---------------------------------------------------------------------------

class TestOutOfBoundsGuard:
    """Verify out-of-bounds event windows raise rather than silently produce garbage."""

    def test_event_window_beyond_spectrogram_raises(self):
        """An event whose columns extend beyond spectrogram width must raise an error.

        start_col = start_window * hop_px; if start_col >= n_time, the extraction
        has no valid columns. The implementation must raise IndexError or ValueError,
        not return NaN-filled features silently.
        """
        n_freq, n_time = 170, 50  # 50 columns
        spec = _make_tonal_spectrogram(n_freq, n_time, active_bin=80)

        # start_window=10, hop_px=10 → start_col=100, which is beyond n_time=50
        event = _make_event(start_window=10, window_count=5)

        with pytest.raises((IndexError, ValueError)):
            extract_event_features(event, spec, hop_px=10)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _assert_all_finite(features: "EventFeatures") -> None:
    """Assert that every numeric field in an EventFeatures instance is finite."""
    for f in fields(features):
        val = getattr(features, f.name)
        assert math.isfinite(float(val)), (
            f"EventFeatures.{f.name} is not finite: {val}"
        )
