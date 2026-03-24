"""Tests for acoustic property extractors (probing targets).

21 test cases covering:
1.  peak_frequency — single-peak column
2.  spectral_centroid — uniform dB column
3.  energy — silence column
4.  energy — loud column
5.  is_voiced — true (loud)
6.  is_voiced — false (silence)
7.  frequency_direction — rising
8.  frequency_direction — falling
9.  frequency_direction — flat
10. bout_position — various positions
11. time_since_last_usv — with and without preceding onsets
12. time_since_last_usv — unsorted onsets handled
13. extract_all_properties — shape and keys
14. extract_all_properties — empty spectrogram
15. extract_all_properties — non-2D spec raises ValueError
16. extract_all_properties — unsorted onsets handled
17. AcousticPropertyConfig — invalid freq range
18. AcousticPropertyConfig — invalid n_freq_bins
19. AcousticPropertyConfig — invalid sample_rate
20. AcousticPropertyConfig — negative direction_threshold
21. AcousticPropertyConfig — zero direction_threshold allowed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Bootstrap
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.analysis.acoustic_properties import (
    AcousticPropertyConfig,
    peak_frequency,
    spectral_centroid,
    energy,
    is_voiced,
    frequency_direction,
    bout_position,
    time_since_last_usv,
    extract_all_properties,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> AcousticPropertyConfig:
    return AcousticPropertyConfig()


@pytest.fixture
def single_peak_column() -> np.ndarray:
    """Column with a single peak at bin 85 (center of 170 bins)."""
    col = np.full(170, -100.0)
    col[85] = -10.0
    return col


@pytest.fixture
def uniform_column() -> np.ndarray:
    """Column with uniform dB across all bins."""
    return np.full(170, -30.0)


@pytest.fixture
def silence_column() -> np.ndarray:
    """Very quiet column — well below any threshold."""
    return np.full(170, -200.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPeakFrequency:
    def test_single_peak(
        self, single_peak_column: np.ndarray, default_config: AcousticPropertyConfig,
    ) -> None:
        """Bin 85 peak → linspace(20000, 120000, 170)[85] ≈ 70000 Hz."""
        freq = peak_frequency(single_peak_column, default_config)
        expected = np.linspace(20_000, 120_000, 170)[85]
        assert abs(freq - expected) < 1.0  # sub-Hz tolerance


class TestSpectralCentroid:
    def test_uniform_column(
        self, uniform_column: np.ndarray, default_config: AcousticPropertyConfig,
    ) -> None:
        """Uniform dB → centroid equals mean frequency = 70000 Hz."""
        cent = spectral_centroid(uniform_column, default_config)
        expected = (20_000 + 120_000) / 2.0  # 70000
        assert abs(cent - expected) < 1.0


class TestEnergy:
    def test_silence(self, silence_column: np.ndarray) -> None:
        """At -200 dB, energy ≈ 0 (10^(-20) per bin)."""
        e = energy(silence_column)
        assert e < 1e-15

    def test_loud(self) -> None:
        """170 bins at -10 dB → 170 * 10^(-1) = 17.0."""
        col = np.full(170, -10.0)
        e = energy(col)
        assert abs(e - 17.0) < 1e-10


class TestIsVoiced:
    def test_voiced_true(
        self, single_peak_column: np.ndarray, default_config: AcousticPropertyConfig,
    ) -> None:
        """Peak at -10 dB > threshold -50 dB → voiced."""
        assert is_voiced(single_peak_column, default_config) is True

    def test_voiced_false(
        self, silence_column: np.ndarray, default_config: AcousticPropertyConfig,
    ) -> None:
        """All at -200 dB < threshold -50 dB → not voiced."""
        assert is_voiced(silence_column, default_config) is False


class TestFrequencyDirection:
    def test_rising(self, default_config: AcousticPropertyConfig) -> None:
        """Peak shifts upward by > threshold → 'rising'."""
        prev = np.full(170, -100.0)
        curr = np.full(170, -100.0)
        prev[50] = -10.0  # lower bin
        curr[80] = -10.0  # higher bin
        result = frequency_direction(prev, curr, default_config)
        assert result == "rising"

    def test_falling(self, default_config: AcousticPropertyConfig) -> None:
        """Peak shifts downward by > threshold → 'falling'."""
        prev = np.full(170, -100.0)
        curr = np.full(170, -100.0)
        prev[80] = -10.0
        curr[50] = -10.0
        result = frequency_direction(prev, curr, default_config)
        assert result == "falling"

    def test_flat(self, default_config: AcousticPropertyConfig) -> None:
        """Identical columns → 'flat'."""
        col = np.full(170, -100.0)
        col[85] = -10.0
        result = frequency_direction(col, col.copy(), default_config)
        assert result == "flat"


class TestBoutPosition:
    def test_various_positions(self) -> None:
        assert bout_position(0, 10) == 0.0
        assert abs(bout_position(5, 10) - 0.5) < 1e-10
        assert bout_position(0, 1) == 0.0


class TestTimeSinceLastUSV:
    def test_with_preceding_onset(
        self, default_config: AcousticPropertyConfig,
    ) -> None:
        """Frame 100, onsets [20, 50, 80] → delta=20 frames → ms."""
        onsets = np.array([20, 50, 80], dtype=np.int64)
        ms = time_since_last_usv(100, onsets, default_config)
        expected_ms = 20 * 128 / 300_000 * 1000.0
        assert abs(ms - expected_ms) < 1e-6

    def test_no_preceding_onset(
        self, default_config: AcousticPropertyConfig,
    ) -> None:
        """Frame 10, onsets [50, 80] → no preceding → -1.0."""
        onsets = np.array([50, 80], dtype=np.int64)
        ms = time_since_last_usv(10, onsets, default_config)
        assert ms == -1.0


class TestExtractAllProperties:
    def test_shape_and_keys(
        self, default_config: AcousticPropertyConfig,
    ) -> None:
        """(170, 10) spectrogram → all arrays length 10 with correct keys."""
        spec = np.random.default_rng(42).uniform(-80, -10, size=(170, 10))
        onsets = np.array([2, 5], dtype=np.int64)
        result = extract_all_properties(spec, default_config, onsets)
        expected_keys = {
            "peak_frequency",
            "spectral_centroid",
            "energy",
            "is_voiced",
            "frequency_direction",
            "bout_position",
            "time_since_last_usv",
        }
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert len(result[key]) == 10, f"{key} has wrong length"

    def test_empty_spectrogram(
        self, default_config: AcousticPropertyConfig,
    ) -> None:
        """Zero-frame 2-D spectrogram → all empty arrays."""
        spec = np.empty((170, 0))
        result = extract_all_properties(spec, default_config)
        for key, arr in result.items():
            assert len(arr) == 0, f"{key} should be empty"

    def test_non_2d_spec_raises(
        self, default_config: AcousticPropertyConfig,
    ) -> None:
        """1-D spec raises ValueError (W4 fix)."""
        spec_1d = np.full(170, -30.0)
        with pytest.raises(ValueError, match="2-D"):
            extract_all_properties(spec_1d, default_config)

    def test_unsorted_onsets_handled(
        self, default_config: AcousticPropertyConfig,
    ) -> None:
        """Unsorted onsets are defensively sorted (W1 fix)."""
        spec = np.random.default_rng(99).uniform(-80, -10, size=(170, 10))
        sorted_onsets = np.array([2, 5], dtype=np.int64)
        unsorted_onsets = np.array([5, 2], dtype=np.int64)
        result_sorted = extract_all_properties(spec, default_config, sorted_onsets)
        result_unsorted = extract_all_properties(spec, default_config, unsorted_onsets)
        np.testing.assert_array_equal(
            result_sorted["time_since_last_usv"],
            result_unsorted["time_since_last_usv"],
        )


class TestTimeSinceLastUSVUnsorted:
    def test_unsorted_onsets_handled(
        self, default_config: AcousticPropertyConfig,
    ) -> None:
        """Scalar function handles unsorted onsets via defensive sort (W1)."""
        sorted_onsets = np.array([20, 50, 80], dtype=np.int64)
        unsorted_onsets = np.array([80, 20, 50], dtype=np.int64)
        ms_sorted = time_since_last_usv(100, sorted_onsets, default_config)
        ms_unsorted = time_since_last_usv(100, unsorted_onsets, default_config)
        assert abs(ms_sorted - ms_unsorted) < 1e-10


class TestConfigValidation:
    def test_invalid_freq_range(self) -> None:
        """freq_min >= freq_max raises ValueError."""
        with pytest.raises(ValueError, match="freq_min_hz"):
            AcousticPropertyConfig(freq_min_hz=120_000.0, freq_max_hz=20_000.0)

    def test_invalid_n_freq_bins(self) -> None:
        with pytest.raises(ValueError, match="n_freq_bins"):
            AcousticPropertyConfig(n_freq_bins=0)

    def test_invalid_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            AcousticPropertyConfig(sample_rate=0)

    def test_negative_direction_threshold(self) -> None:
        """Negative direction_threshold raises ValueError (W2 fix)."""
        with pytest.raises(ValueError, match="direction_threshold"):
            AcousticPropertyConfig(direction_threshold=-500.0)

    def test_zero_direction_threshold_allowed(self) -> None:
        """Zero is a valid direction_threshold (any delta is rising/falling)."""
        cfg = AcousticPropertyConfig(direction_threshold=0.0)
        assert cfg.direction_threshold == 0.0
