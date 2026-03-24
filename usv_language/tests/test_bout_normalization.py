"""Tests for normalization module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.data.normalization import (
    NormalizationStats,
    compute_normalization_stats,
    load_normalization_stats,
    normalize,
    save_normalization_stats,
)

N_FREQ = 170


@pytest.fixture
def synthetic_specs():
    """Create a list of synthetic spectrograms for testing."""
    rng = np.random.RandomState(42)
    specs = []
    for _ in range(5):
        n_frames = rng.randint(100, 500)
        # Simulate dB values roughly in [-100, 0] range
        spec = rng.randn(N_FREQ, n_frames).astype(np.float32) * 20 - 60
        specs.append(spec)
    return specs


class TestComputeNormalizationStats:
    def test_stats_shape(self, synthetic_specs):
        """Mean and std should be (170,) shaped."""
        stats = compute_normalization_stats(synthetic_specs)
        assert stats.mean.shape == (N_FREQ,)
        assert stats.std.shape == (N_FREQ,)

    def test_count_correct(self, synthetic_specs):
        """Count should equal total number of frames across all specs."""
        stats = compute_normalization_stats(synthetic_specs)
        expected_count = sum(s.shape[1] for s in synthetic_specs)
        assert stats.count == expected_count

    def test_normalized_mean_near_zero(self, synthetic_specs):
        """After normalization, mean per bin should be approximately 0."""
        stats = compute_normalization_stats(synthetic_specs)
        # Normalize all specs and check aggregate mean
        all_normalized = np.concatenate(
            [normalize(s, stats) for s in synthetic_specs], axis=1,
        )
        per_bin_mean = all_normalized.mean(axis=1)
        assert np.allclose(per_bin_mean, 0.0, atol=0.1)

    def test_normalized_std_near_one(self, synthetic_specs):
        """After normalization, std per bin should be approximately 1."""
        stats = compute_normalization_stats(synthetic_specs)
        all_normalized = np.concatenate(
            [normalize(s, stats) for s in synthetic_specs], axis=1,
        )
        per_bin_std = all_normalized.std(axis=1)
        assert np.allclose(per_bin_std, 1.0, atol=0.1)

    def test_constant_frequency_bin_handled(self):
        """Frequency bins with constant value should get eps std, not zero."""
        spec = np.zeros((N_FREQ, 100), dtype=np.float32)
        spec[50, :] = -60.0  # Constant value in bin 50
        stats = compute_normalization_stats([spec])
        # std should be at least eps
        assert stats.std[50] >= 1e-8

    def test_empty_list_raises(self):
        """Empty spectrogram list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_normalization_stats([])


class TestSaveLoadStats:
    def test_save_and_load_roundtrip(self, tmp_path, synthetic_specs):
        """Stats should survive a save/load roundtrip."""
        stats = compute_normalization_stats(synthetic_specs)
        path = tmp_path / "stats.npz"
        save_normalization_stats(stats, path)
        loaded = load_normalization_stats(path)

        np.testing.assert_array_almost_equal(loaded.mean, stats.mean)
        np.testing.assert_array_almost_equal(loaded.std, stats.std)
        assert loaded.count == stats.count


class TestNormalize:
    def test_apply_normalization(self):
        """normalize() should produce (spec - mean) / (std + eps)."""
        rng = np.random.RandomState(42)
        spec = rng.randn(N_FREQ, 50).astype(np.float32)
        mean = rng.randn(N_FREQ).astype(np.float32)
        std = np.abs(rng.randn(N_FREQ).astype(np.float32)) + 0.1

        stats = NormalizationStats(mean=mean, std=std, count=1000)
        eps = 1e-8
        result = normalize(spec, stats, eps=eps)

        expected = (spec - mean[:, None]) / (std[:, None] + eps)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
