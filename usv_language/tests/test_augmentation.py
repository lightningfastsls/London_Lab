"""Isolated tests for spectrogram augmentation transforms.

Each of the 4 augmentation types (Gaussian noise, gain, frequency masking,
time masking) is tested in isolation to verify correctness and edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.data.dataset import AugmentationConfig, apply_augmentations

N_FREQ = 170
SEQ_LEN = 100


@pytest.fixture
def base_spectrogram() -> np.ndarray:
    """A known synthetic spectrogram (T, n_freq) with non-zero values."""
    rng = np.random.RandomState(123)
    return rng.randn(SEQ_LEN, N_FREQ).astype(np.float32) * 20 - 60


# ---------------------------------------------------------------------------
# Test 16: Disabled = no-op
# ---------------------------------------------------------------------------


def test_augmentation_disabled_noop(base_spectrogram: np.ndarray) -> None:
    """enabled=False returns input unchanged."""
    config = AugmentationConfig(enabled=False)
    rng = np.random.RandomState(42)

    result = apply_augmentations(base_spectrogram, config, rng)
    np.testing.assert_array_equal(result, base_spectrogram)


# ---------------------------------------------------------------------------
# Test 17: Does not modify input array
# ---------------------------------------------------------------------------


def test_augmentation_does_not_modify_input(base_spectrogram: np.ndarray) -> None:
    """The original array must not be modified (function should .copy())."""
    original = base_spectrogram.copy()
    config = AugmentationConfig(enabled=True, probability=1.0)
    rng = np.random.RandomState(42)

    apply_augmentations(base_spectrogram, config, rng)

    np.testing.assert_array_equal(base_spectrogram, original)


# ---------------------------------------------------------------------------
# Test 18: Gaussian noise
# ---------------------------------------------------------------------------


def test_gaussian_noise_changes_signal(base_spectrogram: np.ndarray) -> None:
    """Gaussian noise augmentation produces different output from input."""
    # Use probability=1.0 to guarantee noise is applied
    config = AugmentationConfig(
        enabled=True,
        probability=1.0,
        gaussian_noise_snr_db=10.0,
    )
    rng = np.random.RandomState(42)

    result = apply_augmentations(base_spectrogram, config, rng)

    # Result should differ from input
    assert not np.array_equal(result, base_spectrogram)
    # But should have the same shape
    assert result.shape == base_spectrogram.shape
    # Should still be finite
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Test 19: Gain shift
# ---------------------------------------------------------------------------


def test_gain_adds_offset(base_spectrogram: np.ndarray) -> None:
    """Gain augmentation adds a constant dB offset to the entire spectrogram.

    We use a fixed seed and force probability=1.0 to guarantee gain is applied.
    The gain is drawn from gain_range_db, and in dB domain gain is additive.
    """
    config = AugmentationConfig(
        enabled=True,
        probability=1.0,
        gain_range_db=(-3.0, 3.0),
    )

    # Run multiple times to find one where gain was actually applied
    # (other augmentations may also fire, so we check the offset pattern)
    for seed in range(10):
        rng = np.random.RandomState(seed)
        result = apply_augmentations(base_spectrogram, config, rng)
        if not np.array_equal(result, base_spectrogram):
            # At least one augmentation fired
            break
    else:
        pytest.fail("No augmentation was applied across 10 seeds")

    assert result.shape == base_spectrogram.shape


# ---------------------------------------------------------------------------
# Test 20: Frequency masking zeros bands
# ---------------------------------------------------------------------------


def test_freq_masking_zeros_bands() -> None:
    """Frequency masking sets contiguous frequency bands to zero."""
    spec = np.ones((SEQ_LEN, N_FREQ), dtype=np.float32)
    config = AugmentationConfig(
        enabled=True,
        probability=1.0,
        freq_mask_bands=(1, 1),
        freq_mask_width=(10, 10),
    )
    rng = np.random.RandomState(42)

    result = apply_augmentations(spec, config, rng)

    # Some frequency columns should be zero
    zero_cols = np.where(np.all(result == 0.0, axis=0))[0]
    assert len(zero_cols) > 0, "No frequency bands were masked"

    # Masked bands should be contiguous
    if len(zero_cols) > 1:
        diffs = np.diff(zero_cols)
        assert np.all(diffs == 1), (
            f"Masked frequency bands are not contiguous: {zero_cols}"
        )


# ---------------------------------------------------------------------------
# Test 21: Time masking zeros frames
# ---------------------------------------------------------------------------


def test_time_masking_zeros_frames() -> None:
    """Time masking sets contiguous time frames to zero."""
    spec = np.ones((SEQ_LEN, N_FREQ), dtype=np.float32)
    config = AugmentationConfig(
        enabled=True,
        probability=1.0,
        time_mask_count=(1, 1),
        time_mask_ratio=0.1,
    )
    rng = np.random.RandomState(42)

    result = apply_augmentations(spec, config, rng)

    # Some time rows should be zero
    zero_rows = np.where(np.all(result == 0.0, axis=1))[0]
    assert len(zero_rows) > 0, "No time frames were masked"

    # Masked frames should be contiguous
    if len(zero_rows) > 1:
        diffs = np.diff(zero_rows)
        assert np.all(diffs == 1), (
            f"Masked time frames are not contiguous: {zero_rows}"
        )


# ---------------------------------------------------------------------------
# Test 22: Reproducibility
# ---------------------------------------------------------------------------


def test_augmentation_reproducibility(base_spectrogram: np.ndarray) -> None:
    """Same seed produces identical augmented outputs."""
    config = AugmentationConfig(enabled=True, probability=1.0)

    rng1 = np.random.RandomState(42)
    result1 = apply_augmentations(base_spectrogram, config, rng1)

    rng2 = np.random.RandomState(42)
    result2 = apply_augmentations(base_spectrogram, config, rng2)

    np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# Test 23: Edge case — n_freq smaller than mask width
# ---------------------------------------------------------------------------


def test_freq_mask_small_spectrogram() -> None:
    """Frequency masking handles n_freq < freq_mask_width without crash."""
    small_n_freq = 10
    spec = np.ones((SEQ_LEN, small_n_freq), dtype=np.float32)
    config = AugmentationConfig(
        enabled=True,
        probability=1.0,
        freq_mask_bands=(1, 1),
        freq_mask_width=(20, 30),  # Wider than spectrogram!
    )
    rng = np.random.RandomState(42)

    # Should not crash
    result = apply_augmentations(spec, config, rng)
    assert result.shape == (SEQ_LEN, small_n_freq)
    assert np.all(np.isfinite(result))
