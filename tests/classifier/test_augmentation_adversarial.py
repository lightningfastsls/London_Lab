"""Adversarial tests for augmentation.py — added by test-hardener (Module 18.3).

Targets gaps NOT covered by the 13 original tests:
  A. Wrong-rank inputs to specaugment: 1-D and 3-D arrays must raise ValueError
     (the implementation unpacks `freq_bins, time_frames = spec.shape` which
     would fail for non-2D shapes — lock in the error rather than leaving it
     as an undocumented crash).
  B. time_mask_max_width_frames >= time_frames (mask wider than spec) — the
     loop uses `continue` when t >= time_frames; verify output is not
     all-zeros (i.e., at most some masking, not a blank output).
  C. inject_cage_noise with prob=1.0 and a real on-disk PNG path must
     change the output (the happy path — never previously tested).
  D. AugmentationConfig with freq_mask_n=0 but large freq_mask_max_width_bins
     -> identity (n=0 means zero iterations of the mask loop).
  E. Reproducibility: same np.random.seed -> same output; different seed ->
     different output (positive and negative control together).
  F. inject_cage_noise boundary: prob=0.0 always identity; prob=1.0 with valid
     path always changes output (no flakiness from random draw).

Total added: 7 tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.augmentation import (  # noqa: E402
    AugmentationConfig,
    inject_cage_noise,
    specaugment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(freq_bins: int = 64, time_frames: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((freq_bins, time_frames)).astype(np.float32)


def _write_synthetic_png(path: Path, height: int = 64, width: int = 128) -> None:
    """Write a synthetic grayscale PNG to disk for inject_cage_noise tests."""
    rng = np.random.default_rng(99)
    arr = (rng.random((height, width)) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


# ===========================================================================
# Section A — Wrong-rank inputs to specaugment
# ===========================================================================

def test_specaugment_1d_input_raises():
    """specaugment on a 1-D array must raise an exception.

    The implementation unpacks `freq_bins, time_frames = spec.shape`, which
    raises ValueError for a 1-D array (not enough values to unpack). This
    test locks in the error behaviour so the failure mode is documented and
    reproducible rather than silently computing wrong results.
    """
    spec_1d = np.random.rand(128).astype(np.float32)
    cfg = AugmentationConfig()
    with pytest.raises((ValueError, TypeError)):
        specaugment(spec_1d, cfg)


def test_specaugment_3d_input_raises():
    """specaugment on a 3-D array must raise an exception.

    A (C, H, W) tensor from a DataLoader is a common mistake — the function
    expects a 2-D (freq, time) spectrogram. Raising here is better than
    silently slicing only the first two dimensions.
    """
    spec_3d = np.random.rand(3, 64, 128).astype(np.float32)
    cfg = AugmentationConfig()
    with pytest.raises((ValueError, TypeError)):
        specaugment(spec_3d, cfg)


# ===========================================================================
# Section B — Mask wider than spectrogram
# ===========================================================================

def test_specaugment_time_mask_wider_than_spec_not_all_zero():
    """When time_mask_max_width_frames >= time_frames, specaugment must not
    zero out the entire output.

    The implementation uses `continue` when `t >= time_frames`, so an
    over-wide mask simply skips that iteration. The output must have at
    least some non-zero values (same as the input, minus any valid masks).
    """
    freq_bins, time_frames = 16, 10
    spec = _make_spec(freq_bins=freq_bins, time_frames=time_frames, seed=5)
    # Set mask width much larger than time_frames — all draws will be skipped
    # by the `if t == 0 or t >= time_frames: continue` guard.
    cfg = AugmentationConfig(
        time_mask_max_width_frames=time_frames + 100,  # always skipped
        time_mask_n=5,
        freq_mask_max_width_bins=0,
        freq_mask_n=0,
    )
    np.random.seed(0)
    out = specaugment(spec, cfg)

    assert out.shape == spec.shape, (
        f"Shape changed under oversized mask: {spec.shape} -> {out.shape}"
    )
    # Output must not be all zeros — the oversized time mask should be skipped.
    assert out.sum() != 0.0, (
        "specaugment produced an all-zero output when time_mask_max_width_frames "
        ">= time_frames. Expected the oversized mask to be skipped (continue), "
        "leaving non-zero values from the input."
    )


# ===========================================================================
# Section C — Happy path: inject_cage_noise with a real PNG file
# ===========================================================================

def test_inject_cage_noise_prob1_with_real_png_changes_output(tmp_path):
    """inject_cage_noise with prob=1.0 and a valid PNG must change the output.

    This is the happy-path test that was never exercised by the original suite.
    With prob=1.0, the RNG draw always passes, a patch is always loaded, and
    the blended output must differ numerically from the input.
    """
    png_path = tmp_path / "noise_patch.png"
    _write_synthetic_png(png_path, height=64, width=128)

    spec = _make_spec(freq_bins=64, time_frames=128, seed=10)
    cfg = AugmentationConfig(
        cage_noise_inject_prob=1.0,
        cage_noise_paths=(str(png_path),),
    )
    rng = np.random.default_rng(42)
    out = inject_cage_noise(spec, cfg, rng)

    assert out.shape == spec.shape, (
        f"inject_cage_noise changed shape: {spec.shape} -> {out.shape}"
    )
    assert not np.array_equal(out, spec), (
        "inject_cage_noise with prob=1.0 and a valid PNG must change the "
        "output. Output is identical to input — noise injection not happening."
    )
    assert np.isfinite(out).all(), (
        "inject_cage_noise produced non-finite values after blending a PNG patch"
    )


# ===========================================================================
# Section D — freq_mask_n=0 with non-zero width is identity
# ===========================================================================

def test_specaugment_freq_mask_n_zero_is_identity():
    """freq_mask_n=0 with a large freq_mask_max_width_bins must be identity.

    The mask-count field controls how many mask iterations are executed.
    With n=0, the for-loop runs zero times regardless of the width setting.
    This tests the logical separation between count and width knobs.
    """
    spec = _make_spec(freq_bins=64, time_frames=128, seed=20)
    cfg = AugmentationConfig(
        freq_mask_n=0,
        freq_mask_max_width_bins=100,  # large width, but n=0 means no masks
        time_mask_n=0,
        time_mask_max_width_frames=0,
    )
    np.random.seed(0)
    out = specaugment(spec, cfg)
    np.testing.assert_array_equal(
        out, spec,
        err_msg=(
            "specaugment with freq_mask_n=0 must be identity regardless of "
            "freq_mask_max_width_bins. Got differences."
        ),
    )


# ===========================================================================
# Section E — Reproducibility: positive and negative controls
# ===========================================================================

def test_specaugment_different_seed_gives_different_output():
    """specaugment with two different seeds must produce different outputs.

    This is the negative-control companion to the existing reproducibility
    test. If specaugment is completely deterministic with no randomness,
    different seeds would produce the same output — which would mean the
    augmentation provides no diversity.
    """
    spec = _make_spec(freq_bins=64, time_frames=128, seed=30)
    cfg = AugmentationConfig(
        time_mask_max_width_frames=20,
        time_mask_n=3,
        freq_mask_max_width_bins=16,
        freq_mask_n=3,
    )

    np.random.seed(111)
    out_a = specaugment(spec.copy(), cfg)
    np.random.seed(999)
    out_b = specaugment(spec.copy(), cfg)

    # They should differ — if the augmentation is non-trivial and seeds differ,
    # the mask positions will land at different locations.
    assert not np.array_equal(out_a, out_b), (
        "specaugment with different random seeds produced identical outputs. "
        "This suggests the random state is not being used, so augmentation "
        "provides no data diversity."
    )


# ===========================================================================
# Section F — inject_cage_noise boundary: prob=0.0 exact identity
# ===========================================================================

def test_inject_cage_noise_prob_zero_exact_identity(tmp_path):
    """inject_cage_noise with prob=0.0 must return the input object unchanged.

    With prob=0.0 the function must short-circuit before even looking at
    cage_noise_paths. We verify identity (same values) even when a valid
    PNG path is supplied — the path must not be loaded.
    """
    png_path = tmp_path / "noise_patch.png"
    _write_synthetic_png(png_path, height=64, width=128)

    spec = _make_spec(freq_bins=64, time_frames=128, seed=40)
    cfg = AugmentationConfig(
        cage_noise_inject_prob=0.0,
        cage_noise_paths=(str(png_path),),
    )
    rng = np.random.default_rng(0)
    out = inject_cage_noise(spec, cfg, rng)

    np.testing.assert_array_equal(
        out, spec,
        err_msg=(
            "inject_cage_noise with prob=0.0 must be exact identity regardless "
            "of cage_noise_paths content. Got differences."
        ),
    )
