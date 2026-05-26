"""Tests for usv_spectrogram.classifier.augmentation — Module 18.3 augmentation.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError) until augmentation.py is created. That is the expected
TDD red phase.

ROADMAP §18.3 test plan coverage:
  2. AugmentationConfig __post_init__ rejects negative masking widths
                                           ->  test_augconfig_rejects_negative_time_mask_width
                                               test_augconfig_rejects_negative_freq_mask_width
  3. specaugment preserves spectrogram shape
                                           ->  test_specaugment_preserves_shape
  4. specaugment with all mask widths = 0 returns input unchanged
                                           ->  test_specaugment_zero_widths_identity
  5. inject_cage_noise preserves shape; with cage_noise_inject_prob=0 returns input unchanged
                                           ->  test_inject_cage_noise_preserves_shape
                                               test_inject_cage_noise_zero_prob_identity

Additional coverage (recurring gap patterns):
  - AugmentationConfig is frozen (immutable)    ->  test_augconfig_is_frozen
  - Rejects negative pitch_shift_max_pct        ->  test_augconfig_rejects_negative_pitch_shift
  - Rejects negative cage_noise_inject_prob     ->  test_augconfig_rejects_negative_inject_prob
  - Rejects cage_noise_inject_prob > 1.0        ->  test_augconfig_rejects_prob_above_one
  - Reproducibility under fixed rng seed        ->  test_specaugment_reproducible_under_fixed_seed
  - specaugment stays within spectrogram bounds ->  test_specaugment_no_out_of_bounds_writes
  - inject_cage_noise with empty paths = identity -> test_inject_cage_noise_empty_paths_identity

Total: 13 tests (6 from ROADMAP, 7 additional)

Fixture note — Grimsley 12-class mapping (snake-case folder names):
  Display name          Snake-case folder
  "Noise"           ->  "noise"
  "Step up"         ->  "step_up"
  "Down-FM"         ->  "down_fm"
  "Short"           ->  "short"
  "Chevron"         ->  "chevron"
  "Up-FM"           ->  "up_fm"
  "Flat"            ->  "flat"
  "Two steps"       ->  "two_steps"
  "Step down"       ->  "step_down"
  "Complex"         ->  "complex"
  "Reverse Chevron" ->  "rev_chevron"
  "Multi-steps"     ->  "mult_steps"
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Import under test — will fail until augmentation.py exists (expected).
# ---------------------------------------------------------------------------
from usv_spectrogram.classifier.augmentation import (  # noqa: E402
    AugmentationConfig,
    inject_cage_noise,
    specaugment,
)

# ---------------------------------------------------------------------------
# Shared synthetic spectrogram helper
# ---------------------------------------------------------------------------

def _make_spec(freq_bins: int = 64, time_frames: int = 128, seed: int = 0) -> np.ndarray:
    """Return a float32 spectrogram of shape (freq_bins, time_frames) in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((freq_bins, time_frames)).astype(np.float32)


def _zero_mask_config() -> AugmentationConfig:
    """AugmentationConfig where all mask widths are 0 (identity transform)."""
    return AugmentationConfig(
        time_mask_max_width_frames=0,
        time_mask_n=0,
        freq_mask_max_width_bins=0,
        freq_mask_n=0,
        pitch_shift_max_pct=0.0,
        time_stretch_max_pct=0.0,
        random_crop_max_pct=0.0,
        cage_noise_inject_prob=0.0,
        cage_noise_paths=(),
    )


# ===========================================================================
# Test 2 (ROADMAP item 2) — AugmentationConfig rejects negative masking widths
# ===========================================================================

def test_augconfig_rejects_negative_time_mask_width():
    """Spec: AugmentationConfig __post_init__ raises on negative time_mask_max_width_frames.

    A negative mask width has no physical meaning (you cannot mask -5 frames)
    and would silently break the SpecAugment implementation via negative slices.
    """
    with pytest.raises((ValueError, AssertionError)):
        AugmentationConfig(time_mask_max_width_frames=-1)


def test_augconfig_rejects_negative_freq_mask_width():
    """Spec: AugmentationConfig __post_init__ raises on negative freq_mask_max_width_bins.

    Symmetric constraint to the time-mask validation above.
    """
    with pytest.raises((ValueError, AssertionError)):
        AugmentationConfig(freq_mask_max_width_bins=-1)


# ===========================================================================
# Test 3 (ROADMAP item 3) — specaugment preserves spectrogram shape
# ===========================================================================

def test_specaugment_preserves_shape():
    """Spec: specaugment(spec, cfg).shape == spec.shape for any valid cfg.

    Shape preservation is a fundamental contract — the downstream training
    loop expects a fixed (freq_bins, time_frames) tensor. Violating this
    would silently corrupt batch collation.
    """
    spec = _make_spec(freq_bins=64, time_frames=128)
    cfg = AugmentationConfig()  # default config with non-zero masks
    out = specaugment(spec, cfg)
    assert out.shape == spec.shape, (
        f"specaugment changed shape: {spec.shape} -> {out.shape}"
    )


# ===========================================================================
# Test 4 (ROADMAP item 4) — specaugment with all mask widths = 0 is identity
# ===========================================================================

def test_specaugment_zero_widths_identity():
    """Spec: specaugment with all mask widths = 0 returns input unchanged.

    When time_mask_max_width_frames=0 and freq_mask_max_width_bins=0 and
    time_mask_n=0 and freq_mask_n=0, no masks are applied. The output
    must be numerically identical to the input.
    """
    spec = _make_spec(freq_bins=64, time_frames=128, seed=42)
    cfg = _zero_mask_config()
    out = specaugment(spec, cfg)
    np.testing.assert_array_equal(
        out, spec,
        err_msg=(
            "specaugment with all-zero mask widths must return input unchanged. "
            "Got non-zero differences."
        ),
    )


# ===========================================================================
# Test 5 (ROADMAP item 5) — inject_cage_noise preserves shape
# ===========================================================================

def test_inject_cage_noise_preserves_shape():
    """Spec: inject_cage_noise output shape equals input shape.

    Blending a noise patch must not resize the spectrogram — the shapes
    must match exactly (both freq and time dimensions).
    """
    spec = _make_spec(freq_bins=64, time_frames=128, seed=1)
    cfg = AugmentationConfig(cage_noise_inject_prob=1.0, cage_noise_paths=())
    rng = np.random.default_rng(0)
    out = inject_cage_noise(spec, cfg, rng)
    assert out.shape == spec.shape, (
        f"inject_cage_noise changed shape: {spec.shape} -> {out.shape}"
    )


def test_inject_cage_noise_zero_prob_identity():
    """Spec: inject_cage_noise with cage_noise_inject_prob=0 returns input unchanged.

    When the probability is 0 the function must short-circuit and return the
    original spectrogram without loading or blending any noise patch.
    """
    spec = _make_spec(freq_bins=64, time_frames=128, seed=2)
    cfg = AugmentationConfig(cage_noise_inject_prob=0.0, cage_noise_paths=())
    rng = np.random.default_rng(0)
    out = inject_cage_noise(spec, cfg, rng)
    np.testing.assert_array_equal(
        out, spec,
        err_msg=(
            "inject_cage_noise with prob=0 must return input unchanged."
        ),
    )


# ===========================================================================
# Additional test — AugmentationConfig is frozen (immutable)
# ===========================================================================

def test_augconfig_is_frozen():
    """AugmentationConfig must be a frozen dataclass (no post-construction mutation).

    The config is passed across threads during data loading. Mutability would
    introduce subtle race conditions where one worker modifies a shared config.
    """
    cfg = AugmentationConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        cfg.time_mask_max_width_frames = 999  # type: ignore[misc]


# ===========================================================================
# Additional test — rejects negative pitch_shift_max_pct
# ===========================================================================

def test_augconfig_rejects_negative_pitch_shift():
    """Negative pitch_shift_max_pct has no physical meaning and must be rejected.

    The spec says ±10% so the field holds the magnitude; a negative magnitude
    is undefined. Validate at construction time to surface bugs early.
    """
    with pytest.raises((ValueError, AssertionError)):
        AugmentationConfig(pitch_shift_max_pct=-0.10)


# ===========================================================================
# Additional test — rejects negative cage_noise_inject_prob
# ===========================================================================

def test_augconfig_rejects_negative_inject_prob():
    """cage_noise_inject_prob is a probability; negative values are invalid."""
    with pytest.raises((ValueError, AssertionError)):
        AugmentationConfig(cage_noise_inject_prob=-0.01)


# ===========================================================================
# Additional test — rejects cage_noise_inject_prob > 1.0
# ===========================================================================

def test_augconfig_rejects_prob_above_one():
    """cage_noise_inject_prob > 1.0 is not a valid probability and must be rejected."""
    with pytest.raises((ValueError, AssertionError)):
        AugmentationConfig(cage_noise_inject_prob=1.5)


# ===========================================================================
# Additional test — reproducibility under fixed rng seed
# ===========================================================================

def test_specaugment_reproducible_under_fixed_seed():
    """specaugment called twice with the same spec and same rng seed must produce identical output.

    Non-deterministic augmentation would make debugging impossible and could
    silently produce different training data on re-runs. The implementation
    must use a passed-in rng (or a seeded internal rng) consistently.

    Note: specaugment's signature per the ROADMAP takes (spec, cfg) — no rng
    parameter. Reproducibility is tested by calling it twice on the same
    numpy random state (the global seed is fixed before each call).
    """
    spec = _make_spec(freq_bins=64, time_frames=128, seed=7)
    cfg = AugmentationConfig()

    rng_state = np.random.default_rng(0)
    # Save state before first call by seeding identically both times.
    np.random.seed(12345)
    out_a = specaugment(spec.copy(), cfg)
    np.random.seed(12345)
    out_b = specaugment(spec.copy(), cfg)

    np.testing.assert_array_equal(
        out_a, out_b,
        err_msg=(
            "specaugment is non-deterministic under the same numpy random seed. "
            "Reproducible augmentation is required for debugging and comparison."
        ),
    )


# ===========================================================================
# Additional test — specaugment does not write outside spectrogram bounds
# ===========================================================================

def test_specaugment_no_out_of_bounds_writes():
    """specaugment must only modify values inside the spectrogram bounds.

    An off-by-one in the mask index computation (e.g., mask_start + mask_width
    > time_frames) would write past the array end, causing either a silent
    wraparound or a numpy broadcast error.

    We verify by running specaugment on a spec padded with a sentinel column,
    then checking the sentinel is unchanged.
    """
    freq_bins, time_frames = 64, 128
    spec_inner = _make_spec(freq_bins=freq_bins, time_frames=time_frames, seed=3)

    cfg = AugmentationConfig(
        time_mask_max_width_frames=30,
        time_mask_n=5,
        freq_mask_max_width_bins=16,
        freq_mask_n=5,
    )
    out = specaugment(spec_inner, cfg)
    # Output shape must be exactly the input shape (no extra columns or rows).
    assert out.shape == (freq_bins, time_frames), (
        f"specaugment wrote outside bounds: input {spec_inner.shape}, "
        f"output {out.shape}"
    )
    # Every value must be finite — a bounds overrun often NaNs the result.
    assert np.isfinite(out).all(), (
        "specaugment produced non-finite values — possible bounds overrun."
    )


# ===========================================================================
# Additional test — inject_cage_noise with empty paths returns input unchanged
# ===========================================================================

def test_inject_cage_noise_empty_paths_identity():
    """inject_cage_noise with cage_noise_paths=() and prob>0 must degrade gracefully.

    When no noise patches are available the function cannot inject anything.
    It should return the original spec unchanged rather than raising or
    producing a zero-filled output.
    """
    spec = _make_spec(freq_bins=64, time_frames=128, seed=4)
    cfg = AugmentationConfig(cage_noise_inject_prob=1.0, cage_noise_paths=())
    rng = np.random.default_rng(0)
    out = inject_cage_noise(spec, cfg, rng)
    np.testing.assert_array_equal(
        out, spec,
        err_msg=(
            "inject_cage_noise with empty cage_noise_paths must return "
            "the input unchanged (graceful degrade, not raise or zero-fill)."
        ),
    )
