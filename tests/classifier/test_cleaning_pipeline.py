"""Tests for cleaning_pipeline.py — written by test-architect BEFORE implementation.

ROADMAP test plan coverage (Module 18.1):
  1. CleaningConfig __post_init__ rejects invalid baseline_mode
     -> test_cleaning_config_rejects_invalid_baseline_mode
  1b. CleaningConfig __post_init__ rejects invalid sample_rate_hz
     -> test_cleaning_config_rejects_invalid_sample_rate
  2. CleaningConfig with apply_soft_notch=True and tonal_library_path=None does not raise
     -> test_cleaning_config_soft_notch_without_tonal_library_is_valid
  3. clean_spectrogram preserves shape: input (n_freq, n_time) -> output (n_freq, n_time)
     -> test_clean_spectrogram_preserves_shape
  4. clean_spectrogram with all layers disabled returns input unchanged within float32 epsilon
     -> test_clean_spectrogram_all_layers_disabled_returns_input_unchanged
  5. clean_spectrogram applies layers in correct order (notch -> baseline -> MAD -> zscore)
     -> test_clean_spectrogram_layer_order_notch_baseline_mad_zscore

Additional coverage (recurring gap patterns):
  - CleaningConfig frozen=True (immutability after creation)
     -> test_cleaning_config_is_immutable_after_creation
  - CleaningConfig default sample_rate_hz is 250_000 (VocalMat-aligned, not corpus default)
     -> test_cleaning_config_default_sample_rate_is_250000
  - CleaningConfig accepts both valid sample rates
     -> test_cleaning_config_accepts_250000_and_300000_sample_rates
  - CleaningConfig baseline_mode defaults to "median_envelope"
     -> test_cleaning_config_default_baseline_mode
  - clean_spectrogram with square input (n_freq == n_time) preserves shape
     -> test_clean_spectrogram_preserves_shape_square_input
  - clean_spectrogram output dtype remains float32 or float64 (not int or complex)
     -> test_clean_spectrogram_output_is_real_valued_float
  - clean_spectrogram with only MAD enabled produces output with finite values
     -> test_clean_spectrogram_mad_only_produces_finite_output
  - Recording ID passed through without mutation
     -> test_clean_spectrogram_recording_id_does_not_affect_shape

Total: 14 tests (6 from ROADMAP [item 1 split into 1a+1b], 8 additional)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import bootstrap: add src/ to sys.path so the classifier package is findable
# once implemented. Follows patterns.md §8 (parents[1] = repo root for tests/).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/classifier/ -> tests/ -> worktree root
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Will raise ImportError until implementation exists — that is expected.
from usv_spectrogram.classifier.cleaning_pipeline import (  # noqa: E402
    CleaningConfig,
    clean_spectrogram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> CleaningConfig:
    """A CleaningConfig with all defaults — used as the baseline for comparison."""
    return CleaningConfig()


@pytest.fixture
def all_disabled_config() -> CleaningConfig:
    """A CleaningConfig with every cleaning layer turned off."""
    return CleaningConfig(
        apply_soft_notch=False,
        apply_baseline_subtraction=False,
        apply_global_mad=False,
        apply_per_recording_zscore=False,
    )


@pytest.fixture
def synthetic_spectrogram_2d() -> np.ndarray:
    """A 2-D float32 spectrogram of shape (64, 128) with known noise statistics.

    Chosen dimensions are intentionally non-square so shape-preservation tests
    catch transposition bugs (64 != 128).
    """
    rng = np.random.default_rng(42)
    spec = rng.normal(loc=-40.0, scale=10.0, size=(64, 128)).astype(np.float32)
    return spec


# ---------------------------------------------------------------------------
# ROADMAP test 1a — CleaningConfig rejects invalid baseline_mode
# ---------------------------------------------------------------------------


def test_cleaning_config_rejects_invalid_baseline_mode():
    """Spec: __post_init__ must raise ValueError for any baseline_mode not in
    {'percentile', 'median_envelope'}.  Verifies the validation branch in the
    config dataclass (ROADMAP test plan item 1).
    """
    with pytest.raises(ValueError, match="baseline_mode"):
        CleaningConfig(baseline_mode="mean")

    with pytest.raises(ValueError, match="baseline_mode"):
        CleaningConfig(baseline_mode="")

    with pytest.raises(ValueError, match="baseline_mode"):
        CleaningConfig(baseline_mode="PERCENTILE")  # case-sensitive


# ---------------------------------------------------------------------------
# ROADMAP test 1b — CleaningConfig rejects invalid sample_rate_hz
# ---------------------------------------------------------------------------


def test_cleaning_config_rejects_invalid_sample_rate():
    """Spec: __post_init__ must raise ValueError for any sample_rate_hz not in
    {250_000, 300_000}.  The two allowed values represent VocalMat-aligned (default)
    and corpus canonical (ADR-001) pipelines (ROADMAP test plan item 1).
    """
    with pytest.raises(ValueError, match="sample_rate_hz"):
        CleaningConfig(sample_rate_hz=44100)

    with pytest.raises(ValueError, match="sample_rate_hz"):
        CleaningConfig(sample_rate_hz=0)

    with pytest.raises(ValueError, match="sample_rate_hz"):
        CleaningConfig(sample_rate_hz=192000)

    # Negative value
    with pytest.raises(ValueError, match="sample_rate_hz"):
        CleaningConfig(sample_rate_hz=-1)


# ---------------------------------------------------------------------------
# ROADMAP test 2 — soft-notch with no tonal_library_path is valid (no-op path)
# ---------------------------------------------------------------------------


def test_cleaning_config_soft_notch_without_tonal_library_is_valid():
    """Spec: CleaningConfig(apply_soft_notch=True, tonal_library_path=None) must NOT raise.
    The spec comment says 'soft-notch will silently no-op' in this case.
    (ROADMAP test plan item 2)
    """
    # Must not raise — this is the valid no-op configuration
    cfg = CleaningConfig(apply_soft_notch=True, tonal_library_path=None)
    assert cfg.apply_soft_notch is True
    assert cfg.tonal_library_path is None


# ---------------------------------------------------------------------------
# ROADMAP test 3 — clean_spectrogram preserves (n_freq, n_time) shape
# ---------------------------------------------------------------------------


def test_clean_spectrogram_preserves_shape(
    default_config: CleaningConfig,
    synthetic_spectrogram_2d: np.ndarray,
):
    """Spec: clean_spectrogram(spec, cfg, recording_id) must return an array with
    the same shape as the input (n_freq_bins, n_time_frames).
    (ROADMAP test plan item 3)
    """
    spec = synthetic_spectrogram_2d  # shape (64, 128)
    result = clean_spectrogram(spec, default_config, recording_id="rec_test")

    assert result.shape == spec.shape, (
        f"Expected output shape {spec.shape}, got {result.shape}. "
        "clean_spectrogram must be shape-preserving."
    )


def test_clean_spectrogram_preserves_shape_square_input(default_config: CleaningConfig):
    """Shape preservation must hold for square spectrograms (n_freq == n_time),
    which catches off-by-one or transposition bugs masked by rectangular shapes.
    """
    rng = np.random.default_rng(7)
    spec = rng.normal(-35.0, 8.0, (80, 80)).astype(np.float32)
    result = clean_spectrogram(spec, default_config, recording_id="square_rec")
    assert result.shape == (80, 80)


# ---------------------------------------------------------------------------
# ROADMAP test 4 — all layers disabled returns input unchanged
# ---------------------------------------------------------------------------


def test_clean_spectrogram_all_layers_disabled_returns_input_unchanged(
    all_disabled_config: CleaningConfig,
    synthetic_spectrogram_2d: np.ndarray,
):
    """Spec: with all four layers disabled, clean_spectrogram must return a value
    numerically identical to the input within float32 machine epsilon.
    This verifies that no cleaning is applied when every flag is False and that
    the function correctly short-circuits rather than applying partial transforms.
    (ROADMAP test plan item 4)
    """
    spec = synthetic_spectrogram_2d.copy()
    result = clean_spectrogram(spec, all_disabled_config, recording_id="passthrough_rec")

    np.testing.assert_array_equal(
        result,
        spec,
        err_msg=(
            "clean_spectrogram with all layers disabled must return input "
            "numerically identical (not just close) to the input array."
        ),
    )


# ---------------------------------------------------------------------------
# ROADMAP test 5 — layers applied in order: notch -> baseline -> MAD -> zscore
# ---------------------------------------------------------------------------


def test_clean_spectrogram_layer_order_notch_baseline_mad_zscore(
    synthetic_spectrogram_2d: np.ndarray,
    tmp_path: Path,
):
    """Spec: layers must be applied in the fixed order notch -> baseline_subtraction ->
    global_MAD -> per_recording_zscore.  Re-ordering is a behavior change and must not
    happen silently.

    Strategy: patch each internal layer function at the boundary where clean_spectrogram
    calls them and verify that the mock call sequence matches the required order.
    Each mock passes its first argument through so subsequent mocks still receive data.

    (ROADMAP test plan item 5)
    """
    call_order: list[str] = []

    def make_passthrough(name: str):
        """Return a side_effect function that records name and returns its input."""
        def _passthrough(spec, *args, **kwargs):
            call_order.append(name)
            return spec  # pass array through unchanged
        return _passthrough

    # Patch the four layer functions at the module boundary.
    # The exact dotted paths must match wherever clean_spectrogram imports them from.
    # We patch them all simultaneously and verify order.
    with (
        patch(
            "usv_spectrogram.classifier.cleaning_pipeline._apply_soft_notch",
            side_effect=make_passthrough("notch"),
        ),
        patch(
            "usv_spectrogram.classifier.cleaning_pipeline._apply_baseline_subtraction",
            side_effect=make_passthrough("baseline"),
        ),
        patch(
            "usv_spectrogram.classifier.cleaning_pipeline._apply_global_mad",
            side_effect=make_passthrough("mad"),
        ),
        patch(
            "usv_spectrogram.classifier.cleaning_pipeline._apply_per_recording_zscore",
            side_effect=make_passthrough("zscore"),
        ),
    ):
        cfg = CleaningConfig(
            apply_soft_notch=True,
            apply_baseline_subtraction=True,
            apply_global_mad=True,
            apply_per_recording_zscore=True,
            tonal_library_path=None,  # no-op allowed per spec
        )
        clean_spectrogram(
            synthetic_spectrogram_2d, cfg, recording_id="order_test"
        )

    assert call_order == ["notch", "baseline", "mad", "zscore"], (
        f"Layer call order was {call_order!r}, expected "
        "['notch', 'baseline', 'mad', 'zscore']. "
        "Spec §'Layer order matters' requires this fixed sequence."
    )


# ---------------------------------------------------------------------------
# Additional: immutability (frozen=True)
# ---------------------------------------------------------------------------


def test_cleaning_config_is_immutable_after_creation(default_config: CleaningConfig):
    """CleaningConfig is declared frozen=True; mutation must raise FrozenInstanceError
    (or AttributeError depending on Python/dataclasses version).
    This prevents accidental mid-run config mutation which would corrupt ablation runs.
    """
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(default_config, "apply_soft_notch", False)


# ---------------------------------------------------------------------------
# Additional: default sample_rate is 250_000, not 300_000
# ---------------------------------------------------------------------------


def test_cleaning_config_default_sample_rate_is_250000():
    """The classifier pipeline is VocalMat-aligned at 250 kHz (not corpus.SAMPLE_RATE_HZ=300 kHz).
    The default must be 250_000 to avoid silent mismatch with VocalMat spectrograms.
    Constraint from ROADMAP §C1 and ADR-001 note.
    """
    cfg = CleaningConfig()
    assert cfg.sample_rate_hz == 250_000, (
        f"Default sample_rate_hz must be 250_000 (VocalMat-aligned), got {cfg.sample_rate_hz}. "
        "Using 300_000 as default would silently misalign with VocalMat frequency bins."
    )


# ---------------------------------------------------------------------------
# Additional: both valid sample rates accepted
# ---------------------------------------------------------------------------


def test_cleaning_config_accepts_250000_and_300000_sample_rates():
    """Both 250_000 and 300_000 must be accepted by __post_init__ without error.
    250_000 = VocalMat-aligned; 300_000 = canonical corpus rate (ADR-001).
    """
    cfg_250 = CleaningConfig(sample_rate_hz=250_000)
    assert cfg_250.sample_rate_hz == 250_000

    cfg_300 = CleaningConfig(sample_rate_hz=300_000)
    assert cfg_300.sample_rate_hz == 300_000


# ---------------------------------------------------------------------------
# Additional: default baseline_mode
# ---------------------------------------------------------------------------


def test_cleaning_config_default_baseline_mode():
    """Baseline mode defaults to 'median_envelope' (Boll 1979 spectral subtraction).
    This is meaningful because 'percentile' mode has different characteristics;
    the implementer should not accidentally swap the default.
    """
    cfg = CleaningConfig()
    assert cfg.baseline_mode == "median_envelope"


# ---------------------------------------------------------------------------
# Additional: output dtype is real float, not int or complex
# ---------------------------------------------------------------------------


def test_clean_spectrogram_output_is_real_valued_float(
    all_disabled_config: CleaningConfig,
    synthetic_spectrogram_2d: np.ndarray,
):
    """clean_spectrogram must return a real-valued float array (float32 or float64).
    Returning integers or complex values would break downstream torch tensor creation.
    """
    result = clean_spectrogram(
        synthetic_spectrogram_2d, all_disabled_config, recording_id="dtype_rec"
    )
    assert np.issubdtype(result.dtype, np.floating), (
        f"Output dtype must be a floating-point type, got {result.dtype}. "
        "Integer or complex output would corrupt downstream ML pipeline."
    )


# ---------------------------------------------------------------------------
# Additional: MAD-only config produces finite output
# ---------------------------------------------------------------------------


def test_clean_spectrogram_mad_only_produces_finite_output():
    """With only global MAD enabled, the output must contain no NaN or Inf values.
    The MAD normalization formula can produce NaN if the spectrogram is constant
    (vmax == vmin) — the implementation must guard this case.
    """
    rng = np.random.default_rng(13)
    spec = rng.normal(-30.0, 12.0, (50, 100)).astype(np.float32)
    cfg = CleaningConfig(
        apply_soft_notch=False,
        apply_baseline_subtraction=False,
        apply_global_mad=True,
        apply_per_recording_zscore=False,
    )
    result = clean_spectrogram(spec, cfg, recording_id="mad_finite_rec")
    assert np.all(np.isfinite(result)), (
        "MAD normalization produced NaN or Inf values. "
        "The implementation must guard the vmax == vmin edge case (divide-by-zero)."
    )


# ---------------------------------------------------------------------------
# Additional: recording_id does not affect output shape
# ---------------------------------------------------------------------------


def test_clean_spectrogram_recording_id_does_not_affect_shape(
    all_disabled_config: CleaningConfig,
):
    """The recording_id parameter is used only for per-recording statistics lookup;
    it must never alter the output shape regardless of its value.
    """
    rng = np.random.default_rng(99)
    spec = rng.normal(-25.0, 5.0, (32, 64)).astype(np.float32)

    result_a = clean_spectrogram(spec.copy(), all_disabled_config, recording_id="rec_A")
    result_b = clean_spectrogram(spec.copy(), all_disabled_config, recording_id="rec_B_longer_name_12345")

    assert result_a.shape == spec.shape
    assert result_b.shape == spec.shape
