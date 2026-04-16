"""Tests for omer_vectorize — written by test-architect BEFORE implementation.

Module under test: src/usv_spectrogram/features/omer_vectorize.py
ROADMAP: Module 17.5 — Oren 80D Vectorization (ROADMAP_SIS_BENCHMARK.md lines ~421-520)

ROADMAP test plan coverage:
  1. vectorize_call on synthetic FM sweep: output shape is (2 * n_steps + 1,)
     -> test_vectorize_call_output_shape_default
  2. Duration is the last element of the vector
     -> test_duration_is_last_element
  3. n_steps=40 -> shape (81,); n_steps=20 -> shape (41,); n_steps=60 -> shape (121,)
     -> test_n_steps_controls_output_shape
  4. NaN-interpolate strategy: NaN in middle gets interpolated, no NaN in output
     -> test_nan_interpolate_fills_gaps
  5. NaN-zero strategy: NaN bins become 0 in output
     -> test_nan_zero_fills_with_zeros
  6. All-NaN ridge: returns zero vector (no crash)
     -> test_all_nan_ridge_returns_zero_vector
  7. FM smoothing window 5 preserves trajectory shape within small tolerance
     -> test_fm_smoothing_preserves_monotone_direction
  8. normalize_vectors minmax: each feature dim has min 0, max 1 across dataset
     -> test_normalize_vectors_minmax_range
  9. normalize_vectors zscore: each feature dim has mean ~0, std ~1
     -> test_normalize_vectors_zscore_statistics
  10. OmerVectorConfig with unknown nan_fill_strategy raises ValueError
     -> test_unknown_nan_fill_strategy_raises

Additional coverage (recurring gap patterns):
  - Empty/null arrays                -> test_vectorize_call_single_column_input
  - normalize_vectors raw mode       -> test_normalize_vectors_raw_unchanged
  - Vector content: FM in second half -> test_fm_values_in_second_half_of_vector
  - AM values in first half          -> test_am_values_in_first_half_of_vector
  - normalize_vectors unknown mode   -> test_normalize_vectors_unknown_mode_raises
  - Duration round-trip              -> test_duration_value_preserved_exactly
  - OmerVectorConfig is frozen       -> test_omerconfig_is_frozen_dataclass
  - OmerVectorConfig defaults        -> test_omerconfig_default_values
  - normalize_vectors single row     -> test_normalize_vectors_minmax_single_row

Total: 19 tests (10 from ROADMAP, 9 additional)

Will pass after src/usv_spectrogram/features/omer_vectorize.py is implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Pattern 8: Import bootstrap — tests live in tests/, repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.features.omer_vectorize import (  # noqa: E402
    OmerVectorConfig,
    normalize_vectors,
    vectorize_call,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sweep(n_cols: int, f_start_hz: float = 50_000.0, f_end_hz: float = 90_000.0) -> np.ndarray:
    """Return a linear FM sweep with no NaNs, shape (n_cols,)."""
    return np.linspace(f_start_hz, f_end_hz, n_cols)


def _make_am(n_cols: int, amplitude: float = 0.5) -> np.ndarray:
    """Return a constant-amplitude AM array, shape (n_cols,)."""
    return np.full(n_cols, amplitude, dtype=float)


# ---------------------------------------------------------------------------
# ROADMAP Test 1 — output shape with default config
# ---------------------------------------------------------------------------

def test_vectorize_call_output_shape_default():
    """Verify vectorize_call returns shape (2*n_steps + 1,) with default OmerVectorConfig.

    Spec: 'Return 1D feature vector of shape (2 * n_steps + 1,)'.
    With default n_steps=40, expected shape is (81,).
    """
    cfg = OmerVectorConfig()
    n_cols = 100
    fm_hz = _make_sweep(n_cols)
    am = _make_am(n_cols)
    result = vectorize_call(fm_hz, am, duration_s=0.050, cfg=cfg)

    assert result.ndim == 1, "vectorize_call must return a 1D array"
    assert result.shape == (2 * cfg.n_steps + 1,), (
        f"Expected shape ({2 * cfg.n_steps + 1},), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 2 — duration is the last element
# ---------------------------------------------------------------------------

def test_duration_is_last_element():
    """Verify the last scalar of the feature vector stores the call duration.

    Spec: 'Append duration as an explicit 81st scalar feature'.
    The concatenation order is [am (n_steps), fm (n_steps), duration_s (1)].
    """
    cfg = OmerVectorConfig(n_steps=40)
    fm_hz = _make_sweep(80)
    am = _make_am(80)
    duration_s = 0.0432

    result = vectorize_call(fm_hz, am, duration_s=duration_s, cfg=cfg)

    assert result[-1] == pytest.approx(duration_s, rel=1e-6), (
        f"Last element should be duration {duration_s}, got {result[-1]}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 3 — n_steps controls vector length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_steps,expected_len", [
    (40, 81),
    (20, 41),
    (60, 121),
])
def test_n_steps_controls_output_shape(n_steps: int, expected_len: int):
    """Verify output shape (2*n_steps + 1,) for all supported n_steps values.

    Spec: 'Sweep n_steps in {20, 30, 40, 60} as a hyperparameter; module accepts it as a config value.'
    """
    cfg = OmerVectorConfig(n_steps=n_steps)
    fm_hz = _make_sweep(120)
    am = _make_am(120)

    result = vectorize_call(fm_hz, am, duration_s=0.05, cfg=cfg)

    assert result.shape == (expected_len,), (
        f"n_steps={n_steps} should give shape ({expected_len},), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 4 — NaN-interpolate strategy fills gaps, no NaN in output
# ---------------------------------------------------------------------------

def test_nan_interpolate_fills_gaps():
    """Verify that NaN values in the middle of a ridge are interpolated.

    Spec: 'Handle NaN per cfg.nan_fill_strategy (interpolate across gaps or zero-fill)'.
    After interpolation, the output vector must contain no NaN values.
    """
    cfg = OmerVectorConfig(n_steps=40, nan_fill_strategy="interpolate")
    n_cols = 100

    # Known-good linear sweep with NaN holes punched in the middle
    fm_hz = _make_sweep(n_cols)
    fm_hz[30:50] = np.nan  # 20 consecutive NaNs in the middle

    am = _make_am(n_cols)
    am[30:50] = np.nan

    result = vectorize_call(fm_hz, am, duration_s=0.05, cfg=cfg)

    assert not np.any(np.isnan(result)), (
        "nan_fill_strategy='interpolate' must eliminate all NaN values from the output"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 5 — NaN-zero strategy fills with zeros
# ---------------------------------------------------------------------------

def test_nan_zero_fills_with_zeros():
    """Verify that NaN values are replaced with zero under nan_fill_strategy='zero'.

    Spec: nan_fill_strategy='zero' => NaN bins become 0 in output.
    We use an entirely NaN FM array so that some resampled output values must be 0.
    """
    cfg = OmerVectorConfig(n_steps=10, nan_fill_strategy="zero")
    n_cols = 40

    # FM is entirely NaN
    fm_hz = np.full(n_cols, np.nan)
    am = np.full(n_cols, np.nan)

    result = vectorize_call(fm_hz, am, duration_s=0.02, cfg=cfg)

    # No NaN in output
    assert not np.any(np.isnan(result)), (
        "nan_fill_strategy='zero' must produce no NaN values in the output"
    )

    # The FM and AM portions (all dims except last) must be zero
    expected_zeros = result[:-1]  # exclude duration
    assert np.all(expected_zeros == 0.0), (
        "With all-NaN input and strategy='zero', all FM+AM elements must be 0.0"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 6 — all-NaN ridge returns zero vector (no crash)
# ---------------------------------------------------------------------------

def test_all_nan_ridge_returns_zero_vector():
    """Verify that an all-NaN ridge produces a zero vector and does not raise.

    Spec: 'All-NaN ridge: returns zero vector (no crash)'.
    This guards against division-by-zero, empty-sequence interpolation, etc.
    """
    cfg = OmerVectorConfig(n_steps=40, nan_fill_strategy="interpolate")
    n_cols = 80

    fm_hz = np.full(n_cols, np.nan)
    am = np.full(n_cols, np.nan)
    duration_s = 0.04

    result = vectorize_call(fm_hz, am, duration_s=duration_s, cfg=cfg)

    assert result.shape == (2 * cfg.n_steps + 1,), "Shape must still be correct for all-NaN input"
    assert not np.any(np.isnan(result)), "Result must not contain NaN"
    # FM+AM portion must be zero; duration scalar is appended separately
    assert np.all(result[:-1] == 0.0), "FM+AM portion must be zero for all-NaN ridge"


# ---------------------------------------------------------------------------
# ROADMAP Test 7 — FM smoothing preserves monotone trajectory direction
# ---------------------------------------------------------------------------

def test_fm_smoothing_preserves_monotone_direction():
    """Verify FM smoothing preserves the overall direction of a monotone sweep.

    Spec: 'Smooth FM (mean, window=fm_smooth_window)'.
    A perfectly linear sweep smoothed with a mean filter should remain monotonically
    non-decreasing from first to last resampled step.
    """
    cfg = OmerVectorConfig(n_steps=20, fm_smooth_window=5, nan_fill_strategy="interpolate")
    n_cols = 200

    # Perfectly linear sweep from 40 kHz -> 90 kHz (no noise)
    fm_hz = _make_sweep(n_cols, f_start_hz=40_000.0, f_end_hz=90_000.0)
    am = _make_am(n_cols)

    result = vectorize_call(fm_hz, am, duration_s=0.085, cfg=cfg)

    # FM portion is the second n_steps elements: indices [n_steps : 2*n_steps]
    n_steps = cfg.n_steps
    fm_portion = result[n_steps : 2 * n_steps]

    assert fm_portion[0] < fm_portion[-1], (
        "FM portion of vector should still increase from start to end "
        f"for an upward sweep; got first={fm_portion[0]:.1f} last={fm_portion[-1]:.1f}"
    )

    # First value should be close to 40 kHz, last to 90 kHz (within 10% absolute tolerance)
    assert abs(fm_portion[0] - 40_000.0) < 5_000.0, (
        f"First FM step should be near 40 kHz, got {fm_portion[0]:.0f}"
    )
    assert abs(fm_portion[-1] - 90_000.0) < 5_000.0, (
        f"Last FM step should be near 90 kHz, got {fm_portion[-1]:.0f}"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 8 — normalize_vectors minmax
# ---------------------------------------------------------------------------

def test_normalize_vectors_minmax_range():
    """Verify normalize_vectors(mode='minmax') maps each feature dim to [0, 1].

    Spec: 'mode=minmax: rescale each of the 2*n_steps+1 feature dimensions to [0,1]'.
    We build a synthetic matrix with known per-column min/max to check the rescaling.
    """
    n_steps = 10
    n_calls = 50
    rng = np.random.default_rng(42)

    # Build a (n_calls, 2*n_steps+1) matrix with different scales per column
    raw = rng.uniform(low=1.0, high=100.0, size=(n_calls, 2 * n_steps + 1))

    normed = normalize_vectors(raw, n_steps=n_steps, mode="minmax")

    assert normed.shape == raw.shape, "normalize_vectors must not change matrix shape"

    col_min = normed.min(axis=0)
    col_max = normed.max(axis=0)

    np.testing.assert_allclose(
        col_min, 0.0, atol=1e-9,
        err_msg="minmax normalization: every feature column minimum must be 0"
    )
    np.testing.assert_allclose(
        col_max, 1.0, atol=1e-9,
        err_msg="minmax normalization: every feature column maximum must be 1"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 9 — normalize_vectors zscore
# ---------------------------------------------------------------------------

def test_normalize_vectors_zscore_statistics():
    """Verify normalize_vectors(mode='zscore') gives ~0 mean and ~1 std per feature.

    Spec: 'mode=zscore: StandardScaler per feature dim'.
    With n_calls=200, the central limit theorem ensures tight convergence.
    """
    n_steps = 10
    n_calls = 200
    rng = np.random.default_rng(7)

    # Each column has different mean and variance
    raw = rng.normal(loc=50.0, scale=10.0, size=(n_calls, 2 * n_steps + 1))
    # Skew some columns more
    raw[:, 5] = rng.normal(loc=1000.0, scale=200.0, size=n_calls)

    normed = normalize_vectors(raw, n_steps=n_steps, mode="zscore")

    assert normed.shape == raw.shape, "normalize_vectors must not change matrix shape"

    col_mean = normed.mean(axis=0)
    col_std = normed.std(axis=0, ddof=0)

    np.testing.assert_allclose(
        col_mean, 0.0, atol=1e-10,
        err_msg="zscore normalization: every feature column mean must be ~0"
    )
    np.testing.assert_allclose(
        col_std, 1.0, atol=1e-10,
        err_msg="zscore normalization: every feature column std must be ~1"
    )


# ---------------------------------------------------------------------------
# ROADMAP Test 10 — unknown nan_fill_strategy raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_nan_fill_strategy_raises():
    """Verify that an unknown nan_fill_strategy raises ValueError.

    Spec: 'nan_fill_strategy: interpolate | zero'.
    The validation surface is either OmerVectorConfig.__post_init__ or vectorize_call —
    both are acceptable. We attempt both and verify at least one raises ValueError.
    """
    bogus_strategy = "median_fill_definitely_not_valid"

    raised = False

    # Path A: construction-time validation (preferred by Pattern 1)
    try:
        cfg = OmerVectorConfig(nan_fill_strategy=bogus_strategy)
    except ValueError:
        raised = True

    if not raised:
        # Path B: call-time validation
        cfg = OmerVectorConfig(nan_fill_strategy=bogus_strategy)
        fm_hz = _make_sweep(40)
        am = _make_am(40)
        with pytest.raises(ValueError):
            vectorize_call(fm_hz, am, duration_s=0.02, cfg=cfg)
    else:
        # Construction raised — that's fine, test passes
        pass

    # If neither raised, we need a hard assertion failure
    if not raised:
        # The context manager above would have caught it; if we reach here, it didn't raise
        # This branch is only reachable if pytest.raises swallowed the no-raise case
        # (pytest.raises re-raises if no exception is thrown, so this is unreachable on failure)
        pass


# ---------------------------------------------------------------------------
# Additional Test: FM values live in the second n_steps slice of the vector
# ---------------------------------------------------------------------------

def test_fm_values_in_second_half_of_vector():
    """Verify the FM trajectory occupies indices [n_steps : 2*n_steps] of the output.

    Spec concatenation order: [am_resampled (n_steps), fm_resampled (n_steps), duration_s (1)].
    A constant-FM input should produce near-constant FM slice values.
    """
    cfg = OmerVectorConfig(n_steps=20)
    n_cols = 100
    constant_freq = 65_000.0
    fm_hz = np.full(n_cols, constant_freq)
    am = _make_am(n_cols, amplitude=0.3)

    result = vectorize_call(fm_hz, am, duration_s=0.045, cfg=cfg)

    n = cfg.n_steps
    fm_slice = result[n : 2 * n]

    assert fm_slice.shape == (n,), f"FM slice must have {n} elements"
    # Each element of the FM slice should be close to 65 kHz (within smoothing tolerance)
    np.testing.assert_allclose(
        fm_slice, constant_freq, rtol=0.01,
        err_msg="Constant-FM input should produce near-constant FM slice"
    )


# ---------------------------------------------------------------------------
# Additional Test: AM values live in the first n_steps slice of the vector
# ---------------------------------------------------------------------------

def test_am_values_in_first_half_of_vector():
    """Verify the AM trajectory occupies indices [0 : n_steps] of the output.

    Spec concatenation order: [am_resampled (n_steps), fm_resampled (n_steps), duration_s (1)].
    A constant-AM input should produce near-constant AM slice values.
    """
    cfg = OmerVectorConfig(n_steps=20)
    n_cols = 100
    constant_am = 0.75
    fm_hz = _make_sweep(n_cols)
    am = np.full(n_cols, constant_am)

    result = vectorize_call(fm_hz, am, duration_s=0.045, cfg=cfg)

    n = cfg.n_steps
    am_slice = result[:n]

    assert am_slice.shape == (n,), f"AM slice must have {n} elements"
    np.testing.assert_allclose(
        am_slice, constant_am, rtol=0.05,
        err_msg="Constant-AM input should produce near-constant AM slice"
    )


# ---------------------------------------------------------------------------
# Additional Test: normalize_vectors raw mode returns unchanged matrix
# ---------------------------------------------------------------------------

def test_normalize_vectors_raw_unchanged():
    """Verify normalize_vectors(mode='raw') returns vectors unchanged.

    Spec: 'mode=raw: return vectors unchanged'.
    """
    n_steps = 10
    rng = np.random.default_rng(99)
    raw = rng.uniform(size=(30, 2 * n_steps + 1))

    result = normalize_vectors(raw.copy(), n_steps=n_steps, mode="raw")

    np.testing.assert_array_equal(
        result, raw,
        err_msg="raw mode must return vectors identical to input"
    )


# ---------------------------------------------------------------------------
# Additional Test: normalize_vectors unknown mode raises ValueError
# ---------------------------------------------------------------------------

def test_normalize_vectors_unknown_mode_raises():
    """Verify normalize_vectors raises ValueError for an unrecognised mode string.

    Spec: 'mode in {raw, minmax, zscore}'.
    """
    n_steps = 10
    rng = np.random.default_rng(0)
    vectors = rng.uniform(size=(20, 2 * n_steps + 1))

    with pytest.raises(ValueError):
        normalize_vectors(vectors, n_steps=n_steps, mode="l2_norm_definitely_invalid")


# ---------------------------------------------------------------------------
# Additional Test: duration value preserved exactly (round-trip)
# ---------------------------------------------------------------------------

def test_duration_value_preserved_exactly():
    """Verify that various duration values are stored without floating-point loss.

    Pattern: round-trip consistency.
    """
    cfg = OmerVectorConfig(n_steps=10)
    fm_hz = _make_sweep(50)
    am = _make_am(50)

    for dur in [0.010, 0.050, 0.100, 0.3141592]:
        result = vectorize_call(fm_hz, am, duration_s=dur, cfg=cfg)
        assert result[-1] == pytest.approx(dur, rel=1e-9), (
            f"Duration {dur} not preserved; got {result[-1]}"
        )


# ---------------------------------------------------------------------------
# Additional Test: OmerVectorConfig is a frozen dataclass (immutable)
# ---------------------------------------------------------------------------

def test_omerconfig_is_frozen_dataclass():
    """Verify OmerVectorConfig follows Pattern 1 (frozen=True).

    Spec: '@dataclass(frozen=True)'. Assigning to any field after construction must raise.
    """
    cfg = OmerVectorConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.n_steps = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Additional Test: OmerVectorConfig default values match spec
# ---------------------------------------------------------------------------

def test_omerconfig_default_values():
    """Verify default field values match the ROADMAP spec exactly.

    Spec: n_steps=40, am_smooth_window=6, fm_smooth_window=5, nan_fill_strategy='interpolate'.
    """
    cfg = OmerVectorConfig()
    assert cfg.n_steps == 40
    assert cfg.am_smooth_window == 6
    assert cfg.fm_smooth_window == 5
    assert cfg.nan_fill_strategy == "interpolate"


# ---------------------------------------------------------------------------
# Additional Test: single-column input (boundary condition)
# ---------------------------------------------------------------------------

def test_vectorize_call_single_column_input():
    """Verify vectorize_call handles a minimal single-sample ridge without crashing.

    Pattern: single-item edge case. A ridge with n_cols=1 is a degenerate but valid input —
    time resampling to n_steps > 1 must not raise.
    """
    cfg = OmerVectorConfig(n_steps=10)
    fm_hz = np.array([70_000.0])
    am = np.array([0.5])

    result = vectorize_call(fm_hz, am, duration_s=0.001, cfg=cfg)

    assert result.shape == (2 * cfg.n_steps + 1,), (
        f"Single-column input must still produce shape ({2 * cfg.n_steps + 1},)"
    )
    assert not np.any(np.isnan(result)), "Single-column result must not contain NaN"


# ---------------------------------------------------------------------------
# Additional Test: normalize_vectors minmax with single row
# ---------------------------------------------------------------------------

def test_normalize_vectors_minmax_single_row():
    """Verify normalize_vectors(mode='minmax') handles a single-row matrix gracefully.

    Boundary condition: when n_calls=1, min==max for every column. The implementation
    must not divide by zero and must return a well-defined result (either 0.0 or 0.5).
    """
    n_steps = 5
    single_row = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.05]])
    assert single_row.shape == (1, 2 * n_steps + 1)

    result = normalize_vectors(single_row, n_steps=n_steps, mode="minmax")

    assert result.shape == single_row.shape, "Shape must be preserved for single-row input"
    assert not np.any(np.isnan(result)), "Single-row minmax must not produce NaN"
