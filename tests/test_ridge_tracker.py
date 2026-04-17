"""Tests for ridge_tracker — written by test-architect BEFORE implementation.

ROADMAP test plan coverage (module 17.3, lines 278-289):
  1. Pure tone at 60 kHz for 100 cols: FM returns ~60 kHz on every column
     -> test_pure_tone_60khz_fm_tracks_correctly
  2. Linearly-sweeping tone 50->80 kHz: FM is monotonic, AM ~constant
     -> test_linear_sweep_fm_is_monotonic_and_am_constant
  3. Spectrogram with harmonic at 2x fundamental: tracker stays on fundamental
     -> test_harmonic_suppression_stays_on_fundamental
  4. Silent column in middle of signal: that column's FM/AM = NaN, surrounding cols intact
     -> test_silent_column_produces_nan_neighbors_intact
  5. All-silent spectrogram: returns all-NaN arrays
     -> test_all_silent_spectrogram_returns_all_nan
  6. Discontinuous pitch jump > max_jump_bins: documents behavior
     -> test_large_discontinuous_jump_behavior
  7. RidgeConfig validation: transition_penalty < 0 raises; max_jump_bins < 1 raises
     -> test_ridgeconfig_rejects_negative_transition_penalty
     -> test_ridgeconfig_rejects_zero_max_jump_bins
  8. Output shapes match input n_time_cols
     -> test_output_shapes_match_n_time_cols
  9. Regression: reconstructed FM within 2 kHz RMSE of ground truth on FM-sweep
     -> test_regression_fm_rmse_within_2khz

Additional coverage (recurring gap patterns):
  - Extra validation: max_jump_bins < 1 raises ValueError
     -> test_ridgeconfig_rejects_zero_max_jump_bins
  - Single time-column edge case
     -> test_single_column_spectrogram_shape_and_value
  - Default RidgeConfig values match spec
     -> test_ridgeconfig_default_values
  - AM values are non-negative on non-silent columns
     -> test_am_nonnegative_on_non_silent_columns
  - NaN propagation: fm and am are NaN on the SAME columns
     -> test_nan_columns_are_consistent_between_fm_and_am

Total: 14 tests (10 from ROADMAP, 4 additional)

Will pass after src/usv_spectrogram/features/ridge_tracker.py is implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# --- Pattern 8 import bootstrap ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.features.ridge_tracker import RidgeConfig, track_ridge  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 300_000        # ADR-001
N_FFT = 512         # ADR-002

# Build a canonical freqs_hz array matching np.fft.rfft output for this SR/N_FFT
# Shape: (N_FFT // 2 + 1,) = (257,)
_FREQS_HZ: np.ndarray = np.fft.rfftfreq(N_FFT, d=1.0 / SR)  # 0 … 150 kHz, 257 bins
_BIN_WIDTH_HZ: float = float(_FREQS_HZ[1] - _FREQS_HZ[0])   # ~585.9 Hz


def _freq_to_bin(freq_hz: float) -> int:
    """Return the closest FFT bin index for a given frequency."""
    return int(round(freq_hz / _BIN_WIDTH_HZ))


def _gaussian_bump(n_bins: int, center_bin: int, sigma_bins: float = 1.5) -> np.ndarray:
    """Return a 1-D Gaussian amplitude profile centred at center_bin."""
    bins = np.arange(n_bins, dtype=float)
    return np.exp(-0.5 * ((bins - center_bin) / sigma_bins) ** 2)


def _build_magnitude(n_bins: int, n_cols: int, center_bins: np.ndarray,
                     amplitude: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Build a (n_bins, n_cols) magnitude array with Gaussian bumps.

    Parameters
    ----------
    center_bins : shape (n_cols,) — per-column bin index of the peak
    amplitude   : peak amplitude of each Gaussian
    sigma       : Gaussian half-width in bins
    """
    mag = np.zeros((n_bins, n_cols), dtype=float)
    for t, cb in enumerate(center_bins):
        mag[:, t] = amplitude * _gaussian_bump(n_bins, int(cb), sigma)
    return mag


# ---------------------------------------------------------------------------
# ROADMAP test 1 — pure 60 kHz tone
# ---------------------------------------------------------------------------

def test_pure_tone_60khz_fm_tracks_correctly() -> None:
    """Spec §17.3 test 1: Pure tone at 60 kHz for 100 cols.

    Constructs a magnitude array where every column has its energy concentrated
    at the FFT bin closest to 60 kHz (bin ~102 at 300 kHz / 512-point FFT).
    The tracker must return FM values within one bin width (~586 Hz) of 60 kHz
    on every column, and all AM values must be positive.
    """
    n_cols = 100
    n_bins = len(_FREQS_HZ)
    target_hz = 60_000.0
    target_bin = _freq_to_bin(target_hz)  # ~102

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert fm_hz.shape == (n_cols,), "FM shape must be (n_time_cols,)"
    assert am.shape == (n_cols,), "AM shape must be (n_time_cols,)"
    assert not np.any(np.isnan(fm_hz)), "No NaN expected on a fully active signal"
    np.testing.assert_allclose(
        fm_hz,
        target_hz,
        atol=_BIN_WIDTH_HZ,
        err_msg=(
            f"FM should be within one bin width ({_BIN_WIDTH_HZ:.1f} Hz) of "
            f"{target_hz} Hz on every column"
        ),
    )
    assert np.all(am > 0), "AM must be positive on non-silent columns"


# ---------------------------------------------------------------------------
# ROADMAP test 2 — linearly-sweeping tone 50→80 kHz
# ---------------------------------------------------------------------------

def test_linear_sweep_fm_is_monotonic_and_am_constant() -> None:
    """Spec §17.3 test 2: Linearly-sweeping 50→80 kHz; FM monotone, AM ~constant.

    Constructs a 100-column magnitude array whose dominant frequency rises
    linearly from ~50 kHz to ~80 kHz.  The tracker must:
      a) produce a monotonically non-decreasing FM trajectory, and
      b) produce AM values that are approximately constant (within 10% of mean).
    """
    n_cols = 100
    n_bins = len(_FREQS_HZ)

    start_bin = _freq_to_bin(50_000.0)  # ~85
    end_bin = _freq_to_bin(80_000.0)    # ~136
    center_bins = np.linspace(start_bin, end_bin, n_cols)

    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0, sigma=1.5)

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert not np.any(np.isnan(fm_hz)), "No NaN expected on fully active signal"

    # Monotonicity: each step should be >= previous (allow equal for plateau)
    diffs = np.diff(fm_hz)
    assert np.all(diffs >= -_BIN_WIDTH_HZ), (
        f"FM should be non-decreasing (monotonic sweep); found drops > one bin: "
        f"min diff = {diffs.min():.1f} Hz"
    )

    # AM ~constant: all values within 10% of mean
    am_mean = float(np.mean(am))
    assert am_mean > 0, "Mean AM must be positive"
    assert np.all(np.abs(am - am_mean) <= 0.1 * am_mean), (
        "AM should be approximately constant on a sweep with uniform Gaussian amplitude"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 3 — harmonic suppression
# ---------------------------------------------------------------------------

def test_harmonic_suppression_stays_on_fundamental() -> None:
    """Spec §17.3 test 3: Tracker stays on fundamental despite 2x harmonic.

    Builds a magnitude array with two peaks per column: one at a 40 kHz
    fundamental (bin ~68) and one at the 80 kHz second harmonic (bin ~136).
    The fundamental is made slightly stronger (amplitude ratio 1.2 : 1.0).
    With a positive transition_penalty the Viterbi path initialized at the
    fundamental should remain on the fundamental throughout — a jump of ~68
    bins exceeds max_jump_bins=10, so the path cannot switch harmonics.

    This tests the core design goal: DP ridge tracking prevents harmonic jumps.
    """
    n_cols = 80
    n_bins = len(_FREQS_HZ)

    fundamental_bin = _freq_to_bin(40_000.0)  # ~68
    harmonic_bin = _freq_to_bin(80_000.0)     # ~136
    harmonic_gap = abs(harmonic_bin - fundamental_bin)  # ~68 bins >> max_jump_bins

    mag = np.zeros((n_bins, n_cols), dtype=float)
    for t in range(n_cols):
        # Stronger fundamental (1.2) + weaker harmonic (1.0)
        mag[:, t] += 1.2 * _gaussian_bump(n_bins, fundamental_bin, sigma_bins=1.5)
        mag[:, t] += 1.0 * _gaussian_bump(n_bins, harmonic_bin, sigma_bins=1.5)

    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    fundamental_hz = _FREQS_HZ[fundamental_bin]
    # Every FM value should be within a few bins of the fundamental, NOT the harmonic
    assert np.all(np.abs(fm_hz - fundamental_hz) <= 3 * _BIN_WIDTH_HZ), (
        f"Tracker should remain on fundamental (~{fundamental_hz/1000:.1f} kHz), "
        f"not jump to harmonic (~{_FREQS_HZ[harmonic_bin]/1000:.1f} kHz). "
        f"Harmonic gap = {harmonic_gap} bins > max_jump_bins=10."
    )


# ---------------------------------------------------------------------------
# ROADMAP test 4 — silent column in middle
# ---------------------------------------------------------------------------

def test_silent_column_produces_nan_neighbors_intact() -> None:
    """Spec §17.3 test 4: Silent column in middle → NaN there, neighbors intact.

    Builds a 50-column pure-tone magnitude array at 70 kHz, then sets column 25
    to all-zero (silent). The tracker must:
      a) return NaN for FM and AM at column 25, and
      b) return valid (non-NaN) numeric FM/AM for columns 24 and 26.
    """
    n_cols = 50
    n_bins = len(_FREQS_HZ)
    silent_col = 25

    target_bin = _freq_to_bin(70_000.0)
    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    # Zero out the silent column completely
    mag[:, silent_col] = 0.0

    cfg = RidgeConfig(silence_threshold=1e-6)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert np.isnan(fm_hz[silent_col]), (
        f"Column {silent_col} has all-zero magnitude; FM must be NaN"
    )
    assert np.isnan(am[silent_col]), (
        f"Column {silent_col} has all-zero magnitude; AM must be NaN"
    )
    assert not np.isnan(fm_hz[silent_col - 1]), (
        f"Column {silent_col - 1} (before silent) must have valid FM"
    )
    assert not np.isnan(fm_hz[silent_col + 1]), (
        f"Column {silent_col + 1} (after silent) must have valid FM"
    )
    assert not np.isnan(am[silent_col - 1]), "Column before silent must have valid AM"
    assert not np.isnan(am[silent_col + 1]), "Column after silent must have valid AM"


# ---------------------------------------------------------------------------
# ROADMAP test 5 — all-silent spectrogram
# ---------------------------------------------------------------------------

def test_all_silent_spectrogram_returns_all_nan() -> None:
    """Spec §17.3 test 5: All-silent spectrogram → all-NaN FM and AM.

    Constructs a magnitude array that is entirely below the silence_threshold.
    Both returned arrays must consist entirely of NaN values.
    """
    n_cols = 40
    n_bins = len(_FREQS_HZ)

    # All values below default silence_threshold=1e-6
    mag = np.zeros((n_bins, n_cols), dtype=float)

    cfg = RidgeConfig(silence_threshold=1e-6)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert fm_hz.shape == (n_cols,), "FM shape must equal n_time_cols"
    assert am.shape == (n_cols,), "AM shape must equal n_time_cols"
    assert np.all(np.isnan(fm_hz)), (
        "All-silent spectrogram: every FM value must be NaN"
    )
    assert np.all(np.isnan(am)), (
        "All-silent spectrogram: every AM value must be NaN"
    )


# ---------------------------------------------------------------------------
# ROADMAP test 6 — discontinuous jump > max_jump_bins
# ---------------------------------------------------------------------------

def test_large_discontinuous_jump_behavior() -> None:
    """Spec §17.3 test 6: Discontinuous pitch jump > max_jump_bins.

    Behavior contract (documented here as the spec does not fully constrain it):
      - The first segment (cols 0-49) has its tone at 40 kHz (~bin 68).
      - The second segment (cols 50-99) has its tone at 100 kHz (~bin 171).
      - The bin gap (~103 bins) exceeds max_jump_bins=10.
      - Because the Viterbi window clips to max_jump_bins, the tracker cannot
        cross the gap in a single step. It is therefore acceptable for the
        tracker to:
            (a) remain near 40 kHz for the second segment (if it cannot follow),
            (b) drift toward 100 kHz over multiple columns, OR
            (c) arrive at 100 kHz within a few columns after the jump.
      - The key invariant that MUST hold: no output value is outside the range
        [min(freqs_hz), max(freqs_hz)], and outputs are non-NaN for all
        non-silent columns.

    This test does NOT assert which bin the tracker ends up on — it only
    asserts the above invariants so that any reasonable, consistent
    implementation passes.
    """
    n_cols = 100
    n_bins = len(_FREQS_HZ)

    low_bin = _freq_to_bin(40_000.0)   # ~68
    high_bin = _freq_to_bin(100_000.0) # ~171

    center_bins = np.empty(n_cols, dtype=float)
    center_bins[:50] = low_bin
    center_bins[50:] = high_bin

    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert not np.any(np.isnan(fm_hz)), "No silent columns → no NaN expected"
    assert np.all(fm_hz >= _FREQS_HZ[0]), "FM must be >= minimum frequency"
    assert np.all(fm_hz <= _FREQS_HZ[-1]), "FM must be <= maximum frequency"
    assert np.all(am >= 0), "AM must be non-negative"


# ---------------------------------------------------------------------------
# ROADMAP test 7a — RidgeConfig rejects negative transition_penalty
# ---------------------------------------------------------------------------

def test_ridgeconfig_rejects_negative_transition_penalty() -> None:
    """Spec §17.3 test 7: RidgeConfig.__post_init__ raises on transition_penalty < 0.

    Per the dataclass spec, a negative penalty is physically meaningless
    (it would encourage jumps instead of discouraging them).
    """
    with pytest.raises(ValueError, match="transition_penalty"):
        RidgeConfig(transition_penalty=-0.01)


# ---------------------------------------------------------------------------
# ROADMAP test 7b — RidgeConfig rejects max_jump_bins < 1
# ---------------------------------------------------------------------------

def test_ridgeconfig_rejects_zero_max_jump_bins() -> None:
    """Spec §17.3 test 7: RidgeConfig.__post_init__ raises on max_jump_bins < 1.

    max_jump_bins=0 would mean the tracker can never move between columns,
    which is pathological.  The spec requires this to raise ValueError.
    """
    with pytest.raises(ValueError, match="max_jump_bins"):
        RidgeConfig(max_jump_bins=0)

    with pytest.raises(ValueError, match="max_jump_bins"):
        RidgeConfig(max_jump_bins=-5)


# ---------------------------------------------------------------------------
# ROADMAP test 8 — output shapes match n_time_cols
# ---------------------------------------------------------------------------

def test_output_shapes_match_n_time_cols() -> None:
    """Spec §17.3 test 8: Output shapes are (n_time_cols,) for arbitrary sizes.

    Checks three different column counts (1, 50, 200) to ensure the output
    shape is always exactly (n_time_cols,) regardless of the input.
    """
    n_bins = len(_FREQS_HZ)
    cfg = RidgeConfig()

    for n_cols in (1, 50, 200):
        center_bins = np.full(n_cols, _freq_to_bin(65_000.0), dtype=float)
        mag = _build_magnitude(n_bins, n_cols, center_bins)
        fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

        assert fm_hz.shape == (n_cols,), (
            f"n_cols={n_cols}: fm_hz.shape={fm_hz.shape}, expected ({n_cols},)"
        )
        assert am.shape == (n_cols,), (
            f"n_cols={n_cols}: am.shape={am.shape}, expected ({n_cols},)"
        )


# ---------------------------------------------------------------------------
# ROADMAP test 9 — regression: FM RMSE < 2 kHz on USV-like FM sweep
# ---------------------------------------------------------------------------

def test_regression_fm_rmse_within_2khz() -> None:
    """Spec §17.3 test 9 / exit criterion: RMSE of reconstructed FM < 2 kHz.

    Constructs a synthetic USV-like FM sweep:
      - 150 time columns
      - Frequency glides from 50 kHz → 90 kHz in the first 100 columns
        (rising sweep, typical for mouse USVs)
      - Then holds at 90 kHz for the remaining 50 columns

    Ground truth FM is known analytically.  After tracking, RMSE of
    fm_hz vs ground_truth_hz must be < 2000 Hz as specified in the exit
    criteria.

    Hand-computed expected bin range:
      50 kHz -> bin 85.3 (bin 85)
      90 kHz -> bin 153.6 (bin 154)
      Sweep covers ~69 bins over 100 columns = ~0.69 bins/column, well within
      max_jump_bins=10, so smooth DP tracking should follow the ground truth
      with sub-bin precision.
    """
    n_bins = len(_FREQS_HZ)
    n_sweep = 100
    n_hold = 50
    n_cols = n_sweep + n_hold

    start_hz = 50_000.0
    end_hz = 90_000.0
    hold_hz = end_hz

    sweep_bins = np.linspace(_freq_to_bin(start_hz), _freq_to_bin(end_hz), n_sweep)
    hold_bins = np.full(n_hold, _freq_to_bin(hold_hz), dtype=float)
    center_bins = np.concatenate([sweep_bins, hold_bins])

    # Build ground truth in Hz from the bin centers (quantized to bin grid)
    ground_truth_hz = _FREQS_HZ[np.round(center_bins).astype(int)]

    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0, sigma=1.5)

    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert not np.any(np.isnan(fm_hz)), "No silent columns → FM must be all-valid"

    rmse = float(np.sqrt(np.mean((fm_hz - ground_truth_hz) ** 2)))
    assert rmse < 2000.0, (
        f"FM RMSE={rmse:.1f} Hz exceeds 2000 Hz exit criterion.  "
        f"The tracker should reconstruct a smooth USV-like sweep within 2 kHz."
    )


# ---------------------------------------------------------------------------
# Additional test: single time-column edge case
# ---------------------------------------------------------------------------

def test_single_column_spectrogram_shape_and_value() -> None:
    """Edge case: spectrogram with exactly one time column.

    The Viterbi forward pass has no previous column to condition on, so the
    first (and only) non-silent column must be seeded directly from argmax.
    Output shapes must be (1,) and FM must match the peak frequency.
    """
    n_bins = len(_FREQS_HZ)
    target_bin = _freq_to_bin(75_000.0)

    mag = np.zeros((n_bins, 1), dtype=float)
    mag[:, 0] = _gaussian_bump(n_bins, target_bin, sigma_bins=1.5)

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert fm_hz.shape == (1,), "Single-column output must have shape (1,)"
    assert am.shape == (1,), "Single-column AM must have shape (1,)"
    assert not np.isnan(fm_hz[0]), "Single non-silent column must not be NaN"
    assert abs(fm_hz[0] - _FREQS_HZ[target_bin]) <= _BIN_WIDTH_HZ, (
        f"Single-column FM {fm_hz[0]/1000:.2f} kHz should be within one bin "
        f"of ground truth {_FREQS_HZ[target_bin]/1000:.2f} kHz"
    )


# ---------------------------------------------------------------------------
# Additional test: default RidgeConfig values match spec
# ---------------------------------------------------------------------------

def test_ridgeconfig_default_values() -> None:
    """Pattern 1 (frozen dataclass): default field values must match the spec.

    The spec (ROADMAP §17.3) specifies:
      transition_penalty = 0.1
      max_jump_bins = 10
      silence_threshold = 1e-6
    """
    cfg = RidgeConfig()
    assert cfg.transition_penalty == 0.1, (
        f"Default transition_penalty should be 0.1, got {cfg.transition_penalty}"
    )
    assert cfg.max_jump_bins == 10, (
        f"Default max_jump_bins should be 10, got {cfg.max_jump_bins}"
    )
    assert cfg.silence_threshold == pytest.approx(1e-6), (
        f"Default silence_threshold should be 1e-6, got {cfg.silence_threshold}"
    )


# ---------------------------------------------------------------------------
# Additional test: AM non-negative on non-silent columns
# ---------------------------------------------------------------------------

def test_am_nonnegative_on_non_silent_columns() -> None:
    """AM(t) = magnitude[ridge_idx[t], t] — magnitude is non-negative, so AM must be.

    This is a structural invariant: magnitude arrays are non-negative (they are
    absolute values of complex spectra).  The tracker must not negate or scale
    the magnitude in a way that produces negative AM.
    """
    n_cols = 60
    n_bins = len(_FREQS_HZ)
    target_bin = _freq_to_bin(55_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=0.5)

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    valid = ~np.isnan(am)
    assert np.any(valid), "Expected at least some non-silent columns"
    assert np.all(am[valid] >= 0.0), (
        f"AM must be non-negative; found min={am[valid].min():.6f}"
    )


# ---------------------------------------------------------------------------
# Additional test: NaN columns consistent between FM and AM
# ---------------------------------------------------------------------------

def test_nan_columns_are_consistent_between_fm_and_am() -> None:
    """FM and AM must be NaN on exactly the same set of columns.

    A column is either silent (both NaN) or active (both numeric).
    Mixed state (one NaN, one finite) would indicate a bug.
    """
    n_cols = 80
    n_bins = len(_FREQS_HZ)
    target_bin = _freq_to_bin(65_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    # Sprinkle several silent columns
    for col in (0, 10, 40, 79):
        mag[:, col] = 0.0

    cfg = RidgeConfig(silence_threshold=1e-6)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    fm_nan = np.isnan(fm_hz)
    am_nan = np.isnan(am)

    np.testing.assert_array_equal(
        fm_nan,
        am_nan,
        err_msg="NaN mask for FM and AM must be identical column-by-column",
    )

# ---------------------------------------------------------------------------
# Hardener additions
# ---------------------------------------------------------------------------

# H1 — Output dtype is float64 regardless of float32 input
# ---------------------------------------------------------------------------

def test_output_dtype_is_float64_for_float32_input() -> None:
    """Reviewer gap #1: float32 magnitude input must still produce float64 outputs.

    track_ridge initialises fm_hz and am via np.full(n_cols, np.nan, dtype=float).
    Python's ``float`` keyword maps to np.float64, so the outputs are always
    float64 regardless of input dtype.  This test anchors that contract so a
    future refactor that changes the dtype keyword would be caught immediately.
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 20
    target_bin = _freq_to_bin(60_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    # Deliberately use float32 input
    mag = _build_magnitude(n_bins, n_cols, center_bins).astype(np.float32)
    assert mag.dtype == np.float32, "Precondition: input must be float32"

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert fm_hz.dtype == np.float64, (
        f"fm_hz must be float64 regardless of input dtype; got {fm_hz.dtype}"
    )
    assert am.dtype == np.float64, (
        f"am must be float64 regardless of input dtype; got {am.dtype}"
    )


# H2 — Two consecutive interior silent columns produce two independent runs
# ---------------------------------------------------------------------------

def test_two_consecutive_interior_silent_columns() -> None:
    """Reviewer gap #2: two adjacent silent columns in the interior.

    Layout: [active×10, silent×2, active×10]
    Expected run segmentation: [(0,10), (12,22)] — two independent runs.
    Each active segment must have valid (non-NaN) FM and AM; the two silent
    columns must be NaN.  Also verifies that each single-column run adjacent
    to the gap uses the argmax seed path (run_len==1 path for cols 9 and 12
    if they are isolated; here they are in runs of length 10 each, confirming
    that the gap truly produces two independent runs rather than one merged run
    with internal NaN placeholders).
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 22
    silent_cols = [10, 11]
    target_bin = _freq_to_bin(65_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)
    for c in silent_cols:
        mag[:, c] = 0.0

    cfg = RidgeConfig(silence_threshold=1e-6)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    # Silent columns must be NaN
    for c in silent_cols:
        assert np.isnan(fm_hz[c]), f"col {c} is silent → fm_hz must be NaN"
        assert np.isnan(am[c]),    f"col {c} is silent → am must be NaN"

    # Active columns on both sides of the gap must be valid
    active_cols = [c for c in range(n_cols) if c not in silent_cols]
    for c in active_cols:
        assert not np.isnan(fm_hz[c]), f"col {c} is active → fm_hz must not be NaN"
        assert not np.isnan(am[c]),    f"col {c} is active → am must not be NaN"

    # Frequency values in each segment should be near the target
    for c in active_cols:
        assert abs(fm_hz[c] - _FREQS_HZ[target_bin]) <= _BIN_WIDTH_HZ, (
            f"col {c}: FM {fm_hz[c]/1000:.2f} kHz deviates > 1 bin from target "
            f"{_FREQS_HZ[target_bin]/1000:.2f} kHz"
        )


# H2b — Three consecutive interior silent columns (single-col runs on both sides)
# ---------------------------------------------------------------------------

def test_three_consecutive_silent_columns_produces_single_col_runs() -> None:
    """Extended gap #2: three silent columns isolate single-active-column runs.

    Layout: [active×1, silent×3, active×1] — 5 columns total.
    Both active runs have run_len == 1, exercising the argmax-seed early return
    path in _track_run for both sides of the gap simultaneously.
    """
    n_bins = len(_FREQS_HZ)

    # Place peaks at different bins on each side so we can verify independently
    left_bin  = _freq_to_bin(60_000.0)
    right_bin = _freq_to_bin(80_000.0)

    mag = np.zeros((n_bins, 5), dtype=float)
    mag[:, 0] = _gaussian_bump(n_bins, left_bin,  sigma_bins=1.5)
    # cols 1, 2, 3 are silent (all-zero)
    mag[:, 4] = _gaussian_bump(n_bins, right_bin, sigma_bins=1.5)

    cfg = RidgeConfig(silence_threshold=1e-6)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    # Silent columns
    for c in (1, 2, 3):
        assert np.isnan(fm_hz[c]), f"col {c} should be NaN"
        assert np.isnan(am[c]),    f"col {c} AM should be NaN"

    # Left single-col run: argmax of col 0 → left_bin
    assert not np.isnan(fm_hz[0]), "col 0 must be non-NaN"
    assert abs(fm_hz[0] - _FREQS_HZ[left_bin]) <= _BIN_WIDTH_HZ, (
        f"col 0 FM {fm_hz[0]/1000:.2f} kHz should be near left peak "
        f"{_FREQS_HZ[left_bin]/1000:.2f} kHz"
    )

    # Right single-col run: argmax of col 4 → right_bin
    assert not np.isnan(fm_hz[4]), "col 4 must be non-NaN"
    assert abs(fm_hz[4] - _FREQS_HZ[right_bin]) <= _BIN_WIDTH_HZ, (
        f"col 4 FM {fm_hz[4]/1000:.2f} kHz should be near right peak "
        f"{_FREQS_HZ[right_bin]/1000:.2f} kHz"
    )


# H3 — max_jump_bins = 1 (tightest legal constraint)
# ---------------------------------------------------------------------------

def test_max_jump_bins_1_tracks_1bin_sweep_rejects_2bin_jumps() -> None:
    """Reviewer gap #3: max_jump_bins=1 is the tightest legal constraint.

    Part A — trackable: a sweep rising by exactly 1 bin per column should be
    followed perfectly.

    Part B — untrackable: a sweep rising by 2 bins per column cannot be
    followed.  The tracker must not crash, must stay within [freq_min, freq_max],
    and must produce no NaN on active columns.  The exact output bin is
    intentionally not asserted (documented unspecified behavior when the path
    cannot follow the true peak).
    """
    n_bins = len(_FREQS_HZ)

    # Part A: 1 bin/col sweep — should track
    n_cols_a = 20
    start_bin = _freq_to_bin(60_000.0)
    # Build center_bins: step exactly 1 bin each column
    center_bins_a = np.arange(start_bin, start_bin + n_cols_a, dtype=float)
    # Clamp to valid range
    center_bins_a = np.clip(center_bins_a, 0, n_bins - 1)
    mag_a = _build_magnitude(n_bins, n_cols_a, center_bins_a,
                             amplitude=1.0, sigma=0.8)

    cfg_tight = RidgeConfig(transition_penalty=0.0, max_jump_bins=1)
    fm_a, am_a = track_ridge(mag_a, _FREQS_HZ, cfg_tight)

    assert not np.any(np.isnan(fm_a)), "1-bin/col sweep: no NaN expected"
    expected_hz_a = _FREQS_HZ[center_bins_a.astype(int)]
    np.testing.assert_allclose(
        fm_a, expected_hz_a, atol=_BIN_WIDTH_HZ,
        err_msg="max_jump_bins=1: 1-bin/col sweep should be tracked exactly"
    )

    # Part B: 2 bins/col sweep — tracker cannot follow, but must not crash
    n_cols_b = 10
    center_bins_b = np.arange(start_bin, start_bin + 2 * n_cols_b, 2, dtype=float)
    center_bins_b = np.clip(center_bins_b, 0, n_bins - 1)
    mag_b = _build_magnitude(n_bins, n_cols_b, center_bins_b,
                             amplitude=1.0, sigma=0.8)

    fm_b, am_b = track_ridge(mag_b, _FREQS_HZ, cfg_tight)

    assert not np.any(np.isnan(fm_b)), "2-bin/col with max_jump=1: no NaN on active"
    assert np.all(fm_b >= _FREQS_HZ[0]), "FM must stay in valid range"
    assert np.all(fm_b <= _FREQS_HZ[-1]), "FM must stay in valid range"


# H4 — transition_penalty=0, max_jump_bins >= n_bins → true per-column argmax
# ---------------------------------------------------------------------------

def test_zero_penalty_unconstrained_window_equals_per_column_argmax() -> None:
    """Reviewer gap #4: degenerate mode documented in W1 fix.

    With transition_penalty=0 AND max_jump_bins >= n_bins, the DP objective
    collapses to per-column argmax (no penalty for any jump, window large enough
    that every bin is reachable from every bin).

    Construct a magnitude array with deliberately DIFFERENT peak bins on each
    column to make this assertion non-trivial.  The DP output must match
    np.argmax(magnitude[:, t]) on every active column.
    """
    n_bins = len(_FREQS_HZ)
    rng = np.random.default_rng(42)
    n_cols = 30

    # Each column has its peak at a random bin to ensure variety
    peak_bins = rng.integers(0, n_bins, size=n_cols)
    mag = np.zeros((n_bins, n_cols), dtype=float)
    for t, pb in enumerate(peak_bins):
        # Tight Gaussian so the peak is unambiguous
        mag[:, t] = _gaussian_bump(n_bins, int(pb), sigma_bins=0.5)
        # Ensure every column is above silence_threshold
        mag[:, t] += 1e-4

    # Use max_jump_bins = n_bins to guarantee all bins reachable from all bins
    cfg = RidgeConfig(transition_penalty=0.0, max_jump_bins=n_bins)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    # Ground truth: per-column argmax
    expected_bins = np.argmax(mag, axis=0)
    expected_hz = _FREQS_HZ[expected_bins]

    assert not np.any(np.isnan(fm_hz)), "No silent columns → no NaN expected"
    np.testing.assert_array_equal(
        fm_hz, expected_hz,
        err_msg=(
            "With penalty=0 and max_jump_bins=n_bins, DP must produce "
            "exactly per-column argmax on every column"
        ),
    )


# H5 — Non-monotonic freqs_hz array
# ---------------------------------------------------------------------------

def test_non_monotonic_freqs_hz_does_not_crash() -> None:
    """Reviewer gap #5: DP logic is bin-index-based, not frequency-order-based.

    The module docstring does not require freqs_hz to be monotonic.  A reversed
    or shuffled freqs_hz must not crash and must still return valid shapes.
    The fm_hz values will be scrambled (mapping bin indices through the reversed
    array), but the tracker must not raise.

    This documents the interface: callers are responsible for passing the
    correct freqs_hz; the module makes no order guarantees.
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 20
    target_bin = _freq_to_bin(60_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    # Reverse the frequency array — non-monotonic
    freqs_reversed = _FREQS_HZ[::-1].copy()
    assert freqs_reversed[0] > freqs_reversed[-1], "Precondition: reversed is descending"

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, freqs_reversed, cfg)

    assert fm_hz.shape == (n_cols,), "Shape must be (n_cols,) with reversed freqs"
    assert am.shape == (n_cols,), "AM shape must be (n_cols,) with reversed freqs"
    # Active columns must have valid (non-NaN) output
    assert not np.any(np.isnan(fm_hz)), "No silent columns → no NaN"
    # FM values must still be drawn from the (reversed) freqs array
    valid_freqs = set(freqs_reversed.tolist())
    for t in range(n_cols):
        assert fm_hz[t] in valid_freqs, (
            f"col {t}: fm_hz={fm_hz[t]} not found in freqs_reversed"
        )


# H5b — Repeated-value freqs_hz array
# ---------------------------------------------------------------------------

def test_repeated_value_freqs_hz_does_not_crash() -> None:
    """Gap #5 extension: freqs_hz with all-identical values must not crash.

    The DP is purely bin-index-based; identical frequency labels are fine.
    The output fm_hz values will all be that repeated frequency, but no
    exception must be raised.
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 10
    target_bin = _freq_to_bin(60_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    freqs_const = np.full(n_bins, 50_000.0)  # all same value

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, freqs_const, cfg)

    assert fm_hz.shape == (n_cols,)
    assert am.shape == (n_cols,)
    assert not np.any(np.isnan(fm_hz)), "No silent columns → no NaN"
    np.testing.assert_array_equal(
        fm_hz,
        np.full(n_cols, 50_000.0),
        err_msg="All-identical freqs_hz: every output FM must equal that frequency",
    )


# H6 — run_len == 2: smallest run exercising full forward-pass + back-trace
# ---------------------------------------------------------------------------

def test_run_length_two_forward_pass_and_backtrace() -> None:
    """Reviewer gap #6: run_len=2 regression anchor.

    The smallest run that exercises the full forward-pass and back-trace path
    (local_t=1 only, then immediate back-trace).  Constructed with a known
    analytically-correct answer to serve as a regression anchor.

    Setup (small 5-bin spectrogram for clarity):
      col 0: peak at bin 2
      col 1: peak at bin 3 (1-bin shift right)
      penalty=0, max_jump_bins=5 → should follow the peaks exactly.
    """
    n_bins = 5
    freqs_small = np.arange(5, dtype=float)  # [0, 1, 2, 3, 4]

    mag = np.zeros((n_bins, 2), dtype=float)
    mag[2, 0] = 1.0   # col 0 peak at bin 2
    mag[3, 1] = 1.0   # col 1 peak at bin 3

    cfg = RidgeConfig(transition_penalty=0.0, max_jump_bins=5,
                      silence_threshold=1e-9)
    fm_hz, am = track_ridge(mag, freqs_small, cfg)

    assert not np.any(np.isnan(fm_hz)), "Both columns active — no NaN expected"
    assert fm_hz[0] == pytest.approx(2.0), (
        f"col 0: expected FM=2.0, got {fm_hz[0]}"
    )
    assert fm_hz[1] == pytest.approx(3.0), (
        f"col 1: expected FM=3.0, got {fm_hz[1]}"
    )
    assert am[0] == pytest.approx(1.0), "col 0 AM must be 1.0"
    assert am[1] == pytest.approx(1.0), "col 1 AM must be 1.0"


# H7 — Consumer invariant: am[t] == magnitude[ridge_bin[t], t] on active cols
# ---------------------------------------------------------------------------

def test_am_equals_magnitude_at_ridge_bin() -> None:
    """Consumer invariant: AM is always the magnitude at the chosen ridge bin.

    am[t] must equal magnitude[ridge_bin[t], t] exactly for every active column.
    This tests the correctness of the final assembly step:
        am[active_cols] = magnitude[active_bins, active_cols]

    We verify this by computing it independently from fm_hz: look up which
    bin in freqs_hz matches fm_hz[t], then compare magnitude[that_bin, t] with am[t].
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 40
    rng = np.random.default_rng(7)

    # Random per-column peaks for a non-trivial test
    peak_bins = rng.integers(10, n_bins - 10, size=n_cols)
    mag = np.zeros((n_bins, n_cols), dtype=float)
    for t, pb in enumerate(peak_bins):
        mag[:, t] = _gaussian_bump(n_bins, int(pb), sigma_bins=2.0)

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    active = ~np.isnan(fm_hz)
    assert np.any(active), "Expected at least some active columns"

    for t in np.nonzero(active)[0]:
        # Recover the ridge bin from the returned fm_hz value
        ridge_bin = int(np.argmin(np.abs(_FREQS_HZ - fm_hz[t])))
        expected_am = float(mag[ridge_bin, t])
        assert am[t] == pytest.approx(expected_am, abs=1e-10), (
            f"col {t}: am={am[t]:.8f} != magnitude[{ridge_bin},{t}]={expected_am:.8f}; "
            f"consumer invariant violated"
        )


# H8 — MAP objective invariant: DP score >= naive per-column argmax score
# ---------------------------------------------------------------------------

def test_dp_score_dominates_naive_argmax_score() -> None:
    """MAP objective invariant: the DP path score must be >= per-column argmax score.

    The DP solves: argmax over paths of  sum(magnitude[f_t, t]) - lambda*sum(|Δf|).
    The naive per-column argmax path achieves score = sum(magnitude[argmax, t]) - lambda*sum(|Δf|).
    For any non-trivial transition_penalty > 0, the per-column argmax path may
    have high |Δf|, making its penalised score <= the smooth DP path.

    Invariant: penalised_score(DP path) >= penalised_score(naive argmax path).

    Construct a multi-peak spectrogram where naive argmax jumps wildly, giving
    a high penalty cost, while the smooth DP path stays near one peak.
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 30
    rng = np.random.default_rng(13)

    # Alternating peaks: even columns at bin A, odd columns at bin B (far away)
    bin_a = _freq_to_bin(40_000.0)   # ~68
    bin_b = _freq_to_bin(100_000.0)  # ~171  (gap >> max_jump_bins=10)
    center_bins = np.where(np.arange(n_cols) % 2 == 0, bin_a, bin_b).astype(float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10)
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    # Compute DP path score
    dp_bins = np.array([int(np.argmin(np.abs(_FREQS_HZ - f))) for f in fm_hz])
    dp_am_sum = float(np.sum(mag[dp_bins, np.arange(n_cols)]))
    dp_jump_penalty = float(cfg.transition_penalty * np.sum(np.abs(np.diff(dp_bins))))
    dp_score = dp_am_sum - dp_jump_penalty

    # Compute naive per-column argmax path score
    naive_bins = np.argmax(mag, axis=0)
    naive_am_sum = float(np.sum(mag[naive_bins, np.arange(n_cols)]))
    naive_jump_penalty = float(cfg.transition_penalty * np.sum(np.abs(np.diff(naive_bins))))
    naive_score = naive_am_sum - naive_jump_penalty

    assert dp_score >= naive_score - 1e-9, (
        f"DP penalised score {dp_score:.6f} must be >= naive argmax score "
        f"{naive_score:.6f} (diff = {dp_score - naive_score:.6f})"
    )


# H9 — silence_threshold = 0: strict-less-than semantics at the zero boundary
# ---------------------------------------------------------------------------

def test_silence_threshold_zero_strict_semantics() -> None:
    """Boundary: silence_threshold=0 documents the strict-less-than semantics.

    The silence condition is ``col_max < silence_threshold`` (strict).
    With silence_threshold=0.0, a column with col_max=0.0 satisfies
    ``0.0 < 0.0 == False``, so it is NOT treated as silent.
    This means silence_threshold=0 effectively disables silence detection
    for any spectrogram with non-negative values.

    This test documents this counter-intuitive behavior as a regression anchor.
    If a caller wants to silence only truly-zero columns, they should use a very
    small positive threshold (e.g., 1e-300) rather than threshold=0.

    Part A: tiny nonzero column → active (not NaN) — same as any threshold.
    Part B: all-zero column WITH threshold=0 → unexpectedly ACTIVE (not NaN),
            because 0.0 < 0.0 is False.  This is the documented behavior.
    Part C: all-zero column with threshold=1e-300 → silent (NaN) as expected.
    """
    n_bins = len(_FREQS_HZ)
    target_bin = _freq_to_bin(60_000.0)

    mag = np.zeros((n_bins, 3), dtype=float)
    mag[target_bin, 0] = 1e-10   # tiny but nonzero
    # col 1: all-zero
    mag[target_bin, 2] = 1.0     # normal

    # Part A + B: with threshold=0, no column is silenced (0.0 < 0.0 is False)
    cfg_zero = RidgeConfig(silence_threshold=0.0)
    fm_zero, _ = track_ridge(mag, _FREQS_HZ, cfg_zero)

    assert not np.isnan(fm_zero[0]), (
        "silence_threshold=0: tiny-nonzero column must be active"
    )
    # Part B: zero column is NOT silenced because 0.0 < 0.0 is False
    assert not np.isnan(fm_zero[1]), (
        "silence_threshold=0: all-zero column is NOT silenced "
        "(0.0 < 0.0 is False — this is the documented strict-lt behavior)"
    )
    assert not np.isnan(fm_zero[2]), (
        "silence_threshold=0: normal column must be active"
    )

    # Part C: with threshold=1e-300, the all-zero column IS silenced
    cfg_tiny = RidgeConfig(silence_threshold=1e-300)
    fm_tiny, _ = track_ridge(mag, _FREQS_HZ, cfg_tiny)

    assert not np.isnan(fm_tiny[0]), (
        "silence_threshold=1e-300: tiny-nonzero column must still be active"
    )
    assert np.isnan(fm_tiny[1]), (
        "silence_threshold=1e-300: all-zero column must be silent "
        "(0.0 < 1e-300 is True)"
    )
    assert not np.isnan(fm_tiny[2]), (
        "silence_threshold=1e-300: normal column must be active"
    )


# H10 — silence_threshold = inf: every column is silent regardless of magnitude
# ---------------------------------------------------------------------------

def test_silence_threshold_infinity_makes_all_columns_silent() -> None:
    """Boundary: silence_threshold=inf treats every column as silent.

    Even a column with magnitude=1e30 is strictly below infinity, so all
    outputs must be NaN.
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 10
    target_bin = _freq_to_bin(60_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1e6)

    cfg = RidgeConfig(silence_threshold=float("inf"))
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert np.all(np.isnan(fm_hz)), (
        "silence_threshold=inf: every column should be treated as silent"
    )
    assert np.all(np.isnan(am)), (
        "silence_threshold=inf: every AM should be NaN"
    )


# H11 — Integer-typed magnitude input
# ---------------------------------------------------------------------------

def test_integer_magnitude_input_does_not_crash() -> None:
    """Pathological input: integer-typed magnitude array.

    Real spectrograms are always float, but a caller might accidentally pass
    an integer array (e.g., from an integer image).  The tracker must not
    crash and must return float64 outputs with the correct shape.
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 15
    target_bin = _freq_to_bin(65_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag_float = _build_magnitude(n_bins, n_cols, center_bins, amplitude=100.0)
    # Quantise to uint16 (mimics image-derived spectrogram)
    mag_int = mag_float.astype(np.uint16)

    cfg = RidgeConfig(silence_threshold=0.5)
    fm_hz, am = track_ridge(mag_int, _FREQS_HZ, cfg)

    assert fm_hz.shape == (n_cols,), "Shape must be (n_cols,) for integer input"
    assert am.shape == (n_cols,), "AM shape must be (n_cols,) for integer input"
    assert fm_hz.dtype == np.float64, "fm_hz must be float64 even for integer input"
    assert am.dtype == np.float64, "am must be float64 even for integer input"


# H12 — max_jump_bins = n_bins - 1 (one below full unconstrained)
# ---------------------------------------------------------------------------

def test_max_jump_bins_n_bins_minus_one_tracks_global_sweep() -> None:
    """Boundary: max_jump_bins = n_bins - 1.

    This is the largest window that does NOT trigger the f_lo >= f_hi guard
    for any shift in [-W, +W].  A sweep from bin 0 to bin n_bins-1 over
    n_bins columns (1 bin/col) must be tracked with zero error.
    """
    n_bins = len(_FREQS_HZ)  # 257
    n_cols = n_bins           # one column per bin

    center_bins = np.arange(n_bins, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0, sigma=0.8)

    cfg = RidgeConfig(
        transition_penalty=0.0,
        max_jump_bins=n_bins - 1,
        silence_threshold=1e-9,
    )
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert not np.any(np.isnan(fm_hz)), "No silent columns expected"
    expected_hz = _FREQS_HZ[np.arange(n_bins)]
    np.testing.assert_allclose(
        fm_hz, expected_hz, atol=_BIN_WIDTH_HZ,
        err_msg="max_jump_bins=n_bins-1: full-range sweep must be tracked bin-by-bin"
    )


# H13 — max_jump_bins > n_bins (f_lo >= f_hi guard exercised for large shifts)
# ---------------------------------------------------------------------------

def test_max_jump_bins_larger_than_n_bins_does_not_crash() -> None:
    """Pathological parameter: max_jump_bins larger than the frequency dimension.

    When max_jump_bins >= n_bins, some shifts in the inner loop produce
    f_lo >= f_hi and trigger the ``continue`` guard (line 175).  The tracker
    must not crash, and the result must be equivalent to the unconstrained case
    (same as penalty=0, max_jump_bins=n_bins for a pure tone).
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 20
    target_bin = _freq_to_bin(60_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    # max_jump_bins = 10 × n_bins — massively oversized
    cfg = RidgeConfig(
        transition_penalty=0.0,
        max_jump_bins=10 * n_bins,
        silence_threshold=1e-9,
    )
    fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)

    assert fm_hz.shape == (n_cols,), "Shape must be correct even with oversized window"
    assert not np.any(np.isnan(fm_hz)), "No NaN expected for oversized window"
    # Must still track the single peak
    np.testing.assert_allclose(
        fm_hz, _FREQS_HZ[target_bin], atol=_BIN_WIDTH_HZ,
        err_msg="Oversized max_jump_bins: pure tone must still be tracked correctly"
    )


# H14 — Non-contiguous (strided) numpy view as input
# ---------------------------------------------------------------------------

def test_strided_numpy_view_input_produces_correct_result() -> None:
    """Robustness: non-contiguous strided numpy view as magnitude input.

    Real callers may pass a slice of a larger spectrogram (e.g.,
    ``magnitude[lo:hi, start:end]``), which produces a non-contiguous array.
    The tracker must handle this without assuming C-contiguity.
    """
    n_bins = len(_FREQS_HZ)
    target_bin = _freq_to_bin(65_000.0)

    # Build a larger array and extract a non-contiguous slice
    n_cols_big = 60
    center_bins = np.full(n_cols_big, target_bin, dtype=float)
    mag_big = _build_magnitude(n_bins, n_cols_big, center_bins, amplitude=1.0)

    # Take every other column → non-contiguous view
    mag_strided = mag_big[:, ::2]   # shape (n_bins, 30), non-contiguous
    assert not mag_strided.flags["C_CONTIGUOUS"], (
        "Precondition: strided view must be non-contiguous"
    )
    n_cols = mag_strided.shape[1]

    cfg = RidgeConfig()
    fm_hz, am = track_ridge(mag_strided, _FREQS_HZ, cfg)

    assert fm_hz.shape == (n_cols,), "Output shape must match strided input"
    assert not np.any(np.isnan(fm_hz)), "No NaN expected on fully active strided input"
    np.testing.assert_allclose(
        fm_hz, _FREQS_HZ[target_bin], atol=_BIN_WIDTH_HZ,
        err_msg="Strided input: pure tone should still be tracked correctly"
    )


# H15 — NaN in magnitude input (pathological)
# ---------------------------------------------------------------------------

def test_nan_in_magnitude_input_does_not_crash() -> None:
    """Pathological input: NaN values in the magnitude spectrogram.

    Real pre-filtered spectrograms should never contain NaN, but a defensive
    test documents the behavior: the tracker must not raise an exception.
    Output behavior (NaN propagation) is intentionally permissive — we only
    assert no crash and correct output shape.
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 10
    target_bin = _freq_to_bin(60_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)

    # Inject NaN into a few interior cells (not entire columns)
    mag[5, 3] = np.nan
    mag[10, 7] = np.nan

    cfg = RidgeConfig()
    # Must not raise
    try:
        fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"track_ridge raised {type(exc).__name__} on NaN-containing input: {exc}")

    assert fm_hz.shape == (n_cols,), "Shape must be correct even with NaN input"
    assert am.shape == (n_cols,), "AM shape must be correct even with NaN input"


# H16 — Negative magnitude values (pathological)
# ---------------------------------------------------------------------------

def test_negative_magnitude_values_do_not_crash() -> None:
    """Pathological input: negative values in the magnitude spectrogram.

    Physical magnitudes are non-negative, but a caller using a signed
    difference-spectrogram or a poorly normalised array might pass negatives.
    The tracker must not raise; output behavior is intentionally unspecified
    (we only assert no crash and correct shapes).
    """
    n_bins = len(_FREQS_HZ)
    n_cols = 10
    target_bin = _freq_to_bin(60_000.0)

    center_bins = np.full(n_cols, target_bin, dtype=float)
    mag = _build_magnitude(n_bins, n_cols, center_bins, amplitude=1.0)
    mag -= 0.5   # shift so lower-energy bins go negative

    cfg = RidgeConfig(silence_threshold=-1.0)  # allow negative-valued columns through
    try:
        fm_hz, am = track_ridge(mag, _FREQS_HZ, cfg)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"track_ridge raised {type(exc).__name__} on negative-valued input: {exc}")

    assert fm_hz.shape == (n_cols,)
    assert am.shape == (n_cols,)
