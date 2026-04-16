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

Total: 13 tests (9 from ROADMAP, 4 additional)

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
