"""Tests for scripts/experiments/build_elastic_fpca.py — written by
test-architect BEFORE implementation exists.

This module computes Fisher-Rao / SRVF elastic amplitude & phase distances and
applies Functional Principal Component Analysis (FPCA) to aligned ridge
functions, enabling warp-invariant shape analysis of USV contours.

ROADMAP / contract test-plan coverage (handoff §3 properties):
  1. elastic_amplitude_distance is non-negative float  -> test_amplitude_distance_returns_nonnegative_float
  2. distance matrix is non-negative everywhere         -> test_distance_matrix_nonnegative
  3. distance matrix is symmetric (symmetrized (D+D.T)/2)  -> test_distance_matrix_symmetric
  4. distance matrix has zero diagonal                  -> test_distance_matrix_zero_diagonal
  5. warp-invariance: chevron vs its own monotone warp < 0.2  -> test_amplitude_distance_warp_invariant_below_threshold
  6. flat vs chevron amplitude distance > 1.0           -> test_amplitude_distance_flat_vs_chevron_above_threshold
  7. flat vs chevron amplitude distance > 0.5 (clearly positive)  -> test_amplitude_distance_flat_vs_chevron_clearly_positive
  8. self-distance is exactly 0.0                       -> test_amplitude_distance_self_is_zero
  9. matrix shape is (n, n)                             -> test_distance_matrix_shape
 10. fpca_reconstruction_errors is monotonically non-increasing  -> test_fpca_reconstruction_errors_monotone
 11. fpca_reconstruction_errors at full rank ≈ 0        -> test_fpca_reconstruction_errors_full_rank_near_zero
 12. amplitude_fpca scores shape is (n, k)              -> test_amplitude_fpca_scores_shape
 13. phase_fpca scores shape is (n, k)                  -> test_phase_fpca_scores_shape

Additional coverage (recurring gap patterns):
  - distance matrix n=1 edge case (trivially zero)      -> test_distance_matrix_single_curve_is_scalar_zero
  - distance matrix n=2 symmetry                        -> test_distance_matrix_n2_is_symmetric
  - fpca_reconstruction_errors length == max_components -> test_fpca_reconstruction_errors_length
  - fpca_reconstruction_errors monotone on flat data    -> test_fpca_reconstruction_errors_all_identical_curves
  - amplitude_fpca returns required dict keys           -> test_amplitude_fpca_returns_required_keys
  - amplitude_fpca components shape is (k, T)           -> test_amplitude_fpca_components_shape
  - amplitude_fpca mean shape is (T,)                   -> test_amplitude_fpca_mean_shape
  - amplitude_fpca recon_errors length == n_components  -> test_amplitude_fpca_recon_errors_length
  - phase_fpca returns required dict keys               -> test_phase_fpca_returns_required_keys
  - phase_fpca components shape is (k, T)               -> test_phase_fpca_components_shape
  - phase_fpca mean shape is (T,)                       -> test_phase_fpca_mean_shape

Total: 24 tests (13 from contract/handoff properties, 11 additional gap patterns)

Spec ambiguities noted:
  - The contract says warp distance chevron-vs-warp = 0.1065, tolerance < 0.2. We test
    BOTH < 0.2 AND that the warp distance is substantially less than flat-vs-chevron (> 1.0),
    giving warp_d / flat_chev_d < 0.2 as an additional invariant.
  - phase_fpca takes 'warps' which per the fdawarp API may be (n, T) or (T, n). We test
    with a (n, T) array (the natural Python convention) and assert the output shape only.
  - amplitude_fpca takes 'aligned_q' (aligned SRVFs); we synthesize plausible (n, T)
    data directly rather than running fdawarp (which would be an integration test). The
    unit contract only requires the PCA shapes to be correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap (mirrors test_eval_shape_human_anchored.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
_REPO = REPO_ROOT
for _p in (_SRC, str(_REPO)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Import the module under test.
# This will fail with ModuleNotFoundError until the implementation exists.
# That is the expected red-phase failure — do NOT add a skip guard here.
# ---------------------------------------------------------------------------
from scripts.experiments.build_elastic_fpca import (  # noqa: E402
    amplitude_fpca,
    elastic_amplitude_distance,
    elastic_amplitude_distance_matrix,
    fpca_reconstruction_errors,
    phase_fpca,
)

# ===========================================================================
# Synthetic curves (deterministic, from the API contract)
# ===========================================================================
# These are the exact curves specified in the contract file and verified by
# running fdasrsf 2.6.9 directly:
#   chevron vs flat:  Dy = 1.2841  (clearly distinguishable shapes)
#   chevron vs warp:  Dy = 0.1065  (same shape, different time grid)
#   self:             Dy = ~0.0    (machine epsilon)

_T50 = np.linspace(0.0, 1.0, 50)
_CHEVRON = 1.0 - np.abs(_T50 - 0.5) * 2.0          # peak-1 at t=0.5, 0 at edges
_FLAT = 0.1 * _T50                                   # nearly flat, slight positive slope
_GAMMA = _T50 ** 1.6                                  # monotone reparameterisation [0,1]->[0,1]
_CHEVRON_WARP = np.interp(_GAMMA, _T50, _CHEVRON)    # same shape, warped time axis


def _make_synthetic_ridge_batch(n: int = 8, T: int = 50, seed: int = 42) -> np.ndarray:
    """Return (n, T) array of smooth synthetic ridge functions.

    Uses a sum of sinusoids with random amplitudes so each row is distinct but
    plausible as a pitch contour. Intentionally tiny for fast unit tests.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    rows = []
    for _ in range(n):
        a1, a2 = rng.uniform(0.5, 1.5, size=2)
        phi = rng.uniform(0.0, np.pi)
        row = a1 * np.sin(np.pi * t + phi) + a2 * 0.3 * np.sin(2 * np.pi * t)
        rows.append(row)
    return np.array(rows, dtype=np.float64)


def _identity_warp(T: int = 50) -> np.ndarray:
    """Return the identity warp gamma(t) = t as a (T,) array."""
    return np.linspace(0.0, 1.0, T)


# ===========================================================================
# elastic_amplitude_distance — scalar distance tests
# ===========================================================================

class TestElasticAmplitudeDistance:
    """Tests for elastic_amplitude_distance(f1, f2, time, lam, method) -> float."""

    def test_amplitude_distance_returns_nonnegative_float(self):
        """Spec §1: elastic_amplitude_distance returns a non-negative float.

        Contract: returns float (or numpy scalar coercible to float), value >= 0.
        """
        d = elastic_amplitude_distance(_CHEVRON, _FLAT, time=_T50)
        assert isinstance(float(d), float), (
            f"Expected float-coercible return, got {type(d)}"
        )
        assert float(d) >= 0.0, f"Distance must be non-negative, got {d}"

    def test_amplitude_distance_self_is_zero(self):
        """Spec §1/§3 (contract verified): self-distance is exactly 0.0 (up to machine epsilon).

        Verified from fdasrsf smoke test: elastic_distance(chevron, chevron, t) -> Dy ~ 6.4e-16.
        The implementation MUST return 0.0 (or < 1e-10) for identical inputs.
        """
        d = elastic_amplitude_distance(_CHEVRON, _CHEVRON, time=_T50)
        assert float(d) < 1e-10, (
            f"Self-distance should be ~0; got {d}"
        )

    def test_amplitude_distance_flat_vs_chevron_clearly_positive(self):
        """Spec §4: flat vs chevron amplitude distance is clearly > 0.5.

        Contract measured value: Dy = 1.2841. Testing > 0.5 as a loose lower bound.
        """
        d = elastic_amplitude_distance(_FLAT, _CHEVRON, time=_T50)
        assert float(d) > 0.5, (
            f"flat-vs-chevron amplitude distance should be > 0.5; got {d}"
        )

    def test_amplitude_distance_flat_vs_chevron_above_threshold(self):
        """Spec §3/§4: flat vs chevron amplitude distance > 1.0.

        Contract measured value: Dy = 1.2841 (chevron-to-flat) and 1.2713 (flat-to-chevron).
        Both exceed 1.0.
        """
        d_cf = elastic_amplitude_distance(_CHEVRON, _FLAT, time=_T50)
        assert float(d_cf) > 1.0, (
            f"chevron->flat distance expected > 1.0 (measured 1.2841), got {d_cf}"
        )

    def test_amplitude_distance_warp_invariant_below_threshold(self):
        """Spec §3: warp-invariance — chevron vs its own monotone time-warp < 0.2.

        Contract measured value: Dy = 0.1065. Tolerance < 0.2.
        This confirms the elastic distance 'warps out' temporal deformation.
        """
        d = elastic_amplitude_distance(_CHEVRON, _CHEVRON_WARP, time=_T50)
        assert float(d) < 0.2, (
            f"Warp-invariance violated: chevron vs warp should be < 0.2, got {d}"
        )

    def test_amplitude_distance_warp_much_less_than_shape_change(self):
        """Warp distance is substantially less than shape-change distance.

        Ratio warp_d / flat_chevron_d should be < 0.20, confirming that
        temporal deformation is 'explained' while shape difference is retained.
        """
        d_warp = float(elastic_amplitude_distance(_CHEVRON, _CHEVRON_WARP, time=_T50))
        d_shape = float(elastic_amplitude_distance(_FLAT, _CHEVRON, time=_T50))
        assert d_shape > 0.0, "Shape distance should be positive"
        ratio = d_warp / d_shape
        assert ratio < 0.20, (
            f"Ratio warp/shape = {ratio:.4f}; expected < 0.20 "
            f"(warp_d={d_warp:.4f}, shape_d={d_shape:.4f})"
        )

    def test_amplitude_distance_uses_default_time_grid(self):
        """elastic_amplitude_distance with time=None should default to linspace(0,1,len(f1)).

        Result with explicit linspace grid must match result with time=None.
        """
        d_explicit = float(elastic_amplitude_distance(_CHEVRON, _FLAT, time=_T50))
        d_default = float(elastic_amplitude_distance(_CHEVRON, _FLAT, time=None))
        assert abs(d_explicit - d_default) < 1e-10, (
            f"Default time grid mismatch: explicit={d_explicit}, default={d_default}"
        )


# ===========================================================================
# elastic_amplitude_distance_matrix — matrix property tests
# ===========================================================================

class TestElasticAmplitudeDistanceMatrix:
    """Tests for elastic_amplitude_distance_matrix(X, lam, time) -> (n, n) ndarray."""

    def _small_X(self, n: int = 6) -> np.ndarray:
        """Return (n, 50) array of synthetic ridge functions."""
        return _make_synthetic_ridge_batch(n=n, T=50, seed=7)

    def test_distance_matrix_shape(self):
        """Spec: matrix shape is (n, n) for input (n, T)."""
        X = self._small_X(n=6)
        D = elastic_amplitude_distance_matrix(X, time=_T50)
        assert D.shape == (6, 6), (
            f"Expected shape (6, 6), got {D.shape}"
        )

    def test_distance_matrix_nonnegative(self):
        """Spec §1: all entries in the distance matrix are non-negative."""
        X = self._small_X(n=5)
        D = elastic_amplitude_distance_matrix(X, time=_T50)
        assert np.all(D >= 0.0), (
            f"Distance matrix has negative entries; min={D.min():.6f}"
        )

    def test_distance_matrix_zero_diagonal(self):
        """Spec §2: diagonal entries are zero (self-distance).

        Contract: elastic_distance(f, f) -> Dy ~ 6.4e-16 ≈ 0.
        """
        X = self._small_X(n=5)
        D = elastic_amplitude_distance_matrix(X, time=_T50)
        diag = np.diag(D)
        assert np.allclose(diag, 0.0, atol=1e-8), (
            f"Diagonal should be 0; got max={diag.max():.2e}"
        )

    def test_distance_matrix_symmetric(self):
        """Spec §2: matrix is symmetric — implementation must symmetrize (D+D.T)/2.

        The raw fdasrsf.elastic_distance is ~1% directional (d(c,f)=1.2841 vs
        d(f,c)=1.2713). The matrix helper MUST return a truly symmetric matrix.
        """
        X = self._small_X(n=5)
        D = elastic_amplitude_distance_matrix(X, time=_T50)
        assert np.allclose(D, D.T, atol=1e-12), (
            f"Distance matrix is not symmetric; max asymmetry={np.abs(D - D.T).max():.2e}"
        )

    def test_distance_matrix_n2_is_symmetric(self):
        """Edge case n=2: the 2×2 matrix must still be symmetric and zero-diagonal."""
        X = np.vstack([_CHEVRON[np.newaxis, :], _FLAT[np.newaxis, :]])
        D = elastic_amplitude_distance_matrix(X, time=_T50)
        assert D.shape == (2, 2)
        assert np.isclose(D[0, 0], 0.0, atol=1e-8), f"D[0,0] = {D[0,0]}"
        assert np.isclose(D[1, 1], 0.0, atol=1e-8), f"D[1,1] = {D[1,1]}"
        assert np.isclose(D[0, 1], D[1, 0], atol=1e-12), (
            f"D[0,1]={D[0,1]:.6f} != D[1,0]={D[1,0]:.6f}"
        )

    def test_distance_matrix_off_diagonal_positive_for_distinct_curves(self):
        """Off-diagonal entries are positive when the curves are genuinely different."""
        X = np.vstack([_CHEVRON[np.newaxis, :], _FLAT[np.newaxis, :]])
        D = elastic_amplitude_distance_matrix(X, time=_T50)
        assert D[0, 1] > 0.5, (
            f"chevron vs flat off-diagonal should be > 0.5; got {D[0, 1]:.4f}"
        )

    def test_distance_matrix_single_curve_is_scalar_zero(self):
        """Edge case n=1: a 1×1 matrix must be [[0.0]] (self-distance only)."""
        X = _CHEVRON[np.newaxis, :]          # shape (1, 50)
        D = elastic_amplitude_distance_matrix(X, time=_T50)
        assert D.shape == (1, 1), f"Expected (1,1), got {D.shape}"
        assert np.isclose(D[0, 0], 0.0, atol=1e-8), f"D[0,0] = {D[0,0]}"


# ===========================================================================
# fpca_reconstruction_errors — monotonicity and full-rank tests
# ===========================================================================

class TestFpcaReconstructionErrors:
    """Tests for fpca_reconstruction_errors(data, max_components) -> ndarray."""

    def _smooth_batch(self, n: int = 10, T: int = 50) -> np.ndarray:
        return _make_synthetic_ridge_batch(n=n, T=T, seed=13)

    def test_fpca_reconstruction_errors_length(self):
        """Return array length must equal max_components."""
        data = self._smooth_batch(n=10, T=50)
        max_k = 6
        errs = fpca_reconstruction_errors(data, max_components=max_k)
        assert len(errs) == max_k, (
            f"Expected length {max_k}, got {len(errs)}"
        )

    def test_fpca_reconstruction_errors_monotone(self):
        """Spec §5: reconstruction error is MONOTONICALLY NON-INCREASING in k.

        Adding more principal components can only reduce (or maintain) the
        mean squared reconstruction error.
        """
        data = self._smooth_batch(n=10, T=50)
        errs = fpca_reconstruction_errors(data, max_components=8)
        for i in range(len(errs) - 1):
            assert errs[i] >= errs[i + 1] - 1e-12, (
                f"Monotone violation at k={i+1}->{i+2}: "
                f"errs[{i}]={errs[i]:.6f} < errs[{i+1}]={errs[i+1]:.6f}"
            )

    def test_fpca_reconstruction_errors_full_rank_near_zero(self):
        """Spec §5: at k = min(n, T) (full rank), error must be ≈ 0.

        With n=8 samples and T=50 time points, the data lies in at most an
        8-dimensional subspace. Using max_components=8 must drive error to ≈ 0.
        """
        n = 8
        data = self._smooth_batch(n=n, T=50)
        errs = fpca_reconstruction_errors(data, max_components=n)
        assert float(errs[-1]) < 1e-6, (
            f"Full-rank reconstruction error should be ≈ 0, got {errs[-1]:.4e}"
        )

    def test_fpca_reconstruction_errors_all_identical_curves(self):
        """All identical curves: 1 component captures everything; error[0] ≈ 0.

        If all rows are the same, the first PC explains 100% of variance.
        errors[0] (k=1) should be ≈ 0.
        """
        base = np.sin(np.linspace(0, np.pi, 50))
        data = np.tile(base, (8, 1))          # (8, 50), all identical
        errs = fpca_reconstruction_errors(data, max_components=4)
        assert float(errs[0]) < 1e-8, (
            f"1 PC should perfectly reconstruct identical rows, got error={errs[0]:.4e}"
        )

    def test_fpca_reconstruction_errors_nonnegative(self):
        """Reconstruction errors must all be non-negative (they are mean squared errors)."""
        data = self._smooth_batch(n=10, T=50)
        errs = fpca_reconstruction_errors(data, max_components=5)
        assert np.all(np.array(errs) >= 0.0), (
            f"Negative reconstruction error detected; values={errs}"
        )

    def test_fpca_reconstruction_errors_first_exceeds_last(self):
        """For non-trivial data, error at k=1 must exceed error at k=max.

        This rules out a degenerate implementation that returns constant zeros.
        """
        data = self._smooth_batch(n=10, T=50)
        errs = fpca_reconstruction_errors(data, max_components=8)
        assert float(errs[0]) > float(errs[-1]), (
            f"First error {errs[0]:.6f} should exceed last {errs[-1]:.6f} for non-trivial data"
        )


# ===========================================================================
# amplitude_fpca — shape tests
# ===========================================================================

class TestAmplitudeFpca:
    """Tests for amplitude_fpca(aligned_q, n_components) -> dict."""

    def _aligned_q(self, n: int = 10, T: int = 50) -> np.ndarray:
        """Return plausible (n, T) aligned SRVF batch."""
        return _make_synthetic_ridge_batch(n=n, T=T, seed=99)

    def test_amplitude_fpca_returns_required_keys(self):
        """Contract: returned dict must contain keys: scores, components, mean, recon_errors."""
        aligned_q = self._aligned_q(n=8, T=50)
        result = amplitude_fpca(aligned_q, n_components=3)
        required_keys = {"scores", "components", "mean", "recon_errors"}
        missing = required_keys - set(result.keys())
        assert not missing, f"amplitude_fpca result missing keys: {missing}"

    def test_amplitude_fpca_scores_shape(self):
        """Contract: scores shape must be (n, k) = (n_curves, n_components)."""
        n, k = 10, 4
        aligned_q = self._aligned_q(n=n, T=50)
        result = amplitude_fpca(aligned_q, n_components=k)
        scores = result["scores"]
        assert scores.shape == (n, k), (
            f"Expected scores shape ({n}, {k}), got {scores.shape}"
        )

    def test_amplitude_fpca_components_shape(self):
        """Contract: components shape must be (k, T) = (n_components, time_points)."""
        n, T, k = 10, 50, 4
        aligned_q = self._aligned_q(n=n, T=T)
        result = amplitude_fpca(aligned_q, n_components=k)
        components = result["components"]
        assert components.shape == (k, T), (
            f"Expected components shape ({k}, {T}), got {components.shape}"
        )

    def test_amplitude_fpca_mean_shape(self):
        """Contract: mean shape must be (T,) = (time_points,)."""
        n, T, k = 10, 50, 3
        aligned_q = self._aligned_q(n=n, T=T)
        result = amplitude_fpca(aligned_q, n_components=k)
        mean = result["mean"]
        assert mean.shape == (T,), (
            f"Expected mean shape ({T},), got {mean.shape}"
        )

    def test_amplitude_fpca_recon_errors_length(self):
        """Contract: recon_errors length must equal n_components."""
        n, k = 10, 4
        aligned_q = self._aligned_q(n=n, T=50)
        result = amplitude_fpca(aligned_q, n_components=k)
        recon_errors = result["recon_errors"]
        assert len(recon_errors) == k, (
            f"Expected recon_errors length {k}, got {len(recon_errors)}"
        )


# ===========================================================================
# phase_fpca — shape tests
# ===========================================================================

class TestPhaseFpca:
    """Tests for phase_fpca(warps, n_components) -> dict."""

    def _warps(self, n: int = 10, T: int = 50) -> np.ndarray:
        """Return (n, T) array of plausible warp functions.

        Each row is a monotone function from 0 to 1 (a valid time-warp gamma).
        Built as cumulative sums of positive increments, normalized to [0, 1].
        """
        rng = np.random.default_rng(17)
        warps = []
        for _ in range(n):
            increments = rng.uniform(0.5, 1.5, size=T)
            cumulative = np.cumsum(increments)
            normalized = cumulative / cumulative[-1]  # maps to [0, 1]
            warps.append(normalized)
        return np.array(warps, dtype=np.float64)

    def test_phase_fpca_returns_required_keys(self):
        """Contract: returned dict must contain keys: scores, components, mean."""
        warps = self._warps(n=8, T=50)
        result = phase_fpca(warps, n_components=2)
        required_keys = {"scores", "components", "mean"}
        missing = required_keys - set(result.keys())
        assert not missing, f"phase_fpca result missing keys: {missing}"

    def test_phase_fpca_scores_shape(self):
        """Contract: scores shape must be (n, k) = (n_curves, n_components)."""
        n, k = 10, 3
        warps = self._warps(n=n, T=50)
        result = phase_fpca(warps, n_components=k)
        scores = result["scores"]
        assert scores.shape == (n, k), (
            f"Expected scores shape ({n}, {k}), got {scores.shape}"
        )

    def test_phase_fpca_components_shape(self):
        """Contract: components shape must be (k, T) = (n_components, time_points)."""
        n, T, k = 10, 50, 3
        warps = self._warps(n=n, T=T)
        result = phase_fpca(warps, n_components=k)
        components = result["components"]
        assert components.shape == (k, T), (
            f"Expected components shape ({k}, {T}), got {components.shape}"
        )

    def test_phase_fpca_mean_shape(self):
        """Contract: mean shape must be (T,) = (time_points,)."""
        n, T, k = 10, 50, 2
        warps = self._warps(n=n, T=T)
        result = phase_fpca(warps, n_components=k)
        mean = result["mean"]
        assert mean.shape == (T,), (
            f"Expected mean shape ({T},), got {mean.shape}"
        )

    def test_phase_fpca_scores_finite(self):
        """Phase FPCA scores must be finite (no NaN or Inf from centering).

        The centering step subtracts the mean warp (identity); degenerate
        warps must not produce non-finite scores.
        """
        warps = self._warps(n=8, T=50)
        result = phase_fpca(warps, n_components=2)
        scores = result["scores"]
        assert np.all(np.isfinite(scores)), (
            f"phase_fpca scores contain non-finite values: {scores}"
        )
