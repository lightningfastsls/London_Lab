"""Unit tests for scripts/experiments/_dist_stats.py (WS-E distance helpers)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments._dist_stats import (
    median_heuristic_gamma,
    mmd_perm_test,
    perm_test,
    rbf_mmd2,
    subsample,
    wasserstein2,
    wasserstein_perm_test,
)

try:
    import ot  # noqa: F401
    HAVE_OT = True
except Exception:
    HAVE_OT = False

needs_ot = pytest.mark.skipif(not HAVE_OT, reason="POT (ot) not installed")


def _two_gaussians(shift, d=4, n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=(n, d)) + shift
    return X, Y


def test_subsample_size_and_no_replacement():
    rng = np.random.default_rng(0)
    X = np.arange(100).reshape(50, 2).astype(float)
    s = subsample(X, 10, rng)
    assert s.shape == (10, 2)
    # rows are a subset of X, unique
    assert len({tuple(r) for r in s}) == 10
    # n >= len returns all
    assert subsample(X, 999, rng).shape == X.shape


def test_median_heuristic_positive():
    X, Y = _two_gaussians(0.0)
    g = median_heuristic_gamma(X, Y, rng=np.random.default_rng(0))
    assert g > 0 and np.isfinite(g)


def test_mmd_zero_for_same_distribution_positive_for_shift():
    Xa, Xb = _two_gaussians(0.0)
    Xc, Yd = _two_gaussians(3.0)
    g = median_heuristic_gamma(Xa, Xb, rng=np.random.default_rng(1))
    mmd_same = rbf_mmd2(Xa, Xb, g)
    mmd_diff = rbf_mmd2(Xc, Yd, g)
    # same-distribution MMD^2 is near zero (unbiased -> can be slightly negative)
    assert abs(mmd_same) < 0.02
    # shifted distributions clearly separate
    assert mmd_diff > 0.1
    assert mmd_diff > mmd_same


@needs_ot
def test_wasserstein_zero_for_same_grows_with_shift():
    Xa, Xb = _two_gaussians(0.0, n=300)
    w0 = wasserstein2(Xa, Xb)
    Xc, Yd = _two_gaussians(2.0, n=300)
    w2 = wasserstein2(Xc, Yd)
    Xe, Yf = _two_gaussians(5.0, n=300)
    w5 = wasserstein2(Xe, Yf)
    # monotone in the shift; same-dist is small relative to a 2-unit shift.
    # (Empirical W2 between two finite samples of the SAME Gaussian is not 0 —
    # sampling noise ~0.8 at d=4, n=300 — so the bound is on that scale.)
    assert w0 < w2 < w5
    assert w0 < 1.2
    # W2 of a pure mean shift recovers the shift-vector norm = s*sqrt(d)
    # (here s=5, d=4 -> 10), plus small sampling noise.
    assert abs(w5 - 10.0) < 1.0


@needs_ot
def test_wasserstein_sinkhorn_close_to_exact():
    X, Y = _two_gaussians(2.0, n=200)
    w_exact = wasserstein2(X, Y)
    w_sink = wasserstein2(X, Y, sinkhorn_reg=0.5)
    # entropic upper-bounds exact but stays in the ballpark
    assert w_sink >= w_exact - 1e-6
    assert abs(w_sink - w_exact) < 1.5


def test_perm_test_generic_addone_bounds():
    X, Y = _two_gaussians(0.0)
    g = median_heuristic_gamma(X, Y, rng=np.random.default_rng(2))
    obs, p, null = perm_test(X, Y, lambda a, b: rbf_mmd2(a, b, g), n_perm=50, seed=3)
    assert null.shape == (50,)
    # add-one estimator is bounded away from 0 and at most 1
    assert 1.0 / 51.0 <= p <= 1.0


def test_mmd_perm_test_detects_difference_and_null():
    # different distributions -> small p
    Xc, Yd = _two_gaussians(3.0)
    g = median_heuristic_gamma(Xc, Yd, rng=np.random.default_rng(4))
    _, p_diff, _ = mmd_perm_test(Xc, Yd, g, n_perm=100, seed=5)
    assert p_diff < 0.05
    # same distribution -> p not significant
    Xa, Xb = _two_gaussians(0.0, seed=10)
    g2 = median_heuristic_gamma(Xa, Xb, rng=np.random.default_rng(6))
    _, p_same, _ = mmd_perm_test(Xa, Xb, g2, n_perm=100, seed=7)
    assert p_same > 0.05


@needs_ot
def test_wasserstein_perm_test_detects_difference():
    Xc, Yd = _two_gaussians(3.0, n=200)
    obs, p, _ = wasserstein_perm_test(Xc, Yd, n_perm=100, seed=8)
    assert obs > 2.0
    assert p < 0.05


def test_perm_test_deterministic_with_seed():
    X, Y = _two_gaussians(1.0)
    g = median_heuristic_gamma(X, Y, rng=np.random.default_rng(0))
    r1 = mmd_perm_test(X, Y, g, n_perm=30, seed=42)
    r2 = mmd_perm_test(X, Y, g, n_perm=30, seed=42)
    assert r1[0] == r2[0]
    assert r1[1] == r2[1]
    assert np.allclose(r1[2], r2[2])
