"""Pure distribution-distance helpers for WS-E (confound-robust cohort comparison).

These are deliberately small, side-effect-free functions so they can be unit
tested in isolation (``tests/experiments/test_dist_stats.py``). The WS-E driver
``harmonize_and_compare.py`` imports them; nothing here reads files or prints.

Two families of distribution distance between two point clouds X, Y in R^d:

* **Wasserstein-2** (optimal transport) via POT (``ot.emd2`` exact, or
  ``ot.sinkhorn2`` entropic). We return the *distance* sqrt(W2^2) so it is in the
  same units as the data.
* **RBF-kernel MMD^2** (maximum mean discrepancy), unbiased U-statistic. Bandwidth
  via the median heuristic by default.

Both come with a label-permutation test that gives an exact-ish p-value for the
null "X and Y are draws from the same distribution".

Note on GAK (handoff §3 asked for a GAK elastic kernel): GAK is for *variable
length raw sequences*. The WS-A elastic-FPCA coordinates we compare are already
fixed 8-vectors in an SRVF warp-aligned shape space, so the elastic geometry is
baked into the coordinates. Applying a sequence kernel to an 8-vector is not
meaningful; the principled choice is an RBF kernel on the FPCA coordinates. This
deviation is documented in the driver and the report.
"""
from __future__ import annotations

import numpy as np

try:  # POT is required by the driver; tests that need OT skip if missing.
    import ot  # type: ignore
    _HAVE_OT = True
except Exception:  # pragma: no cover - environment guard
    _HAVE_OT = False


def subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` rows of ``X`` without replacement (or all rows if n>=len)."""
    X = np.asarray(X, dtype=float)
    if n >= len(X):
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray | None = None,
                           rng: np.random.Generator | None = None,
                           max_n: int = 1000) -> float:
    """RBF gamma = 1 / (2 * median_pairwise_sqdist) on the pooled sample.

    The median heuristic is the standard default bandwidth for kernel-MMD. We
    cap the pooled sample at ``max_n`` for the O(n^2) median computation.
    """
    Z = X if Y is None else np.vstack([X, Y])
    Z = np.asarray(Z, dtype=float)
    if rng is not None and len(Z) > max_n:
        Z = Z[rng.choice(len(Z), size=max_n, replace=False)]
    elif len(Z) > max_n:
        Z = Z[:max_n]
    # pairwise squared distances, upper triangle
    sq = _pairwise_sqdist(Z, Z)
    iu = np.triu_indices(len(Z), k=1)
    med = np.median(sq[iu])
    if med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med)


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix between rows of A and B."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    sq = a2 + b2 - 2.0 * A @ B.T
    return np.maximum(sq, 0.0)


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased estimate of MMD^2 with RBF kernel k(a,b)=exp(-gamma*||a-b||^2).

    Uses U-statistics (diagonal excluded) for Kxx, Kyy and the standard biased
    mean for Kxy. Can be slightly negative for identical distributions — that is
    the expected behaviour of the unbiased estimator near zero.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    m, n = len(X), len(Y)
    Kxx = np.exp(-gamma * _pairwise_sqdist(X, X))
    Kyy = np.exp(-gamma * _pairwise_sqdist(Y, Y))
    Kxy = np.exp(-gamma * _pairwise_sqdist(X, Y))
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_yy = Kyy.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_xy = Kxy.mean()
    return float(term_xx + term_yy - 2.0 * term_xy)


def wasserstein2(X: np.ndarray, Y: np.ndarray, sinkhorn_reg: float | None = None) -> float:
    """W2 distance (sqrt of optimal-transport cost with squared-euclidean ground).

    Uniform marginals. Exact ``ot.emd2`` by default; entropic ``ot.sinkhorn2``
    when ``sinkhorn_reg`` is given (faster for large n / permutation nulls).
    """
    if not _HAVE_OT:  # pragma: no cover
        raise RuntimeError("POT (ot) not installed")
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    M = _pairwise_sqdist(X, Y)  # squared euclidean ground cost
    if sinkhorn_reg is None:
        cost = ot.emd2(a, b, M)
    else:
        cost = ot.sinkhorn2(a, b, M, reg=sinkhorn_reg)
    return float(np.sqrt(max(cost, 0.0)))


def perm_test(X: np.ndarray, Y: np.ndarray, stat_fn, n_perm: int,
              seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Two-sample label-permutation test for an arbitrary distance ``stat_fn``.

    Returns ``(observed, p_value, null_distribution)``. p = (1 + #{null >=
    obs}) / (1 + n_perm) — the standard add-one estimator (never 0).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    rng = np.random.default_rng(seed)
    obs = stat_fn(X, Y)
    pooled = np.vstack([X, Y])
    m = len(X)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(pooled))
        null[i] = stat_fn(pooled[perm[:m]], pooled[perm[m:]])
    p = (1.0 + np.sum(null >= obs)) / (1.0 + n_perm)
    return float(obs), float(p), null


def wasserstein_perm_test(X: np.ndarray, Y: np.ndarray, n_perm: int = 200,
                          seed: int = 0, sinkhorn_reg: float | None = None
                          ) -> tuple[float, float, np.ndarray]:
    """Permutation test with the W2 distance as the statistic."""
    return perm_test(X, Y, lambda a, b: wasserstein2(a, b, sinkhorn_reg),
                     n_perm=n_perm, seed=seed)


def mmd_perm_test(X: np.ndarray, Y: np.ndarray, gamma: float, n_perm: int = 200,
                  seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Permutation test with RBF MMD^2 as the statistic."""
    return perm_test(X, Y, lambda a, b: rbf_mmd2(a, b, gamma),
                     n_perm=n_perm, seed=seed)
