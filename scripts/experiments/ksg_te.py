"""Pure-Python KSG mutual information, conditional MI, and transfer entropy estimators.

WS-B of the USV shape-analysis program. Pure numpy + scipy.spatial.cKDTree.
NO Java / jidt / idtxl / jpype dependency.

================================================================================
ESTIMATOR VARIANT & THE THREE RESOLVED AMBIGUITIES (flagged by test-architect)
================================================================================

(a) WHICH KSG VARIANT — KSG Algorithm 1 (Kraskov, Stoegbauer & Grassberger, 2004,
    Phys. Rev. E 69, 066138). For MI:

        I(X;Y) = psi(k) + psi(n) - <psi(n_x + 1) + psi(n_y + 1)>

    where psi is the digamma function, the k-th nearest neighbour is found in the
    JOINT (X,Y) space under the CHEBYSHEV (L-inf, max) metric, eps is the distance
    to that k-th neighbour, and n_x / n_y count points strictly within eps in each
    marginal subspace (Algorithm 1 uses strict "< eps", with the marginal radius
    equal to the joint k-NN radius). This is the standard, symmetric Algorithm-1
    estimator; it can read slightly negative on independent data (finite-sample
    bias) — that is expected, not a bug.

    For CMI we use the Frenzel-Pompe (2007) / KSG-style decomposition, which is the
    natural Algorithm-1 generalisation:

        I(X;Y|Z) = psi(k) - < psi(n_xz + 1) + psi(n_yz + 1) - psi(n_z + 1) >

    where the k-th neighbour is found in the joint (X,Y,Z) space (Chebyshev), eps is
    that radius, and n_xz, n_yz, n_z count points strictly within eps in the (X,Z),
    (Y,Z), and (Z) marginal subspaces respectively. Note the psi(n) and psi(k)
    bookkeeping differs from MI: CMI has no global psi(n) term, and uses psi(k)
    (not psi(k)+psi(n)).

(b) lag=1 SEMANTICS — lag=1 means a SINGLE-STEP source delay. TE is

        TE(source->target) = I( target_future ; source_past | target_past )

    with target_future = target[t], target_past = target[t-1], source_past =
    source[t-lag] = source[t-1] for lag=1. Higher lag shifts the source-past
    index back by `lag` steps (target embedding stays at history length 1 here).

(c) `cond` ARGUMENT — `cond` is APPENDED to the conditioning (target_past) block.
    So TE(source->target | cond) = I(target_future ; source_past | [target_past, cond_past]),
    where cond_past is aligned to the same t-lag index as source_past.

================================================================================
BIAS-FLOOR AWARENESS
================================================================================
KSG TE/CMI carry a finite-sample bias that does NOT vanish to exactly 0 on
independent data (it is typically small and positive for TE, can be slightly
negative for MI). Downstream analysis MUST calibrate the estimator zero on
shuffled / circular-shift surrogate data and report TE relative to that null
offset — never compare a raw KSG TE against literal 0. See grammar_te.py.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


# ---------------------------------------------------------------------------
# Input hygiene
# ---------------------------------------------------------------------------

def _as_2d(a: np.ndarray) -> np.ndarray:
    """Coerce a 1-D array (n,) to (n,1); leave (n,d) untouched. Returns float64."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got ndim={arr.ndim}")
    return arr


def _check_k(k: int) -> None:
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer >= 1, got k={k!r}")


def _count_within_radius(data: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """For each point, count OTHER points strictly within its radius (Chebyshev).

    Uses cKDTree.query_ball_point with p=inf. The count excludes the point itself
    (KSG Algorithm 1 marginal counts n_x exclude the point; the standard formula
    uses n_x+1 with n_x = number of strictly-closer neighbours, so we subtract the
    self-match). Radii are shrunk by a tiny epsilon so the boundary point (the
    k-th neighbour that defines eps in the OTHER marginal) is excluded — i.e.
    strict "< eps" as in Algorithm 1.
    """
    tree = cKDTree(data)
    # Strict inequality: shrink radius slightly. KSG Algorithm 1 counts points with
    # marginal distance STRICTLY less than the joint k-NN distance eps.
    shrunk = radii * (1.0 - 1e-12) - 1e-15
    counts = tree.query_ball_point(data, r=shrunk, p=np.inf, return_length=True)
    # query_ball_point includes the point itself (distance 0 < r) -> subtract 1.
    # Clamp at 0: with exact-duplicate points eps can be 0, leaving only the self
    # match within the shrunk radius (count 1 -> 0). The KSG formula then uses
    # digamma(n+1)=digamma(1), which is finite. Negative counts (impossible
    # mathematically, but a float artefact of the shrink) would yield digamma(0)=-inf
    # and poison the mean with NaN; clamp guards against that.
    return np.maximum(np.asarray(counts, dtype=np.float64) - 1.0, 0.0)


# ---------------------------------------------------------------------------
# KSG mutual information (Algorithm 1)
# ---------------------------------------------------------------------------

def ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 4) -> float:
    """KSG Algorithm-1 mutual information I(X;Y) in nats.

    X: (n, dx) or (n,)   Y: (n, dy) or (n,).  1-D arrays treated as (n,1).
    Returns a Python float. Symmetric in X, Y by construction.
    """
    _check_k(k)
    X2 = _as_2d(X)
    Y2 = _as_2d(Y)
    if X2.shape[0] != Y2.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples; got {X2.shape[0]} vs {Y2.shape[0]}"
        )
    n = X2.shape[0]
    if n <= k:
        # Not enough neighbours; gracefully degrade to 0 (finite, not NaN/Inf).
        return 0.0

    joint = np.hstack([X2, Y2])
    tree = cKDTree(joint)
    # k+1 because the nearest neighbour (distance 0) is the point itself.
    dists, _ = tree.query(joint, k=k + 1, p=np.inf)
    eps = dists[:, k]  # distance to k-th neighbour (excluding self)

    nx = _count_within_radius(X2, eps)
    ny = _count_within_radius(Y2, eps)

    mi = (
        digamma(k)
        + digamma(n)
        - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0))
    )
    return float(mi)


# ---------------------------------------------------------------------------
# KSG / Frenzel-Pompe conditional mutual information
# ---------------------------------------------------------------------------

def ksg_cmi(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, k: int = 4) -> float:
    """Conditional mutual information I(X;Y|Z) in nats (Frenzel-Pompe / KSG).

    X, Y, Z: (n, d_*) or (n,).  1-D arrays treated as (n,1).
    Returns a Python float.
    """
    _check_k(k)
    X2 = _as_2d(X)
    Y2 = _as_2d(Y)
    Z2 = _as_2d(Z)
    n = X2.shape[0]
    if not (Y2.shape[0] == n and Z2.shape[0] == n):
        raise ValueError(
            f"X, Y, Z must share the sample count; got {X2.shape[0]}, {Y2.shape[0]}, {Z2.shape[0]}"
        )
    if n <= k:
        return 0.0

    xyz = np.hstack([X2, Y2, Z2])
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1, p=np.inf)
    eps = dists[:, k]

    xz = np.hstack([X2, Z2])
    yz = np.hstack([Y2, Z2])

    n_xz = _count_within_radius(xz, eps)
    n_yz = _count_within_radius(yz, eps)
    n_z = _count_within_radius(Z2, eps)

    cmi = (
        digamma(k)
        - np.mean(digamma(n_xz + 1.0) + digamma(n_yz + 1.0) - digamma(n_z + 1.0))
    )
    return float(cmi)


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------

def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 4,
    lag: int = 1,
    cond: np.ndarray | None = None,
) -> float:
    """Transfer entropy TE(source -> target) in nats.

    TE = I( target_future ; source_past | target_past[, cond_past] )

    - target_future = target[t]
    - target_past   = target[t-1]   (history length 1)
    - source_past   = source[t-lag]
    - cond_past (optional) = cond[t-lag], APPENDED to the conditioning block.

    source, target, cond: (n,) or (n, d). 1-D treated as (n,1).
    Returns a Python float. Implemented as a CMI call, so it inherits the KSG
    Algorithm-1 bias floor — calibrate against surrogates downstream.
    """
    _check_k(k)
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    src = _as_2d(source)
    tgt = _as_2d(target)
    n = src.shape[0]
    if tgt.shape[0] != n:
        raise ValueError(
            f"source and target must have the same length; got {src.shape[0]} vs {tgt.shape[0]}"
        )
    if cond is not None:
        cnd = _as_2d(cond)
        if cnd.shape[0] != n:
            raise ValueError(
                f"cond must match source/target length; got {cnd.shape[0]} vs {n}"
            )
    else:
        cnd = None

    # Valid time indices t run from `lag` .. n-1 so that t-lag and t-1 are in range.
    start = max(lag, 1)
    if n - start <= k:
        return 0.0

    t_idx = np.arange(start, n)

    target_future = tgt[t_idx]            # target[t]
    target_past = tgt[t_idx - 1]          # target[t-1]
    source_past = src[t_idx - lag]        # source[t-lag]

    cond_block = target_past
    if cnd is not None:
        cond_past = cnd[t_idx - lag]
        cond_block = np.hstack([target_past, cond_past])

    # TE = I(target_future ; source_past | cond_block)
    return ksg_cmi(target_future, source_past, cond_block, k=k)
