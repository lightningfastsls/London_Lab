"""Shared benchmark harness for the shape-invariance bake-off.

`benchmark(...)` scores ONE representation (an (N,d) embedding OR an (N,N)
precomputed distance matrix) against human shape families with leave-one-out
kNN retrieval purity + 1000x bootstrap 95% CIs, in 4 settings:

    {pooled, within-stratum} x {invariant-only, invariant + z-scored side-channels}

Decision rule everywhere: NON-overlapping CIs, never point estimates.

REUSE of SPEC functions (scripts/experiments/eval_shape_human_anchored.py):
  - POOLED settings call `bootstrap_purity_ci` / `bootstrap_purity_ci_from_distance`
    and `loo_knn_purity` / `knn_purity_from_distance` DIRECTLY.
  - WITHIN-STRATUM settings cannot use those (they have no pool-restriction arg),
    so per-point purity is computed here per stratum and aggregated N-weighted,
    then bootstrapped with the IDENTICAL resample-the-target-points + 2.5/97.5
    percentile recipe the SPEC bootstrap uses. This is a faithful extension, not
    a different metric. (Documented in CONTRACT.md.)

within-stratum = restrict each query's neighbour pool to its own `stratum`
(default = cohort/cage), compute per-point purity inside each stratum, then pool
all target points across strata (N-weighted by construction) and bootstrap.

For kind='distance', side-channels are NOT applicable (no embedding to
concatenate to) -> the two side-channel settings carry a note instead of scores.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/experiments
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from eval_shape_human_anchored import (  # noqa: E402
    bootstrap_purity_ci,
    bootstrap_purity_ci_from_distance,
    knn_purity_from_distance,
    loo_knn_purity,
)


# ---------------------------------------------------------------------------
# side-channel z-scoring (handoff rule 2): z-score the (N,3) side array, leave
# the invariant matrix as-is, then hstack.
# ---------------------------------------------------------------------------
def zscore_side(side: np.ndarray) -> np.ndarray:
    side = np.asarray(side, dtype=np.float64)
    mu = side.mean(axis=0)
    sd = side.std(axis=0)
    sd[sd == 0] = 1.0
    return (side - mu) / sd


def _with_side(X: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.hstack([np.asarray(X, dtype=np.float64), zscore_side(side)])


# ---------------------------------------------------------------------------
# within-stratum per-point purity (faithful extension of the SPEC core)
# ---------------------------------------------------------------------------
def _within_stratum_per_point(M, labels, strata, target, k, kind, min_stratum=2):
    """Per-point purity for `target`, neighbour pool restricted to each query's
    own stratum, pooled across strata. Returns (per_point_array, n_target).
    Strata with < max(min_stratum, 2) points or no target points are skipped.
    """
    labels = np.asarray(labels)
    strata = np.asarray(strata)
    per_all = []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) < max(min_stratum, 2):
            continue
        ls = labels[idx]
        tmask = ls == target
        if int(tmask.sum()) == 0:
            continue
        kk = min(k, len(idx) - 1)
        if kk < 1:
            continue
        if kind == "embedding":
            Xs = np.asarray(M, dtype=np.float64)[idx]
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xs)
            _, nbr = nn.kneighbors(Xs)
            nbr = nbr[:, 1:]
        else:  # distance
            Ds = np.array(M, dtype=np.float64)[np.ix_(idx, idx)]
            np.fill_diagonal(Ds, np.inf)
            nbr = np.argsort(Ds, axis=1)[:, :kk]
        per = (ls[nbr[tmask]] == target).mean(axis=1)
        per_all.append(per.astype(np.float64))
    if not per_all:
        return np.array([], dtype=np.float64), 0
    per = np.concatenate(per_all)
    return per, len(per)


def _bootstrap_from_per(per, n_boot=1000, seed=42):
    """IDENTICAL recipe to the SPEC bootstrap: resample target points with
    replacement, re-average, take 2.5/97.5 percentiles."""
    nt = len(per)
    if nt == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(per.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot[b] = per[rng.integers(0, nt, nt)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _within_stratum_setting(M, family, strata, families, k, kind, n_boot, seed):
    out = {}
    for f in families:
        per, nt = _within_stratum_per_point(M, family, strata, f, k, kind)
        out[f] = list(_bootstrap_from_per(per, n_boot=n_boot, seed=seed))
    return out


def _pooled_setting_embedding(X, family, families, k, n_boot, seed):
    return {f: list(bootstrap_purity_ci(X, family, f, k=k, n_boot=n_boot, seed=seed))
            for f in families}


def _pooled_setting_distance(D, family, families, k, n_boot, seed):
    return {f: list(bootstrap_purity_ci_from_distance(D, family, f, k=k, n_boot=n_boot, seed=seed))
            for f in families}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def benchmark(X_or_D, *, kind="embedding", meta, families, k=10, ks=(1, 5, 15),
              side=None, n_boot=1000, seed=42):
    """Score one representation in all 4 settings.

    Parameters
    ----------
    X_or_D : (N,d) embedding if kind=='embedding', else (N,N) distance matrix.
    kind   : 'embedding' | 'distance'.
    meta   : the loader dict; must contain 'family' and 'stratum' (==cohort by
             default). 'side' read from here if `side` arg is None.
    families : list of family names to score.
    k      : primary kNN k (CI'd, all settings). ks : extra k for the pooled_invariant
             k-sweep (point estimates only).
    side   : (N,3) RAW side-channel array; z-scored internally. Defaults to
             meta['side']. Ignored for kind=='distance'.

    Returns
    -------
    dict with keys: pooled_invariant, pooled_sidechannel, withinstratum_invariant,
    withinstratum_sidechannel (each {family: [point, lo, hi]}), and k_sweep
    ({k: {family: point}} for the pooled invariant setting).
    """
    family = np.asarray(meta["family"])
    strata = np.asarray(meta["stratum"])
    if side is None:
        side = meta.get("side")

    result = {}

    if kind == "embedding":
        X = np.asarray(X_or_D, dtype=np.float64)
        result["pooled_invariant"] = _pooled_setting_embedding(X, family, families, k, n_boot, seed)
        result["withinstratum_invariant"] = _within_stratum_setting(
            X, family, strata, families, k, "embedding", n_boot, seed)
        if side is not None:
            Xs = _with_side(X, side)
            result["pooled_sidechannel"] = _pooled_setting_embedding(Xs, family, families, k, n_boot, seed)
            result["withinstratum_sidechannel"] = _within_stratum_setting(
                Xs, family, strata, families, k, "embedding", n_boot, seed)
        else:
            note = {"_note": "no side-channel array provided"}
            result["pooled_sidechannel"] = note
            result["withinstratum_sidechannel"] = note
        # k-sweep on pooled invariant (point estimates only)
        ksweep = {}
        for kk in ks:
            ksweep[str(kk)] = {f: loo_knn_purity(X, family, f, k=kk)[0] for f in families}
        result["k_sweep"] = ksweep

    elif kind == "distance":
        D = np.asarray(X_or_D, dtype=np.float64)
        result["pooled_invariant"] = _pooled_setting_distance(D, family, families, k, n_boot, seed)
        result["withinstratum_invariant"] = _within_stratum_setting(
            D, family, strata, families, k, "distance", n_boot, seed)
        note = {"_note": "side-channels N/A for distance-native methods (no embedding to concatenate)"}
        result["pooled_sidechannel"] = note
        result["withinstratum_sidechannel"] = note
        ksweep = {}
        for kk in ks:
            ksweep[str(kk)] = {f: knn_purity_from_distance(D, family, f, k=kk) for f in families}
        result["k_sweep"] = ksweep
    else:
        raise ValueError(f"kind must be 'embedding' or 'distance', got {kind!r}")

    return result
