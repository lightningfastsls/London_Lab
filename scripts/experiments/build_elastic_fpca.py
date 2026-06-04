#!/usr/bin/env python
"""Elastic FPCA (Fisher-Rao / SRVF with warp alignment) — shared core + full-corpus producer.

WS-A Phase 1 of ``PLAN_continuum_repertoire_program.md``. See handoff
``docs/handoffs/2026-06-04_ws-a-elastic-fpca-implementation.md``.

This module is BOTH:
  (1) the shared elastic-distance core imported by the standing gate harness
      ``scripts/experiments/eval_shape_human_anchored.py`` (via a thin private wrapper), and
  (2) a standalone full-corpus producer that aligns all registered ridges to the
      elastic Karcher mean and emits amplitude + phase FPCA scores as a PARALLEL
      artifact for WS-B/C/D/E to consume (incumbents untouched).

Why elastic FPCA: USV shape is a navigable continuum and per-pair *warp alignment*
(soft-DTW) was the only lever that beat registration on `jump`. The principled
generalization is the Fisher-Rao elastic metric via SRVF *with* warp optimization:
SRVF aligns in velocity space, so a jump (a spike in f') is cheap to align. Our
earlier SRVF test lost only because it was the pointwise q-transform with NO
alignment — the ``min over gamma`` step is the active ingredient.

Verified ``fdasrsf`` 2.6.9 facts (smoke-tested 2026-06-04, do not re-derive):
  - ``elastic_distance(f1, f2, time, method='DP2', lam=0.0)`` returns ``(Dy, Dx)`` =
    (amplitude, phase). Use index [0] for amplitude.
  - It is directional (~1% asymmetric: "f1 aligned to f2") -> the matrix helper
    symmetrizes ``(D + D.T) / 2``.
  - ``lam`` raises rigidity (phase distance shrinks as lam grows). Low lam = max elastic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_TIME_POINTS = 50
TIME_GRID = np.linspace(0.0, 1.0, DEFAULT_TIME_POINTS)
DEFAULT_LAMBDA = 0.0          # placeholder; tuned in GATE A, re-defaulted afterward
DEFAULT_METHOD = "DP2"


# --------------------------------------------------------------------------- #
# Shared elastic-distance core (imported by the gate harness)
# --------------------------------------------------------------------------- #
def elastic_amplitude_distance(f1, f2, time=None, lam: float = DEFAULT_LAMBDA,
                               method: str = DEFAULT_METHOD) -> float:
    """Fisher-Rao / SRVF elastic AMPLITUDE distance between two curves, WITH warp
    optimization (the ``min over gamma`` step) and elasticity penalty ``lam``.

    Returns the amplitude component (``Dy``) of ``fdasrsf.elastic_distance``. This is
    directional in the raw library (``f1`` is aligned to ``f2``); use
    :func:`elastic_amplitude_distance_matrix` for a symmetric matrix.

    Parameters
    ----------
    f1, f2 : array_like, shape (T,)
        The two curves (registered ridges) sampled on a common grid.
    time : array_like or None
        Sample points. Defaults to ``np.linspace(0, 1, len(f1))``.
    lam : float
        Elasticity penalty (>= 0). 0 = maximally elastic.
    method : str
        ``fdasrsf`` warp optimizer ("DP2" default).
    """
    from fdasrsf import elastic_distance

    f1 = np.asarray(f1, dtype=np.float64).ravel()
    f2 = np.asarray(f2, dtype=np.float64).ravel()
    if time is None:
        time = np.linspace(0.0, 1.0, len(f1))
    else:
        time = np.asarray(time, dtype=np.float64).ravel()

    dy, _dx = elastic_distance(f1, f2, time, method=method, lam=lam)
    # Numerical guard: tiny negative round-off -> 0.
    return float(max(dy, 0.0))


def elastic_amplitude_distance_matrix(X, lam: float = DEFAULT_LAMBDA, time=None,
                                      method: str = DEFAULT_METHOD) -> np.ndarray:
    """Pairwise elastic amplitude distance matrix over a batch of curves.

    Parameters
    ----------
    X : array_like, shape (n, T)
        ``n`` registered ridges, each sampled on a common ``T``-point grid.
    lam, method : see :func:`elastic_amplitude_distance`.
    time : array_like or None
        Sample points; defaults to ``np.linspace(0, 1, T)``.

    Returns
    -------
    D : ndarray, shape (n, n), float64
        NON-NEGATIVE, SYMMETRIC (``(D + D.T) / 2``), ZERO on the diagonal.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, T); got shape {X.shape}")
    n, T = X.shape
    if time is None:
        time = np.linspace(0.0, 1.0, T)

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = elastic_amplitude_distance(X[i], X[j], time=time, lam=lam, method=method)
            D[i, j] = d
            D[j, i] = d
    # Symmetrize defensively (raw library is ~1% directional; we filled both
    # halves from one call, so this is a no-op here but documents the invariant).
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------------------- #
# FPCA helpers (vertical / amplitude and a generic reconstruction-error probe)
# --------------------------------------------------------------------------- #
def _svd_components(data):
    """Mean-center ``data`` (n, T) and return (mean, U, S, Vt) of the centered matrix.

    Components are the right singular vectors Vt (rows = principal directions over T).
    """
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return mean, U, S, Vt


def fpca_reconstruction_errors(data, max_components: int) -> np.ndarray:
    """Mean squared reconstruction error using the top-k principal components,
    for k = 1 .. ``max_components``.

    Guaranteed MONOTONICALLY NON-INCREASING in k by the Eckart-Young theorem
    (truncated SVD is the optimal rank-k approximation). Element ``[k-1]`` is the
    error using ``k`` components. At full rank the error is ~0.

    Parameters
    ----------
    data : array_like, shape (n, T)
    max_components : int

    Returns
    -------
    errs : ndarray, shape (max_components,)
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    mean, U, S, Vt = _svd_components(data)
    centered = data - mean
    total_sq = float(np.sum(centered ** 2))
    n_elems = centered.size
    rank = len(S)

    errs = np.empty(max_components, dtype=np.float64)
    for k in range(1, max_components + 1):
        use = min(k, rank)
        # Residual energy after top-`use` singular values = sum of squared tail singvals.
        captured = float(np.sum(S[:use] ** 2))
        residual = max(total_sq - captured, 0.0)
        errs[k - 1] = residual / n_elems     # mean squared error per element
    return errs


def amplitude_fpca(aligned_q, n_components: int) -> dict:
    """Vertical / amplitude FPCA on aligned SRVFs (or aligned functions).

    Parameters
    ----------
    aligned_q : array_like, shape (n, T)
        Aligned SRVFs (``fdawarp.qn`` transposed to row-per-curve) or any (n, T) batch.
    n_components : int

    Returns
    -------
    dict with keys:
        scores      : (n, k)   projection of each curve onto the top-k PCs
        components   : (k, T)   principal directions
        mean         : (T,)     mean curve
        recon_errors : (k,)     reconstruction error using 1..k components
    """
    data = np.asarray(aligned_q, dtype=np.float64)
    n, T = data.shape
    mean, U, S, Vt = _svd_components(data)
    k = int(n_components)
    rank = len(S)

    components = np.zeros((k, T), dtype=np.float64)
    components[:min(k, rank)] = Vt[:min(k, rank)]

    centered = data - mean
    scores = centered @ components.T            # (n, k)

    recon_errors = fpca_reconstruction_errors(data, max_components=k)
    return {"scores": scores, "components": components, "mean": mean,
            "recon_errors": recon_errors}


def _warp_to_psi(warps):
    """Map monotone warps gamma (n, T) to their square-root-density (SRVF) psi = sqrt(d gamma/dt).

    This is the Fisher-Rao-correct representation for the phase (warping) component:
    the space of psi with the L2 metric is the positive orthant of the Hilbert sphere.
    ``np.gradient`` keeps length T so component shape stays (k, T).
    """
    warps = np.asarray(warps, dtype=np.float64)
    t = np.linspace(0.0, 1.0, warps.shape[1])
    dgamma = np.gradient(warps, t, axis=1)
    dgamma = np.clip(dgamma, 0.0, None)         # monotone -> non-negative derivative
    return np.sqrt(dgamma)


def phase_fpca(warps, n_components: int) -> dict:
    """Horizontal / phase FPCA on warp functions gamma (n, T).

    FPCA is run in the square-root-density (psi) space (see :func:`_warp_to_psi`),
    the principled tangent representation for warps.

    Returns
    -------
    dict with keys: scores (n, k), components (k, T), mean (T,).
    """
    psi = _warp_to_psi(warps)
    n, T = psi.shape
    mean, U, S, Vt = _svd_components(psi)
    k = int(n_components)
    rank = len(S)

    components = np.zeros((k, T), dtype=np.float64)
    components[:min(k, rank)] = Vt[:min(k, rank)]

    centered = psi - mean
    scores = centered @ components.T
    return {"scores": scores, "components": components, "mean": mean}


# --------------------------------------------------------------------------- #
# Full-corpus producer
# --------------------------------------------------------------------------- #
def elastic_karcher_align(X, lam: float = DEFAULT_LAMBDA, max_itr: int = 20,
                          parallel: bool = True, verbose: bool = False) -> dict:
    """Align all curves to the elastic Karcher mean via ``fdasrsf.fdawarp.srsf_align``.

    Parameters
    ----------
    X : array_like, shape (n, T)   row-per-curve registered ridges.

    Returns
    -------
    dict: mean_f (T,), aligned_f (n, T), aligned_q (n, T), warps (n, T).
    """
    from fdasrsf import fdawarp

    X = np.asarray(X, dtype=np.float64)
    n, T = X.shape
    time = np.linspace(0.0, 1.0, T)
    f_colmajor = X.T.copy()                      # fdawarp wants (T, n)
    warp = fdawarp(f_colmajor, time)
    warp.srsf_align(method="mean", omethod=DEFAULT_METHOD, lam=lam,
                    parallel=parallel, verbose=verbose, MaxItr=max_itr)

    # fdawarp attrs are column-major (T, n) / (T, n); transpose to row-per-curve.
    aligned_f = np.asarray(warp.fn, dtype=np.float64).T
    aligned_q = np.asarray(warp.qn, dtype=np.float64).T
    gam = np.asarray(warp.gam, dtype=np.float64)
    warps = gam.T if gam.shape[0] == T else gam   # normalize to (n, T)
    mean_f = np.asarray(warp.fmean, dtype=np.float64).ravel()
    return {"mean_f": mean_f, "aligned_f": aligned_f, "aligned_q": aligned_q,
            "warps": warps}


def build(meta_npz: str, out_joblib: str, out_parquet: str,
          lam: float = DEFAULT_LAMBDA, n_amp: int = 5, n_phase: int = 3,
          seed: int = 42, subset: int | None = None,
          max_itr: int = 20, parallel: bool = True) -> dict:
    """Full-corpus producer: align all ridges to the elastic Karcher mean, run
    amplitude + phase FPCA, write a joblib model + per-call parquet of scores.

    Parallel artifact — does NOT overwrite incumbents.
    """
    import joblib
    import pandas as pd

    print("=" * 88)
    print("ELASTIC FPCA — FULL-CORPUS PRODUCER")
    print("=" * 88)
    print("PARAMETERS")
    print(f"  meta       = {meta_npz}")
    print(f"  lam        = {lam}   n_amp = {n_amp}   n_phase = {n_phase}   seed = {seed}")
    print(f"  subset     = {subset}   max_itr = {max_itr}   parallel = {parallel}")

    m = np.load(meta_npz, allow_pickle=True)
    Sh = m["shapes"].astype(np.float64)
    ws = m["wav_stem"].astype(str)
    cid = m["call_id"]
    coh = m["cohort"].astype(str)
    print(f"  ridges     = {Sh.shape}   cohorts = {dict(zip(*np.unique(coh, return_counts=True)))}")

    if subset is not None:
        Sh, ws, cid, coh = Sh[:subset], ws[:subset], cid[:subset], coh[:subset]
        print(f"  [SUBSET] using first {subset} rows -> {Sh.shape}")

    print(f"\n  [ALIGN] elastic Karcher mean align of {Sh.shape[0]} ridges (lam={lam})...")
    al = elastic_karcher_align(Sh, lam=lam, max_itr=max_itr, parallel=parallel)

    print(f"  [FPCA-amp] vertical FPCA on aligned SRVFs -> {n_amp} axes")
    amp = amplitude_fpca(al["aligned_q"], n_components=n_amp)
    print(f"  [FPCA-amp] recon_errors (1..{n_amp}) = {np.round(amp['recon_errors'], 5)}")

    print(f"  [FPCA-phase] horizontal FPCA on warps (psi space) -> {n_phase} axes")
    pha = phase_fpca(al["warps"], n_components=n_phase)

    # ---- per-call parquet ----
    cols = {"wav_stem": ws, "call_id": np.asarray(cid), "cohort": coh}
    for j in range(n_amp):
        cols[f"amp_pc{j + 1}"] = amp["scores"][:, j]
    for j in range(n_phase):
        cols[f"phase_pc{j + 1}"] = pha["scores"][:, j]
    df = pd.DataFrame(cols)

    Path(out_joblib).parent.mkdir(parents=True, exist_ok=True)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    model = {
        "lam": lam, "seed": seed, "n_amp": n_amp, "n_phase": n_phase,
        "time_grid": TIME_GRID,
        "amp_mean": amp["mean"], "amp_components": amp["components"],
        "amp_recon_errors": amp["recon_errors"],
        "phase_mean": pha["mean"], "phase_components": pha["components"],
        "karcher_mean_f": al["mean_f"],
        "n_rows": int(df.shape[0]),
    }
    joblib.dump(model, out_joblib)
    df.to_parquet(out_parquet, index=False)
    print(f"\n  [WRITE] model   -> {out_joblib}")
    print(f"  [WRITE] scores  -> {out_parquet}   ({df.shape[0]} rows, {df.shape[1]} cols)")
    print("  [DONE]")
    return {"model": model, "scores_df": df}


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="/home/shachar/.claude/jobs/57976676/tmp/shape_data/true_registered_ridges_meta.npz")
    ap.add_argument("--out-joblib", default="models/shape_fpca/elastic_fpca.joblib")
    ap.add_argument("--out-parquet", default="models/shape_fpca/elastic_fpca_scores.parquet")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--n-amp", type=int, default=5)
    ap.add_argument("--n-phase", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=None, help="use first N rows (smoke)")
    ap.add_argument("--max-itr", type=int, default=20)
    ap.add_argument("--no-parallel", action="store_true")
    args = ap.parse_args()
    build(args.meta, args.out_joblib, args.out_parquet, lam=args.lam,
          n_amp=args.n_amp, n_phase=args.n_phase, seed=args.seed,
          subset=args.subset, max_itr=args.max_itr, parallel=not args.no_parallel)


if __name__ == "__main__":
    _main()
