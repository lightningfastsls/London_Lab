"""Syntax Information Score (SIS) baselines at depth 1.

Module 17.1 of ``ROADMAP_SIS_BENCHMARK.md``. Computes ``MI(X_n ; X_{n-1})`` —
the mutual information between consecutive syllable labels — for an arbitrary
labeling of a USV call sequence. Serves as the decision-gate baseline for the
rest of Phase 17: the existing labelings (Scattoni-7, DeepSqueak-27, HDBSCAN-3)
are scored against Hertz 2020's published iMSA value of 0.22 bits to decide
whether new feature engineering is warranted.

Reuses ``usv_language.analysis.sequence_analysis.mutual_information_at_lag``
to preserve numerical continuity with the Phase A2 result (0.093 bits on
Scattoni-7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical source: sequence_analysis. information_theory.py re-exports the
# same function, so either import path resolves to the same callable.
from usv_language.analysis.sequence_analysis import mutual_information_at_lag


@dataclass(frozen=True)
class SISResult:
    """SIS result for a single labeling.

    Immutable; constructed only via :func:`compute_sis_depth_1`.
    """

    name: str
    n_calls: int
    n_labels: int
    mi_at_lag_1: float
    marginal_entropy: float
    conditional_entropy: float
    entropy_reduction_pct: float


def compute_sis_depth_1(
    labels: np.ndarray,
    name: str,
    sort_by_time: np.ndarray | None = None,
) -> SISResult:
    """Compute depth-1 SIS for a sequence of labels.

    Parameters
    ----------
    labels:
        1D array of labels. May be strings or integers — factorized internally.
    name:
        Stored verbatim on the returned :class:`SISResult`.
    sort_by_time:
        Optional 1D array with the same length as ``labels``. If given, labels
        are reordered by ``np.argsort(sort_by_time, kind="stable")`` before
        the MI is computed. Use for CSVs that are not yet chronologically
        sorted.

    Returns
    -------
    SISResult
        Holds n_calls, n_labels, marginal entropy, conditional entropy, MI at
        lag 1 (all in bits, log2), and entropy_reduction_pct = (MI / H) * 100
        (guarded: returns 0.0 when H = 0).
    """

    labels = np.asarray(labels)
    n_calls = int(labels.shape[0])

    if n_calls == 0:
        return SISResult(name, 0, 0, 0.0, 0.0, 0.0, 0.0)

    if sort_by_time is not None:
        time_arr = np.asarray(sort_by_time)
        if time_arr.shape[0] != n_calls:
            raise ValueError(
                f"sort_by_time length ({time_arr.shape[0]}) does not match "
                f"labels length ({n_calls})"
            )
        order = np.argsort(time_arr, kind="stable")
        labels = labels[order]

    codes, unique = pd.factorize(labels, sort=True)
    codes = np.ascontiguousarray(codes, dtype=np.intp)
    k = int(len(unique))

    if k <= 1 or n_calls < 2:
        return SISResult(name, n_calls, k, 0.0, 0.0, 0.0, 0.0)

    counts = np.bincount(codes, minlength=k).astype(np.float64)
    probs = counts / counts.sum()
    nonzero = probs > 0.0
    marginal_h = float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))

    mi = float(mutual_information_at_lag(codes, k, lag=1))

    conditional_h = marginal_h - mi
    pct = (mi / marginal_h * 100.0) if marginal_h > 0.0 else 0.0

    return SISResult(name, n_calls, k, mi, marginal_h, conditional_h, pct)
