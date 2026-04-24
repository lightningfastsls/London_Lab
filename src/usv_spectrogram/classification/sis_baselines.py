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
from usv_language.analysis.sequence_analysis import (
    mutual_information_at_lag,
    mutual_information_within_bouts,
)


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


def compute_sis_depth_1_bout_aware(
    labels: np.ndarray,
    ici_gap_s: np.ndarray,
    bout_threshold_s: float,
    name: str,
) -> tuple[SISResult, int, int]:
    """Bout-aware depth-1 SIS for a chronologically-sorted label sequence.

    Identical contract to :func:`compute_sis_depth_1` (returns the same
    7-field :class:`SISResult`), except the MI is computed only over
    within-bout pairs: pairs whose silent gap exceeds ``bout_threshold_s``
    are excluded from the joint-count matrix. This matches Phase A2's
    methodology (``scripts/analyze_sequential_structure.py``) and Hertz
    2020's *family* of methods (Hertz uses 160 ms ISI; we use the
    data-derived 600 ms threshold registered in ``data/corpus_facts``).

    Parameters
    ----------
    labels:
        1D array of already-chronologically-sorted labels. Strings or
        integers — factorized internally. Do NOT supply ``sort_by_time``
        here; the caller must sort both ``labels`` and ``ici_gap_s`` into
        the same order before calling.
    ici_gap_s:
        1D length ``len(labels) - 1`` array of per-pair silent gaps in
        seconds (end-to-start). Must be in the same order as ``labels``.
    bout_threshold_s:
        Silent-gap threshold. Pairs with gap ``>=`` threshold are treated
        as cross-bout and excluded.
    name:
        Stored verbatim on the returned :class:`SISResult`.

    Returns
    -------
    tuple[SISResult, int, int]
        ``(result, n_within_pairs, n_excluded_pairs)``. The SISResult uses
        the *within-bout* transitions for ``mi_at_lag_1``, but
        ``marginal_entropy`` is still computed on the **full** label
        marginal (the marginal distribution over syllable types isn't
        bout-dependent). ``conditional_entropy`` and
        ``entropy_reduction_pct`` follow from ``H - MI`` as usual.
    """
    labels_arr = np.asarray(labels)
    ici_arr = np.asarray(ici_gap_s)
    n_calls = int(labels_arr.shape[0])

    if n_calls == 0:
        return SISResult(name, 0, 0, 0.0, 0.0, 0.0, 0.0), 0, 0

    if ici_arr.shape[0] != max(0, n_calls - 1):
        raise ValueError(
            f"ici_gap_s length ({ici_arr.shape[0]}) must equal "
            f"len(labels) - 1 = {n_calls - 1}"
        )

    codes, unique = pd.factorize(labels_arr, sort=True)
    codes = np.ascontiguousarray(codes, dtype=np.intp)
    k = int(len(unique))

    if k <= 1 or n_calls < 2:
        return SISResult(name, n_calls, k, 0.0, 0.0, 0.0, 0.0), 0, 0

    counts = np.bincount(codes, minlength=k).astype(np.float64)
    probs = counts / counts.sum()
    nonzero = probs > 0.0
    marginal_h = float(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))

    mi, n_within, n_excluded = mutual_information_within_bouts(
        codes, ici_arr, bout_threshold_s, k, lag=1
    )

    conditional_h = marginal_h - mi
    pct = (mi / marginal_h * 100.0) if marginal_h > 0.0 else 0.0

    return (
        SISResult(name, n_calls, k, mi, marginal_h, conditional_h, pct),
        n_within,
        n_excluded,
    )
