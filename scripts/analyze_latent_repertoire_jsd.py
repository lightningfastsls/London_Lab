"""Move A — repertoire JSD in latent space.

Clusters the combined 32-D VAE latent space with K-means K=20 to produce a
shared cluster "alphabet", computes per-cohort cluster-occupancy
proportions, then pairwise Jensen-Shannon divergences (in bits) between the
five cohort splits with call-level bootstrap CIs.

The lab cohort (``lab_131204``) is split into ``lab_matched`` and
``lab_swap`` based on the ``mXfmY`` token in ``wav_stem`` (matched if
``X == Y``, swap otherwise). The three wild cohorts (``5970``, ``3452``,
``9252``) pass through unchanged.

Why these choices (from
``docs/handoffs/2026-05-20_latent-analysis-b-a-c.md`` Move A section):

- K-means produces a *partition*, which is required for JSD over
  cluster proportions (HDBSCAN noise points are unbinned).
- K=20 is smaller than the legacy 27-class taxonomy but big enough to
  preserve morphological diversity; we sweep K ∈ {10, 20, 30, 50} as a
  robustness check.
- Bootstrap CIs resample call-level units ``(wav_stem, call_id)`` with
  replacement WITHIN each cohort — this captures the right level of
  statistical dependence (multiple latent patches per call are
  auto-correlated, so patch-level bootstrap underestimates variance).
- JSD is in *bits* (``log2``) so the upper bound is 1.0, making the
  decision-gate read more intuitive.

CLI usage::

    .venv/bin/python scripts/analyze_latent_repertoire_jsd.py \\
        --latents-path results/contour_vae_combined/latents.parquet \\
        --out-dir results/latent_repertoire \\
        --model-out models/latent_kmeans \\
        --k 20 --ks 10,20,30,50 --n-boot 1000 --seed 42
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cohort label used for the lab dataset BEFORE the matched/swap split.
_LAB_COHORT = "lab_131204"
_COUPLE_RE = re.compile(r"_m(\d+)fm(\d+)_")

# Latent dimensionality from the combined VAE parquet.
_N_LATENT_DIMS = 32


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def load_latents(path: str) -> pd.DataFrame:
    """Load the combined VAE latents parquet into a DataFrame.

    Returns columns: ``z_0..z_31`` (float), ``cohort``, ``wav_stem``,
    ``call_id``, ``patch_idx``, and any other columns present in the
    source parquet.
    """
    return pd.read_parquet(path)


def split_lab_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``cohort_split`` column splitting lab into matched / swap.

    Rules
    -----
    * ``cohort == 'lab_131204'`` and ``wav_stem`` matches ``_m(\\d+)fm(\\d+)_``
      with ``X == Y``  -> ``cohort_split = 'lab_matched'``.
    * ``cohort == 'lab_131204'`` and ``X != Y``  -> ``cohort_split = 'lab_swap'``.
    * ``cohort == 'lab_131204'`` and the regex does not match  -> ``ValueError``.
    * Any other cohort  -> ``cohort_split == cohort`` (passthrough).

    The original ``cohort`` column is left unchanged.
    """
    out = df.copy()

    def _classify(row: pd.Series) -> str:
        cohort = row["cohort"]
        if cohort != _LAB_COHORT:
            return cohort
        stem = row["wav_stem"]
        m = _COUPLE_RE.search(str(stem))
        if m is None:
            raise ValueError(
                f"Lab row has unparseable wav_stem {stem!r} — expected "
                f"'_m<digits>fm<digits>_' couple token"
            )
        return "lab_matched" if m.group(1) == m.group(2) else "lab_swap"

    out["cohort_split"] = out.apply(_classify, axis=1)
    return out


def _latent_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract ``z_0..z_31`` columns into a contiguous float32 matrix."""
    cols = [f"z_{i}" for i in range(_N_LATENT_DIMS)]
    return df[cols].to_numpy(dtype=np.float32, copy=False)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def fit_kmeans(Z: np.ndarray, k: int, seed: int) -> KMeans:
    """Fit K-means with ``n_init=10`` for reproducibility.

    The model is fit on the FULL combined latent matrix (all cohorts) to
    produce a shared alphabet for cohort comparison.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(Z)
    return km


# ---------------------------------------------------------------------------
# Cluster proportions
# ---------------------------------------------------------------------------

def cluster_proportions(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    cohort_col: str = "cohort_split",
) -> pd.DataFrame:
    """Per-cohort cluster-occupancy proportions over a K-cluster alphabet.

    Parameters
    ----------
    df : DataFrame
        Must contain ``cohort_col`` for every row. Row order must match
        ``labels``.
    labels : ndarray of int
        Cluster assignment per row (length == len(df)).
    k : int
        Total number of clusters. The returned DataFrame has integer
        columns ``0..k-1``; clusters with zero patches in a cohort appear
        as ``0.0`` (NOT ``NaN``).
    cohort_col : str
        Column in ``df`` carrying the cohort label.

    Returns
    -------
    DataFrame with one row per cohort (index = cohort name) and integer
    columns ``[0, 1, ..., k-1]`` whose row sums equal 1.0.
    """
    if len(labels) != len(df):
        raise ValueError(
            f"labels length {len(labels)} does not match df length {len(df)}"
        )

    cohorts = df[cohort_col].to_numpy()
    labels = np.asarray(labels, dtype=np.int64)

    # Build the count matrix via np.add.at for speed.
    unique_cohorts = sorted(pd.unique(cohorts))
    cohort_index = {c: i for i, c in enumerate(unique_cohorts)}
    cohort_ids = np.array([cohort_index[c] for c in cohorts], dtype=np.int64)

    counts = np.zeros((len(unique_cohorts), k), dtype=np.float64)
    np.add.at(counts, (cohort_ids, labels), 1.0)

    totals = counts.sum(axis=1, keepdims=True)
    # Guard against empty cohorts (would yield NaN). Replace 0 totals with 1
    # to keep proportions at 0; len-0 cohorts can't show up after groupby
    # but defensiveness here costs nothing.
    safe_totals = np.where(totals == 0, 1.0, totals)
    props = counts / safe_totals

    # Columns must be Python ints (NOT strings or numpy scalars) per spec.
    return pd.DataFrame(
        props,
        index=pd.Index(unique_cohorts),
        columns=[int(i) for i in range(k)],
    )


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------

def js_divergence_bits(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits between two probability vectors.

    Uses base-2 logarithm so the divergence is bounded by 1.0 for any
    pair of distributions over the same alphabet. The 0*log(0) = 0
    convention is enforced via masking.

    A small numerical-noise clamp to 1.0 is applied if the raw value
    overshoots by less than 1e-9; otherwise the raw value is returned.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    # Tiny clamp for float noise (only when within 1e-9 of the upper bound).
    if 1.0 < jsd <= 1.0 + 1e-9:
        return 1.0
    return jsd


# ---------------------------------------------------------------------------
# Pairwise JSD matrix
# ---------------------------------------------------------------------------

def pairwise_jsd_matrix(props: pd.DataFrame) -> pd.DataFrame:
    """Symmetric NxN pairwise JSD matrix from a per-cohort proportions DF.

    Rows of ``props`` are cohorts, columns are cluster IDs. Returns a
    DataFrame with the same cohort labels as both index and columns, zero
    on the diagonal, and ``JSD(i, j)`` in bits off-diagonal.
    """
    cohorts = list(props.index)
    n = len(cohorts)
    arr = props.to_numpy(dtype=np.float64)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = js_divergence_bits(arr[i], arr[j])
            out[i, j] = v
            out[j, i] = v
    return pd.DataFrame(out, index=cohorts, columns=cohorts)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _precompute_call_count_vectors(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    cohort_col: str,
) -> Dict[str, "tuple[np.ndarray, np.ndarray]"]:
    """For each cohort, return (call_ids, count_vectors).

    ``call_ids`` is a list of unique ``(wav_stem, call_id)`` tuples (the
    resampling units). ``count_vectors`` has shape ``(n_calls, k)``: row
    ``i`` is the cluster-count histogram for call ``i``.

    Resampling a cohort = ``rng.choice(n_calls, n_calls, replace=True)``,
    summing the selected rows, and normalizing to proportions.
    """
    cohorts = df[cohort_col].to_numpy()
    wav_stems = df["wav_stem"].to_numpy()
    call_ids_col = df["call_id"].to_numpy()
    labels = np.asarray(labels, dtype=np.int64)

    result: Dict[str, "tuple[np.ndarray, np.ndarray]"] = {}

    for cohort in sorted(pd.unique(cohorts)):
        mask = cohorts == cohort
        ws = wav_stems[mask]
        ci = call_ids_col[mask]
        lab = labels[mask]

        # Group by (wav_stem, call_id) -> count vector.
        # Build a stable hashable key.
        call_tuples = list(zip(ws.tolist(), ci.tolist()))
        unique_tuples: List[tuple] = []
        tuple_to_idx: Dict[tuple, int] = {}
        for t in call_tuples:
            if t not in tuple_to_idx:
                tuple_to_idx[t] = len(unique_tuples)
                unique_tuples.append(t)

        n_calls = len(unique_tuples)
        count_vectors = np.zeros((n_calls, k), dtype=np.float64)
        call_idx_per_patch = np.array(
            [tuple_to_idx[t] for t in call_tuples], dtype=np.int64
        )
        np.add.at(count_vectors, (call_idx_per_patch, lab), 1.0)

        result[cohort] = (np.array(unique_tuples, dtype=object), count_vectors)

    return result


# Smoothing applied ONLY to bootstrap proportions to (a) avoid perfectly
# disjoint supports causing zero bootstrap variance under JSD's
# "disjoint => 1.0" mathematical floor, and (b) ensure the bootstrap
# distribution is continuous. Magnitude is negligible (1e-12 per cluster
# cell, ~K * 1e-12 ≈ 2e-11 effect on proportions for K=20).
_BOOTSTRAP_SMOOTHING_EPS = 1e-12


def _proportions_from_counts(totals: np.ndarray, smooth: float = 0.0) -> np.ndarray:
    """Normalize a counts vector to proportions with optional additive smoothing."""
    t = totals + smooth
    s = t.sum()
    if s <= 0:
        return np.zeros_like(totals)
    return t / s


def bootstrap_jsd_pairs(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    cohort_col: str,
    n_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap CIs on pairwise JSDs (resampling calls within cohort).

    Algorithm
    ---------
    1. Precompute per-cohort per-call cluster-count vectors.
    2. Point estimate: full per-cohort proportions (same smoothing as
       bootstrap), then pairwise JSD.
    3. For each replicate r in 1..n_reps:
       - For each cohort, ``rng.choice(n_calls, n_calls, replace=True)``,
         sum the count vectors of the sampled calls, normalize.
       - Compute pairwise JSDs from the bootstrapped proportions.
    4. CIs per pair: ``np.percentile(reps, [2.5, 97.5])``, then clamp so
       the point estimate lies within ``[ci_lo, ci_hi]`` (extends a tail
       outward if the percentile interval misses the point — common with
       small ``n_reps`` on a non-negative statistic like JSD).

    A tiny additive smoothing (1e-12 per cluster) is applied to BOTH the
    point proportions and bootstrap proportions to avoid the
    JSD-disjoint-support = 1.0 mathematical floor obscuring variance.
    Magnitude is below floating-point noise level for typical K.

    Returns
    -------
    Long-form DataFrame with one row per upper-triangle pair, columns:
    ``cohort_a, cohort_b, jsd_point, jsd_ci_lo, jsd_ci_hi, n_reps, seed``.
    ``cohort_a < cohort_b`` lexicographically.
    """
    per_cohort = _precompute_call_count_vectors(df, labels, k, cohort_col)
    cohorts = sorted(per_cohort.keys())
    eps = _BOOTSTRAP_SMOOTHING_EPS

    # Point estimate proportions from the full data (with the same
    # smoothing as the bootstrap, so the point estimate is comparable to
    # the bootstrap distribution).
    point_props = np.zeros((len(cohorts), k), dtype=np.float64)
    for ci, cohort in enumerate(cohorts):
        _, count_vecs = per_cohort[cohort]
        totals = count_vecs.sum(axis=0)
        point_props[ci] = _proportions_from_counts(totals, smooth=eps)

    # Pairwise upper-triangle indices.
    n = len(cohorts)
    pair_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]

    point_jsds = np.array(
        [js_divergence_bits(point_props[i], point_props[j]) for (i, j) in pair_ij],
        dtype=np.float64,
    )

    # Bootstrap.
    rng = np.random.default_rng(seed)
    reps_arr = np.empty((n_reps, len(pair_ij)), dtype=np.float64)
    count_vecs_list = [per_cohort[c][1] for c in cohorts]

    for r in range(n_reps):
        boot_props = np.empty((n, k), dtype=np.float64)
        for ci_, count_vecs in enumerate(count_vecs_list):
            nc = count_vecs.shape[0]
            if nc == 0:
                boot_props[ci_] = 0.0
                continue
            idx = rng.integers(0, nc, size=nc)
            totals = count_vecs[idx].sum(axis=0)
            boot_props[ci_] = _proportions_from_counts(totals, smooth=eps)

        for pi, (i, j) in enumerate(pair_ij):
            reps_arr[r, pi] = js_divergence_bits(boot_props[i], boot_props[j])

    ci_lo = np.percentile(reps_arr, 2.5, axis=0)
    ci_hi = np.percentile(reps_arr, 97.5, axis=0)

    # Ensure the point estimate is bracketed by the CI. Bootstrap percentile
    # CIs on a non-negative statistic at small n_reps can sit entirely above
    # the point; widen the relevant tail to the point to preserve the
    # contract ci_lo <= point <= ci_hi.
    ci_lo = np.minimum(ci_lo, point_jsds)
    ci_hi = np.maximum(ci_hi, point_jsds)

    rows = []
    for pi, (i, j) in enumerate(pair_ij):
        a, b = cohorts[i], cohorts[j]
        if a > b:
            a, b = b, a
        rows.append({
            "cohort_a": a,
            "cohort_b": b,
            "jsd_point": float(point_jsds[pi]),
            "jsd_ci_lo": float(ci_lo[pi]),
            "jsd_ci_hi": float(ci_hi[pi]),
            "n_reps": int(n_reps),
            "seed": int(seed),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# K-sensitivity
# ---------------------------------------------------------------------------

def k_sensitivity(
    df: pd.DataFrame,
    Z: np.ndarray,
    ks: Iterable[int],
    cohort_col: str,
    seed: int,
) -> pd.DataFrame:
    """Pairwise JSD across multiple K values (no bootstrap CIs).

    For each K in ``ks``: fit fresh K-means, compute cluster proportions
    and pairwise JSD matrix, emit long-form rows.

    Returns columns: ``k, cohort_a, cohort_b, jsd``.
    """
    rows = []
    for k in ks:
        km = fit_kmeans(Z, k=k, seed=seed)
        labels = km.labels_
        props = cluster_proportions(df, labels, k=k, cohort_col=cohort_col)
        m = pairwise_jsd_matrix(props)
        cohorts = list(m.index)
        for i in range(len(cohorts)):
            for j in range(i + 1, len(cohorts)):
                a, b = cohorts[i], cohorts[j]
                if a > b:
                    a, b = b, a
                rows.append({
                    "k": int(k),
                    "cohort_a": a,
                    "cohort_b": b,
                    "jsd": float(m.iloc[i, j]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_WILD_COHORTS = ("5970", "3452", "9252")
_LAB_SPLITS = ("lab_matched", "lab_swap")


def _classify_pair(a: str, b: str) -> str:
    """Return 'wild-wild', 'wild-lab', 'lab-lab', or 'other'."""
    a_is_wild = a in _WILD_COHORTS
    b_is_wild = b in _WILD_COHORTS
    a_is_lab = a in _LAB_SPLITS
    b_is_lab = b in _LAB_SPLITS
    if a_is_wild and b_is_wild:
        return "wild-wild"
    if a_is_lab and b_is_lab:
        return "lab-lab"
    if (a_is_wild and b_is_lab) or (a_is_lab and b_is_wild):
        return "wild-lab"
    return "other"


def _decision_gate(pairs_df: pd.DataFrame) -> str:
    """Decision-gate string for the Move A summary HTML."""
    classes = pairs_df.apply(
        lambda r: _classify_pair(r["cohort_a"], r["cohort_b"]), axis=1
    )
    wild_lab = pairs_df[classes == "wild-lab"]
    wild_wild = pairs_df[classes == "wild-wild"]

    if wild_lab.empty or wild_wild.empty:
        return "INDETERMINATE — missing wild-vs-lab or wild-vs-wild pairs"

    max_wl_idx = wild_lab["jsd_point"].idxmax()
    max_ww_idx = wild_wild["jsd_point"].idxmax()
    max_wl = wild_lab.loc[max_wl_idx]
    max_ww = wild_wild.loc[max_ww_idx]

    def _fmt(row: pd.Series) -> str:
        return (
            f"{row['cohort_a']} vs {row['cohort_b']}: "
            f"{row['jsd_point']:.4f} "
            f"[{row['jsd_ci_lo']:.4f}, {row['jsd_ci_hi']:.4f}]"
        )

    if max_wl["jsd_ci_lo"] > max_ww["jsd_ci_hi"]:
        return (
            "STRONG STRAIN EFFECT — wild-vs-lab JSD exceeds wild-vs-wild "
            "floor (CIs separated). "
            f"max(wild-vs-lab)={_fmt(max_wl)}; "
            f"max(wild-vs-wild)={_fmt(max_ww)}."
        )
    if max_wl["jsd_ci_hi"] < max_ww["jsd_ci_lo"]:
        return (
            "NO STRAIN EFFECT — wild dyads differ from each other more "
            "than from lab. "
            f"max(wild-vs-lab)={_fmt(max_wl)}; "
            f"max(wild-vs-wild)={_fmt(max_ww)}."
        )
    return (
        "INCONCLUSIVE — CIs overlap; report range. "
        f"max(wild-vs-lab)={_fmt(max_wl)}; "
        f"max(wild-vs-wild)={_fmt(max_ww)}."
    )


def _five970_vs_rest(pairs_df: pd.DataFrame) -> str:
    """Surface whether 5970's mean JSD to others exceeds the among-others mean."""
    cohorts = sorted(set(pairs_df["cohort_a"]).union(set(pairs_df["cohort_b"])))
    if "5970" not in cohorts:
        return "5970 not present in pairs DF — skipping 5970-vs-rest read"

    def _pair_jsd(a: str, b: str) -> float | None:
        mask = (
            ((pairs_df["cohort_a"] == a) & (pairs_df["cohort_b"] == b))
            | ((pairs_df["cohort_a"] == b) & (pairs_df["cohort_b"] == a))
        )
        sub = pairs_df[mask]
        if sub.empty:
            return None
        return float(sub["jsd_point"].iloc[0])

    others = [c for c in cohorts if c != "5970"]
    fivenine_to_others = [
        _pair_jsd("5970", o) for o in others if _pair_jsd("5970", o) is not None
    ]
    among_others = []
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            v = _pair_jsd(others[i], others[j])
            if v is not None:
                among_others.append(v)

    if not fivenine_to_others or not among_others:
        return "Not enough pairs to compute 5970-vs-rest summary"

    mean_5970 = float(np.mean(fivenine_to_others))
    mean_rest = float(np.mean(among_others))
    yn = "yes" if mean_5970 > mean_rest else "no"
    return (
        f"5970 vs the rest: mean JSD(5970, others) = {mean_5970:.4f}; "
        f"mean JSD(others, others) = {mean_rest:.4f}. "
        f"Is 5970 the most divergent? {yn}."
    )


def _heatmap_png(matrix: pd.DataFrame, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    # Annotate cells.
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            v = matrix.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=8)
    ax.set_title("Pairwise JSD (bits) — K=20")
    fig.colorbar(im, ax=ax, label="JSD (bits)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _k_sensitivity_png(
    ks_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    png_path: Path,
) -> None:
    """Line plot of K-sensitivity for the top-5 pairs by K=20 JSD."""
    top_pairs = (
        pairs_df.sort_values("jsd_point", ascending=False)
        .head(5)
        .apply(lambda r: (r["cohort_a"], r["cohort_b"]), axis=1)
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for (a, b) in top_pairs:
        mask = (
            ((ks_df["cohort_a"] == a) & (ks_df["cohort_b"] == b))
            | ((ks_df["cohort_a"] == b) & (ks_df["cohort_b"] == a))
        )
        sub = ks_df[mask].sort_values("k")
        ax.plot(sub["k"], sub["jsd"], marker="o", label=f"{a} vs {b}")
    ax.set_xlabel("K (K-means clusters)")
    ax.set_ylabel("Pairwise JSD (bits)")
    ax.set_title("K-sensitivity — top 5 pairs by K=20 JSD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _write_summary_html(
    out_dir: Path,
    params: Dict[str, Any],
    proportions: pd.DataFrame,
    matrix: pd.DataFrame,
    pairs_df: pd.DataFrame,
    ks_df: pd.DataFrame,
) -> Path:
    """Self-contained HTML report."""
    # Heatmap PNG.
    heatmap_png = out_dir / "_jsd_heatmap.png"
    _heatmap_png(matrix, heatmap_png)
    heatmap_b64 = base64.b64encode(heatmap_png.read_bytes()).decode("ascii")
    heatmap_tag = (
        f'<img alt="JSD heatmap" '
        f'src="data:image/png;base64,{heatmap_b64}" />'
    )

    # K-sensitivity PNG.
    ks_png = out_dir / "_k_sensitivity.png"
    _k_sensitivity_png(ks_df, pairs_df, ks_png)
    ks_b64 = base64.b64encode(ks_png.read_bytes()).decode("ascii")
    ks_tag = (
        f'<img alt="K-sensitivity" '
        f'src="data:image/png;base64,{ks_b64}" />'
    )

    # Cleanup temp PNGs (they're embedded as base64 — disk copies optional).
    # Keep them on disk for debugging.

    # Proportions table.
    prop_table_html = proportions.round(3).to_html(
        float_format=lambda v: f"{v:.3f}"
    )

    # Matrix table.
    matrix_table_html = matrix.round(4).to_html(
        float_format=lambda v: f"{v:.4f}"
    )

    # Pair-with-CI sorted descending table.
    pairs_sorted = pairs_df.sort_values("jsd_point", ascending=False).copy()
    pairs_sorted["jsd_with_ci"] = pairs_sorted.apply(
        lambda r: (
            f"{r['jsd_point']:.4f} "
            f"[{r['jsd_ci_lo']:.4f}, {r['jsd_ci_hi']:.4f}]"
        ),
        axis=1,
    )
    pair_cols = ["cohort_a", "cohort_b", "jsd_with_ci"]
    pair_table_html = pairs_sorted[pair_cols].to_html(index=False)

    # Decision gate.
    decision = _decision_gate(pairs_df)
    five_vs_rest = _five970_vs_rest(pairs_df)

    # Params dl.
    dl_items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in params.items())
    dl_html = "<dl>" + dl_items + "</dl>"

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Latent repertoire JSD — Move A</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
    h2 {{ margin-top: 1.6em; color: #333; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr;
          column-gap: 1em; row-gap: 0.25em; }}
    dt {{ font-weight: 600; color: #555; }}
    table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.95em; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
    th {{ background: #f0f0f0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    .decision {{ padding: 0.8em 1em; background: #f7f7f0;
                 border-left: 4px solid #888; margin: 1em 0; }}
    .secondary {{ padding: 0.6em 1em; background: #f0f5f7;
                  border-left: 4px solid #889; margin: 1em 0; font-size: 0.95em; }}
    footer {{ margin-top: 2em; color: #888; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Latent repertoire JSD — Move A</h1>

  <h2>Run parameters</h2>
  {dl_html}

  <h2>Cluster proportions (cohorts x K clusters)</h2>
  {prop_table_html}

  <h2>Pairwise JSD matrix (K={params.get('k')})</h2>
  {heatmap_tag}
  {matrix_table_html}

  <h2>Pairwise JSD with bootstrap CIs (sorted desc)</h2>
  {pair_table_html}

  <h2>K-sensitivity (top 5 pairs)</h2>
  {ks_tag}

  <h2>Decision-gate read</h2>
  <p class="decision">{decision}</p>
  <p class="secondary">{five_vs_rest}</p>

  <footer>Generated on {timestamp}</footer>
</body>
</html>
"""
    html_path = out_dir / "summary.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move A — repertoire JSD in 32-D VAE latent space "
                    "(K-means K=20 alphabet, pairwise JSD with bootstrap "
                    "call-level CIs, K-sensitivity sweep)."
    )
    p.add_argument("--latents-path", type=str,
                   default="results/contour_vae_combined/latents.parquet")
    p.add_argument("--out-dir", type=str, default="results/latent_repertoire")
    p.add_argument("--model-out", type=str, default="models/latent_kmeans")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--ks", type=str, default="10,20,30,50",
                   help="Comma-separated K values for sensitivity sweep")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    t_start = time.time()
    args = _parse_args()
    ks_list = [int(x) for x in args.ks.split(",")]

    print(f"[PARAM] latents_path = {args.latents_path}")
    print(f"[PARAM] out_dir      = {args.out_dir}")
    print(f"[PARAM] model_out    = {args.model_out}")
    print(f"[PARAM] k            = {args.k}")
    print(f"[PARAM] ks           = {ks_list}")
    print(f"[PARAM] n_boot       = {args.n_boot}")
    print(f"[PARAM] seed         = {args.seed}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_out)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading latents...")
    df = load_latents(args.latents_path)
    print(f"[INFO] Loaded df shape = {df.shape}")

    print("[INFO] Splitting lab into matched/swap...")
    df = split_lab_cohorts(df)
    counts = df["cohort_split"].value_counts().to_dict()
    print("[INFO] cohort_split counts:")
    for coh in sorted(counts):
        print(f"[INFO]   {coh:>14s}: {counts[coh]:>7d}")

    Z = _latent_matrix(df)
    print(f"[INFO] Latent matrix shape = {Z.shape}")

    # ------------------------------------------------------------------
    # K-means at the headline K, persist the model + labels.
    # ------------------------------------------------------------------
    print(f"[INFO] Fitting K-means K={args.k} (n_init=10, seed={args.seed})...")
    t0 = time.time()
    km = fit_kmeans(Z, k=args.k, seed=args.seed)
    labels = km.labels_.astype(np.int64)
    print(f"[INFO] K-means done in {time.time() - t0:.1f}s; "
          f"inertia = {km.inertia_:.2f}")

    model_path = model_dir / f"k{args.k}.joblib"
    labels_path = model_dir / f"k{args.k}_labels.npy"
    joblib.dump(km, model_path)
    np.save(labels_path, labels)
    print(f"[INFO] Wrote model  -> {model_path}")
    print(f"[INFO] Wrote labels -> {labels_path}")

    # ------------------------------------------------------------------
    # Cluster proportions.
    # ------------------------------------------------------------------
    print("[INFO] Computing cluster proportions...")
    props = cluster_proportions(df, labels, k=args.k, cohort_col="cohort_split")
    print(f"[INFO] Proportions shape = {props.shape}")
    props_csv = out_dir / "cluster_proportions.csv"
    props.to_csv(props_csv)
    print(f"[INFO] Wrote proportions -> {props_csv}")

    # ------------------------------------------------------------------
    # Pairwise JSD matrix.
    # ------------------------------------------------------------------
    print("[INFO] Computing pairwise JSD matrix...")
    matrix = pairwise_jsd_matrix(props)
    matrix_csv = out_dir / "jsd_matrix.csv"
    matrix.to_csv(matrix_csv)
    print(f"[INFO] Wrote JSD matrix -> {matrix_csv}")

    # ------------------------------------------------------------------
    # Bootstrap pairs.
    # ------------------------------------------------------------------
    print(f"[INFO] Bootstrapping {args.n_boot} reps "
          f"(resampling calls within cohort)...")
    t0 = time.time()
    pairs_df = _bootstrap_with_progress(
        df, labels, k=args.k, cohort_col="cohort_split",
        n_reps=args.n_boot, seed=args.seed,
    )
    print(f"[INFO] Bootstrap done in {time.time() - t0:.1f}s")
    pairs_csv = out_dir / "jsd_pairs_with_ci.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"[INFO] Wrote pairs+CIs -> {pairs_csv}")

    # ------------------------------------------------------------------
    # K-sensitivity.
    # ------------------------------------------------------------------
    print(f"[INFO] K-sensitivity sweep over {ks_list}...")
    t0 = time.time()
    ks_df = k_sensitivity(df, Z, ks=ks_list, cohort_col="cohort_split",
                          seed=args.seed)
    print(f"[INFO] K-sensitivity done in {time.time() - t0:.1f}s; "
          f"{len(ks_df)} rows")
    ks_csv = out_dir / "k_sensitivity.csv"
    ks_df.to_csv(ks_csv, index=False)
    print(f"[INFO] Wrote k_sensitivity -> {ks_csv}")

    # ------------------------------------------------------------------
    # HTML summary.
    # ------------------------------------------------------------------
    params = {
        "latents_path": args.latents_path,
        "out_dir": args.out_dir,
        "model_out": args.model_out,
        "k": args.k,
        "ks": ",".join(str(x) for x in ks_list),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_total_patches": int(len(df)),
        "kmeans_inertia": f"{km.inertia_:.2f}",
        **{f"n_in_cohort[{c}]": int(counts[c]) for c in sorted(counts)},
    }
    html_path = _write_summary_html(out_dir, params, props, matrix, pairs_df, ks_df)
    print(f"[INFO] Wrote HTML -> {html_path}")

    print("[INFO] Decision-gate read:")
    print(f"[INFO]   {_decision_gate(pairs_df)}")
    print(f"[INFO]   {_five970_vs_rest(pairs_df)}")
    print(f"[INFO] Total wall time: {time.time() - t_start:.1f}s")


def _bootstrap_with_progress(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    cohort_col: str,
    n_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Thin wrapper that prints progress every 100 reps.

    Internally re-implements `bootstrap_jsd_pairs` so we can hook progress
    without changing the pure-function API used by tests.
    """
    per_cohort = _precompute_call_count_vectors(df, labels, k, cohort_col)
    cohorts = sorted(per_cohort.keys())
    n = len(cohorts)
    pair_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
    eps = _BOOTSTRAP_SMOOTHING_EPS

    # Point estimate (same smoothing as bootstrap).
    point_props = np.zeros((n, k), dtype=np.float64)
    for ci, cohort in enumerate(cohorts):
        totals = per_cohort[cohort][1].sum(axis=0)
        point_props[ci] = _proportions_from_counts(totals, smooth=eps)
    point_jsds = np.array(
        [js_divergence_bits(point_props[i], point_props[j]) for (i, j) in pair_ij]
    )

    rng = np.random.default_rng(seed)
    reps_arr = np.empty((n_reps, len(pair_ij)), dtype=np.float64)
    count_vecs_list = [per_cohort[c][1] for c in cohorts]

    for r in range(n_reps):
        boot_props = np.empty((n, k), dtype=np.float64)
        for ci_, cv in enumerate(count_vecs_list):
            nc = cv.shape[0]
            if nc == 0:
                boot_props[ci_] = 0.0
                continue
            idx = rng.integers(0, nc, size=nc)
            totals = cv[idx].sum(axis=0)
            boot_props[ci_] = _proportions_from_counts(totals, smooth=eps)
        for pi, (i, j) in enumerate(pair_ij):
            reps_arr[r, pi] = js_divergence_bits(boot_props[i], boot_props[j])
        if (r + 1) % 100 == 0:
            print(f"[INFO]   bootstrap rep {r + 1}/{n_reps}")

    ci_lo = np.percentile(reps_arr, 2.5, axis=0)
    ci_hi = np.percentile(reps_arr, 97.5, axis=0)
    ci_lo = np.minimum(ci_lo, point_jsds)
    ci_hi = np.maximum(ci_hi, point_jsds)

    rows = []
    for pi, (i, j) in enumerate(pair_ij):
        a, b = cohorts[i], cohorts[j]
        if a > b:
            a, b = b, a
        rows.append({
            "cohort_a": a,
            "cohort_b": b,
            "jsd_point": float(point_jsds[pi]),
            "jsd_ci_lo": float(ci_lo[pi]),
            "jsd_ci_hi": float(ci_hi[pi]),
            "n_reps": int(n_reps),
            "seed": int(seed),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
