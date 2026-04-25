#!/usr/bin/env python3
"""Stream 5 — Bout threshold sensitivity + file-aware logic prototype.

Addresses Q1 in docs/questions-for-mickey.md quantitatively: how much do
bout counts, MI lag-1, and entropy reductions change as the ICI bout-gap
threshold varies, and how does the choice of file-aware vs non-file-aware
logic shift that curve?

Outputs (per dataset suffix ``<ds>`` — ``5970`` or ``3452``):
    results/bout_threshold_sensitivity/sweep_no_file_aware_<ds>.csv
    results/bout_threshold_sensitivity/sweep_file_aware_<ds>.csv
    results/bout_threshold_sensitivity/sweep_no_file_aware_<ds>.png
    results/bout_threshold_sensitivity/sweep_file_aware_<ds>.png
    results/bout_threshold_sensitivity/comparison_<ds>.png
    results/bout_threshold_sensitivity/gap_distributions_<ds>.png

Canonical files (data/corpus_facts/*.json, src/usv_spectrogram/corpus.py,
usv_language/analysis/sequence_analysis.py) are NOT modified — this script
is sensitivity analysis only.

Usage:
    python scripts/bout_threshold_sensitivity.py \
        --csv results/traditional_taxonomy/classified_traditional.csv \
        --dataset 5970

    python scripts/bout_threshold_sensitivity.py \
        --csv results/traditional_taxonomy_3452/classified_traditional.csv \
        --dataset 3452
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.mixture import GaussianMixture

# Project path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from usv_language.analysis.sequence_analysis import (  # noqa: E402
    mutual_information_from_sequences,
)
from scripts.analyze_sequential_structure import (  # noqa: E402
    SYLLABLE_TYPES,
    TYPE_TO_CODE,
    K,
    load_and_enrich,
)

# ── Constants ──────────────────────────────────────────────────────────────

# Sweep grid — sensitivity-analysis evaluation points, NOT a canonical
# declaration. 0.60 is included deliberately so the non-file-aware row at
# that threshold acts as a drift check against data/corpus_facts/5970.json.
THRESHOLDS_S = [0.10, 0.143, 0.20, 0.25, 0.40, 0.60, 0.80, 1.00, 2.00]

# Reference-annotation value from prior within-file mixture fit (cited in
# docs/questions-for-mickey.md Q1 background). The script recomputes its own
# mixture crossover — this constant only draws a vertical reference line on
# sweep plots. Not a corpus_facts entry.
PRIOR_MIXTURE_CROSSOVER_S = 0.143

BOOTSTRAP_N = 1000
RNG_SEED = 20260424

# Layer 2 lookup — consumed from corpus_facts at startup, never redeclared.
CORPUS_FACTS_PATH = PROJECT_ROOT / "data" / "corpus_facts" / "5970.json"


def load_canonical_threshold_s() -> float:
    """Read bout_detection_a2.threshold_s from corpus_facts/5970.json."""
    with CORPUS_FACTS_PATH.open() as f:
        facts = json.load(f)
    return float(facts["bout_detection_a2"]["threshold_s"])

# ── Bout segmentation ─────────────────────────────────────────────────────


def compute_gaps_and_files(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gap_s array length N-1, file_changed array length N-1, codes length N).

    Gap convention = start(next) - end(current), matching A2's ``detect_bouts``.
    A negative gap (10 overlapping cases in 5970) is retained as-is — it will
    never exceed any positive threshold, so it is always treated as
    within-bout by the gap rule alone. The file-aware rule may still mark a
    boundary there if the overlap spans a file change.
    """
    start_times = df["absolute_time"].values.astype("datetime64[ns]").astype(np.float64) / 1e9
    durations = (df["end_time_s"] - df["begin_time_s"]).values
    end_times = start_times + durations
    gaps = start_times[1:] - end_times[:-1]
    files = df["file"].values
    file_changed = files[1:] != files[:-1]
    codes = df["type_code"].values.astype(np.int64)
    return gaps, file_changed, codes


def segment_bouts(
    codes: np.ndarray,
    gaps: np.ndarray,
    file_changed: np.ndarray,
    threshold_s: float,
    file_aware: bool,
) -> list[np.ndarray]:
    """Return list of per-bout code arrays.

    ``threshold_s = inf`` with ``file_aware=True`` means "file = bout"
    (no within-file gap splits). Non-file-aware logic never uses file info.
    Strict ``>`` matches A2's convention (gap equal to threshold stays
    within bout).
    """
    n = len(codes)
    if n == 0:
        return []
    if n == 1:
        return [codes.copy()]

    gap_break = gaps > threshold_s
    if file_aware:
        boundaries_bool = gap_break | file_changed
    else:
        boundaries_bool = gap_break

    boundary_idx = np.where(boundaries_bool)[0] + 1
    cuts = np.concatenate([[0], boundary_idx, [n]])
    return [codes[cuts[k]:cuts[k + 1]] for k in range(len(cuts) - 1)]


# ── MI + entropies on a bout list ──────────────────────────────────────────


def marginal_entropy_from_bouts(bouts: list[np.ndarray], k: int) -> float:
    """Marginal entropy of codes pooled across all bouts (singletons included)."""
    counts = np.zeros(k, dtype=np.float64)
    for seq in bouts:
        for c in seq:
            counts[c] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def conditional_entropy_from_bouts(bouts: list[np.ndarray], k: int) -> float:
    """H(C_{t+1} | C_t) from within-bout pairs only."""
    counts = np.zeros((k, k), dtype=np.float64)
    for seq in bouts:
        for i in range(len(seq) - 1):
            counts[seq[i], seq[i + 1]] += 1
    row_sums = counts.sum(axis=1)
    total = row_sums.sum()
    if total == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = np.where(row_sums[:, None] > 0, counts / row_sums[:, None], 0.0)
    marginal = row_sums / total
    h = 0.0
    for i in range(k):
        if marginal[i] == 0:
            continue
        row_ent = 0.0
        for j in range(k):
            if trans[i, j] > 0:
                row_ent -= trans[i, j] * np.log2(trans[i, j])
        h += marginal[i] * row_ent
    return float(h)


def bootstrap_mi_ci(
    bouts: list[np.ndarray],
    k: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (mi_point, ci_low_2.5pct, ci_high_97.5pct).

    Resamples BOUTS (not pairs) with replacement — bouts are the independent
    observation unit; resampling pairs would treat within-bout transitions
    as i.i.d. and underestimate MI variance. Singleton bouts contribute no
    pairs, which is fine.
    """
    mi_point, _ = mutual_information_from_sequences(bouts, k, lag=1)
    if len(bouts) == 0:
        return mi_point, 0.0, 0.0
    indices = np.arange(len(bouts))
    bs_values = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sampled = [bouts[i] for i in sample_idx]
        bs_mi, _ = mutual_information_from_sequences(sampled, k, lag=1)
        bs_values[b] = bs_mi
    ci_low, ci_high = np.quantile(bs_values, [0.025, 0.975])
    return float(mi_point), float(ci_low), float(ci_high)


# ── One sweep row ──────────────────────────────────────────────────────────


def sweep_one(
    codes: np.ndarray,
    gaps: np.ndarray,
    file_changed: np.ndarray,
    start_times_s: np.ndarray,
    end_times_s: np.ndarray,
    threshold_s: float,
    file_aware: bool,
    rng: np.random.Generator,
) -> dict:
    bouts = segment_bouts(codes, gaps, file_changed, threshold_s, file_aware)
    n_bouts = len(bouts)
    n_multi = sum(1 for b in bouts if len(b) >= 2)
    n_within_pairs = sum(max(0, len(b) - 1) for b in bouts)
    n_total_pairs = max(0, len(codes) - 1)
    n_cross_pairs = n_total_pairs - n_within_pairs

    marg_h = marginal_entropy_from_bouts(bouts, K)
    cond_h = conditional_entropy_from_bouts(bouts, K)
    mi, mi_lo, mi_hi = bootstrap_mi_ci(bouts, K, BOOTSTRAP_N, rng)

    # Bout duration = time span from first-call start to last-call end
    # within each bout, computed from the flat index ranges we just cut.
    # Rebuild boundaries to get per-bout index slices.
    n = len(codes)
    if file_aware:
        boundaries_bool = (gaps > threshold_s) | file_changed
    else:
        boundaries_bool = gaps > threshold_s
    boundary_idx = np.where(boundaries_bool)[0] + 1
    cuts = np.concatenate([[0], boundary_idx, [n]])

    durations = []
    sizes = []
    for k_ in range(len(cuts) - 1):
        lo, hi = cuts[k_], cuts[k_ + 1]
        sizes.append(hi - lo)
        if hi > lo:
            durations.append(end_times_s[hi - 1] - start_times_s[lo])
    mean_bout_duration = float(np.mean(durations)) if durations else 0.0
    mean_calls_per_bout = float(np.mean(sizes)) if sizes else 0.0

    return {
        "threshold_s": threshold_s,
        "file_aware": file_aware,
        "n_bouts": n_bouts,
        "n_multi_call_bouts": n_multi,
        "n_within_bout_pairs": n_within_pairs,
        "n_cross_bout_pairs": n_cross_pairs,
        "mi_lag1_bits": round(mi, 6),
        "mi_ci_low_bits": round(mi_lo, 6),
        "mi_ci_high_bits": round(mi_hi, 6),
        "marginal_entropy_bits": round(marg_h, 6),
        "conditional_entropy_bits": round(cond_h, 6),
        "mean_bout_duration_s": round(mean_bout_duration, 4),
        "mean_calls_per_bout": round(mean_calls_per_bout, 4),
    }


# ── Plotting ───────────────────────────────────────────────────────────────


def _format_threshold_label(t: float) -> str:
    return "∞" if not np.isfinite(t) else f"{t:g}"


def plot_sweep(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Two-panel plot: (top) MI vs threshold with 95% CI band, (bottom) n_bouts."""
    finite_mask = np.isfinite(df["threshold_s"].values)
    finite = df[finite_mask].sort_values("threshold_s")
    inf_rows = df[~finite_mask]

    fig, (ax_mi, ax_n) = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    x = finite["threshold_s"].values
    y = finite["mi_lag1_bits"].values
    lo = finite["mi_ci_low_bits"].values
    hi = finite["mi_ci_high_bits"].values

    ax_mi.fill_between(x, lo, hi, alpha=0.25, color="#228833", label="95% bootstrap CI")
    ax_mi.plot(x, y, "o-", color="#228833", linewidth=2, markersize=7, label="MI lag 1")
    canonical = load_canonical_threshold_s()
    ax_mi.axvline(canonical, color="#AA3377", linestyle="--",
                  alpha=0.7, label=f"canonical {canonical}s")
    ax_mi.axvline(PRIOR_MIXTURE_CROSSOVER_S, color="#4477AA", linestyle=":",
                  alpha=0.7, label=f"prior mixture crossover {PRIOR_MIXTURE_CROSSOVER_S}s")
    ax_mi.set_xscale("log")
    ax_mi.set_xlabel("Bout threshold (s, log scale)")
    ax_mi.set_ylabel("MI lag 1 (bits)")
    ax_mi.set_title(title)
    ax_mi.grid(True, alpha=0.3)
    ax_mi.legend(fontsize=8, loc="best")

    # N_bouts panel
    ax_n.plot(x, finite["n_bouts"].values, "s-", color="#EE6677",
              linewidth=2, markersize=7, label="n_bouts")
    ax_n.plot(x, finite["n_within_bout_pairs"].values, "^-", color="#4477AA",
              linewidth=2, markersize=6, label="n_within_bout_pairs")
    ax_n.axvline(canonical, color="#AA3377", linestyle="--", alpha=0.7)
    ax_n.axvline(PRIOR_MIXTURE_CROSSOVER_S, color="#4477AA", linestyle=":", alpha=0.7)
    ax_n.set_xscale("log")
    ax_n.set_xlabel("Bout threshold (s, log scale)")
    ax_n.set_ylabel("count")
    ax_n.grid(True, alpha=0.3)
    ax_n.legend(fontsize=9, loc="best")

    if len(inf_rows) > 0:
        inf_row = inf_rows.iloc[0]
        ax_mi.text(
            0.02, 0.98,
            f"threshold=∞ (file=bout):\n  MI={inf_row['mi_lag1_bits']:.4f} "
            f"[{inf_row['mi_ci_low_bits']:.4f}, {inf_row['mi_ci_high_bits']:.4f}]\n"
            f"  n_bouts={inf_row['n_bouts']}  n_pairs={inf_row['n_within_bout_pairs']}",
            transform=ax_mi.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison(df_no: pd.DataFrame, df_yes: pd.DataFrame, dataset: str,
                    out_path: Path) -> None:
    """Overlay MI vs threshold under the two logics on one axis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, color, label, marker in [
        (df_no, "#BB5566", "non-file-aware", "o"),
        (df_yes[np.isfinite(df_yes["threshold_s"].values)], "#228833", "file-aware", "s"),
    ]:
        finite = df[np.isfinite(df["threshold_s"].values)].sort_values("threshold_s")
        x = finite["threshold_s"].values
        y = finite["mi_lag1_bits"].values
        lo = finite["mi_ci_low_bits"].values
        hi = finite["mi_ci_high_bits"].values
        ax.fill_between(x, lo, hi, alpha=0.2, color=color)
        ax.plot(x, y, marker=marker, linestyle="-", color=color, linewidth=2,
                markersize=7, label=label)

    canonical = load_canonical_threshold_s()
    ax.axvline(canonical, color="#AA3377", linestyle="--", alpha=0.7,
               label=f"canonical {canonical}s")
    ax.axvline(PRIOR_MIXTURE_CROSSOVER_S, color="#4477AA", linestyle=":", alpha=0.7,
               label=f"prior mixture crossover {PRIOR_MIXTURE_CROSSOVER_S}s")

    inf_rows = df_yes[~np.isfinite(df_yes["threshold_s"].values)]
    if len(inf_rows) > 0:
        inf_row = inf_rows.iloc[0]
        ax.text(
            0.98, 0.98,
            f"file-aware threshold=∞:\n  MI={inf_row['mi_lag1_bits']:.4f} "
            f"[{inf_row['mi_ci_low_bits']:.4f}, {inf_row['mi_ci_high_bits']:.4f}]",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bout threshold (s, log scale)")
    ax.set_ylabel("MI lag 1 (bits)")
    ax.set_title(f"{dataset}: MI vs threshold — non-file-aware vs file-aware")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gap_distributions(
    gaps: np.ndarray,
    file_changed: np.ndarray,
    dataset: str,
    out_path: Path,
) -> dict:
    """Three-panel: within-file hist+mixture, cross-file hist, KS annotation.

    Returns a dict with mixture params and KS result (for the memo).
    """
    within = gaps[~file_changed]
    cross = gaps[file_changed]

    # Positive-only for log-space fitting and plotting
    within_pos = within[within > 0]
    cross_pos = cross[cross > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: within-file w/ 2-component GMM on log-ICI ───────
    ax = axes[0]
    if len(within_pos) >= 2:
        log_w = np.log10(within_pos).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=RNG_SEED, max_iter=500)
        gmm.fit(log_w)
        means_log = gmm.means_.flatten()
        stds_log = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_
        # Sort by mean (log) ascending
        order = np.argsort(means_log)
        means_log = means_log[order]
        stds_log = stds_log[order]
        weights = weights[order]
        means_s = 10 ** means_log

        bins = np.logspace(np.log10(max(1e-4, within_pos.min())),
                           np.log10(within_pos.max()), 60)
        ax.hist(within_pos, bins=bins, density=True, alpha=0.5,
                color="#4477AA", edgecolor="black", linewidth=0.3,
                label=f"within-file ({len(within_pos)})")

        xs_log = np.linspace(log_w.min(), log_w.max(), 400).reshape(-1, 1)
        log_pdf = gmm.score_samples(xs_log)
        # Convert density from log-space to linear space: p_lin(x) = p_log(log10 x)/(x ln10)
        xs = 10 ** xs_log.flatten()
        pdf_linear = np.exp(log_pdf) / (xs * np.log(10))
        ax.plot(xs, pdf_linear, color="#AA3377", linewidth=2,
                label=f"2-comp GMM (μ₁={means_s[0]*1000:.1f}ms w={weights[0]:.2f}; "
                      f"μ₂={means_s[1]*1000:.1f}ms w={weights[1]:.2f})")

        # Crossover: where the two component densities are equal in log space
        # Solve numerically over a dense grid
        from scipy.stats import norm as scinorm
        def comp_pdf(xlog, i):
            return weights[i] * scinorm.pdf(xlog, means_log[i], stds_log[i])
        grid = np.linspace(means_log[0], means_log[1], 500)
        diff = comp_pdf(grid, 0) - comp_pdf(grid, 1)
        sign_change = np.where(np.diff(np.sign(diff)))[0]
        crossover_s = 10 ** grid[sign_change[0]] if len(sign_change) > 0 else np.nan
        if np.isfinite(crossover_s):
            ax.axvline(crossover_s, color="#228833", linestyle="--",
                       label=f"crossover {crossover_s*1000:.0f}ms")
    else:
        crossover_s = np.nan
        means_s = np.array([np.nan, np.nan])
        weights = np.array([np.nan, np.nan])

    canonical = load_canonical_threshold_s()
    ax.axvline(canonical, color="#AA3377", linestyle="--", alpha=0.6,
               label=f"canonical {canonical}s")
    ax.set_xscale("log")
    ax.set_xlabel("within-file gap (s, log)")
    ax.set_ylabel("density")
    ax.set_title(f"{dataset}: within-file gaps + 2-comp GMM")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: cross-file gaps ────────────────────────
    ax = axes[1]
    if len(cross_pos) > 0:
        bins = np.logspace(np.log10(max(1e-4, cross_pos.min())),
                           np.log10(cross_pos.max()), 60)
        ax.hist(cross_pos, bins=bins, density=True, alpha=0.6,
                color="#EE6677", edgecolor="black", linewidth=0.3,
                label=f"cross-file ({len(cross_pos)})")
        n_under_1s = int((cross_pos < 1.0).sum())
        ax.text(0.02, 0.98, f"cross-file gaps <1s: {n_under_1s}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.axvline(canonical, color="#AA3377", linestyle="--", alpha=0.6,
               label=f"canonical {canonical}s")
    ax.axvline(2.0, color="#228833", linestyle=":", alpha=0.6,
               label="2s recorder timeout")
    ax.set_xscale("log")
    ax.set_xlabel("cross-file gap (s, log)")
    ax.set_ylabel("density")
    ax.set_title(f"{dataset}: cross-file gaps")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: KS test + ECDFs ────────────────────────
    ax = axes[2]
    ks_stat, ks_p = (np.nan, np.nan)
    if len(within_pos) > 0 and len(cross_pos) > 0:
        ks_stat, ks_p = scipy_stats.ks_2samp(np.log10(within_pos), np.log10(cross_pos))

        for data, color, label in [
            (within_pos, "#4477AA", f"within-file ({len(within_pos)})"),
            (cross_pos, "#EE6677", f"cross-file ({len(cross_pos)})"),
        ]:
            sorted_d = np.sort(data)
            ecdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.plot(sorted_d, ecdf, color=color, linewidth=2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("gap (s, log)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"KS test on log(gap)\nD={ks_stat:.3f}, p={ks_p:.2e}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {
        "mixture_mu1_s": float(means_s[0]) if means_s.size > 0 else np.nan,
        "mixture_mu2_s": float(means_s[1]) if means_s.size > 1 else np.nan,
        "mixture_w1": float(weights[0]) if weights.size > 0 else np.nan,
        "mixture_w2": float(weights[1]) if weights.size > 1 else np.nan,
        "mixture_crossover_s": float(crossover_s),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "n_within_file_gaps": int(len(within)),
        "n_cross_file_gaps": int(len(cross)),
        "n_cross_file_gaps_under_1s": int((cross_pos < 1.0).sum()) if len(cross_pos) else 0,
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Bout threshold sensitivity sweep")
    parser.add_argument("--csv", required=True, help="Classified CSV path")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g. 5970, 3452)")
    parser.add_argument(
        "--output-dir",
        default="results/bout_threshold_sensitivity",
        help="Output directory (files get _<dataset> suffix)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = args.dataset
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print(f"Stream 5 — Bout threshold sensitivity sweep")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  dataset           : {ds}")
    print(f"  csv               : {args.csv}")
    print(f"  output_dir        : {out_dir}")
    print(f"  thresholds_s      : {THRESHOLDS_S}")
    print(f"  n_bootstrap       : {args.n_bootstrap}")
    canonical = load_canonical_threshold_s()
    print(f"  canonical (from corpus_facts/5970.json) : {canonical}s")
    print(f"  prior mixture crossover (Q1 background) : {PRIOR_MIXTURE_CROSSOVER_S}s")
    print(f"  seed              : {args.seed}")
    print(f"  codebook          : {SYLLABLE_TYPES} (K={K})")
    print()

    df = load_and_enrich(args.csv)
    n_calls = len(df)

    # Precompute gap + file_changed arrays
    gaps, file_changed, codes = compute_gaps_and_files(df)
    start_times_s = df["absolute_time"].values.astype("datetime64[ns]").astype(np.float64) / 1e9
    durations = (df["end_time_s"] - df["begin_time_s"]).values
    end_times_s = start_times_s + durations

    print(f"Loaded {n_calls} calls ({df['file'].nunique()} files)")
    print(f"  gaps: {len(gaps)} ({int(file_changed.sum())} cross-file, "
          f"{int((~file_changed).sum())} within-file)")
    print(f"  negative gaps (overlaps): {int((gaps < 0).sum())}")
    print()

    # ── Sweep both logics ─────────────────────────────
    print("── Sweep: non-file-aware ──")
    rows_no = []
    for t in THRESHOLDS_S:
        row = sweep_one(codes, gaps, file_changed, start_times_s, end_times_s,
                        t, file_aware=False, rng=rng)
        rows_no.append(row)
        print(f"  t={t:>5.3f}s: n_bouts={row['n_bouts']:>4d}  "
              f"pairs_in={row['n_within_bout_pairs']:>4d}  "
              f"pairs_ex={row['n_cross_bout_pairs']:>3d}  "
              f"MI={row['mi_lag1_bits']:.4f} "
              f"[{row['mi_ci_low_bits']:.4f},{row['mi_ci_high_bits']:.4f}]")
    df_no = pd.DataFrame(rows_no)

    print()
    print("── Sweep: file-aware ──")
    rows_yes = []
    for t in THRESHOLDS_S + [float("inf")]:
        row = sweep_one(codes, gaps, file_changed, start_times_s, end_times_s,
                        t, file_aware=True, rng=rng)
        rows_yes.append(row)
        tlabel = _format_threshold_label(t)
        print(f"  t={tlabel:>5}s: n_bouts={row['n_bouts']:>4d}  "
              f"pairs_in={row['n_within_bout_pairs']:>4d}  "
              f"pairs_ex={row['n_cross_bout_pairs']:>3d}  "
              f"MI={row['mi_lag1_bits']:.4f} "
              f"[{row['mi_ci_low_bits']:.4f},{row['mi_ci_high_bits']:.4f}]")
    df_yes = pd.DataFrame(rows_yes)

    # ── Save CSVs ─────────────────────────────────────
    no_csv = out_dir / f"sweep_no_file_aware_{ds}.csv"
    yes_csv = out_dir / f"sweep_file_aware_{ds}.csv"
    df_no.to_csv(no_csv, index=False)
    df_yes.to_csv(yes_csv, index=False)
    print(f"\n  Saved: {no_csv}")
    print(f"  Saved: {yes_csv}")

    # ── Plots ─────────────────────────────────────────
    print("\n── Plots ──")
    plot_sweep(df_no, f"{ds}: sweep (non-file-aware)",
               out_dir / f"sweep_no_file_aware_{ds}.png")
    plot_sweep(df_yes, f"{ds}: sweep (file-aware)",
               out_dir / f"sweep_file_aware_{ds}.png")
    plot_comparison(df_no, df_yes, ds, out_dir / f"comparison_{ds}.png")
    gap_stats = plot_gap_distributions(gaps, file_changed, ds,
                                        out_dir / f"gap_distributions_{ds}.png")

    # ── Drift check (5970 only) ───────────────────────
    if ds == "5970":
        canonical_row = df_no[df_no["threshold_s"] == canonical].iloc[0]
        canonical_mi = canonical_row["mi_lag1_bits"]
        with CORPUS_FACTS_PATH.open() as f:
            facts = json.load(f)
        expected_mi = float(
            facts["sequential_structure_mi"]["scattoni_7_bout_aware"]["mi_lag1_bits"]
        )
        delta = abs(canonical_mi - expected_mi)
        print(f"\n── Drift check vs data/corpus_facts/5970.json ──")
        print(f"  expected MI @ {canonical}s (non-file-aware) : {expected_mi}")
        print(f"  observed MI                         : {canonical_mi}")
        print(f"  |Δ|                                 : {delta:.6f} bits "
              f"({'OK' if delta < 1e-3 else 'DRIFT — INVESTIGATE'})")

    # ── Print gap diagnostic summary ──────────────────
    print(f"\n── Gap distribution diagnostic ({ds}) ──")
    for k_, v in gap_stats.items():
        print(f"  {k_}: {v}")

    print(f"\n── Done. All outputs in {out_dir}/ (suffix _{ds}) ──")


if __name__ == "__main__":
    main()
