#!/usr/bin/env python3
"""Phase A2: Sequential structure analysis of USV call data.

Analyzes the 5970 dataset (usv_lmt_034) for sequential patterns:
- Transition matrix P(type_B | type_A)
- Entropy rate convergence across n-gram orders
- Mutual information at lag (sequence "memory")
- Zipf rank-frequency distribution
- Idiom detection (recurring n-grams above chance)
- Within-bout vs full-sequence comparison

Outputs figures and summary CSV to results/sequential_structure/.
"""

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from usv_language.analysis.information_theory import (
    zipf_exponent_mle,
    ngram_idioms,
)


# ── Constants ───────────────────────────────────────────────────────────────

SYLLABLE_TYPES = ["Flat", "Down", "Chevron", "Short", "Complex", "Frequency_Jump", "Up"]
TYPE_TO_CODE = {t: i for i, t in enumerate(SYLLABLE_TYPES)}
CODE_TO_TYPE = {i: t for t, i in TYPE_TO_CODE.items()}
K = len(SYLLABLE_TYPES)

TYPE_COLORS = {
    "Flat": "#4477AA",
    "Down": "#EE6677",
    "Chevron": "#228833",
    "Short": "#CCBB44",
    "Complex": "#AA3377",
    "Frequency_Jump": "#66CCEE",
    "Up": "#BBBBBB",
}

BOUT_THRESHOLD_S = 0.6  # From A1 findings


# ── Data loading (reused from A1) ──────────────────────────────────────────

def parse_filename_timestamp(filename: str) -> datetime:
    """Extract datetime from WAV filename like '2024-09-30_11-18-17_0000001'."""
    match = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_(\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse timestamp from filename: {filename}")
    date_str = match.group(1)
    h, m, s = match.group(2), match.group(3), match.group(4)
    return datetime.strptime(f"{date_str} {h}:{m}:{s}", "%Y-%m-%d %H:%M:%S")


def load_and_enrich(csv_path: str) -> pd.DataFrame:
    """Load classified CSV and add absolute_time + integer code columns."""
    df = pd.read_csv(csv_path)

    n_before = len(df)
    df = df.dropna(subset=["file"]).copy()
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing 'file' values")

    file_timestamps = df["file"].map(parse_filename_timestamp)
    df["file_datetime"] = file_timestamps
    df["absolute_time"] = file_timestamps + pd.to_timedelta(df["begin_time_s"], unit="s")
    df = df.sort_values("absolute_time").reset_index(drop=True)

    # Map syllable types to integer codes
    df["type_code"] = df["syllable_type"].map(TYPE_TO_CODE)
    unknown = df["type_code"].isna().sum()
    if unknown > 0:
        print(f"  WARNING: {unknown} calls with unknown syllable type — dropping")
        df = df.dropna(subset=["type_code"]).reset_index(drop=True)
    df["type_code"] = df["type_code"].astype(int)

    t0, t1 = df["absolute_time"].iloc[0], df["absolute_time"].iloc[-1]
    print(f"Loaded {len(df)} calls, timeline: {t0} → {t1}")
    print(f"Type distribution: {df['syllable_type'].value_counts().to_dict()}")

    return df


def detect_bouts(df: pd.DataFrame, threshold_s: float = BOUT_THRESHOLD_S) -> pd.DataFrame:
    """Assign bout IDs based on inter-call interval (silent gap) threshold.

    ICI is computed as the gap between end of one call and start of the next
    (end-to-start), not onset-to-onset.  Falls back to onset-to-onset if
    end_time_s is unavailable.
    """
    start_times = df["absolute_time"].values.astype("datetime64[ns]").astype(np.float64) / 1e9

    if "end_time_s" in df.columns and "begin_time_s" in df.columns:
        # Gap = start(next) - end(current), where end = start + (end_time_s - begin_time_s)
        durations = (df["end_time_s"] - df["begin_time_s"]).values
        end_times = start_times + durations
        icis = start_times[1:] - end_times[:-1]
    else:
        icis = np.diff(start_times)

    bout_ids = np.zeros(len(df), dtype=int)
    bout_id = 0
    for i, ici in enumerate(icis):
        if ici > threshold_s:
            bout_id += 1
        bout_ids[i + 1] = bout_id

    df = df.copy()
    df["bout_id"] = bout_ids
    return df


# ── Analysis functions ──────────────────────────────────────────────────────

def extract_bout_sequences(df: pd.DataFrame) -> list[np.ndarray]:
    """Return a list of per-bout code arrays (only bouts with 2+ calls)."""
    sequences = []
    for _, bout_df in df.groupby("bout_id"):
        codes = bout_df.sort_values("absolute_time")["type_code"].values
        if len(codes) >= 2:
            sequences.append(codes)
    return sequences


def collect_bigrams_from_bouts(bout_sequences: list[np.ndarray]) -> list[tuple[int, int]]:
    """Extract all consecutive (A, B) pairs, respecting bout boundaries."""
    bigrams = []
    for seq in bout_sequences:
        for i in range(len(seq) - 1):
            bigrams.append((seq[i], seq[i + 1]))
    return bigrams


def transition_matrix_from_bouts(bout_sequences: list[np.ndarray], n_labels: int) -> np.ndarray:
    """Row-stochastic transition matrix counting only within-bout pairs."""
    counts = np.zeros((n_labels, n_labels), dtype=np.float64)
    for seq in bout_sequences:
        for i in range(len(seq) - 1):
            counts[seq[i], seq[i + 1]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(row_sums > 0, counts / row_sums, 0.0)
    return matrix, counts


def entropy_rate_from_bouts(bout_sequences: list[np.ndarray], max_order: int = 5) -> list[float]:
    """Entropy rate pooling n-gram counts across bouts (no cross-bout n-grams)."""
    rates = []
    for n in range(1, max_order + 1):
        ngram_counts: Counter = Counter()
        for seq in bout_sequences:
            if len(seq) < n:
                continue
            for i in range(len(seq) - n + 1):
                ngram_counts[tuple(seq[i:i + n])] += 1

        total = sum(ngram_counts.values())
        if total == 0:
            rates.append(0.0)
            continue

        ent = 0.0
        for count in ngram_counts.values():
            p = count / total
            if p > 0:
                ent -= p * np.log2(p)
        rates.append(ent / n)
    return rates


def conditional_entropy_from_bouts(bout_sequences: list[np.ndarray], n_labels: int) -> float:
    """H(C_{t+1} | C_t) counting only within-bout transitions."""
    counts = np.zeros((n_labels, n_labels), dtype=np.float64)
    for seq in bout_sequences:
        for i in range(len(seq) - 1):
            counts[seq[i], seq[i + 1]] += 1

    row_sums = counts.sum(axis=1)
    total = row_sums.sum()
    if total == 0:
        return 0.0

    # Row-stochastic transition matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = np.where(row_sums[:, None] > 0, counts / row_sums[:, None], 0.0)

    # Weighted row entropy
    marginal = row_sums / total
    h = 0.0
    for i in range(n_labels):
        if marginal[i] == 0:
            continue
        row_ent = 0.0
        for j in range(n_labels):
            if trans[i, j] > 0:
                row_ent -= trans[i, j] * np.log2(trans[i, j])
        h += marginal[i] * row_ent
    return h


def mi_at_lag_from_bouts(bout_sequences: list[np.ndarray], n_labels: int, lag: int) -> float:
    """MI(T, T+lag) counting only within-bout pairs at given lag."""
    joint = np.zeros((n_labels, n_labels), dtype=np.float64)
    for seq in bout_sequences:
        if len(seq) <= lag:
            continue
        for i in range(len(seq) - lag):
            joint[seq[i], seq[i + lag]] += 1

    total = joint.sum()
    if total == 0:
        return 0.0

    joint /= total
    marginal_x = joint.sum(axis=1)
    marginal_y = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_labels):
        for j in range(n_labels):
            if joint[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (marginal_x[i] * marginal_y[j]))
    return mi


def compute_mi_curve(bout_sequences: list[np.ndarray], n_labels: int, max_lag: int = 10) -> list[float]:
    """Mutual information at lags 1..max_lag, bout-aware."""
    return [mi_at_lag_from_bouts(bout_sequences, n_labels, lag) for lag in range(1, max_lag + 1)]


def idioms_from_bouts(bout_sequences: list[np.ndarray], n_labels: int,
                      max_n: int = 5, n_shuffles: int = 200, fdr_alpha: float = 0.01):
    """Run idiom detection on concatenated bout codes, but shuffle within bouts."""
    # For idiom detection we still use the library function, but feed it
    # a sentinel-separated sequence so n-grams can't span bout boundaries.
    # Strategy: use a sentinel code (K) that won't match any real n-gram.
    sentinel = n_labels  # One beyond valid codes
    parts = []
    for seq in bout_sequences:
        parts.extend(seq.tolist())
        # Insert (max_n - 1) sentinels to break any n-gram crossing
        parts.extend([sentinel] * (max_n - 1))

    combined = np.array(parts, dtype=np.int64)
    # Run idiom detection with expanded alphabet (K+1), then filter out
    # any idioms containing the sentinel
    results = ngram_idioms(combined, n_labels + 1, max_n=max_n,
                           n_shuffles=n_shuffles, fdr_alpha=fdr_alpha,
                           sentinel=sentinel)
    # Filter: keep only idioms where all codes are real (< n_labels)
    return [r for r in results if all(c < n_labels for c in r.ngram)]


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_transition_matrix(matrix: np.ndarray, labels: list[str], output_path: Path,
                           title: str = "Syllable Transition Probabilities P(B|A)"):
    """7×7 heatmap of transition probabilities."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=matrix.max(), aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Next call (B)", fontsize=11)
    ax.set_ylabel("Current call (A)", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Annotate cells with probability values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="P(B|A)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_entropy_convergence(rates: list[float], output_path: Path):
    """Entropy rate vs n-gram order."""
    orders = list(range(1, len(rates) + 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(orders, rates, "o-", color="#4477AA", linewidth=2, markersize=8)
    ax.set_xlabel("N-gram order", fontsize=11)
    ax.set_ylabel("Entropy rate (bits/symbol)", fontsize=11)
    ax.set_title("Entropy Rate Convergence", fontsize=13)
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.3)

    # Annotate the value at each point
    for o, r in zip(orders, rates):
        ax.annotate(f"{r:.3f}", (o, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_mi_curve(mi_values: list[float], output_path: Path):
    """Mutual information at lag k=1..10."""
    lags = list(range(1, len(mi_values) + 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(lags, mi_values, color="#228833", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Lag (k)", fontsize=11)
    ax.set_ylabel("MI(T, T+k) (bits)", fontsize=11)
    ax.set_title("Mutual Information at Lag — Sequence Memory", fontsize=13)
    ax.set_xticks(lags)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_zipf(codes: np.ndarray, labels: list[str], zipf_result, output_path: Path):
    """Rank-frequency plot with Zipf MLE fit."""
    counts = Counter(codes)
    # Sort by frequency descending
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    ranks = np.arange(1, len(sorted_items) + 1)
    freqs = np.array([c for _, c in sorted_items])
    type_labels = [labels[code] for code, _ in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar chart colored by type
    colors = [TYPE_COLORS.get(lbl, "#999999") for lbl in type_labels]
    bars = ax.bar(ranks, freqs, color=colors, edgecolor="black", linewidth=0.5)

    # Add type labels on bars
    for r, f, lbl in zip(ranks, freqs, type_labels):
        ax.text(r, f + max(freqs) * 0.01, lbl, ha="center", va="bottom",
                fontsize=8, rotation=30)

    # Zipf fit line (rank-frequency): f(r) = C * r^(-alpha)
    alpha_rank = zipf_result.rank_alpha
    if np.isfinite(alpha_rank) and alpha_rank > 0:
        fit_ranks = np.linspace(1, len(sorted_items), 100)
        C = freqs[0]  # Normalize so fit passes through rank 1
        fit_freqs = C * fit_ranks ** (-alpha_rank)
        ax.plot(fit_ranks, fit_freqs, "r--", linewidth=2,
                label=f"Zipf fit (α_rank={alpha_rank:.2f})")
        ax.legend(fontsize=10)

    ax.set_xlabel("Rank", fontsize=11)
    ax.set_ylabel("Frequency (count)", fontsize=11)
    ax.set_title(f"Rank-Frequency Distribution (Zipf α={zipf_result.alpha:.2f}, "
                 f"p={zipf_result.p_value:.3f})", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main analysis ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A2: Sequential structure analysis")
    parser.add_argument(
        "--csv",
        default="results/traditional_taxonomy/classified_traditional.csv",
        help="Path to classified CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sequential_structure",
        help="Output directory",
    )
    parser.add_argument(
        "--bout-threshold",
        type=float,
        default=BOUT_THRESHOLD_S,
        help="Inter-call interval threshold for bout detection (seconds)",
    )
    parser.add_argument(
        "--max-ngram-order",
        type=int,
        default=5,
        help="Maximum n-gram order for entropy rate",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=10,
        help="Maximum lag for mutual information",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=200,
        help="Number of shuffle surrogates for idiom detection",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data & detect bouts ─────────────────────────────────────────
    print("=" * 60)
    print("A2: Sequential Structure Analysis (bout-aware)")
    print("=" * 60)

    df = load_and_enrich(args.csv)
    df = detect_bouts(df, threshold_s=args.bout_threshold)
    codes = df["type_code"].values

    # Extract bout sequences — ALL sequential analyses use these
    bout_seqs = extract_bout_sequences(df)
    n_bouts = len(bout_seqs)
    n_within_pairs = sum(len(s) - 1 for s in bout_seqs)
    n_single_call = int((df.groupby("bout_id").size() == 1).sum())
    print(f"\nBout detection (threshold={args.bout_threshold}s):")
    print(f"  Total bouts: {df['bout_id'].nunique()}")
    print(f"  Multi-call bouts: {n_bouts}")
    print(f"  Single-call bouts: {n_single_call} (excluded from sequential analysis)")
    print(f"  Within-bout transition pairs: {n_within_pairs}")
    print(f"  Cross-bout gaps excluded: {len(codes) - 1 - n_within_pairs}")

    # ── 1. Transition matrix (bout-aware) ──────────────────────────────────
    print("\n── 1. Transition Matrix (within-bout only) ──")
    trans_mat, trans_counts = transition_matrix_from_bouts(bout_seqs, K)
    plot_transition_matrix(trans_mat, SYLLABLE_TYPES,
                           output_dir / "transition_matrix.png",
                           title="Within-Bout Transition Probabilities P(B|A)")

    # Identify strongest non-self transitions
    trans_mat_noselfdiag = trans_mat.copy()
    np.fill_diagonal(trans_mat_noselfdiag, 0)
    top_idx = np.unravel_index(
        np.argsort(trans_mat_noselfdiag.ravel())[-5:], trans_mat.shape
    )
    print("  Top 5 non-self transitions:")
    for i, j in zip(top_idx[0][::-1], top_idx[1][::-1]):
        print(f"    {SYLLABLE_TYPES[i]} → {SYLLABLE_TYPES[j]}: "
              f"P={trans_mat[i, j]:.3f} (n={int(trans_counts[i, j])})")

    # ── 2. Entropy rate convergence (bout-aware) ──────────────────────────
    print("\n── 2. Entropy Rate (within-bout) ──")
    rates = entropy_rate_from_bouts(bout_seqs, max_order=args.max_ngram_order)
    plot_entropy_convergence(rates, output_dir / "entropy_convergence.png")

    cond_ent = conditional_entropy_from_bouts(bout_seqs, K)
    marginal_ent = rates[0]
    print(f"  Marginal entropy H(C):        {marginal_ent:.4f} bits")
    print(f"  Conditional entropy H(C|C-1): {cond_ent:.4f} bits")
    if marginal_ent > 0:
        reduction_pct = (1 - cond_ent / marginal_ent) * 100
        print(f"  Entropy reduction:            {marginal_ent - cond_ent:.4f} bits "
              f"({reduction_pct:.1f}% predictability gain)")
    for i, r in enumerate(rates):
        print(f"  Order {i+1}: {r:.4f} bits/symbol")

    # ── 3. Mutual information at lag (bout-aware) ─────────────────────────
    print("\n── 3. Mutual Information at Lag (within-bout) ──")
    mi_values = compute_mi_curve(bout_seqs, K, max_lag=args.max_lag)
    plot_mi_curve(mi_values, output_dir / "mutual_information_lag.png")

    for lag, mi in enumerate(mi_values, 1):
        print(f"  Lag {lag:2d}: MI = {mi:.4f} bits")

    # Find where MI drops below 10% of lag-1 (practical memory limit)
    mi_threshold = mi_values[0] * 0.10 if mi_values[0] > 0 else 0
    memory_depth = next(
        (lag for lag, mi in enumerate(mi_values, 1) if mi < mi_threshold),
        len(mi_values),
    )
    print(f"  Sequence memory depth (MI < 10% of lag-1): ~{memory_depth} calls")

    # ── 4. Zipf distribution (uses overall counts, not sequential) ────────
    print("\n── 4. Zipf Distribution ──")
    zipf_result = zipf_exponent_mle(codes)
    plot_zipf(codes, SYLLABLE_TYPES, zipf_result, output_dir / "zipf_distribution.png")
    print(f"  Zipf alpha (count-distribution): {zipf_result.alpha:.3f}")
    print(f"  Zipf rank_alpha:                 {zipf_result.rank_alpha:.3f}")
    print(f"  p-value:                         {zipf_result.p_value:.3f}")
    print(f"  Log-likelihood ratio vs exp:     {zipf_result.log_likelihood_ratio:.3f}")

    # ── 5. Idiom detection (bout-aware) ───────────────────────────────────
    print(f"\n── 5. Idiom Detection ({args.n_shuffles} shuffles, bout-aware) ──")
    idioms = idioms_from_bouts(bout_seqs, K, max_n=5, n_shuffles=args.n_shuffles)
    print(f"  Found {len(idioms)} significant idioms")

    # Build idiom report
    idiom_rows = []
    for idiom in idioms:
        ngram_str = " → ".join(CODE_TO_TYPE[c] for c in idiom.ngram)
        idiom_rows.append({
            "ngram": ngram_str,
            "ngram_codes": str(idiom.ngram),
            "n": idiom.n,
            "observed": idiom.observed_count,
            "expected": round(idiom.expected_count, 1),
            "z_score": round(idiom.z_score, 2),
            "p_value": idiom.p_value,
        })
        if len(idiom_rows) <= 20:
            print(f"    [{idiom.n}-gram] {ngram_str}: "
                  f"obs={idiom.observed_count}, exp={idiom.expected_count:.1f}, "
                  f"z={idiom.z_score:.2f}, p={idiom.p_value:.4f}")

    idiom_df = pd.DataFrame(idiom_rows)
    idiom_path = output_dir / "idiom_report.csv"
    idiom_df.to_csv(idiom_path, index=False)
    print(f"  Saved: {idiom_path}")

    # ── 6. Summary CSV ─────────────────────────────────────────────────────
    print("\n── 6. Summary ──")

    # Compute self-transition rates from bout-aware matrix
    self_trans = {SYLLABLE_TYPES[i]: trans_mat[i, i] for i in range(K)}
    mean_self_trans = np.mean(list(self_trans.values()))

    # Independence baseline: Σ(pᵢ²) using the "next call" marginal from
    # within-bout transitions.  This is the expected self-transition rate if
    # consecutive calls were independent but preserved marginal frequencies.
    next_marginal = trans_counts.sum(axis=0)
    total_trans = next_marginal.sum()
    if total_trans > 0:
        p_next = next_marginal / total_trans
        chance_self = float(np.sum(p_next ** 2))
    else:
        chance_self = 1.0 / K

    summary = {
        "n_calls": len(codes),
        "n_syllable_types": K,
        "bout_threshold_s": args.bout_threshold,
        "n_multi_call_bouts": n_bouts,
        "n_within_bout_pairs": n_within_pairs,
        "n_cross_bout_gaps_excluded": len(codes) - 1 - n_within_pairs,
        "marginal_entropy_bits": round(rates[0], 4),
        "conditional_entropy_bits": round(cond_ent, 4),
        "entropy_reduction_pct": round((1 - cond_ent / rates[0]) * 100, 2) if rates[0] > 0 else 0,
        "entropy_rate_order1": round(rates[0], 4),
        "entropy_rate_order2": round(rates[1], 4),
        "entropy_rate_order3": round(rates[2], 4),
        "entropy_rate_order5": round(rates[4], 4) if len(rates) >= 5 else None,
        "mi_lag1_bits": round(mi_values[0], 4),
        "mi_lag2_bits": round(mi_values[1], 4),
        "mi_lag5_bits": round(mi_values[4], 4) if len(mi_values) >= 5 else None,
        "memory_depth_calls": memory_depth,
        "zipf_alpha": round(zipf_result.alpha, 3),
        "zipf_rank_alpha": round(zipf_result.rank_alpha, 3) if np.isfinite(zipf_result.rank_alpha) else None,
        "zipf_p_value": round(zipf_result.p_value, 3),
        "n_significant_idioms": len(idioms),
        "top_idiom": idiom_rows[0]["ngram"] if idiom_rows else None,
        "top_idiom_z_score": idiom_rows[0]["z_score"] if idiom_rows else None,
        "mean_self_transition_prob": round(mean_self_trans, 4),
        "chance_self_transition": round(chance_self, 4),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "sequential_structure_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"  Marginal entropy:     {rates[0]:.3f} bits "
          f"(max possible: {np.log2(K):.3f} bits for {K} types)")
    reduction_str = f"{(1 - cond_ent / rates[0]) * 100:.1f}%" if rates[0] > 0 else "N/A"
    print(f"  Conditional entropy:  {cond_ent:.3f} bits "
          f"→ {reduction_str} predictability from prior call")
    print(f"  Sequence memory:      ~{memory_depth} calls deep (MI criterion)")
    print(f"  Mean self-transition: {mean_self_trans:.1%} "
          f"(independence baseline Σpᵢ² = {chance_self:.1%})")
    print(f"  Significant idioms:   {len(idioms)}")
    if idiom_rows:
        print(f"  Top idiom:            {idiom_rows[0]['ngram']} (z={idiom_rows[0]['z_score']})")
    print(f"  Zipf fit:             α={zipf_result.alpha:.2f} (p={zipf_result.p_value:.3f})")
    print()

    # Interpretation
    if rates[0] > 0 and (1 - cond_ent / rates[0]) > 0.05:
        print("  → SEQUENTIAL STRUCTURE DETECTED: Knowing the current call type")
        print("    reduces uncertainty about the next call beyond chance.")
    else:
        print("  → WEAK/NO sequential structure: Next call is approximately")
        print("    independent of current call.")

    if mean_self_trans > 1.5 * chance_self:
        print(f"  → SELF-REPETITION BIAS: Calls tend to repeat (self-transition "
              f"{mean_self_trans:.1%} vs independence baseline {chance_self:.1%}).")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
