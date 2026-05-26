"""Cross-cohort analysis of DeepSqueak VAE embeddings.

Reads `vae_embeddings.csv` produced by `scripts/deepsqueak_train_vae.m` and runs
the downstream analysis needed for the wild-vs-lab comparison deck:

    - UMAP (n_neighbors=15, min_dist=0.1) on the 32D latent
    - HDBSCAN (min_cluster_size=50) on UMAP 2D and on raw 32D
    - Per-dim Cohen's d between cohorts (effect size, no p-values: N=1 couple)
    - Per-dim 1-Wasserstein / EMD between cohort marginals
    - Jensen-Shannon divergence between cohorts on the 2D UMAP density
    - Per-dim std diagnostic (catches mode collapse / dead latent dims)
    - K-NN sanity in latent space

Outputs land at --out-dir/ as:

    figs/01..10_*.png          - deck-ready figures
    umap_embeddings.csv        - UMAP 2D coords + cohort + cluster joined
    knn_sanity.csv             - K-NN query/neighbor table
    cross_cohort_summary.json  - all numerical results in one JSON

The script is safe to run before real embeddings exist: pass --smoke-test to
generate synthetic 32D embeddings with controllable cohort overlap and exercise
the full pipeline.

Usage:
    .venv/bin/python scripts/analyze_vae_embeddings.py \\
        --embeddings results/vae_5970_lab/vae_embeddings.csv \\
        --scattoni-csv results/traditional_taxonomy/classified_traditional.csv \\
        --out-dir results/vae_analysis/

    # Smoke test before real embeddings arrive:
    .venv/bin/python scripts/analyze_vae_embeddings.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Matplotlib in headless mode — important for background runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import umap  # noqa: E402
import hdbscan  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402


# ============================================================================
# Data loading
# ============================================================================

@dataclass
class EmbeddingData:
    df: pd.DataFrame           # metadata columns (cohort, mat_file, call_id, ...)
    Z: np.ndarray              # latent embeddings (N, D)
    z_cols: list[str]
    cohorts: list[str]         # unique cohort labels, sorted

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def d(self) -> int:
        return self.Z.shape[1]


def load_embeddings(path: Path) -> EmbeddingData:
    """Read vae_embeddings.csv. Detects z_* columns automatically."""
    df = pd.read_csv(path)
    z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                    key=lambda c: int(c.split("_")[1]))
    if not z_cols:
        raise ValueError(f"No z_* columns found in {path}")
    if "cohort" not in df.columns:
        raise ValueError(f"Missing 'cohort' column in {path}")
    Z = df[z_cols].to_numpy(dtype=np.float64)
    cohorts = sorted(df["cohort"].unique().tolist())
    print(f"  Loaded {len(df)} embeddings ({len(z_cols)}D) from {path}")
    print(f"  Cohorts: {cohorts}")
    for c in cohorts:
        n_c = (df["cohort"] == c).sum()
        print(f"    {c}: {n_c}")
    return EmbeddingData(df=df, Z=Z, z_cols=z_cols, cohorts=cohorts)


def join_scattoni(data: EmbeddingData, scattoni_path: Path) -> EmbeddingData:
    """Left-join Scattoni-7 labels from `classified_traditional.csv` using
    (cohort-aware) mat_file + begin_s as the merge key.

    This is a best-effort merge: if the column names don't align, the script
    proceeds without Scattoni labels and skips the type-colored plots.
    """
    if not scattoni_path.exists():
        print(f"  [skip] Scattoni CSV not found: {scattoni_path}")
        return data
    sc = pd.read_csv(scattoni_path)
    # Heuristic column matching — depends on which dataset's classified CSV.
    # `mat_file` matches the WAV stem; the classified CSV usually has `filename`
    # (the WAV stem) and `begin_time_s`.
    candidate_cols = {
        "filename": "mat_file",
        "wav_stem": "mat_file",
        "begin_time_s": "begin_s",
        "begin_s": "begin_s",
    }
    rename = {k: v for k, v in candidate_cols.items() if k in sc.columns}
    sc = sc.rename(columns=rename)
    if "mat_file" not in sc.columns or "begin_s" not in sc.columns:
        print(f"  [skip] Scattoni CSV missing expected columns after rename: {sc.columns.tolist()}")
        return data
    keep = [c for c in ("mat_file", "begin_s", "scattoni_type", "scattoni7",
                        "label", "tonality", "call_length_s", "duration_s")
            if c in sc.columns]
    sc = sc[keep].copy()
    if "scattoni_type" not in sc.columns:
        for alt in ("scattoni7", "label"):
            if alt in sc.columns:
                sc = sc.rename(columns={alt: "scattoni_type"})
                break
    # Tolerance-based join — VAE begin_s may differ from CNN begin_s by <2ms.
    merged = pd.merge_asof(
        data.df.sort_values(["mat_file", "begin_s"]),
        sc.sort_values(["mat_file", "begin_s"]),
        by="mat_file",
        on="begin_s",
        tolerance=0.005,  # 5 ms — generous given DS's frame rate
        direction="nearest",
    ).reset_index(drop=True)
    # Re-attach Z in the new order
    # (merge_asof preserves row count but reorders by sort key)
    Z_new = merged[data.z_cols].to_numpy(dtype=np.float64)
    n_joined = merged["scattoni_type"].notna().sum() if "scattoni_type" in merged.columns else 0
    print(f"  Scattoni join: {n_joined}/{len(merged)} calls matched")
    return EmbeddingData(df=merged, Z=Z_new, z_cols=data.z_cols, cohorts=data.cohorts)


# ============================================================================
# UMAP / HDBSCAN
# ============================================================================

def run_umap(Z: np.ndarray, seed: int = 42) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=seed,
        metric="euclidean",
    )
    return reducer.fit_transform(Z)


def run_hdbscan(X: np.ndarray, min_cluster_size: int = 50) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        prediction_data=False,
    )
    return clusterer.fit_predict(X)


# ============================================================================
# Cross-cohort statistics
# ============================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Per-dim Cohen's d using pooled SD. N=1-couple-aware: this is an effect
    size, not an inferential statistic."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = a.size, b.size
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def per_dim_cohens_d(Z: np.ndarray, cohort: np.ndarray, a: str, b: str) -> np.ndarray:
    A, B = Z[cohort == a], Z[cohort == b]
    return np.array([cohens_d(A[:, k], B[:, k]) for k in range(Z.shape[1])])


def per_dim_emd(Z: np.ndarray, cohort: np.ndarray, a: str, b: str) -> np.ndarray:
    A, B = Z[cohort == a], Z[cohort == b]
    return np.array([wasserstein_distance(A[:, k], B[:, k]) for k in range(Z.shape[1])])


def umap_density_jsd(umap_xy: np.ndarray, cohort: np.ndarray,
                     a: str, b: str, n_bins: int = 50) -> float:
    """JSD between two cohorts on a shared 2D histogram of the UMAP plane.

    Uses scipy.spatial.distance.jensenshannon, which returns sqrt(JSD); we
    square it to recover the divergence proper.
    """
    Ax, Ay = umap_xy[cohort == a, 0], umap_xy[cohort == a, 1]
    Bx, By = umap_xy[cohort == b, 0], umap_xy[cohort == b, 1]
    if len(Ax) == 0 or len(Bx) == 0:
        return float("nan")
    x_min, x_max = umap_xy[:, 0].min(), umap_xy[:, 0].max()
    y_min, y_max = umap_xy[:, 1].min(), umap_xy[:, 1].max()
    edges = (np.linspace(x_min, x_max, n_bins + 1),
             np.linspace(y_min, y_max, n_bins + 1))
    Ha, _, _ = np.histogram2d(Ax, Ay, bins=edges)
    Hb, _, _ = np.histogram2d(Bx, By, bins=edges)
    pa = (Ha.ravel() + 1e-12) / (Ha.sum() + 1e-12 * Ha.size)
    pb = (Hb.ravel() + 1e-12) / (Hb.sum() + 1e-12 * Hb.size)
    return float(jensenshannon(pa, pb) ** 2)


def knn_sanity(Z: np.ndarray, df: pd.DataFrame,
               n_queries_per_cohort: int = 10, k: int = 5,
               seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cohort in sorted(df["cohort"].unique()):
        idx_cohort = df.index[df["cohort"] == cohort].tolist()
        if not idx_cohort:
            continue
        n_pick = min(n_queries_per_cohort, len(idx_cohort))
        queries = rng.choice(idx_cohort, size=n_pick, replace=False)
        for q in queries:
            d = np.linalg.norm(Z - Z[q], axis=1)
            # Exclude self (distance == 0), keep the next k
            d_sorted = np.argsort(d)
            neighbors = [i for i in d_sorted if i != q][:k]
            for rank, nbr in enumerate(neighbors, 1):
                rows.append({
                    "query_cohort": cohort,
                    "query_idx": int(q),
                    "query_mat_file": df.at[q, "mat_file"] if "mat_file" in df.columns else "",
                    "query_begin_s": float(df.at[q, "begin_s"]) if "begin_s" in df.columns else float("nan"),
                    "rank": rank,
                    "neighbor_idx": int(nbr),
                    "neighbor_cohort": df.at[nbr, "cohort"],
                    "neighbor_mat_file": df.at[nbr, "mat_file"] if "mat_file" in df.columns else "",
                    "neighbor_begin_s": float(df.at[nbr, "begin_s"]) if "begin_s" in df.columns else float("nan"),
                    "latent_distance": float(d[nbr]),
                })
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================

COHORT_COLORS = {
    "5970": "#1f77b4",         # wild — blue
    "lab_131204": "#d62728",   # lab  — red
    "3452": "#2ca02c",
    "9252": "#9467bd",
}


def _color(c: str) -> str:
    return COHORT_COLORS.get(c, "#7f7f7f")


def plot_umap_by_cohort(umap_xy: np.ndarray, cohort: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    for c in sorted(np.unique(cohort)):
        m = cohort == c
        ax.scatter(umap_xy[m, 0], umap_xy[m, 1], s=4, alpha=0.4,
                   c=_color(c), label=f"{c} (N={m.sum()})", linewidths=0)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("VAE latent → UMAP, by cohort")
    ax.legend(loc="best", fontsize=9, markerscale=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_umap_by_scattoni(umap_xy: np.ndarray, scattoni: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    s = scattoni.fillna("unknown")
    for typ in sorted(s.unique()):
        m = (s == typ).to_numpy()
        ax.scatter(umap_xy[m, 0], umap_xy[m, 1], s=4, alpha=0.4,
                   label=f"{typ} (N={m.sum()})", linewidths=0)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("VAE latent → UMAP, by Scattoni-7 (joined)")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_umap_by_continuous(umap_xy: np.ndarray, vals: pd.Series,
                            title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    v = vals.to_numpy(dtype=float)
    m = np.isfinite(v)
    sc = ax.scatter(umap_xy[m, 0], umap_xy[m, 1], s=4, alpha=0.6,
                    c=v[m], cmap="viridis", linewidths=0)
    fig.colorbar(sc, ax=ax, label=title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"VAE latent → UMAP, colored by {title}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_umap_density(umap_xy: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.hexbin(umap_xy[:, 0], umap_xy[:, 1], gridsize=60, cmap="magma", mincnt=1)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP density (continuum view)")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_umap_cohort_split(umap_xy: np.ndarray, cohort: np.ndarray, out: Path) -> None:
    cohorts = sorted(np.unique(cohort).tolist())
    n = len(cohorts)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=120, squeeze=False)
    x_lim = umap_xy[:, 0].min(), umap_xy[:, 0].max()
    y_lim = umap_xy[:, 1].min(), umap_xy[:, 1].max()
    for ax, c in zip(axes.flat, cohorts):
        m = cohort == c
        ax.hexbin(umap_xy[m, 0], umap_xy[m, 1], gridsize=50, cmap="magma",
                  mincnt=1, extent=(*x_lim, *y_lim))
        ax.set_title(f"{c} (N={m.sum()})")
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_cohens_d(d: np.ndarray, a: str, b: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    order = np.argsort(np.abs(d))[::-1]
    bars = ax.bar(range(len(d)), d[order], color="#1f77b4")
    for thresh, style in [(0.2, ":"), (0.5, "--"), (0.8, "-")]:
        ax.axhline(thresh, color="grey", linestyle=style, alpha=0.6)
        ax.axhline(-thresh, color="grey", linestyle=style, alpha=0.6)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"z{i}" for i in order], rotation=90, fontsize=7)
    ax.set_ylabel(f"Cohen's d  ({a} - {b})")
    ax.set_title(f"Per-dim effect size {a} vs {b}  (|d|≥0.2 small, ≥0.5 med, ≥0.8 large)")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_emd(emd: np.ndarray, a: str, b: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    order = np.argsort(emd)[::-1]
    ax.bar(range(len(emd)), emd[order], color="#d62728")
    ax.set_xticks(range(len(emd)))
    ax.set_xticklabels([f"z{i}" for i in order], rotation=90, fontsize=7)
    ax.set_ylabel(f"1-Wasserstein ({a} - {b})")
    ax.set_title(f"Per-dim EMD  ({a} vs {b})")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_hdbscan(umap_xy: np.ndarray, labels: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    uniq = sorted(np.unique(labels).tolist())
    for k in uniq:
        m = labels == k
        if k == -1:
            ax.scatter(umap_xy[m, 0], umap_xy[m, 1], s=3, alpha=0.2,
                       c="lightgrey", label=f"noise (N={m.sum()})", linewidths=0)
        else:
            ax.scatter(umap_xy[m, 0], umap_xy[m, 1], s=4, alpha=0.5,
                       label=f"c{k} (N={m.sum()})", linewidths=0)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("HDBSCAN on UMAP")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_dead_dims(Z: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    stds = Z.std(axis=0)
    ax.bar(range(len(stds)), stds, color="#2ca02c")
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.6,
               label="dead-dim threshold (std=0.05)")
    ax.set_xticks(range(len(stds)))
    ax.set_xticklabels([f"z{i}" for i in range(len(stds))], rotation=90, fontsize=7)
    ax.set_ylabel("std across dataset")
    ax.set_title("Per-dim latent activity  (low std = mode collapse / dead dim)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# ============================================================================
# Smoke test — synthetic embeddings
# ============================================================================

def make_synthetic_embeddings(n_per_cohort: int = 5000, d: int = 32,
                              seed: int = 42) -> pd.DataFrame:
    """Two cohorts with controlled overlap. Cohort A is N(0, I), cohort B is
    N(mu, I) with mu_k = 0.5 for k<8 and 0 elsewhere. This gives roughly
    Cohen's d=0.5 on dims 0-7 and ~0 elsewhere — exactly the per-dim story
    we expect the real wild-vs-lab plot to tell."""
    rng = np.random.default_rng(seed)
    Za = rng.standard_normal((n_per_cohort, d))
    Zb = rng.standard_normal((n_per_cohort, d))
    mu = np.zeros(d); mu[:8] = 0.5
    Zb += mu
    z_cols = [f"z_{i}" for i in range(d)]
    df_a = pd.DataFrame(Za, columns=z_cols)
    df_a.insert(0, "cohort", "5970")
    df_a.insert(1, "mat_file", [f"synthA_{i:05d}" for i in range(n_per_cohort)])
    df_a.insert(2, "call_id", np.arange(n_per_cohort))
    df_a.insert(3, "begin_s", rng.uniform(0, 1.5, n_per_cohort))
    df_a.insert(4, "end_s", df_a["begin_s"] + rng.uniform(0.02, 0.15, n_per_cohort))
    df_b = pd.DataFrame(Zb, columns=z_cols)
    df_b.insert(0, "cohort", "lab_131204")
    df_b.insert(1, "mat_file", [f"synthB_{i:05d}" for i in range(n_per_cohort)])
    df_b.insert(2, "call_id", np.arange(n_per_cohort))
    df_b.insert(3, "begin_s", rng.uniform(0, 1.5, n_per_cohort))
    df_b.insert(4, "end_s", df_b["begin_s"] + rng.uniform(0.02, 0.15, n_per_cohort))
    return pd.concat([df_a, df_b], ignore_index=True)


# ============================================================================
# Main
# ============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Analyze DS VAE embeddings cross-cohort.")
    p.add_argument("--embeddings", type=Path,
                   default=Path("results/vae_5970_lab/vae_embeddings.csv"),
                   help="Path to vae_embeddings.csv (default: results/vae_5970_lab/...)")
    p.add_argument("--scattoni-csv", type=Path,
                   default=Path("results/traditional_taxonomy/classified_traditional.csv"),
                   help="Optional Scattoni-7 CSV for type-colored UMAP")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/vae_analysis"),
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="Generate synthetic embeddings instead of reading from disk")
    p.add_argument("--min-cluster-size", type=int, default=50,
                   help="HDBSCAN min_cluster_size (default 50)")
    args = p.parse_args(argv)

    print("=" * 70)
    print(" Analyze VAE embeddings")
    print("=" * 70)
    print(f"  embeddings:     {args.embeddings}")
    print(f"  scattoni_csv:   {args.scattoni_csv}")
    print(f"  out_dir:        {args.out_dir}")
    print(f"  seed:           {args.seed}")
    print(f"  smoke_test:     {args.smoke_test}")
    print(f"  min_cluster_size: {args.min_cluster_size}")
    print("=" * 70)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = args.out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    # ----- Load -----
    if args.smoke_test:
        print("\n[smoke test] Generating synthetic embeddings (2 cohorts × 5000)")
        df = make_synthetic_embeddings(seed=args.seed)
        synth_path = args.out_dir / "synthetic_embeddings.csv"
        df.to_csv(synth_path, index=False)
        print(f"  Wrote {synth_path}")
        data = load_embeddings(synth_path)
    else:
        if not args.embeddings.exists():
            print(f"\nERROR: {args.embeddings} does not exist.", file=sys.stderr)
            print("Run with --smoke-test to exercise the pipeline against synthetic data.",
                  file=sys.stderr)
            return 2
        data = load_embeddings(args.embeddings)
        if args.scattoni_csv and not args.smoke_test:
            data = join_scattoni(data, args.scattoni_csv)

    # ----- UMAP -----
    print("\n[UMAP] fitting 32D → 2D ...")
    umap_xy = run_umap(data.Z, seed=args.seed)
    print(f"  UMAP done: shape={umap_xy.shape}")
    cohort = data.df["cohort"].to_numpy()

    # ----- HDBSCAN -----
    print(f"\n[HDBSCAN] min_cluster_size={args.min_cluster_size}")
    hdb_umap = run_hdbscan(umap_xy, min_cluster_size=args.min_cluster_size)
    hdb_32d = run_hdbscan(data.Z, min_cluster_size=args.min_cluster_size)
    n_clusters_umap = int((np.unique(hdb_umap) >= 0).sum())
    n_noise_umap = int((hdb_umap == -1).sum())
    n_clusters_32d = int((np.unique(hdb_32d) >= 0).sum())
    n_noise_32d = int((hdb_32d == -1).sum())
    print(f"  on UMAP: {n_clusters_umap} clusters, {n_noise_umap} noise ({n_noise_umap/len(hdb_umap):.1%})")
    print(f"  on 32D:  {n_clusters_32d} clusters, {n_noise_32d} noise ({n_noise_32d/len(hdb_32d):.1%})")

    # ----- Cross-cohort statistics -----
    print("\n[stats] per-dim Cohen's d, EMD, UMAP-density JSD")
    pair_results: dict[str, dict] = {}
    if len(data.cohorts) >= 2:
        a, b = data.cohorts[0], data.cohorts[1]
        d_cohens = per_dim_cohens_d(data.Z, cohort, a, b)
        emd_per_d = per_dim_emd(data.Z, cohort, a, b)
        jsd_2d = umap_density_jsd(umap_xy, cohort, a, b)
        n_active_dims = int(np.sum(np.abs(d_cohens) >= 0.2))
        print(f"  {a} vs {b}: max|d|={np.max(np.abs(d_cohens)):.3f}  "
              f"#dims |d|≥0.2: {n_active_dims}/{len(d_cohens)}  "
              f"sum EMD: {emd_per_d.sum():.3f}  UMAP-JSD: {jsd_2d:.4f}")
        pair_results[f"{a}_vs_{b}"] = {
            "cohort_a": a,
            "cohort_b": b,
            "n_a": int((cohort == a).sum()),
            "n_b": int((cohort == b).sum()),
            "cohens_d_per_dim": d_cohens.tolist(),
            "emd_per_dim": emd_per_d.tolist(),
            "sum_emd": float(emd_per_d.sum()),
            "max_abs_cohens_d": float(np.max(np.abs(d_cohens))),
            "n_dims_d_above_0p2": n_active_dims,
            "umap_density_jsd": jsd_2d,
        }
        plot_cohens_d(d_cohens, a, b, figs_dir / "06_cohen_d_per_dim.png")
        plot_emd(emd_per_d, a, b, figs_dir / "07_emd_per_dim.png")
    else:
        print(f"  Only one cohort ({data.cohorts}); skipping pairwise stats.")

    # ----- K-NN sanity -----
    print("\n[k-NN] sanity (10 queries per cohort, k=5)")
    knn_df = knn_sanity(data.Z, data.df, n_queries_per_cohort=10, k=5, seed=args.seed)
    knn_path = args.out_dir / "knn_sanity.csv"
    knn_df.to_csv(knn_path, index=False)
    # Quick sanity readout
    same_cohort_frac = (knn_df["query_cohort"] == knn_df["neighbor_cohort"]).mean()
    print(f"  Same-cohort fraction in NN sets: {same_cohort_frac:.3f}")
    print(f"  (1.0 = total separation; ~0.5 = full overlap.  Wrote {knn_path})")

    # ----- Plots -----
    print("\n[plots] writing figures to", figs_dir)
    plot_umap_by_cohort(umap_xy, cohort, figs_dir / "01_umap_cohort.png")
    plot_umap_density(umap_xy, figs_dir / "04_density_continuum.png")
    plot_umap_cohort_split(umap_xy, cohort, figs_dir / "05_cohort_split.png")
    plot_hdbscan(umap_xy, hdb_umap, figs_dir / "08_hdbscan_clusters.png")
    plot_dead_dims(data.Z, figs_dir / "10_dead_dim_diagnostic.png")
    if "scattoni_type" in data.df.columns and data.df["scattoni_type"].notna().any():
        plot_umap_by_scattoni(umap_xy, data.df["scattoni_type"],
                              figs_dir / "02_umap_scattoni.png")
    duration_col = next((c for c in ("call_length_s", "duration_s") if c in data.df.columns), None)
    if duration_col is not None:
        plot_umap_by_continuous(umap_xy, data.df[duration_col],
                                "call duration (s)", figs_dir / "03_umap_duration.png")
    if "tonality" in data.df.columns:
        plot_umap_by_continuous(umap_xy, data.df["tonality"],
                                "tonality", figs_dir / "03b_umap_tonality.png")

    # ----- Save UMAP coords -----
    umap_df = data.df[[c for c in ("cohort", "mat_file", "call_id", "begin_s", "end_s")
                       if c in data.df.columns]].copy()
    umap_df["umap_1"] = umap_xy[:, 0]
    umap_df["umap_2"] = umap_xy[:, 1]
    umap_df["hdbscan_umap"] = hdb_umap
    umap_df["hdbscan_32d"] = hdb_32d
    umap_path = args.out_dir / "umap_embeddings.csv"
    umap_df.to_csv(umap_path, index=False)
    print(f"  Wrote {umap_path}")

    # ----- Summary JSON -----
    summary = {
        "embeddings_path": str(args.embeddings) if not args.smoke_test else "synthetic",
        "smoke_test": args.smoke_test,
        "n_total": data.n,
        "latent_dim": data.d,
        "cohorts": {c: int((cohort == c).sum()) for c in data.cohorts},
        "umap_params": {"n_neighbors": 15, "min_dist": 0.1, "random_state": args.seed},
        "hdbscan_on_umap": {
            "min_cluster_size": args.min_cluster_size,
            "n_clusters": n_clusters_umap,
            "n_noise": n_noise_umap,
            "noise_frac": n_noise_umap / len(hdb_umap),
        },
        "hdbscan_on_32d": {
            "min_cluster_size": args.min_cluster_size,
            "n_clusters": n_clusters_32d,
            "n_noise": n_noise_32d,
            "noise_frac": n_noise_32d / len(hdb_32d),
        },
        "knn_same_cohort_frac": float(same_cohort_frac),
        "dead_dim_count_at_std_0p05": int((data.Z.std(axis=0) < 0.05).sum()),
        "per_dim_std": data.Z.std(axis=0).tolist(),
        "pairwise": pair_results,
    }
    summary_path = args.out_dir / "cross_cohort_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {summary_path}")

    print("\n[done] Outputs in", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
