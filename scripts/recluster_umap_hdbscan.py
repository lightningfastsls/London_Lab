#!/usr/bin/env python3
"""Re-cluster USV detections using UMAP + HDBSCAN on acoustic features.

Reads classified_detections_full.csv (7,921 rows with 27 k-means clusters from
DeepSqueak), applies UMAP dimensionality reduction and HDBSCAN density-based
clustering on acoustic features, and outputs comparison visualizations.

UMAP is run twice: 2D for visualization (optimized for layout) and higher-dim
for HDBSCAN input (preserves more metric structure for density estimation).

Output: results/recluster_umap_hdbscan/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ACOUSTIC_FEATURES = [
    "call_length_s",
    "principal_freq_hz",
    "low_freq_hz",
    "high_freq_hz",
    "bandwidth_hz",
    "freq_std_dev_hz",
    "slope",
    "sinuosity",
    "tonality",
    "mean_power_db",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-cluster USV detections with UMAP + HDBSCAN",
    )
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "classified_detections_full.csv"),
        help="Input CSV with acoustic features and k-means labels",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "recluster_umap_hdbscan"),
        help="Output directory for CSV, figures, gallery",
    )
    # UMAP parameters
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--umap-components-cluster", type=int, default=8,
                        help="UMAP dimensions for HDBSCAN input (default 8)")
    # HDBSCAN parameters
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="HDBSCAN min_cluster_size (default 50)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="HDBSCAN min_samples (default 10)")
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-features", nargs="*", default=[],
                        help="Features to exclude (e.g. mean_power_db)")
    parser.add_argument("--skip-gallery", action="store_true",
                        help="Skip spectrogram gallery generation")
    parser.add_argument("--gallery-n-per-cluster", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def load_and_prepare(
    csv_path: str,
    exclude_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Load CSV, select features, normalize.

    Returns (df_full, df_valid, X_scaled, feature_names).
    Rows with NaN in any acoustic feature are excluded from df_valid/X_scaled
    but kept in df_full for later reassembly.
    """
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path}")

    features = [f for f in ACOUSTIC_FEATURES if f not in exclude_features]
    if exclude_features:
        log.info(f"Excluded features: {exclude_features}")
    log.info(f"Using {len(features)} features: {features}")

    mask_valid = df[features].notna().all(axis=1)
    df_valid = df[mask_valid].copy()
    n_nan = (~mask_valid).sum()
    log.info(f"Valid rows: {len(df_valid)}, NaN rows excluded: {n_nan}")

    X = df_valid[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if log.isEnabledFor(logging.DEBUG):
        for i, f in enumerate(features):
            log.debug(f"  {f}: raw [{X[:, i].min():.3f}, {X[:, i].max():.3f}] "
                      f"-> scaled [{X_scaled[:, i].min():.2f}, {X_scaled[:, i].max():.2f}]")

    return df, df_valid, X_scaled, features


# ---------------------------------------------------------------------------
# UMAP + HDBSCAN
# ---------------------------------------------------------------------------

def run_umap(
    X_scaled: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    import umap

    log.info(f"Running UMAP (n_components={n_components}, n_neighbors={n_neighbors}, "
             f"min_dist={min_dist}) on {X_scaled.shape[0]} points...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="euclidean",
    )
    embedding = reducer.fit_transform(X_scaled)
    log.info(f"UMAP {n_components}D done in {time.time() - t0:.1f}s -> shape {embedding.shape}")
    return embedding


def run_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HDBSCAN clustering. Returns (labels, probabilities)."""
    import hdbscan

    log.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, "
             f"min_samples={min_samples}) on {embedding.shape} embedding...")
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        core_dist_n_jobs=1,  # deterministic
    )
    clusterer.fit(embedding)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    log.info(f"HDBSCAN done in {time.time() - t0:.1f}s: "
             f"{n_clusters} clusters, {n_noise} noise points ({n_noise / len(labels) * 100:.1f}%)")

    # Log cluster size distribution
    for lbl in sorted(set(labels)):
        count = (labels == lbl).sum()
        name = "noise" if lbl == -1 else f"cluster {lbl}"
        log.info(f"  {name}: {count} points")

    return labels, probabilities


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_output_df(
    df_full: pd.DataFrame,
    df_valid: pd.DataFrame,
    emb_2d: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Merge clustering results back into the full DataFrame."""
    df_valid = df_valid.copy()
    df_valid["umap_x"] = emb_2d[:, 0]
    df_valid["umap_y"] = emb_2d[:, 1]
    df_valid["hdbscan_label"] = labels
    df_valid["hdbscan_probability"] = probabilities

    # NaN rows get noise label
    nan_idx = df_full.index.difference(df_valid.index)
    n_nan = len(nan_idx)
    if n_nan == 0:
        df_out = df_valid.sort_index()
    else:
        # Build NaN-row additions with explicit dtypes matching df_valid
        # to avoid FutureWarning from pd.concat on all-NA columns
        df_nan = df_full.loc[nan_idx].copy()
        df_nan["umap_x"] = np.nan
        df_nan["umap_y"] = np.nan
        df_nan["hdbscan_label"] = -1
        df_nan["hdbscan_probability"] = 0.0
        for col in df_valid.columns:
            if col in df_nan.columns:
                try:
                    df_nan[col] = df_nan[col].astype(df_valid[col].dtype)
                except (ValueError, TypeError):
                    pass
        df_out = pd.concat([df_valid, df_nan]).sort_index()
    log.info(f"Output DataFrame: {len(df_out)} rows "
             f"({len(df_valid)} clustered + {n_nan} NaN -> noise)")
    return df_out


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_umap_scatter(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
):
    """UMAP 2D scatter plot colored by cluster labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    unique_labels = sorted(set(labels))
    has_noise = -1 in unique_labels
    cluster_labels = [l for l in unique_labels if l != -1]

    # Build colormap for clusters
    if len(cluster_labels) <= 20:
        cmap = plt.cm.tab20
    else:
        # Concatenate tab20 + tab20b for >20 clusters
        colors_a = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_b = plt.cm.tab20b(np.linspace(0, 1, 20))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(np.vstack([colors_a, colors_b]))

    # Plot noise first (underneath)
    if has_noise:
        mask = labels == -1
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=1, alpha=0.15, c="#cccccc", label=f"noise ({mask.sum()})",
            rasterized=True,
        )

    # Plot each cluster
    for i, lbl in enumerate(cluster_labels):
        mask = labels == lbl
        color = cmap(i / max(len(cluster_labels) - 1, 1))
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=3, alpha=0.6, c=[color], label=f"{lbl} ({mask.sum()})",
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=11)

    # Legend: outside plot if many clusters, otherwise inside
    n_legend = len(unique_labels)
    if n_legend <= 15:
        ax.legend(fontsize=7, markerscale=3, loc="best")
    else:
        ax.legend(fontsize=6, markerscale=3, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), ncol=1 + n_legend // 30)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved scatter plot: {output_path}")


def compute_cluster_summary(
    df_valid: pd.DataFrame,
    features: list[str],
    label_col: str = "hdbscan_label",
) -> pd.DataFrame:
    """Per-cluster feature statistics."""
    grouped = df_valid.groupby(label_col)[features]
    summary = grouped.agg(["count", "mean", "std", "median"]).reset_index()
    # Flatten MultiIndex columns
    summary.columns = [
        f"{feat}_{stat}" if stat != "" else feat
        for feat, stat in summary.columns
    ]
    # The count is the same for all features — keep just one
    count_cols = [c for c in summary.columns if c.endswith("_count")]
    if count_cols:
        summary.insert(1, "count", summary[count_cols[0]])
        summary.drop(columns=count_cols, inplace=True)

    # Log a readable summary
    log.info("=== Cluster Summary ===")
    for _, row in summary.iterrows():
        lbl = int(row.iloc[0])
        n = int(row["count"])
        name = "noise" if lbl == -1 else f"cluster {lbl}"
        log.info(f"  {name}: n={n}")

    return summary


def plot_contingency_matrix(
    old_labels: np.ndarray,
    new_labels: np.ndarray,
    output_path: Path,
):
    """Heatmap of old k-means labels vs new HDBSCAN labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    contingency = pd.crosstab(
        pd.Series(old_labels, name="k-means cluster"),
        pd.Series(new_labels, name="HDBSCAN cluster"),
    )
    # Sort columns: noise (-1) first, then numeric
    col_order = sorted(contingency.columns, key=lambda x: (x != -1, x))
    contingency = contingency[col_order]

    n_old = len(contingency.index)
    n_new = len(contingency.columns)
    fig_w = max(10, n_new * 0.7 + 3)
    fig_h = max(8, n_old * 0.4 + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    # Annotate only if matrix is small enough to be readable
    annot = n_old * n_new <= 600
    sns.heatmap(
        contingency, annot=annot, fmt="d", cmap="YlOrRd",
        linewidths=0.3, ax=ax, cbar_kws={"label": "count"},
    )
    ax.set_title("k-means (27 clusters) vs HDBSCAN re-clustering", fontsize=11)
    ax.set_ylabel("Original k-means cluster")
    ax.set_xlabel("HDBSCAN cluster")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved contingency matrix: {output_path}")


# ---------------------------------------------------------------------------
# Gallery generation (reuses generate_cluster_gallery functions)
# ---------------------------------------------------------------------------

def generate_gallery(
    df_out: pd.DataFrame,
    gallery_dir: Path,
    n_per_cluster: int,
    seed: int,
):
    """Generate spectrogram PNGs for each HDBSCAN cluster.

    Imports rendering functions from generate_cluster_gallery.py.
    """
    # Import from sibling script
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_cluster_gallery import build_wav_lookup, render_call_spectrogram

    rng = np.random.default_rng(seed)

    # Build WAV lookup
    search_dirs = [
        REPO_ROOT / "5970_reviewed",
        REPO_ROOT / "5970 USV",
    ]
    log.info("Building WAV lookup for gallery...")
    wav_lookup = build_wav_lookup(search_dirs)
    log.info(f"Found {len(wav_lookup)} unique WAV stems")

    # Only rows with valid clustering and WAV info
    df_gallery = df_out.dropna(subset=["wav_stem", "begin_time_s", "end_time_s"])
    labels = sorted(df_gallery["hdbscan_label"].unique())
    log.info(f"Generating gallery for {len(labels)} label groups, "
             f"{n_per_cluster} examples each")

    total_generated = 0

    for lbl in labels:
        lbl_int = int(lbl)
        cluster_name = "HDBSCAN_noise" if lbl_int == -1 else f"HDBSCAN_{lbl_int}"
        cluster_df = df_gallery[df_gallery["hdbscan_label"] == lbl_int]

        # Filter to rows where we have a WAV file
        available = cluster_df[cluster_df["wav_stem"].isin(wav_lookup)]
        if len(available) == 0:
            log.warning(f"{cluster_name}: no WAV files available, skipping gallery")
            continue

        n_sample = min(n_per_cluster, len(available))
        sample = available.sample(n=n_sample, random_state=rng.integers(0, 2**31))

        cluster_dir = gallery_dir / cluster_name
        generated = 0

        for idx, (_, row) in enumerate(sample.iterrows()):
            wav_path = wav_lookup[row["wav_stem"]]
            begin_s = row["begin_time_s"]
            end_s = row["end_time_s"]
            freq_info = {
                "principal_freq_hz": row.get("principal_freq_hz"),
                "low_freq_hz": row.get("low_freq_hz"),
                "high_freq_hz": row.get("high_freq_hz"),
            }
            prob = row.get("hdbscan_probability", 0)
            fname = f"{idx + 1:02d}_{row['wav_stem']}_{begin_s:.3f}s.png"
            title = f"{cluster_name} (p={prob:.2f}) | {row['wav_stem']} @ {begin_s:.3f}s"

            ok = render_call_spectrogram(
                wav_path, begin_s, end_s, cluster_dir / fname, title, freq_info,
            )
            if ok:
                generated += 1

        total_generated += generated
        log.info(f"{cluster_name}: {generated}/{n_sample} PNGs "
                 f"(from {len(cluster_df)} total calls)")

    log.info(f"Gallery done: {total_generated} PNGs in {gallery_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare
    df_full, df_valid, X_scaled, features = load_and_prepare(
        args.csv, args.exclude_features,
    )

    # 2. UMAP: 2D for visualization
    emb_2d = run_umap(X_scaled, n_components=2,
                      n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                      seed=args.seed)

    # 3. UMAP: higher-dim for HDBSCAN
    emb_nd = run_umap(X_scaled, n_components=args.umap_components_cluster,
                      n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                      seed=args.seed)

    # 4. HDBSCAN on higher-dim embedding
    labels, probabilities = run_hdbscan(
        emb_nd, args.min_cluster_size, args.min_samples,
    )

    # 5. Assemble output DataFrame
    df_out = build_output_df(df_full, df_valid, emb_2d, labels, probabilities)
    csv_path = output_dir / "reclassified_detections.csv"
    df_out.to_csv(csv_path, index=False)
    log.info(f"Saved {len(df_out)} rows to {csv_path}")

    # 6. UMAP scatter: HDBSCAN labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    plot_umap_scatter(
        emb_2d, labels,
        output_dir / "umap_hdbscan_scatter.png",
        f"UMAP + HDBSCAN: {n_clusters} clusters, {n_noise} noise points",
    )

    # 7. UMAP scatter: original k-means labels
    old_label_nums = (
        df_valid["label"]
        .str.extract(r"Cluster_(\d+)", expand=False)
        .fillna(-1)
        .astype(int)
        .values
    )
    plot_umap_scatter(
        emb_2d, old_label_nums,
        output_dir / "umap_kmeans_scatter.png",
        "UMAP colored by original DeepSqueak k-means (27 clusters)",
    )

    # 8. Cluster feature summary
    df_valid_with_labels = df_valid.copy()
    df_valid_with_labels["hdbscan_label"] = labels
    summary = compute_cluster_summary(df_valid_with_labels, features)
    summary_path = output_dir / "cluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved cluster summary: {summary_path}")

    # 9. Contingency matrix
    plot_contingency_matrix(
        old_label_nums, labels,
        output_dir / "contingency_matrix.png",
    )

    # 10. Gallery
    if not args.skip_gallery:
        generate_gallery(
            df_out, output_dir / "gallery",
            args.gallery_n_per_cluster, args.seed,
        )
    else:
        log.info("Gallery generation skipped (--skip-gallery)")

    elapsed = time.time() - t_start
    log.info(f"All done in {elapsed:.1f}s. Output: {output_dir}")


if __name__ == "__main__":
    main()
