#!/usr/bin/env python3
"""A3: Acoustic feature deep-dive for USV calls.

Analyzes the 10 acoustic features from the traditional taxonomy classification
to answer: does the 7-type taxonomy cut the continuous acoustic space at
meaningful boundaries, or does it impose arbitrary categories on a continuum?

Analyses:
  1. Feature correlation matrix (clustermap with hierarchical clustering)
  2. PCA biplot + scree plot + loadings table
  3. UMAP embedding colored by type and by individual features
  4. Within-type violin plots with classification threshold reference lines
  5. Boundary case analysis (low-confidence calls)
  6. Auto-generated summary of key findings

Output: results/acoustic_feature_analysis/

Usage:
    python scripts/analyze_acoustic_features.py [--input CSV] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False

REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Feature configuration ─────────────────────────────────────────────────
# The 10 acoustic features to analyze. Values are in kHz despite "_hz" suffix.
ACOUSTIC_FEATURES = [
    "call_length_s",
    "principal_freq_hz",
    "low_freq_hz",
    "high_freq_hz",
    "bandwidth_hz",
    "freq_std_dev_hz",
    "slope",
    "sinuosity",
    "mean_power_db",
    "tonality",
]

# Human-readable labels for plots
FEATURE_LABELS = {
    "call_length_s": "Duration (s)",
    "principal_freq_hz": "Principal freq (kHz)",
    "low_freq_hz": "Low freq (kHz)",
    "high_freq_hz": "High freq (kHz)",
    "bandwidth_hz": "Bandwidth (kHz)",
    "freq_std_dev_hz": "Freq SD (kHz)",
    "slope": "Slope",
    "sinuosity": "Sinuosity",
    "mean_power_db": "Mean power (dB)",
    "tonality": "Tonality",
}

# Classification thresholds from classify_traditional_taxonomy.py
# Used as reference lines on violin plots
THRESHOLDS = {
    "call_length_s": [("Short < 15ms", 0.015)],
    "sinuosity": [
        ("Complex > 3.5", 3.5),
        ("Chevron > 1.8", 1.8),
    ],
    "bandwidth_hz": [
        ("Chevron > 25", 25.0),
        ("Freq Jump > 55", 55.0),
    ],
    "slope": [
        ("Up > 200", 200.0),
        ("Down < -200", -200.0),
    ],
}

# Syllable type ordering and colors (consistent with taxonomy script)
TYPE_ORDER = ["Short", "Flat", "Up", "Down", "Chevron", "Complex", "Frequency_Jump"]
TYPE_COLORS = {
    "Short": "#e41a1c",
    "Flat": "#377eb8",
    "Up": "#4daf4a",
    "Down": "#984ea3",
    "Chevron": "#ff7f00",
    "Complex": "#a65628",
    "Frequency_Jump": "#f781bf",
}


# ── Data loading ──────────────────────────────────────────────────────────

def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load classified CSV, select features, standardize.

    Returns:
        df: Full dataframe (rows with NaN features dropped)
        features_raw: DataFrame of the 10 raw acoustic features
        features_scaled: Standardized feature array (zero mean, unit variance)
    """
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    # Drop rows missing any acoustic feature
    initial = len(df)
    df = df.dropna(subset=ACOUSTIC_FEATURES).copy()
    dropped = initial - len(df)
    if dropped:
        log.info("Dropped %d rows with NaN features (%d remaining)", dropped, len(df))

    # Filter to classified calls only
    df = df[df["syllable_type"].isin(TYPE_ORDER)].copy()
    log.info("Using %d classified calls (excluded 'unclassified')", len(df))

    features_raw = df[ACOUSTIC_FEATURES].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw)

    return df, features_raw, features_scaled


# ── Analysis 1: Feature correlation ───────────────────────────────────────

def plot_correlation_matrix(features_raw: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute and plot feature correlation clustermap."""
    log.info("Computing feature correlations...")
    labels = [FEATURE_LABELS.get(f, f) for f in features_raw.columns]
    corr = features_raw.corr()
    corr.index = labels
    corr.columns = labels

    g = sns.clustermap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        figsize=(12, 10),
        linewidths=0.5,
        dendrogram_ratio=0.15,
    )
    g.fig.suptitle("Acoustic Feature Correlation Matrix (hierarchically clustered)", y=1.02)
    g.savefig(output_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    log.info("Saved correlation_matrix.png")
    return corr


# ── Analysis 2: PCA ───────────────────────────────────────────────────────

def run_pca(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """PCA on standardized features: scree plot, biplot, loadings CSV."""
    log.info("Running PCA...")
    n_components = len(ACOUSTIC_FEATURES)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(features_scaled)

    labels = [FEATURE_LABELS.get(f, f) for f in ACOUSTIC_FEATURES]

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 5))
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)
    x = np.arange(1, n_components + 1)
    ax.bar(x, explained, color="#377eb8", alpha=0.7, label="Individual")
    ax.plot(x, cumulative, "o-", color="#e41a1c", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Scree Plot — Acoustic Features")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_scree.png", dpi=150)
    plt.close(fig)

    # Biplot (PC1 vs PC2 with loading arrows)
    fig, ax = plt.subplots(figsize=(12, 10))
    types = df["syllable_type"].values
    for stype in TYPE_ORDER:
        mask = types == stype
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=TYPE_COLORS[stype],
            label=stype,
            alpha=0.3,
            s=10,
            rasterized=True,
        )

    # Loading arrows
    loadings = pca.components_[:2, :]  # (2, n_features)
    scale = np.abs(scores[:, :2]).max() * 0.8
    for i, label in enumerate(labels):
        ax.annotate(
            "",
            xy=(loadings[0, i] * scale, loadings[1, i] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )
        ax.text(
            loadings[0, i] * scale * 1.1,
            loadings[1, i] * scale * 1.1,
            label,
            fontsize=9,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.set_title("PCA Biplot — Acoustic Features by Syllable Type")
    ax.legend(loc="upper right", markerscale=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_biplot.png", dpi=150)
    plt.close(fig)

    # Loadings table
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=ACOUSTIC_FEATURES,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings_df["feature_label"] = [FEATURE_LABELS.get(f, f) for f in ACOUSTIC_FEATURES]
    loadings_df.to_csv(output_dir / "pca_loadings.csv")
    log.info("Saved pca_scree.png, pca_biplot.png, pca_loadings.csv")

    return pca.explained_variance_ratio_, loadings_df


# ── Analysis 3: UMAP embedding ───────────────────────────────────────────

def compute_umap(features_scaled: np.ndarray) -> np.ndarray:
    """Compute 2D embedding via UMAP (or t-SNE fallback)."""
    if HAS_UMAP:
        log.info("Computing UMAP embedding (n=%d)...", len(features_scaled))
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(features_scaled)
    else:
        log.warning("umap-learn not installed, falling back to t-SNE")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding = reducer.fit_transform(features_scaled)
    return embedding


def plot_umap_by_type(
    embedding: np.ndarray,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """UMAP colored by syllable type."""
    fig, ax = plt.subplots(figsize=(10, 8))
    types = df["syllable_type"].values
    for stype in TYPE_ORDER:
        mask = types == stype
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=TYPE_COLORS[stype],
            label=f"{stype} (n={mask.sum()})",
            alpha=0.4,
            s=8,
            rasterized=True,
        )
    method = "UMAP" if HAS_UMAP else "t-SNE"
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{method} of 10 Acoustic Features — Colored by Syllable Type")
    ax.legend(loc="upper right", markerscale=3, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "umap_by_type.png", dpi=150)
    plt.close(fig)
    log.info("Saved umap_by_type.png")


def plot_umap_by_features(
    embedding: np.ndarray,
    features_raw: pd.DataFrame,
    output_dir: Path,
) -> None:
    """UMAP colored by each of 6 key features (2x3 grid)."""
    key_features = [
        "call_length_s", "slope", "sinuosity",
        "bandwidth_hz", "mean_power_db", "tonality",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    method = "UMAP" if HAS_UMAP else "t-SNE"

    for ax, feat in zip(axes.flat, key_features):
        vals = features_raw[feat].values
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=vals,
            cmap="viridis",
            alpha=0.4,
            s=5,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlabel(f"{method} 1", fontsize=8)
        ax.set_ylabel(f"{method} 2", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"{method} of Acoustic Features — Colored by Individual Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "umap_by_feature.png", dpi=150)
    plt.close(fig)
    log.info("Saved umap_by_feature.png")


def save_umap_coordinates(
    embedding: np.ndarray,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save UMAP coordinates for reuse."""
    method = "umap" if HAS_UMAP else "tsne"
    coord_df = pd.DataFrame({
        "id": df["id"].values if "id" in df.columns else range(len(df)),
        f"{method}_x": embedding[:, 0],
        f"{method}_y": embedding[:, 1],
        "syllable_type": df["syllable_type"].values,
    })
    coord_df.to_csv(output_dir / "umap_coordinates.csv", index=False)
    log.info("Saved umap_coordinates.csv (%d rows)", len(coord_df))


# ── Analysis 4: Within-type violin plots ──────────────────────────────────

def plot_within_type_violins(
    df: pd.DataFrame,
    features_raw: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    """Violin plots per feature split by syllable type, with threshold lines."""
    log.info("Generating within-type violin plots...")
    fig, axes = plt.subplots(2, 5, figsize=(28, 12))

    # Compute coefficient of variation per type per feature for summary
    type_cv = {}

    for ax, feat in zip(axes.flat, ACOUSTIC_FEATURES):
        plot_df = pd.DataFrame({
            "value": features_raw[feat].values,
            "type": df["syllable_type"].values,
        })

        sns.violinplot(
            data=plot_df,
            x="type",
            y="value",
            hue="type",
            order=TYPE_ORDER,
            hue_order=TYPE_ORDER,
            palette=TYPE_COLORS,
            inner="quartile",
            cut=0,
            ax=ax,
            density_norm="width",
            legend=False,
        )

        # Add threshold reference lines
        if feat in THRESHOLDS:
            for label, val in THRESHOLDS[feat]:
                ax.axhline(y=val, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
                ax.text(
                    len(TYPE_ORDER) - 0.5, val, f"  {label}",
                    color="red", fontsize=7, va="bottom",
                )

        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

        # Track within-type CV for the loosest type
        for stype in TYPE_ORDER:
            vals = features_raw.loc[df["syllable_type"] == stype, feat]
            if len(vals) > 1 and vals.mean() != 0:
                cv = vals.std() / abs(vals.mean())
                key = f"{feat}|{stype}"
                type_cv[key] = cv

    fig.suptitle(
        "Within-Type Feature Distributions (red dashed = classification thresholds)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "within_type_violins.png", dpi=150)
    plt.close(fig)
    log.info("Saved within_type_violins.png")
    return type_cv


# ── Analysis 5: Boundary case analysis ────────────────────────────────────

def analyze_boundary_cases(
    df: pd.DataFrame,
    features_raw: pd.DataFrame,
    embedding: np.ndarray,
    output_dir: Path,
) -> dict:
    """Analyze low-confidence boundary cases."""
    log.info("Analyzing boundary cases...")
    is_low = df["classification_confidence"] == "low"
    n_low = is_low.sum()
    n_total = len(df)
    log.info("Low-confidence calls: %d / %d (%.1f%%)", n_low, n_total, 100 * n_low / n_total)

    method = "UMAP" if HAS_UMAP else "t-SNE"

    fig = plt.figure(figsize=(18, 10))

    # Panel 1: UMAP with low-confidence highlighted
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(
        embedding[~is_low, 0],
        embedding[~is_low, 1],
        c="lightgray",
        alpha=0.2,
        s=5,
        label=f"High/medium (n={n_total - n_low})",
        rasterized=True,
    )
    # Color low-confidence by type
    for stype in TYPE_ORDER:
        mask = is_low & (df["syllable_type"] == stype).values
        if mask.sum() > 0:
            ax1.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=TYPE_COLORS[stype],
                alpha=0.6,
                s=15,
                label=f"{stype} low (n={mask.sum()})",
                rasterized=True,
            )
    ax1.set_xlabel(f"{method} 1")
    ax1.set_ylabel(f"{method} 2")
    ax1.set_title(f"Low-Confidence Calls on {method} (colored by type)")
    ax1.legend(loc="upper right", fontsize=8, markerscale=2)
    ax1.grid(alpha=0.2)

    # Panel 2: Feature distributions — low vs high/medium confidence
    key_features = ["call_length_s", "slope", "sinuosity", "bandwidth_hz"]
    ax_grid = [fig.add_subplot(2, 4, i) for i in range(5, 9)]

    for ax, feat in zip(ax_grid, key_features):
        vals_high = features_raw.loc[~is_low, feat]
        vals_low = features_raw.loc[is_low, feat]
        ax.hist(vals_high, bins=50, alpha=0.5, density=True, color="steelblue", label="High/Med")
        ax.hist(vals_low, bins=50, alpha=0.5, density=True, color="crimson", label="Low")
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    fig.suptitle("Boundary Case Analysis — Low-Confidence Classifications", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_cases.png", dpi=150)
    plt.close(fig)
    log.info("Saved boundary_cases.png")

    # Summary stats
    low_by_type = df.loc[is_low, "syllable_type"].value_counts().to_dict()
    return {
        "n_low": n_low,
        "n_total": n_total,
        "pct_low": 100 * n_low / n_total,
        "low_by_type": low_by_type,
    }


# ── Analysis 6: Auto-generated summary ───────────────────────────────────

def write_summary(
    output_dir: Path,
    corr: pd.DataFrame,
    explained_var: np.ndarray,
    loadings_df: pd.DataFrame,
    type_cv: dict[str, float],
    boundary_stats: dict,
    n_calls: int,
) -> None:
    """Write analysis_summary.md with key findings."""
    log.info("Writing analysis summary...")

    # Strong correlations (|r| > 0.7, excluding diagonal)
    strong_pairs = []
    raw_labels = corr.index.tolist()
    for i in range(len(raw_labels)):
        for j in range(i + 1, len(raw_labels)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                strong_pairs.append((raw_labels[i], raw_labels[j], r))

    # PCA top loadings
    pc1_loadings = loadings_df["PC1"].abs().sort_values(ascending=False)
    pc2_loadings = loadings_df["PC2"].abs().sort_values(ascending=False)

    # Loosest types (highest mean CV)
    type_mean_cv = {}
    for key, cv in type_cv.items():
        _, stype = key.split("|")
        type_mean_cv.setdefault(stype, []).append(cv)
    type_mean_cv = {k: np.mean(v) for k, v in type_mean_cv.items()}
    loosest = sorted(type_mean_cv.items(), key=lambda x: x[1], reverse=True)

    method = "UMAP" if HAS_UMAP else "t-SNE"

    summary = textwrap.dedent(f"""\
    # A3: Acoustic Feature Deep-Dive — Summary

    **Dataset:** {n_calls} classified USV calls (7 traditional types)
    **Features:** {len(ACOUSTIC_FEATURES)} acoustic features, standardized for PCA/{method}
    **Embedding:** {"UMAP (n_neighbors=15, min_dist=0.1)" if HAS_UMAP else "t-SNE (perplexity=30)"}

    ## 1. Feature Correlations

    **Strong correlations (|r| > 0.7):** {len(strong_pairs)} pairs
    """)

    if strong_pairs:
        for f1, f2, r in sorted(strong_pairs, key=lambda x: abs(x[2]), reverse=True):
            summary += f"- {f1} <-> {f2}: r = {r:.3f}\n"
    else:
        summary += "- No feature pairs with |r| > 0.7\n"

    cum2 = (explained_var[0] + explained_var[1]) * 100
    summary += textwrap.dedent(f"""
    ## 2. PCA Results

    **Variance explained by PC1 + PC2:** {cum2:.1f}%
    - PC1 ({explained_var[0]*100:.1f}%): dominated by {pc1_loadings.index[0]} ({FEATURE_LABELS.get(pc1_loadings.index[0], pc1_loadings.index[0])})
    - PC2 ({explained_var[1]*100:.1f}%): dominated by {pc2_loadings.index[0]} ({FEATURE_LABELS.get(pc2_loadings.index[0], pc2_loadings.index[0])})

    **Top PC1 loadings:**
    """)
    for feat in pc1_loadings.index[:3]:
        summary += f"- {FEATURE_LABELS.get(feat, feat)}: {loadings_df.loc[feat, 'PC1']:.3f}\n"

    summary += "\n**Top PC2 loadings:**\n"
    for feat in pc2_loadings.index[:3]:
        summary += f"- {FEATURE_LABELS.get(feat, feat)}: {loadings_df.loc[feat, 'PC2']:.3f}\n"

    summary += textwrap.dedent(f"""
    ## 3. {method} Embedding

    See `umap_by_type.png` and `umap_by_feature.png` for visual assessment of
    whether the 7 traditional types map onto discrete clusters or a continuum.

    ## 4. Within-Type Variability

    **Loosest types (highest mean coefficient of variation):**
    """)
    for stype, cv in loosest[:3]:
        summary += f"- {stype}: mean CV = {cv:.3f}\n"

    summary += "\n**Tightest types:**\n"
    for stype, cv in loosest[-3:]:
        summary += f"- {stype}: mean CV = {cv:.3f}\n"

    bs = boundary_stats
    summary += textwrap.dedent(f"""
    ## 5. Boundary Cases

    **Low-confidence calls:** {bs['n_low']} / {bs['n_total']} ({bs['pct_low']:.1f}%)

    **Leakiest type boundaries (most low-confidence calls):**
    """)
    for stype, count in sorted(bs["low_by_type"].items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / bs["n_low"]
        summary += f"- {stype}: {count} ({pct:.1f}% of all low-confidence)\n"

    summary += textwrap.dedent("""
    ## Interpretation Guide

    - If violin plots show threshold lines cutting through density peaks rather than
      valleys, the taxonomy boundary is arbitrary at that point.
    - If UMAP shows smooth color gradients rather than discrete patches for types,
      the acoustic space is a continuum (consistent with Goffinet et al. 2021).
    - High within-type CV suggests that type is a catch-all rather than a coherent category.
    - Many low-confidence calls at a type boundary suggest the rule cascade is ambiguous there.
    """)

    (output_dir / "analysis_summary.md").write_text(summary)
    log.info("Saved analysis_summary.md")


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "traditional_taxonomy" / "classified_traditional.csv",
        help="Input CSV with classified calls",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "acoustic_feature_analysis",
        help="Output directory for plots and tables",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", args.output_dir)

    # Step 1: Load and prepare
    df, features_raw, features_scaled = load_and_prepare(args.input)

    # Step 2: Correlation matrix
    corr = plot_correlation_matrix(features_raw, args.output_dir)

    # Step 3: PCA
    explained_var, loadings_df = run_pca(df, features_scaled, args.output_dir)

    # Step 4: UMAP
    embedding = compute_umap(features_scaled)
    plot_umap_by_type(embedding, df, args.output_dir)
    plot_umap_by_features(embedding, features_raw, args.output_dir)
    save_umap_coordinates(embedding, df, args.output_dir)

    # Step 5: Within-type violins
    type_cv = plot_within_type_violins(df, features_raw, args.output_dir)

    # Step 6: Boundary cases
    boundary_stats = analyze_boundary_cases(df, features_raw, embedding, args.output_dir)

    # Step 7: Summary
    write_summary(
        args.output_dir,
        corr,
        explained_var,
        loadings_df,
        type_cv,
        boundary_stats,
        n_calls=len(df),
    )

    log.info("A3 analysis complete. %d files in %s", len(list(args.output_dir.iterdir())), args.output_dir)


if __name__ == "__main__":
    main()
