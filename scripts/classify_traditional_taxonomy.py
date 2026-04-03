#!/usr/bin/env python3
"""Classify USV calls into traditional syllable types using rule-based taxonomy.

Applies Holy & Guo (2005) / Scattoni et al. (2008) syllable categories using
deterministic rules on acoustic features from DeepSqueak. No ML or training data
required — pure feature thresholds calibrated to the data distributions.

Traditional types (priority-ordered cascade):
  1. Short    — duration < 15ms
  2. Complex  — sinuosity > 3.5 (multiple direction changes)
  3. Chevron  — sinuosity > 1.8 AND bandwidth > 25 kHz (inverted-U)
  4. Freq Jump — bandwidth > 55 kHz AND sinuosity < 1.8 (abrupt step)
  5. Up       — slope > 200 (rising frequency)
  6. Down     — slope < -200 (falling frequency)
  7. Flat     — default (low slope, low sinuosity)

References:
  - Holy & Guo (2005) — ultrasonic songs of male mice, Nature
  - Scattoni et al. (2008) — USV repertoire in infant mice
  - Grimsley et al. (2011) — temporal lobe responses to mouse USVs

Output: results/traditional_taxonomy/
  - classified_traditional.csv (all original columns + syllable_type, classification_confidence)
  - type_distribution.png (bar chart)
  - feature_summary.png (mean ± std per type)
  - cluster_vs_type_heatmap.png (cross-tabulation)
  - gallery/<Type>/*.png (spectrogram examples per type)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import spectrogram rendering from cluster gallery
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from generate_cluster_gallery import build_wav_lookup, render_call_spectrogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Classification thresholds ──────────────────────────────────────────────
# Calibrated against classified_detections_full.csv distributions (7,921 rows).
# Note: frequency columns are in kHz despite "_hz" suffix in column names.

THRESH_SHORT_DURATION_S = 0.015    # 15ms — natural break near 10th–15th percentile
THRESH_COMPLEX_SINUOSITY = 3.5     # Between 75th (2.0) and 90th (3.6) percentile
THRESH_CHEVRON_SINUOSITY = 1.8     # Moderate sinuosity
THRESH_CHEVRON_BANDWIDTH = 25.0    # kHz — moderate bandwidth for inverted-U shape
THRESH_FREQJUMP_BANDWIDTH = 55.0   # kHz — ~80th percentile, abrupt step
THRESH_FREQJUMP_SINUOSITY = 1.8    # Low sinuosity (step, not modulation)
THRESH_SLOPE_DIRECTIONAL = 200.0   # Separates up/down from flat

# Confidence margins (fraction of threshold to consider "borderline")
CONFIDENCE_MARGIN = 0.20  # 20% of threshold


def classify_call(row: pd.Series) -> tuple[str, str]:
    """Classify a single USV call into a traditional syllable type.

    Returns (syllable_type, confidence) where confidence is
    'high', 'medium', or 'low'.
    """
    duration = row.get("call_length_s")
    slope = row.get("slope")
    sinuosity = row.get("sinuosity")
    bandwidth = row.get("bandwidth_hz")

    # NaN guard — can't classify without features
    if any(pd.isna(v) for v in [duration, slope, sinuosity, bandwidth]):
        return ("unclassified", "none")

    # Priority 1: Short
    if duration < THRESH_SHORT_DURATION_S:
        conf = _confidence(duration, THRESH_SHORT_DURATION_S, below=True)
        return ("Short", conf)

    # Priority 2: Complex (high sinuosity)
    if sinuosity > THRESH_COMPLEX_SINUOSITY:
        conf = _confidence(sinuosity, THRESH_COMPLEX_SINUOSITY, below=False)
        return ("Complex", conf)

    # Priority 3: Chevron (moderate sinuosity + wide bandwidth)
    if sinuosity > THRESH_CHEVRON_SINUOSITY and bandwidth > THRESH_CHEVRON_BANDWIDTH:
        # Confidence based on how clearly both criteria are met
        sin_conf = _confidence(sinuosity, THRESH_CHEVRON_SINUOSITY, below=False)
        bw_conf = _confidence(bandwidth, THRESH_CHEVRON_BANDWIDTH, below=False)
        conf = _min_confidence(sin_conf, bw_conf)
        return ("Chevron", conf)

    # Priority 4: Frequency Jump (wide bandwidth, low sinuosity)
    if bandwidth > THRESH_FREQJUMP_BANDWIDTH and sinuosity < THRESH_FREQJUMP_SINUOSITY:
        conf = _confidence(bandwidth, THRESH_FREQJUMP_BANDWIDTH, below=False)
        return ("Frequency_Jump", conf)

    # Priority 5: Up (positive slope)
    if slope > THRESH_SLOPE_DIRECTIONAL:
        conf = _confidence(slope, THRESH_SLOPE_DIRECTIONAL, below=False)
        return ("Up", conf)

    # Priority 6: Down (negative slope)
    if slope < -THRESH_SLOPE_DIRECTIONAL:
        conf = _confidence(abs(slope), THRESH_SLOPE_DIRECTIONAL, below=False)
        return ("Down", conf)

    # Priority 7: Flat (default)
    # Confidence is higher when slope is very low and sinuosity is very low
    abs_slope = abs(slope)
    if abs_slope < THRESH_SLOPE_DIRECTIONAL * (1 - CONFIDENCE_MARGIN) and sinuosity < 1.3:
        conf = "high"
    elif abs_slope < THRESH_SLOPE_DIRECTIONAL * (1 - CONFIDENCE_MARGIN * 0.5):
        conf = "medium"
    else:
        conf = "low"
    return ("Flat", conf)


def _confidence(value: float, threshold: float, below: bool) -> str:
    """Determine confidence based on distance from threshold."""
    margin = threshold * CONFIDENCE_MARGIN
    if below:
        distance = threshold - value
    else:
        distance = value - threshold
    if distance > margin * 2:
        return "high"
    elif distance > margin * 0.5:
        return "medium"
    else:
        return "low"


def _min_confidence(a: str, b: str) -> str:
    """Return the lower of two confidence levels."""
    order = {"high": 2, "medium": 1, "low": 0}
    reverse = {2: "high", 1: "medium", 0: "low"}
    return reverse[min(order[a], order[b])]


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply classification to entire DataFrame."""
    if df.empty:
        return df.assign(syllable_type=pd.Series(dtype=str), classification_confidence=pd.Series(dtype=str))
    results = df.apply(classify_call, axis=1, result_type="expand")
    results.columns = ["syllable_type", "classification_confidence"]
    return pd.concat([df, results], axis=1)


# ── Figure generation ──────────────────────────────────────────────────────

TYPE_ORDER = ["Short", "Flat", "Up", "Down", "Chevron", "Complex", "Frequency_Jump", "unclassified"]
TYPE_COLORS = {
    "Short": "#e74c3c",
    "Flat": "#3498db",
    "Up": "#2ecc71",
    "Down": "#f39c12",
    "Chevron": "#9b59b6",
    "Complex": "#e67e22",
    "Frequency_Jump": "#1abc9c",
    "unclassified": "#95a5a6",
}


def generate_distribution_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart showing syllable type distribution."""
    counts = df["syllable_type"].value_counts()
    # Reorder by TYPE_ORDER, keeping only present types
    types_present = [t for t in TYPE_ORDER if t in counts.index]
    counts = counts[types_present]
    total = len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [TYPE_COLORS.get(t, "#999999") for t in types_present]
    bars = ax.bar(range(len(types_present)), counts.values, color=colors, edgecolor="white", linewidth=0.5)

    # Add count and percentage labels
    for i, (bar, count) in enumerate(zip(bars, counts.values)):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(types_present)))
    ax.set_xticklabels(types_present, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Traditional Syllable Type Distribution (n={total})")
    ax.set_ylim(0, counts.max() * 1.15)
    sns.despine()

    fig.tight_layout()
    fig.savefig(output_dir / "type_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved type_distribution.png")


def generate_feature_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Table figure showing mean +/- std of key features per type."""
    features = ["call_length_s", "principal_freq_hz", "slope", "sinuosity", "bandwidth_hz", "tonality"]
    feature_labels = ["Duration (s)", "Principal Freq (kHz)", "Slope", "Sinuosity", "Bandwidth (kHz)", "Tonality"]

    types_present = [t for t in TYPE_ORDER if t in df["syllable_type"].values and t != "unclassified"]

    # Build summary table
    rows = []
    for stype in types_present:
        subset = df[df["syllable_type"] == stype]
        row = {"Type": stype, "n": len(subset)}
        for feat, label in zip(features, feature_labels):
            vals = subset[feat].dropna()
            row[label] = f"{vals.mean():.2f} +/- {vals.std():.2f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Render as table figure
    fig, ax = plt.subplots(figsize=(14, len(types_present) * 0.6 + 1.5))
    ax.axis("off")
    ax.set_title("Acoustic Feature Summary by Traditional Syllable Type", fontsize=12, pad=20)

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(list(range(len(summary_df.columns))))

    # Color header
    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", weight="bold")

    # Color type column
    for i in range(len(types_present)):
        table[i + 1, 0].set_facecolor(TYPE_COLORS.get(types_present[i], "#eeeeee"))
        table[i + 1, 0].set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved feature_summary.png")

    # Also save as CSV for programmatic use
    summary_df.to_csv(output_dir / "feature_summary.csv", index=False)


def generate_cross_tabulation(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap showing DeepSqueak cluster vs traditional type."""
    classified = df.dropna(subset=["label", "syllable_type"])
    if classified.empty:
        log.warning("No data for cross-tabulation")
        return

    ct = pd.crosstab(classified["label"], classified["syllable_type"])

    # Sort clusters numerically
    cluster_order = sorted(ct.index, key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
    type_cols = [t for t in TYPE_ORDER if t in ct.columns]
    ct = ct.reindex(index=cluster_order, columns=type_cols, fill_value=0)

    # Normalize by row to show proportions within each cluster
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(cluster_order) * 0.35)))

    # Raw counts
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax1, linewidths=0.5, cbar_kws={"shrink": 0.6})
    ax1.set_title("Counts: DeepSqueak Cluster vs Traditional Type")
    ax1.set_ylabel("DeepSqueak Cluster")
    ax1.set_xlabel("Traditional Syllable Type")

    # Proportions
    sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2, linewidths=0.5, cbar_kws={"shrink": 0.6})
    ax2.set_title("Proportions (row-normalized)")
    ax2.set_ylabel("")
    ax2.set_xlabel("Traditional Syllable Type")

    fig.suptitle("Cross-Tabulation: k-means Clusters vs Traditional Taxonomy", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_vs_type_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cluster_vs_type_heatmap.png")

    # Also save raw cross-tab as CSV
    ct.to_csv(output_dir / "cluster_vs_type_crosstab.csv")


def generate_gallery(
    df: pd.DataFrame,
    wav_lookup: dict[str, Path],
    output_dir: Path,
    n_per_type: int,
    seed: int,
) -> None:
    """Generate spectrogram gallery PNGs organized by syllable type."""
    rng = np.random.default_rng(seed)
    gallery_dir = output_dir / "gallery"
    total_generated = 0

    types_present = [t for t in TYPE_ORDER if t in df["syllable_type"].values and t != "unclassified"]

    for stype in types_present:
        type_df = df[df["syllable_type"] == stype]
        available = type_df[type_df["wav_stem"].isin(wav_lookup)]

        if len(available) == 0:
            log.warning(f"{stype}: no WAV files available, skipping gallery")
            continue

        n_sample = min(n_per_type, len(available))
        sample = available.sample(n=n_sample, random_state=rng.integers(0, 2**31))

        type_dir = gallery_dir / stype
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

            conf = row["classification_confidence"]
            fname = f"{idx + 1:02d}_{row['wav_stem']}_{begin_s:.3f}s.png"
            out_path = type_dir / fname

            title = f"{stype} ({conf}) | {row['wav_stem']} @ {begin_s:.3f}s"
            ok = render_call_spectrogram(wav_path, begin_s, end_s, out_path, title, freq_info)
            if ok:
                generated += 1

        total_generated += generated
        log.info(f"{stype}: {generated}/{n_sample} gallery PNGs (from {len(type_df)} calls)")

    log.info(f"Gallery complete: {total_generated} PNGs in {gallery_dir}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classify USV calls into traditional syllable types (Holy & Guo 2005)",
    )
    parser.add_argument(
        "--csv", default=str(REPO_ROOT / "classified_detections_full.csv"),
        help="Path to classified_detections_full.csv",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "results" / "traditional_taxonomy"),
        help="Output directory",
    )
    parser.add_argument("--n-per-type", type=int, default=5, help="Gallery examples per type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-gallery", action="store_true", help="Skip gallery PNG generation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and classify
    log.info(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    log.info("Applying traditional taxonomy classification...")
    df = classify_dataframe(df)

    # Summary stats
    type_counts = df["syllable_type"].value_counts()
    total = len(df)
    log.info("=" * 60)
    log.info("SYLLABLE TYPE DISTRIBUTION")
    log.info("=" * 60)
    for stype in TYPE_ORDER:
        if stype in type_counts.index:
            count = type_counts[stype]
            pct = count / total * 100
            log.info(f"  {stype:18s}  {count:5d}  ({pct:5.1f}%)")
    log.info(f"  {'TOTAL':18s}  {total:5d}")

    # Confidence breakdown
    conf_counts = df["classification_confidence"].value_counts()
    log.info("\nCONFIDENCE BREAKDOWN")
    for conf in ["high", "medium", "low", "none"]:
        if conf in conf_counts.index:
            log.info(f"  {conf:10s}  {conf_counts[conf]:5d}  ({conf_counts[conf] / total * 100:5.1f}%)")

    # Save classified CSV
    csv_path = output_dir / "classified_traditional.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nSaved classified CSV: {csv_path}")

    # Generate figures
    log.info("\nGenerating figures...")
    generate_distribution_chart(df, output_dir)
    generate_feature_summary(df, output_dir)
    generate_cross_tabulation(df, output_dir)

    # Gallery
    if not args.skip_gallery:
        log.info("\nGenerating gallery PNGs...")
        search_dirs = [
            REPO_ROOT / "5970_reviewed",
            REPO_ROOT / "5970 USV",
        ]
        wav_lookup = build_wav_lookup(search_dirs)
        log.info(f"Found {len(wav_lookup)} WAV stems")
        generate_gallery(df, wav_lookup, output_dir, args.n_per_type, args.seed)
    else:
        log.info("\nSkipping gallery (--skip-gallery)")

    log.info("\nDone!")

    # Validation checks
    n_nan = df["syllable_type"].isna().sum()
    if n_nan > 0:
        log.error(f"VALIDATION FAIL: {n_nan} rows have NaN syllable_type")
    types_above_5pct = sum(1 for t in type_counts.index if type_counts[t] / total > 0.05 and t != "unclassified")
    if types_above_5pct < 4:
        log.warning(f"VALIDATION WARNING: Only {types_above_5pct} types have > 5% of calls (expected >= 4)")


if __name__ == "__main__":
    main()
