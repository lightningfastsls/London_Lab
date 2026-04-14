#!/usr/bin/env python3
"""Phase A1: Temporal dynamics analysis of USV call data.

Analyzes the 5970 dataset (usv_lmt_034) for temporal patterns:
- Call rate over time (hourly bins)
- Syllable type composition shifts
- Bout detection via inter-call intervals
- Within-bout structure (first-call type distribution)

Outputs figures and summary CSV to results/temporal_dynamics/.
"""

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ── Data loading & timestamp parsing ──────────────────────────────────────

def parse_filename_timestamp(filename: str) -> datetime:
    """Extract datetime from WAV filename like '2024-09-30_11-18-17_0000001'."""
    match = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_(\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse timestamp from filename: {filename}")
    date_str = match.group(1)
    h, m, s = match.group(2), match.group(3), match.group(4)
    return datetime.strptime(f"{date_str} {h}:{m}:{s}", "%Y-%m-%d %H:%M:%S")


def load_and_enrich(csv_path: str) -> pd.DataFrame:
    """Load classified CSV and add absolute_time column."""
    df = pd.read_csv(csv_path)

    # Drop rows with missing file identifiers
    n_before = len(df)
    df = df.dropna(subset=["file"]).copy()
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing 'file' values")

    # Parse filename timestamps and compute absolute time
    file_timestamps = df["file"].map(parse_filename_timestamp)
    df["file_datetime"] = file_timestamps
    df["absolute_time"] = file_timestamps + pd.to_timedelta(df["begin_time_s"], unit="s")

    df = df.sort_values("absolute_time").reset_index(drop=True)

    # Summary
    t0, t1 = df["absolute_time"].iloc[0], df["absolute_time"].iloc[-1]
    span = t1 - t0
    print(f"Loaded {len(df)} calls from {df['file'].nunique()} files")
    print(f"Timeline: {t0} → {t1} ({span})")
    print(f"Syllable types: {df['syllable_type'].value_counts().to_dict()}")

    return df


# ── Analysis functions ────────────────────────────────────────────────────

def call_rate_over_time(df: pd.DataFrame, bin_minutes: int = 60) -> pd.DataFrame:
    """Bin calls into time windows and count per bin."""
    df = df.copy()
    df["time_bin"] = df["absolute_time"].dt.floor(f"{bin_minutes}min")
    rate = df.groupby("time_bin").size().reset_index(name="call_count")

    # Fill gaps where no calls occurred
    full_range = pd.date_range(
        rate["time_bin"].min(),
        rate["time_bin"].max(),
        freq=f"{bin_minutes}min",
    )
    rate = rate.set_index("time_bin").reindex(full_range, fill_value=0).reset_index()
    rate.columns = ["time_bin", "call_count"]
    return rate


def type_composition_over_time(
    df: pd.DataFrame, bin_minutes: int = 60
) -> pd.DataFrame:
    """Syllable type proportions per time bin."""
    df = df.copy()
    df["time_bin"] = df["absolute_time"].dt.floor(f"{bin_minutes}min")
    counts = df.groupby(["time_bin", "syllable_type"]).size().unstack(fill_value=0)

    # Fill gaps
    full_range = pd.date_range(
        counts.index.min(), counts.index.max(), freq=f"{bin_minutes}min"
    )
    counts = counts.reindex(full_range, fill_value=0)

    # Convert to proportions (avoid div by zero for empty bins)
    row_sums = counts.sum(axis=1)
    proportions = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    return proportions


def detect_bouts(
    df: pd.DataFrame, threshold_s: float | None = None
) -> tuple[pd.DataFrame, dict]:
    """Detect bouts as clusters of calls separated by gaps > threshold.

    If threshold_s is None, uses median(ICI) * 3 as a heuristic
    (more conservative than mean/2 from burstiness_coefficient).
    """
    times = df["absolute_time"].values.astype("datetime64[ns]").astype(np.float64) / 1e9
    icis = np.diff(times)

    if threshold_s is None:
        # Use 3x median ICI — robust to outliers from long silent gaps
        threshold_s = float(np.median(icis) * 3)
        print(f"Bout threshold (auto): {threshold_s:.1f}s (3× median ICI)")

    # Assign bout IDs: new bout starts when gap > threshold
    bout_ids = np.zeros(len(df), dtype=int)
    bout_id = 0
    bout_ids[0] = bout_id
    for i, ici in enumerate(icis):
        if ici > threshold_s:
            bout_id += 1
        bout_ids[i + 1] = bout_id

    df = df.copy()
    df["bout_id"] = bout_ids

    # Bout statistics
    bout_stats = df.groupby("bout_id").agg(
        n_calls=("absolute_time", "size"),
        start=("absolute_time", "min"),
        end=("absolute_time", "max"),
        first_type=("syllable_type", "first"),
    )
    bout_stats["duration_s"] = (
        bout_stats["end"] - bout_stats["start"]
    ).dt.total_seconds()

    inter_bout = np.diff(
        bout_stats["start"].values.astype("datetime64[ns]").astype(np.float64) / 1e9
    )

    summary = {
        "n_bouts": len(bout_stats),
        "threshold_s": threshold_s,
        "mean_bout_calls": bout_stats["n_calls"].mean(),
        "median_bout_calls": bout_stats["n_calls"].median(),
        "mean_bout_duration_s": bout_stats["duration_s"].mean(),
        "median_bout_duration_s": bout_stats["duration_s"].median(),
        "mean_inter_bout_s": float(np.mean(inter_bout)) if len(inter_bout) > 0 else 0,
        "median_inter_bout_s": float(np.median(inter_bout)) if len(inter_bout) > 0 else 0,
        "single_call_bouts": int((bout_stats["n_calls"] == 1).sum()),
        "single_call_bout_pct": float((bout_stats["n_calls"] == 1).mean() * 100),
    }

    return df, bout_stats, summary


def ici_distribution(df: pd.DataFrame) -> np.ndarray:
    """Compute inter-call intervals in seconds."""
    times = df["absolute_time"].values.astype("datetime64[ns]").astype(np.float64) / 1e9
    return np.diff(times)


# ── Plotting ──────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "Flat": "#4477AA",
    "Down": "#EE6677",
    "Chevron": "#228833",
    "Short": "#CCBB44",
    "Complex": "#AA3377",
    "Frequency_Jump": "#66CCEE",
    "Up": "#BBBBBB",
}


def plot_call_rate(rate: pd.DataFrame, output_dir: Path):
    """Call rate timeline (calls per hour)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(rate["time_bin"], rate["call_count"], width=timedelta(minutes=55),
           color="#4477AA", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Time")
    ax.set_ylabel("Calls per hour")
    ax.set_title("USV Call Rate Over Time (5970 / lmt_034)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "call_rate_hourly.png", dpi=150)
    plt.close(fig)
    print(f"  Saved call_rate_hourly.png")


def plot_type_composition(proportions: pd.DataFrame, output_dir: Path):
    """Stacked area chart of syllable type proportions."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Order types by overall frequency (most common at bottom)
    type_order = proportions.sum().sort_values(ascending=False).index.tolist()
    colors = [TYPE_COLORS.get(t, "#999999") for t in type_order]

    ax.stackplot(
        proportions.index,
        [proportions[t].values for t in type_order],
        labels=type_order,
        colors=colors,
        alpha=0.85,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Proportion")
    ax.set_title("Syllable Type Composition Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "type_composition_hourly.png", dpi=150)
    plt.close(fig)
    print(f"  Saved type_composition_hourly.png")


def plot_ici_histogram(icis: np.ndarray, threshold_s: float, output_dir: Path):
    """Inter-call interval distribution with bout threshold marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale (clipped to show structure)
    clip = min(60, np.percentile(icis, 99))
    axes[0].hist(icis[icis < clip], bins=100, color="#4477AA", alpha=0.8,
                 edgecolor="none")
    axes[0].axvline(threshold_s, color="red", linestyle="--", linewidth=1.5,
                    label=f"Bout threshold ({threshold_s:.1f}s)")
    axes[0].set_xlabel("Inter-call interval (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("ICI Distribution (linear, clipped)")
    axes[0].legend(fontsize=8)

    # Log scale (full range)
    axes[1].hist(icis, bins=np.logspace(np.log10(0.01), np.log10(icis.max()), 100),
                 color="#4477AA", alpha=0.8, edgecolor="none")
    axes[1].axvline(threshold_s, color="red", linestyle="--", linewidth=1.5,
                    label=f"Bout threshold ({threshold_s:.1f}s)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Inter-call interval (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("ICI Distribution (log scale)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "ici_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ici_distribution.png")


def plot_bout_structure(bout_stats: pd.DataFrame, df_with_bouts: pd.DataFrame,
                        output_dir: Path):
    """Bout size distribution and first-call type distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Bout size distribution
    max_display = int(bout_stats["n_calls"].quantile(0.98))
    axes[0].hist(bout_stats["n_calls"].clip(upper=max_display), bins=50,
                 color="#228833", alpha=0.8, edgecolor="none")
    axes[0].set_xlabel("Calls per bout")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Bout Size Distribution (N={len(bout_stats)})")

    # First-call type distribution within multi-call bouts
    multi_call_bouts = bout_stats[bout_stats["n_calls"] > 1]
    first_counts = multi_call_bouts["first_type"].value_counts()
    overall_counts = df_with_bouts["syllable_type"].value_counts(normalize=True)

    # Normalize to compare
    first_norm = first_counts / first_counts.sum()
    type_order = overall_counts.sort_values(ascending=False).index
    colors = [TYPE_COLORS.get(t, "#999999") for t in type_order]

    x = np.arange(len(type_order))
    w = 0.35
    axes[1].bar(x - w/2, [overall_counts.get(t, 0) for t in type_order], w,
                label="Overall", color=colors, alpha=0.5, edgecolor="none")
    axes[1].bar(x + w/2, [first_norm.get(t, 0) for t in type_order], w,
                label="Bout-initial", color=colors, alpha=1.0, edgecolor="black",
                linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(type_order, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Bout-Initial vs Overall Type Distribution")
    axes[1].legend(fontsize=8)

    # Bout duration distribution
    multi_durations = multi_call_bouts["duration_s"]
    max_dur = multi_durations.quantile(0.98)
    axes[2].hist(multi_durations.clip(upper=max_dur), bins=50,
                 color="#EE6677", alpha=0.8, edgecolor="none")
    axes[2].set_xlabel("Bout duration (s)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Bout Duration (multi-call, N={len(multi_call_bouts)})")

    fig.tight_layout()
    fig.savefig(output_dir / "bout_structure.png", dpi=150)
    plt.close(fig)
    print(f"  Saved bout_structure.png")


def plot_call_raster(df: pd.DataFrame, output_dir: Path):
    """Temporal raster — each call as a tick mark, colored by type."""
    fig, ax = plt.subplots(figsize=(14, 2.5))
    for stype, group in df.groupby("syllable_type"):
        color = TYPE_COLORS.get(stype, "#999999")
        ax.scatter(group["absolute_time"], [1] * len(group),
                   c=color, s=0.3, alpha=0.4, label=stype, marker="|")
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Call Raster (5970 / lmt_034)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=7, ncol=4,
              markerscale=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "call_raster.png", dpi=150)
    plt.close(fig)
    print(f"  Saved call_raster.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase A1: Temporal dynamics analysis")
    parser.add_argument(
        "--input",
        default="results/traditional_taxonomy/classified_traditional.csv",
        help="Path to classified CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="results/temporal_dynamics",
        help="Output directory for figures and stats",
    )
    parser.add_argument(
        "--bin-minutes",
        type=int,
        default=60,
        help="Time bin size in minutes (default: 60)",
    )
    parser.add_argument(
        "--bout-threshold",
        type=float,
        default=None,
        help="Bout gap threshold in seconds (default: auto = 3× median ICI)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_and_enrich(args.input)

    # Call rate
    print("\nComputing call rate...")
    rate = call_rate_over_time(df, bin_minutes=args.bin_minutes)
    peak = rate.loc[rate["call_count"].idxmax()]
    print(f"  Peak hour: {peak['time_bin']} ({peak['call_count']} calls)")
    quiet_hours = (rate["call_count"] == 0).sum()
    print(f"  Silent hours: {quiet_hours}/{len(rate)}")

    # Type composition
    print("\nComputing type composition...")
    proportions = type_composition_over_time(df, bin_minutes=args.bin_minutes)

    # ICI & bout detection
    print("\nAnalyzing inter-call intervals...")
    icis = ici_distribution(df)
    print(f"  Median ICI: {np.median(icis):.2f}s, Mean ICI: {np.mean(icis):.2f}s")
    print(f"  Min ICI: {np.min(icis):.4f}s, Max ICI: {np.max(icis):.1f}s")

    print("\nDetecting bouts...")
    df_bouts, bout_stats, bout_summary = detect_bouts(df, threshold_s=args.bout_threshold)

    # Generate figures
    print("\nGenerating figures...")
    plot_call_rate(rate, output_dir)
    plot_type_composition(proportions, output_dir)
    plot_ici_histogram(icis, bout_summary["threshold_s"], output_dir)
    plot_bout_structure(bout_stats, df_bouts, output_dir)
    plot_call_raster(df, output_dir)

    # Save summary stats
    summary_rows = [
        ("total_calls", len(df)),
        ("total_files", df["file"].nunique()),
        ("timeline_start", df["absolute_time"].iloc[0]),
        ("timeline_end", df["absolute_time"].iloc[-1]),
        ("timeline_hours", (df["absolute_time"].iloc[-1] - df["absolute_time"].iloc[0]).total_seconds() / 3600),
        ("bin_minutes", args.bin_minutes),
        ("peak_hour", peak["time_bin"]),
        ("peak_hour_calls", peak["call_count"]),
        ("silent_hours", quiet_hours),
        ("median_ici_s", np.median(icis)),
        ("mean_ici_s", np.mean(icis)),
    ]
    for k, v in bout_summary.items():
        summary_rows.append((k, v))

    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_df.to_csv(output_dir / "temporal_summary.csv", index=False)
    print(f"\nSaved temporal_summary.csv")

    # Print bout summary
    print(f"\n{'='*50}")
    print(f"BOUT SUMMARY")
    print(f"{'='*50}")
    for k, v in bout_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
