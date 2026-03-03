"""
Generate a Gantt chart and project timeline for the USV Detection & Analysis Pipeline.

Usage:
    .venv/Scripts/python.exe scripts/generate_project_timeline.py
    .venv/Scripts/python.exe scripts/generate_project_timeline.py --output docs/timeline.png
    .venv/Scripts/python.exe scripts/generate_project_timeline.py --style dark

Produces a publication-quality Gantt chart PNG suitable for PI presentations.
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np


# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "done":        "#2ecc71",  # green
    "in_progress": "#f39c12",  # amber
    "blocked":     "#e74c3c",  # red
    "future":      "#95a5a6",  # gray
    "ready":       "#3498db",  # blue
}

DARK_COLORS = {
    "done":        "#27ae60",
    "in_progress": "#e67e22",
    "blocked":     "#c0392b",
    "future":      "#7f8c8d",
    "ready":       "#2980b9",
}

# ── Phase data ──────────────────────────────────────────────────────────────
# Each entry: (label, start_date, end_date, status, group)
# Dates are approximate for early phases based on implementation log.
# For blocked/future phases, end_date is projected estimate.

PHASES = [
    # ── Foundation (Phases 1-7) ──────────────────────────────────────────────
    ("1: Energy Detector",          date(2026, 1, 16), date(2026, 1, 19), "done", "Foundation"),
    ("2: Spectrogram Extraction",   date(2026, 1, 19), date(2026, 1, 22), "done", "Foundation"),
    ("3: Labeling Tool + Data",     date(2026, 1, 22), date(2026, 2, 3),  "done", "Foundation"),
    ("4: Dataset Preparation",      date(2026, 2, 3),  date(2026, 2, 7),  "done", "Foundation"),
    ("5: CNN Classifier",           date(2026, 2, 7),  date(2026, 2, 9),  "done", "Foundation"),
    ("6: Desktop Detection App",    date(2026, 2, 7),  date(2026, 2, 14), "done", "Foundation"),
    ("7: Clustering Exploration",   date(2026, 2, 9),  date(2026, 2, 14), "done", "Foundation"),

    # ── Transformer + VQ-VAE (Phase 8) ──────────────────────────────────────
    ("8.1: Bout Data Pipeline",     date(2026, 2, 18), date(2026, 2, 18), "done", "Transformer"),
    ("8.2: Autoregressive Transformer", date(2026, 2, 20), date(2026, 2, 20), "done", "Transformer"),
    ("8.3: Hidden-State VQ-VAE",    date(2026, 2, 20), date(2026, 2, 20), "done", "Transformer"),
    ("8.4: Analysis & Interpretation", date(2026, 2, 21), date(2026, 2, 21), "done", "Transformer"),

    # ── Training Infrastructure (Phase 9-10) ────────────────────────────────
    ("9.1: Dataset Assembler",      date(2026, 2, 21), date(2026, 2, 21), "done", "Infrastructure"),
    ("10.1: Active Learning Runner", date(2026, 2, 21), date(2026, 2, 21), "done", "Infrastructure"),

    # ── Real Data Execution (Phase 11) ──────────────────────────────────────
    ("11.1: Bout Extraction (real)", date(2026, 2, 22), date(2026, 2, 22), "done", "Real Data"),
    ("11.2: Transformer Training",  date(2026, 3, 1),  date(2026, 3, 21), "blocked", "Real Data"),
    ("11.3: VQ-VAE on Real States", date(2026, 3, 22), date(2026, 3, 28), "blocked", "Real Data"),
    ("11.4: Full Analysis",         date(2026, 3, 29), date(2026, 4, 5),  "blocked", "Real Data"),

    # ── DeepSqueak Classification Bridge (Phase 14) ─────────────────────────
    ("14.1: Raven Export",          date(2026, 2, 22), date(2026, 2, 22), "done", "Classification"),
    ("14.2: MATLAB Import/Cluster", date(2026, 2, 23), date(2026, 3, 5),  "in_progress", "Classification"),
    ("14.3: Repertoire Statistics",  date(2026, 2, 25), date(2026, 2, 25), "done", "Classification"),

    # ── Vacation Workstreams ────────────────────────────────────────────────
    ("Info Theory Metrics",         date(2026, 2, 24), date(2026, 2, 24), "done", "Advanced Analysis"),
    ("Null Model Generators",       date(2026, 2, 24), date(2026, 2, 24), "done", "Advanced Analysis"),
    ("Statistical Comparison",      date(2026, 2, 24), date(2026, 2, 24), "done", "Advanced Analysis"),
    ("Acoustic Property Extractors", date(2026, 2, 24), date(2026, 2, 24), "done", "Advanced Analysis"),
    ("Probing Framework",           date(2026, 2, 24), date(2026, 2, 24), "done", "Advanced Analysis"),
    ("LMT Data Access Layer",      date(2026, 2, 24), date(2026, 2, 24), "done", "Advanced Analysis"),

    # ── Future Phases ───────────────────────────────────────────────────────
    ("12: Cross-Population Comparison", date(2026, 4, 6), date(2026, 4, 20), "future", "Future"),
    ("13: Batch Detection Pipeline", date(2026, 4, 20), date(2026, 5, 4), "future", "Future"),
    ("Syllable Classification",     date(2026, 4, 1),  date(2026, 5, 1),  "future", "Future"),
]

# ── Summary statistics ──────────────────────────────────────────────────────
STATS = {
    "project_start": date(2026, 1, 16),
    "today": date(2026, 2, 28),
    "total_tests": 600,  # approximate across all modules
    "total_modules": 45,
    "cnn_f1": "91.7%",
    "cnn_precision": "89.7%",
    "cnn_recall": "93.8%",
    "transformer_params": "25.6M",
    "vqvae_params": "820K",
    "labels_collected": 840,
    "recordings_300khz": "6,491",
}


def build_chart(phases, output_path, style="light", dpi=150):
    """Generate a horizontal Gantt chart."""
    colors = DARK_COLORS if style == "dark" else COLORS

    if style == "dark":
        plt.style.use("dark_background")
        bg_color = "#1e1e1e"
        text_color = "#e0e0e0"
        grid_color = "#333333"
        group_bg = "#2a2a2a"
    else:
        plt.rcdefaults()
        bg_color = "#fafafa"
        text_color = "#2c3e50"
        grid_color = "#dfe6e9"
        group_bg = "#ecf0f1"

    # Group phases
    groups = []
    seen = set()
    for _, _, _, _, g in phases:
        if g not in seen:
            groups.append(g)
            seen.add(g)

    # Build y-positions with group headers
    y_labels = []
    y_positions = []
    bar_data = []  # (y, start, width, color)
    group_spans = {}  # group -> (y_min, y_max)

    y = 0
    for group in groups:
        group_phases = [(l, s, e, st) for l, s, e, st, g in phases if g == group]
        group_start_y = y
        for label, start, end, status in group_phases:
            # Ensure minimum 1-day width for single-day items
            duration = max((end - start).days, 1)
            bar_data.append((y, start, duration, colors[status]))
            y_labels.append(label)
            y_positions.append(y)
            y += 1
        group_spans[group] = (group_start_y - 0.5, y - 0.5)
        y += 0.3  # gap between groups

    # Create figure
    fig_height = max(10, len(y_labels) * 0.38 + 2)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Draw group backgrounds
    for group, (ymin, ymax) in group_spans.items():
        ax.axhspan(ymin, ymax, color=group_bg, alpha=0.5, zorder=0)

    # Draw bars
    for y_pos, start, duration, color in bar_data:
        ax.barh(
            y_pos, duration, left=mdates.date2num(start),
            height=0.6, color=color, edgecolor="white",
            linewidth=0.5, alpha=0.9, zorder=2
        )

    # Today line
    today = STATS["today"]
    today_num = mdates.date2num(today)
    ax.axvline(today_num, color="#e74c3c", linewidth=2, linestyle="--",
               alpha=0.7, zorder=3, label=f"Today ({today.strftime('%b %d')})")

    # Configure axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.5, color=text_color)
    ax.invert_yaxis()

    # X-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # Mondays
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    # Set x range
    all_starts = [s for _, s, _, _ in bar_data]
    all_ends = [mdates.date2num(s) + d for _, s, d, _ in bar_data]
    x_min = mdates.date2num(min(all_starts)) - 3
    x_max = max(all_ends) + 5
    ax.set_xlim(x_min, x_max)

    plt.xticks(fontsize=8, color=text_color, rotation=45, ha="right")
    ax.tick_params(axis="y", which="both", left=False)
    ax.grid(axis="x", color=grid_color, linewidth=0.5, alpha=0.7, zorder=1)

    # Group labels on the right
    for group, (ymin, ymax) in group_spans.items():
        ymid = (ymin + ymax) / 2
        ax.annotate(
            group, xy=(x_max - 1, ymid),
            fontsize=8, fontweight="bold", color=text_color,
            alpha=0.6, ha="right", va="center"
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors["done"], label="Completed"),
        mpatches.Patch(color=colors["in_progress"], label="In Progress"),
        mpatches.Patch(color=colors["blocked"], label="Blocked (GPU)"),
        mpatches.Patch(color=colors["ready"], label="Ready"),
        mpatches.Patch(color=colors["future"], label="Future"),
    ]
    ax.legend(
        handles=legend_patches, loc="upper right",
        fontsize=8, framealpha=0.9, edgecolor=grid_color
    )

    # Title and subtitle
    ax.set_title(
        "USV Detection & Analysis Pipeline — Project Timeline",
        fontsize=14, fontweight="bold", color=text_color, pad=15
    )
    fig.text(
        0.5, 0.97,
        f"Started {STATS['project_start'].strftime('%b %d, %Y')}  |  "
        f"{STATS['total_tests']}+ tests  |  "
        f"CNN F1 {STATS['cnn_f1']}  |  "
        f"Transformer {STATS['transformer_params']} params  |  "
        f"{STATS['labels_collected']} labeled USVs",
        ha="center", fontsize=9, color=text_color, alpha=0.7,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Timeline saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate USV project timeline Gantt chart")
    parser.add_argument(
        "--output", "-o", default="docs/timeline.png",
        help="Output PNG path (default: docs/timeline.png)"
    )
    parser.add_argument(
        "--style", choices=["light", "dark"], default="light",
        help="Chart style (default: light)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output resolution (default: 150)"
    )
    args = parser.parse_args()

    build_chart(PHASES, args.output, style=args.style, dpi=args.dpi)

    # Print summary stats
    done = sum(1 for _, _, _, s, _ in PHASES if s == "done")
    in_progress = sum(1 for _, _, _, s, _ in PHASES if s == "in_progress")
    blocked = sum(1 for _, _, _, s, _ in PHASES if s == "blocked")
    future = sum(1 for _, _, _, s, _ in PHASES if s in ("future", "ready"))
    total = len(PHASES)
    elapsed = (STATS["today"] - STATS["project_start"]).days

    print(f"\n{'='*55}")
    print(f"  USV Pipeline Project Summary ({STATS['today'].strftime('%b %d, %Y')})")
    print(f"{'='*55}")
    print(f"  Elapsed:     {elapsed} days ({elapsed // 7} weeks)")
    print(f"  Phases:      {done}/{total} done, {in_progress} in progress, {blocked} blocked, {future} future")
    print(f"  Tests:       {STATS['total_tests']}+")
    print(f"  CNN:         F1 {STATS['cnn_f1']} (P={STATS['cnn_precision']}, R={STATS['cnn_recall']})")
    print(f"  Transformer: {STATS['transformer_params']} params (architecture complete)")
    print(f"  VQ-VAE:      {STATS['vqvae_params']} params (architecture complete)")
    print(f"  Labels:      {STATS['labels_collected']} USVs labeled")
    print(f"  Recordings:  {STATS['recordings_300khz']} WAV files @ 300 kHz")
    print(f"  Blocker:     GPU/HPC access needed for Phase 11.2+")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
