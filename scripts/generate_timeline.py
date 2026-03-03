"""
Generate a presentation-quality Gantt chart of the USV Detection & Analysis project.

Usage:
    python scripts/generate_timeline.py [--output timeline.png] [--dpi 150] [--dark]

Outputs a color-coded timeline showing completed, active, blocked, and future phases.
Designed for PI presentations and lab meetings.
"""
import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np


# ── Status definitions ──────────────────────────────────────────────────────

@dataclass
class Phase:
    name: str
    status: str          # "done", "active", "blocked", "future"
    start: date
    end: date
    group: str           # for grouping/coloring by pipeline stage
    detail: str = ""     # hover/annotation text
    tests: int = 0       # test count (shown in annotation)


# ── Project data ────────────────────────────────────────────────────────────
# Dates are approximate — based on commit history and implementation records

PHASES = [
    # --- Core Pipeline (Phases 1-7) ---
    Phase("1. Energy Detector", "done",
          date(2025, 10, 1), date(2025, 10, 20),
          "Core Pipeline", "42 tests, ADR-001/002/003", 42),
    Phase("2. Spectrogram Extraction", "done",
          date(2025, 10, 20), date(2025, 11, 5),
          "Core Pipeline", "PNG extraction pipeline"),
    Phase("3. Labeling Tool (Streamlit)", "done",
          date(2025, 11, 5), date(2025, 11, 25),
          "Core Pipeline", "~840 labels collected"),
    Phase("4. Dataset Preparation", "done",
          date(2025, 11, 25), date(2025, 12, 15),
          "Core Pipeline", "Recording-based splits, 71 tests", 71),
    Phase("5. CNN Classifier", "done",
          date(2025, 12, 15), date(2026, 1, 10),
          "Core Pipeline", "F1=91.7%, 38 tests", 38),
    Phase("6. Desktop App (PyQt6)", "done",
          date(2026, 1, 10), date(2026, 1, 30),
          "Core Pipeline", "Detection + labeling GUI"),
    Phase("7. Clustering Exploration", "done",
          date(2026, 1, 30), date(2026, 2, 5),
          "Core Pipeline", "k-means, HDBSCAN, GMM"),

    # --- Transformer + VQ-VAE (Phase 8) ---
    Phase("8.1 Bout Data Preparation", "done",
          date(2026, 2, 5), date(2026, 2, 10),
          "Transformer + VQ-VAE", "PyTorch datasets, bucketed batching"),
    Phase("8.2 Causal Transformer", "done",
          date(2026, 2, 10), date(2026, 2, 14),
          "Transformer + VQ-VAE", "~25.6M params, 11 tests", 11),
    Phase("8.3 Hidden-State VQ-VAE", "done",
          date(2026, 2, 14), date(2026, 2, 18),
          "Transformer + VQ-VAE", "~820K params, K=64 codebook, 21 tests", 21),
    Phase("8.4 Analysis Tools", "done",
          date(2026, 2, 18), date(2026, 2, 21),
          "Transformer + VQ-VAE", "9 analysis modules, 17 tests", 17),

    # --- Infrastructure (Phases 9-10) ---
    Phase("9.1 Dataset Assembly", "done",
          date(2026, 2, 21), date(2026, 2, 21),
          "Infrastructure", "DatasetAssembler, 10 tests", 10),
    Phase("10.1 Active Learning Runner", "done",
          date(2026, 2, 21), date(2026, 2, 21),
          "Infrastructure", "Cycle automation"),

    # --- Real Data Execution (Phase 11) ---
    Phase("11.1 Bout Extraction (real)", "done",
          date(2026, 2, 22), date(2026, 2, 22),
          "Real Data", "Extracted from 300kHz WAVs"),
    Phase("11.2 Transformer Training", "blocked",
          date(2026, 3, 1), date(2026, 4, 15),
          "Real Data", "BLOCKED: needs HPC/cloud GPU"),
    Phase("11.3 VQ-VAE on Real States", "blocked",
          date(2026, 4, 15), date(2026, 5, 1),
          "Real Data", "Needs trained transformer"),
    Phase("11.4 Analysis on Real Codes", "blocked",
          date(2026, 5, 1), date(2026, 5, 15),
          "Real Data", "Needs VQ-VAE codes"),

    # --- DeepSqueak Bridge (Phase 14) ---
    Phase("14.1 Raven Table Export", "done",
          date(2026, 2, 22), date(2026, 2, 22),
          "DeepSqueak Bridge", "93 detections, 5 WAVs, 33 tests", 33),
    Phase("14.2 MATLAB Classification", "active",
          date(2026, 2, 25), date(2026, 3, 10),
          "DeepSqueak Bridge", "Import + unsupervised clustering"),
    Phase("14.3 Results Ingestion", "blocked",
          date(2026, 3, 10), date(2026, 3, 20),
          "DeepSqueak Bridge", "Needs MATLAB output"),
    Phase("14.4 Repertoire Statistics", "blocked",
          date(2026, 3, 20), date(2026, 4, 1),
          "DeepSqueak Bridge", "Cross-recording comparison"),

    # --- Future work ---
    Phase("12. Cross-Population", "future",
          date(2026, 5, 15), date(2026, 7, 1),
          "Future", "Compare USV repertoires across groups"),
    Phase("13. Batch Detection", "future",
          date(2026, 7, 1), date(2026, 8, 1),
          "Future", "Large-scale automated detection"),
]


# ── Color scheme ────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "done":    "#2ecc71",   # green
    "active":  "#3498db",   # blue
    "blocked": "#e74c3c",   # red
    "future":  "#95a5a6",   # gray
}

STATUS_LABELS = {
    "done":    "Completed",
    "active":  "In Progress",
    "blocked": "Blocked",
    "future":  "Future",
}

GROUP_HATCH = {
    "Core Pipeline":        "",
    "Transformer + VQ-VAE": "",
    "Infrastructure":       "",
    "Real Data":            "",
    "DeepSqueak Bridge":    "",
    "Future":               "//",
}


def generate_gantt(output_path: str = "timeline.png", dpi: int = 150,
                   dark: bool = False):
    """Generate the Gantt chart and save to file."""

    if dark:
        plt.style.use("dark_background")
        edge_color = "#cccccc"
        today_color = "#f39c12"
        text_color = "#eeeeee"
        grid_color = "#444444"
    else:
        edge_color = "#333333"
        today_color = "#e67e22"
        text_color = "#222222"
        grid_color = "#dddddd"

    fig, ax = plt.subplots(figsize=(18, 12))

    # Y positions (reversed so Phase 1 is at top)
    n = len(PHASES)
    y_positions = list(range(n - 1, -1, -1))

    bar_height = 0.6

    for i, phase in enumerate(PHASES):
        y = y_positions[i]
        start_num = mdates.date2num(phase.start)
        end_num = mdates.date2num(phase.end)
        duration = max(end_num - start_num, 3)  # minimum bar width of 3 days

        color = STATUS_COLORS[phase.status]
        hatch = GROUP_HATCH.get(phase.group, "")
        alpha = 0.4 if phase.status == "future" else 0.85

        # Draw bar
        bar = ax.barh(
            y, duration, left=start_num, height=bar_height,
            color=color, alpha=alpha, edgecolor=edge_color,
            linewidth=0.8, hatch=hatch, zorder=3
        )

        # Add test count annotation inside bar if space allows
        if phase.tests > 0 and duration > 10:
            mid = start_num + duration / 2
            ax.text(mid, y, f"{phase.tests}t",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white", zorder=4)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([p.name for p in PHASES], fontsize=9)

    # Format x-axis as dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))

    # Today line
    today_num = mdates.date2num(date.today())
    ax.axvline(today_num, color=today_color, linewidth=2, linestyle="--",
               zorder=5, label=f"Today ({date.today().strftime('%b %d')})")

    # Grid
    ax.xaxis.grid(True, which="major", color=grid_color, linewidth=0.5, zorder=1)
    ax.xaxis.grid(True, which="minor", color=grid_color, linewidth=0.2, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Limits
    ax.set_xlim(
        mdates.date2num(date(2025, 9, 15)),
        mdates.date2num(date(2026, 8, 15))
    )

    # Title and subtitle
    ax.set_title(
        "USV Detection & Analysis Pipeline — Project Timeline",
        fontsize=16, fontweight="bold", pad=20, color=text_color
    )
    fig.text(0.5, 0.935,
             "Mouse Ultrasonic Vocalization Analysis | 300 kHz Recordings | "
             f"434+ Tests | {sum(1 for p in PHASES if p.status == 'done')}/{len(PHASES)} Phases Complete",
             ha="center", fontsize=10, color=text_color, alpha=0.7)

    # Legend
    legend_patches = [
        mpatches.Patch(color=STATUS_COLORS[s], label=STATUS_LABELS[s], alpha=0.85)
        for s in ["done", "active", "blocked", "future"]
    ]
    legend_patches.append(
        plt.Line2D([0], [0], color=today_color, linewidth=2,
                   linestyle="--", label=f"Today ({date.today().strftime('%b %d')})")
    )
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor=grid_color)

    # Blocker annotation
    blocker_y = None
    for i, p in enumerate(PHASES):
        if p.name == "11.2 Transformer Training":
            blocker_y = y_positions[i]
            blocker_x = mdates.date2num(p.start)
            break

    if blocker_y is not None:
        ax.annotate(
            "GPU ACCESS\nNEEDED",
            xy=(blocker_x, blocker_y),
            xytext=(blocker_x - 30, blocker_y + 2),
            fontsize=8, fontweight="bold", color=STATUS_COLORS["blocked"],
            arrowprops=dict(arrowstyle="->", color=STATUS_COLORS["blocked"],
                            lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white" if not dark else "#333",
                      edgecolor=STATUS_COLORS["blocked"], alpha=0.9),
            zorder=6
        )

    # Group separators
    group_boundaries = []
    current_group = PHASES[0].group
    for i, phase in enumerate(PHASES):
        if phase.group != current_group:
            group_boundaries.append(i)
            current_group = phase.group

    for boundary_idx in group_boundaries:
        sep_y = (y_positions[boundary_idx] + y_positions[boundary_idx - 1]) / 2
        ax.axhline(sep_y, color=grid_color, linewidth=0.8, linestyle=":",
                   zorder=2, alpha=0.6)

    # Key metrics box
    done_count = sum(1 for p in PHASES if p.status == "done")
    total_tests = sum(p.tests for p in PHASES if p.tests > 0)
    metrics_text = (
        f"Progress: {done_count}/{len(PHASES)} phases\n"
        f"Tests: 434+ across pipeline\n"
        f"CNN: F1=91.7% (P=89.7%, R=93.8%)\n"
        f"Transformer: ~25.6M params\n"
        f"VQ-VAE: K=64 codebook\n"
        f"Labels: ~840 human-reviewed"
    )
    props = dict(boxstyle="round,pad=0.5",
                 facecolor="white" if not dark else "#2d2d2d",
                 edgecolor=grid_color, alpha=0.95)
    ax.text(0.01, 0.02, metrics_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=props, color=text_color, family="monospace")

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Save
    output = Path(output_path)
    fig.savefig(output, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Timeline saved to: {output.resolve()}")

    # Also save PDF if requested via extension
    if output.suffix == ".png":
        pdf_path = output.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"PDF version saved to: {pdf_path.resolve()}")

    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate USV project timeline Gantt chart"
    )
    parser.add_argument(
        "--output", "-o", default="timeline.png",
        help="Output file path (default: timeline.png)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Image DPI (default: 150)"
    )
    parser.add_argument(
        "--dark", action="store_true",
        help="Use dark background theme"
    )
    args = parser.parse_args()
    generate_gantt(args.output, args.dpi, args.dark)


if __name__ == "__main__":
    main()
