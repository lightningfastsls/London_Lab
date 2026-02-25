"""Event-triggered USV rate analysis (PETH) from LMT + detection CSV.

Computes peri-event time histograms for each behavioral event type,
bridging LMT behavioral annotations with USV detection output.

Usage:
    python run_event_triggered_analysis.py \
        --lmt-db path/to/lmt_database.sqlite \
        --detections path/to/detections.csv \
        --output path/to/output_dir/ \
        --event-types "Oral-oral Contact" "Rearing" \
        --n-permutations 1000

Input:
    - LMT SQLite database (behavioral events)
    - Detection CSV with 'start_ms' column (from batch detection pipeline)

Output:
    - results.json            JSON-serialized PETHResult per event type
    - event_triggered_report.md   Markdown summary
    - peth_all.png            Grid plot of all PETHs
    - peth_{EventType}.png    Individual PETH plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Bootstrap: add src/ to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.lmt.db_loader import LMTDatabaseLoader
from usv_spectrogram.lmt.event_triggered import (
    PETHConfig,
    PETHResult,
    compute_all_peths,
    plot_all_peths,
    plot_peth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_usv_times_from_csv(csv_path: Path) -> list[float]:
    """Load USV onset times from a detection CSV.

    Reads the 'start_ms' column and converts to seconds.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with at least a 'start_ms' column.

    Returns
    -------
    list[float]
        USV onset times in seconds, sorted.
    """
    times = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "start_ms" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV must have a 'start_ms' column, found: {reader.fieldnames}"
            )
        for row in reader:
            times.append(float(row["start_ms"]) / 1000.0)
    return sorted(times)


def group_events_by_type(
    events: list,
) -> dict[str, list[float]]:
    """Group behavioral events by type, extracting onset times.

    Parameters
    ----------
    events : list[BehavioralEvent]
        Events from LMTDatabaseLoader.get_events().

    Returns
    -------
    dict[str, list[float]]
        Mapping from event type name to sorted list of onset times (seconds).
    """
    groups: dict[str, list[float]] = {}
    for ev in events:
        groups.setdefault(ev.event_type, []).append(ev.start_time_s)
    # Sort each group
    for key in groups:
        groups[key].sort()
    return groups


def _result_to_dict(result: PETHResult) -> dict:
    """Convert a PETHResult to a JSON-serializable dict."""
    return {
        "event_type": result.event_type,
        "time_bins_s": list(result.time_bins_s),
        "rate_hz": list(result.rate_hz),
        "rate_ci_lower_hz": list(result.rate_ci_lower_hz),
        "rate_ci_upper_hz": list(result.rate_ci_upper_hz),
        "baseline_rate_hz": result.baseline_rate_hz,
        "n_events": result.n_events,
        "p_value": result.p_value,
        "significant": result.significant,
        "peak_time_s": result.peak_time_s,
        "peak_rate_hz": result.peak_rate_hz,
    }


def _generate_report(
    results: dict[str, PETHResult],
    config: PETHConfig,
    n_usvs: int,
    recording_duration_s: float,
) -> str:
    """Generate a Markdown report summarizing PETH results."""
    lines = [
        "# Event-Triggered USV Rate Analysis (PETH)",
        "",
        "## Parameters",
        "",
        f"- Window: -{config.window_before_s}s to +{config.window_after_s}s",
        f"- Bin size: {config.bin_size_s}s",
        f"- Permutations: {config.n_permutations}",
        f"- Baseline method: {config.baseline_method}",
        f"- Min events: {config.min_events}",
        "",
        "## Data Summary",
        "",
        f"- Total USVs: {n_usvs}",
        f"- Recording duration: {recording_duration_s:.1f}s",
        f"- Overall USV rate: {n_usvs / recording_duration_s:.2f} Hz"
        if recording_duration_s > 0 else "- Overall USV rate: N/A",
        f"- Event types analyzed: {len(results)}",
        "",
        "## Results",
        "",
        "| Event Type | n Events | Peak Rate (Hz) | Peak Time (s) | Baseline (Hz) | p-value | Sig. |",
        "|------------|----------|----------------|---------------|---------------|---------|------|",
    ]

    for event_type in sorted(results.keys()):
        r = results[event_type]
        sig_marker = "**yes**" if r.significant else "no"
        lines.append(
            f"| {event_type} | {r.n_events} | {r.peak_rate_hz:.2f} | "
            f"{r.peak_time_s:+.2f} | {r.baseline_rate_hz:.2f} | "
            f"{r.p_value:.3f} | {sig_marker} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Significant (p < 0.05)**: USV rate around this event type is unlikely "
        "to arise from chance temporal alignment. The circular-shift permutation "
        "test preserves USV autocorrelation while destroying cross-correlation.",
        "- **Peak time**: Positive = after event onset; negative = before. "
        "Anticipatory peaks (negative) may indicate sensory-driven vocalization.",
        "- **Peak-to-baseline ratio > 2**: Strong modulation.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute peri-event time histograms (PETH) for USV-behavior correlation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_event_triggered_analysis.py \\
      --lmt-db experiment.sqlite --detections usvs.csv --output results/

  python run_event_triggered_analysis.py \\
      --lmt-db experiment.sqlite --detections usvs.csv --output results/ \\
      --event-types "Oral-oral Contact" "Rearing" --bin-size 0.05
        """,
    )
    parser.add_argument(
        "--lmt-db", required=True, type=Path,
        help="Path to LMT SQLite database",
    )
    parser.add_argument(
        "--detections", required=True, type=Path,
        help="Path to detection CSV with 'start_ms' column",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--window-before", type=float, default=2.0,
        help="Window before event onset in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--window-after", type=float, default=2.0,
        help="Window after event onset in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--bin-size", type=float, default=0.1,
        help="Bin size in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--event-types", nargs="+", default=None,
        help="Filter to these event types (default: all)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Number of permutations for significance test (default: 1000)",
    )
    parser.add_argument(
        "--min-events", type=int, default=5,
        help="Minimum events required per type (default: 5)",
    )
    parser.add_argument(
        "--recording-duration", type=float, default=None,
        help="Recording duration in seconds (auto-detect from max USV time if not set)",
    )
    parser.add_argument(
        "--time-offset", type=float, default=0.0,
        help="Time offset to add to LMT event times in seconds (default: 0.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- Load USV times ---
    print(f"Loading USV detections from {args.detections}")
    usv_times = load_usv_times_from_csv(args.detections)
    print(f"  Loaded {len(usv_times)} USV detections")

    if len(usv_times) == 0:
        print("Error: No USV detections found in CSV.")
        return 1

    # --- Recording duration ---
    if args.recording_duration is not None:
        recording_duration = args.recording_duration
    else:
        # Auto-detect: use max USV time + a buffer
        recording_duration = max(usv_times) + 5.0
        print(f"  WARNING: Recording duration auto-detected as {recording_duration:.1f}s "
              f"(max USV time + 5s buffer).")
        print(f"  This may be shorter than the actual recording. Use --recording-duration")
        print(f"  for accurate baseline rate and permutation test. The whole-recording")
        print(f"  baseline will be inflated if events extend beyond {recording_duration:.1f}s.")

    # --- Load LMT events ---
    print(f"Loading LMT events from {args.lmt_db}")
    with LMTDatabaseLoader(args.lmt_db) as loader:
        events = loader.get_events(event_types=args.event_types)
    print(f"  Loaded {len(events)} events")

    # Apply time offset
    if args.time_offset != 0.0:
        print(f"  Applying time offset: {args.time_offset:+.3f}s")

    # Group events by type
    events_by_type = group_events_by_type(events)

    # Adjust onset times by offset
    if args.time_offset != 0.0:
        events_by_type = {
            k: [t + args.time_offset for t in v]
            for k, v in events_by_type.items()
        }

    print(f"  Event types: {sorted(events_by_type.keys())}")
    for etype, times in sorted(events_by_type.items()):
        print(f"    {etype}: {len(times)} events")

    # --- Compute PETHs ---
    config = PETHConfig(
        window_before_s=args.window_before,
        window_after_s=args.window_after,
        bin_size_s=args.bin_size,
        n_permutations=args.n_permutations,
        min_events=args.min_events,
    )

    print(f"\nComputing PETHs (window: -{config.window_before_s}s to "
          f"+{config.window_after_s}s, bin: {config.bin_size_s}s, "
          f"permutations: {config.n_permutations})...")

    results = compute_all_peths(
        usv_times_s=usv_times,
        events_by_type=events_by_type,
        recording_duration_s=recording_duration,
        config=config,
    )

    if not results:
        print("No event types had enough events to compute PETH.")
        return 1

    # --- Output ---
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    json_path = output_dir / "results.json"
    json_data = {k: _result_to_dict(v) for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Markdown report
    report = _generate_report(results, config, len(usv_times), recording_duration)
    report_path = output_dir / "event_triggered_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Grid plot
        fig = plot_all_peths(results)
        if fig is not None:
            grid_path = output_dir / "peth_all.png"
            fig.savefig(grid_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Grid plot saved to {grid_path}")

        # Individual plots
        for event_type, result in sorted(results.items()):
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_peth(result, ax=ax)
            safe_name = event_type.replace(" ", "_").replace("/", "-")
            ind_path = output_dir / f"peth_{safe_name}.png"
            fig.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Individual plots saved to {output_dir}/peth_*.png")

    except ImportError:
        print("Warning: matplotlib not available, skipping plots")

    # --- Summary ---
    print("\n--- Summary ---")
    for event_type in sorted(results.keys()):
        r = results[event_type]
        sig = "SIGNIFICANT" if r.significant else "not significant"
        print(
            f"  {event_type}: peak={r.peak_rate_hz:.2f} Hz at {r.peak_time_s:+.2f}s, "
            f"p={r.p_value:.3f} ({sig})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
