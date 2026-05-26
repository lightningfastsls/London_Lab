"""Render side-by-side PNGs for each previously-labeled lab event.

For every event the user already eyeballed at
``results/batch_lab_131204_full/audit_2026-05-08/{bucket}/*.png``, produce a
2-panel comparison PNG:

    [TOP]    ORIGINAL spectrogram (no subtraction) with baseline event window
    [BOTTOM] SUBTRACTED spectrogram (post-CNN-input preprocessing) with same
             event window

Title encodes:
- Baseline bucket the event was sampled into (cleanest / typical / borderline / ...)
- Baseline metrics: max_prob, SEF, tier
- Pilot status:
    KEPT-AA       → event still detected, host chunk still auto_accept
    DEMOTED-MR    → event still detected, chunk fell to manual_review
    DEMOTED-AR    → no event in this time window in the pilot run
    GONE          → no detection at all on this chunk in the pilot
    NEW-CHUNK-ONLY→ chunk produced new events that didn't exist in baseline
                    (relevant only when scanning the pilot side)

Files are sorted into status subdirectories so the user can browse
"events that disappeared" or "events still kept" in one click.

This script does NOT relabel — it just produces the comparison PNGs the
user will eyeball to update their understanding of what subtraction did
to events they already have ground-truth notes on.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from usv_spectrogram.app.core.audio_loader import AudioLoader  # noqa: E402

# Tolerance for matching a baseline event to a pilot event on the same chunk.
# Events are merged by start time; 0.05 s = half the chunk overlap window
# the merge step uses for dedup, so it's a natural "same event" tolerance.
TIME_MATCH_TOL_S = 0.05

# Filename pattern: <bucket>_sef<...>_dur<...>ms_<chunk_stem>_ev<idx>.png
PNG_RE = re.compile(
    r"^(?P<bucket>[a-z_]+)_sef[\d.]+_dur\d+ms_(?P<stem>.+)_ev(?P<idx>\d+)\.png$"
)


def parse_audit_pngs(audit_dir: Path) -> pd.DataFrame:
    """Walk audit_2026-05-08/<bucket>/*.png and return a DataFrame of labeled events."""
    rows = []
    for bucket_dir in sorted(audit_dir.iterdir()):
        if not bucket_dir.is_dir():
            continue
        for png in bucket_dir.glob("*.png"):
            m = PNG_RE.match(png.name)
            if not m:
                print(f"[skip] could not parse {png.name}")
                continue
            rows.append({
                "bucket": m.group("bucket"),
                "chunk_stem": m.group("stem"),
                "baseline_ev_idx": int(m.group("idx")),
                "baseline_png": str(png),
            })
    return pd.DataFrame(rows)


def find_pilot_match(
    baseline_event: pd.Series,
    pilot_events: pd.DataFrame,
) -> pd.Series | None:
    """Find the pilot event on the same chunk closest in time to baseline.

    Returns the matching row, or None if no event lies within tolerance.
    """
    same_chunk = pilot_events[pilot_events["chunk_stem"] == baseline_event["chunk_stem"]]
    if same_chunk.empty:
        return None
    deltas = (same_chunk["original_begin_time_s"] - baseline_event["original_begin_time_s"]).abs()
    best_idx = deltas.idxmin()
    if deltas.loc[best_idx] > TIME_MATCH_TOL_S:
        return None
    return same_chunk.loc[best_idx]


def status_label(
    matched_pilot: pd.Series | None,
    pilot_chunk_tier: str | None,
) -> str:
    """Compress to one of: KEPT-AA, DEMOTED-MR, DEMOTED-AR, GONE."""
    if matched_pilot is None:
        # No event in pilot at this time — but is the chunk completely gone?
        if pilot_chunk_tier is None or pilot_chunk_tier == "auto_reject":
            return "GONE"
        # Chunk has events, but none at this time = this specific event vanished
        return "DEMOTED-AR"
    if pilot_chunk_tier == "auto_accept":
        return "KEPT-AA"
    if pilot_chunk_tier == "manual_review":
        return "DEMOTED-MR"
    return "DEMOTED-AR"


def render_pair(
    chunk_wav: Path,
    baseline_event: pd.Series,
    matched_pilot: pd.Series | None,
    bucket: str,
    pilot_status: str,
    pilot_chunk_tier: str | None,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plain = AudioLoader().load(chunk_wav)
    cleaned = AudioLoader(subtract_baseline=True).load(chunk_wav)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=140, sharex=True)

    extent = [
        plain.times[0], plain.times[-1],
        plain.frequencies[0] / 1000.0, plain.frequencies[-1] / 1000.0,
    ]
    # Shared color scale across panels so subtraction's visual effect is
    # "stationary bands disappear; everything else looks the same intensity."
    # Without shared scaling, each panel autoscales; the cleaned panel's 5th
    # percentile sits at the eps-floor (-200 dB by construction since 10% of
    # cells are pinned there) which crushes the meaningful range into ~10% of
    # the colormap — purely cosmetic, but reads as "the spectrogram changed".
    # Using the ORIGINAL panel's 5th–99th percentile range matches what the
    # CNN's MAD-normalized input actually exposes downstream.
    vmin = float(np.percentile(plain.spectrogram_db, 5))
    vmax = float(np.percentile(plain.spectrogram_db, 99))
    axes[0].imshow(
        plain.spectrogram_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent, vmin=vmin, vmax=vmax,
    )
    axes[1].imshow(
        cleaned.spectrogram_db, aspect="auto", origin="lower", cmap="magma",
        extent=extent, vmin=vmin, vmax=vmax,
    )

    # Baseline event window — drawn on both panels
    start_t = baseline_event["original_begin_time_s"] - baseline_event["start_s_in_original"]
    end_t = baseline_event["original_end_time_s"] - baseline_event["start_s_in_original"]
    for ax in axes:
        ax.axvspan(start_t, end_t, alpha=0.20, color="cyan", linewidth=0)
        ax.axvline(start_t, color="cyan", linewidth=1.0, alpha=0.9)
        ax.axvline(end_t, color="cyan", linewidth=1.0, alpha=0.9)

    # If pilot has a matched event at slightly different bounds, mark it lime
    if matched_pilot is not None:
        ps = matched_pilot["original_begin_time_s"] - matched_pilot["start_s_in_original"]
        pe = matched_pilot["original_end_time_s"] - matched_pilot["start_s_in_original"]
        if abs(ps - start_t) > 1e-6 or abs(pe - end_t) > 1e-6:
            axes[1].axvline(ps, color="lime", linewidth=1.0, alpha=0.9, linestyle="--")
            axes[1].axvline(pe, color="lime", linewidth=1.0, alpha=0.9, linestyle="--")

    # Titles
    base_sef = baseline_event.get("stationary_energy_fraction", float("nan"))
    base_dur_ms = int(round(baseline_event["duration_s"] * 1000))
    axes[0].set_title(
        f"ORIGINAL  [{bucket}]  {baseline_event['chunk_stem']}  "
        f"baseline_ev{int(baseline_event['chunk_detection_idx'])}  "
        f"max_prob={baseline_event['max_probability']:.3f}  "
        f"SEF={base_sef:.3f}  dur={base_dur_ms}ms  "
        f"baseline_tier={baseline_event.get('tier', '?')}",
        fontsize=9,
    )
    axes[0].set_ylabel("Freq (kHz)")

    if matched_pilot is not None:
        pilot_dur_ms = int(round(matched_pilot["duration_s"] * 1000))
        pilot_title = (
            f"max_prob={matched_pilot['max_probability']:.3f}  "
            f"SEF={matched_pilot.get('stationary_energy_fraction', float('nan')):.3f}  "
            f"dur={pilot_dur_ms}ms"
        )
    else:
        pilot_title = "no matched pilot event in this time window"

    axes[1].set_title(
        f"SUBTRACTED  [{pilot_status}]  pilot_chunk_tier={pilot_chunk_tier}  {pilot_title}",
        fontsize=9,
    )
    axes[1].set_xlabel("Time in chunk (s)")
    axes[1].set_ylabel("Freq (kHz)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline-audit-dir",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_full/audit_2026-05-08",
    )
    ap.add_argument(
        "--baseline-events-with-filter",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_full/merged_events_with_filter.parquet",
    )
    ap.add_argument(
        "--pilot-events-with-filter",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_subtracted_pilot/merged_events_with_filter.parquet",
    )
    ap.add_argument(
        "--pilot-summary",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_subtracted_pilot/summary.parquet",
    )
    ap.add_argument(
        "--chunks-dir",
        type=Path,
        default=REPO_ROOT / "USV_lab_131204_chunked_2s_full",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/batch_lab_131204_subtracted_pilot/labeled_paired_comparison",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    labeled = parse_audit_pngs(args.baseline_audit_dir)
    print(f"[load] {len(labeled)} labeled (chunk, event) pairs from baseline audit")

    baseline_events = pd.read_parquet(args.baseline_events_with_filter)
    print(f"[load] {len(baseline_events):,} baseline events")

    pilot_events = pd.read_parquet(args.pilot_events_with_filter)
    print(f"[load] {len(pilot_events):,} pilot events")

    pilot_summary = pd.read_parquet(args.pilot_summary)
    pilot_summary["stem"] = pilot_summary["filepath"].apply(lambda p: Path(p).stem)
    pilot_chunk_tier = dict(zip(pilot_summary["stem"], pilot_summary["tier"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    n_rendered = 0
    n_skipped = 0
    for _, row in labeled.iterrows():
        bk = baseline_events[
            (baseline_events["chunk_stem"] == row["chunk_stem"])
            & (baseline_events["chunk_detection_idx"] == row["baseline_ev_idx"])
        ]
        if bk.empty:
            print(f"[skip] {row['chunk_stem']} ev{row['baseline_ev_idx']}  baseline event not found")
            n_skipped += 1
            continue
        baseline_event = bk.iloc[0]

        chunk_wav = args.chunks_dir / f"{row['chunk_stem']}.wav"
        if not chunk_wav.exists():
            print(f"[skip] {row['chunk_stem']} — WAV not found")
            n_skipped += 1
            continue

        chunk_tier_pilot = pilot_chunk_tier.get(row["chunk_stem"])
        matched = find_pilot_match(baseline_event, pilot_events)
        status = status_label(matched, chunk_tier_pilot)

        status_dir = args.out_dir / status
        status_dir.mkdir(parents=True, exist_ok=True)
        out_path = status_dir / (
            f"{row['bucket']}_{row['chunk_stem']}_ev{row['baseline_ev_idx']:03d}.png"
        )
        render_pair(
            chunk_wav, baseline_event, matched, row["bucket"],
            status, chunk_tier_pilot, out_path,
        )
        n_rendered += 1
        summary_rows.append({
            "bucket": row["bucket"],
            "chunk_stem": row["chunk_stem"],
            "baseline_ev_idx": row["baseline_ev_idx"],
            "baseline_max_prob": float(baseline_event["max_probability"]),
            "baseline_sef": float(baseline_event.get("stationary_energy_fraction", float("nan"))),
            "baseline_tier": baseline_event.get("tier"),
            "pilot_chunk_tier": chunk_tier_pilot,
            "pilot_event_max_prob": (float(matched["max_probability"]) if matched is not None else None),
            "pilot_event_sef": (
                float(matched["stationary_energy_fraction"])
                if matched is not None and "stationary_energy_fraction" in matched.index
                else None
            ),
            "status": status,
            "out_png": str(out_path),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.out_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[done] rendered {n_rendered} comparison PNGs (skipped {n_skipped})")
    print(f"[done] summary: {summary_csv}")

    print("\n=== STATUS DISTRIBUTION ===")
    print(summary_df["status"].value_counts())
    print("\n=== STATUS × BUCKET CROSSTAB ===")
    print(pd.crosstab(summary_df["bucket"], summary_df["status"], margins=True, margins_name="TOTAL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
