"""Unified lab full-corpus merge: per-chunk triage + per-event detections.

Spans the entire 26,309-chunk lab run (both the pre-resume 9,543 chunks and
the post-resume 16,766 chunks). Re-derives the triage tier from the per-chunk
JSONs alone, since the original ``summary.parquet`` only contains the 16,766
chunks processed in the resumed run.

The tier reconstruction is sound because ``postprocessing/triage.py``
assigns tier from event ``peak_probability`` only; the per-window probability
array is used for a QC *flag* (``high_noise_floor``) but not for tier
assignment. The script validates this by cross-checking the derived tier
against the original ``summary.parquet`` on the 16,766-chunk overlap; any
mismatch is a bug.

Outputs:
    results/batch_lab_131204_full/merged_summary_full.parquet   per-chunk
    results/batch_lab_131204_full/merged_events_full.parquet    per-event,
        deduped across the 0.1 s chunk overlap, with original-file timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Import canonical triage thresholds rather than redeclaring (avoids drift if
# TriageConfig defaults ever change in postprocessing/triage.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usv_spectrogram.postprocessing.triage import TriageConfig  # noqa: E402

_TIER_CONFIG = TriageConfig()


def derive_tier(events: list[dict]) -> str:
    """Re-derive triage tier from event list alone.

    Re-implements the tier branch of ``postprocessing/triage.py:triage_recording``.
    The ``prob_max <= auto_reject_max_window`` branch is unreachable when
    n_events > 0 (events come from hysteresis on the same probability array,
    so prob_max >= max event peak >= hysteresis low threshold), so we omit it.
    The ``high_noise_floor`` QC flag does not affect tier and is dropped.
    """
    if not events:
        return "auto_reject"
    if all(e["max_probability"] >= _TIER_CONFIG.auto_accept_min_peak for e in events):
        return "auto_accept"
    return "manual_review"


def aggregate_events(events: list[dict]) -> dict:
    n = len(events)
    if n == 0:
        return dict(
            n_events=0,
            max_confidence=0.0,
            mean_event_confidence=0.0,
            total_usv_duration_ms=0.0,
        )
    peaks = [e["max_probability"] for e in events]
    durations_ms = [(e["end_time_s"] - e["start_time_s"]) * 1000.0 for e in events]
    return dict(
        n_events=n,
        max_confidence=max(peaks),
        mean_event_confidence=sum(peaks) / n,
        total_usv_duration_ms=sum(durations_ms),
    )


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["chunk_stem"] = df["chunk_filename"].str.replace(r"\.wav$", "", regex=True)
    return df.set_index("chunk_stem")


def build_tables(detections_dir: Path, manifest: pd.DataFrame):
    chunk_rows: list[dict] = []
    event_rows: list[dict] = []
    n_orphan = 0

    json_files = sorted(detections_dir.glob("*.json"))
    print(f"[json] files: {len(json_files)}")

    for jp in json_files:
        stem = jp.stem
        if stem not in manifest.index:
            n_orphan += 1
            continue
        m = manifest.loc[stem]
        events = json.loads(jp.read_text())
        if not isinstance(events, list):
            raise ValueError(f"{jp}: expected list at top level")

        agg = aggregate_events(events)
        tier = derive_tier(events)

        chunk_rows.append(
            {
                "chunk_stem": stem,
                "filepath": f"USV_lab_131204_chunked_2s_full/{stem}.wav",
                "original_filename": m["original_filename"],
                "original_chunk_index": int(m["chunk_index"]),
                "start_s_in_original": float(m["start_s_in_original"]),
                "tier": tier,
                **agg,
            }
        )

        for det_idx, ev in enumerate(events):
            event_rows.append(
                {
                    "chunk_stem": stem,
                    "chunk_detection_idx": det_idx,
                    "original_filename": m["original_filename"],
                    "original_chunk_index": int(m["chunk_index"]),
                    "start_s_in_original": float(m["start_s_in_original"]),
                    "original_begin_time_s": float(m["start_s_in_original"]) + ev["start_time_s"],
                    "original_end_time_s": float(m["start_s_in_original"]) + ev["end_time_s"],
                    "duration_s": ev["duration_s"],
                    "max_probability": ev["max_probability"],
                    "mean_probability": ev["mean_probability"],
                    "start_col": ev.get("start_col"),
                    "end_col": ev.get("end_col"),
                    "tier": tier,
                }
            )

    if n_orphan:
        print(f"[warn] {n_orphan} JSONs had no manifest entry (skipped)")

    return pd.DataFrame(chunk_rows), pd.DataFrame(event_rows)


def dedup_overlap(events: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    """Drop overlap-region duplicates between adjacent chunks (same as
    ``scripts/merge_chunked_lab_detections.py``)."""
    if events.empty:
        return events

    events = events.sort_values(
        ["original_filename", "original_begin_time_s"]
    ).reset_index(drop=True)

    drop_idx: set[int] = set()
    for _, group in events.groupby("original_filename", sort=False):
        for i, row_i in enumerate(group.itertuples(index=True)):
            if row_i.Index in drop_idx:
                continue
            for row_j in group.iloc[i + 1:].itertuples(index=True):
                if row_j.Index in drop_idx:
                    continue
                dt = row_j.original_begin_time_s - row_i.original_begin_time_s
                if dt > tolerance_s:
                    break
                if abs(row_j.original_chunk_index - row_i.original_chunk_index) != 1:
                    continue
                if row_i.max_probability >= row_j.max_probability:
                    drop_idx.add(row_j.Index)
                else:
                    drop_idx.add(row_i.Index)
                    break

    keep = [idx for idx in events.index if idx not in drop_idx]
    out = events.loc[keep].reset_index(drop=True)
    print(f"[dedup] tolerance_s={tolerance_s}  dropped={len(events) - len(out)}  kept={len(out)}")
    return out


def validate_tier(chunks: pd.DataFrame, summary_path: Path) -> None:
    if not summary_path.exists():
        print("[validate] no summary.parquet found, skipping cross-check")
        return
    orig = pd.read_parquet(summary_path)
    orig["chunk_stem"] = (
        orig["filepath"].str.split("/").str[-1].str.replace(r"\.wav$", "", regex=True)
    )
    joined = chunks.merge(
        orig[["chunk_stem", "tier"]], on="chunk_stem", suffixes=("_derived", "_orig")
    )
    print(f"[validate] overlap with original summary: {len(joined)} chunks")
    if joined.empty:
        return
    match = (joined["tier_derived"] == joined["tier_orig"]).sum()
    pct = 100.0 * match / len(joined)
    print(f"[validate] tier match: {match}/{len(joined)} ({pct:.2f}%)")
    if match < len(joined):
        mismatches = joined[joined["tier_derived"] != joined["tier_orig"]]
        print("[validate] FIRST 10 MISMATCHES:")
        print(mismatches.head(10).to_string())
        raise SystemExit("[validate] tier reconstruction is wrong; aborting")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("results/batch_lab_131204_full/detections"),
    )
    ap.add_argument(
        "--chunk-manifest",
        type=Path,
        default=Path("USV_lab_131204_chunked_2s_full/chunk_manifest.csv"),
    )
    ap.add_argument(
        "--summary-parquet",
        type=Path,
        default=Path("results/batch_lab_131204_full/summary.parquet"),
        help="Original summary.parquet (resumed-run only); used for tier validation.",
    )
    ap.add_argument(
        "--out-summary",
        type=Path,
        default=Path("results/batch_lab_131204_full/merged_summary_full.parquet"),
    )
    ap.add_argument(
        "--out-events",
        type=Path,
        default=Path("results/batch_lab_131204_full/merged_events_full.parquet"),
    )
    ap.add_argument("--dedup-tolerance-s", type=float, default=0.05)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    manifest = load_manifest(args.chunk_manifest)
    print(
        f"[manifest] rows={len(manifest)}  "
        f"unique_originals={manifest['original_filename'].nunique()}"
    )

    chunks, events = build_tables(args.detections_dir, manifest)

    print(f"\n[chunks] rows={len(chunks)}")
    print("[chunks] tier distribution:")
    print(chunks["tier"].value_counts().to_string())

    validate_tier(chunks, args.summary_parquet)

    events = dedup_overlap(events, args.dedup_tolerance_s)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(args.out_summary, index=False)
    events.to_parquet(args.out_events, index=False)
    print(f"\n[write] {args.out_summary}  ({len(chunks)} rows)")
    print(f"[write] {args.out_events}  ({len(events)} rows)")

    print("\n[summary] per-tier event totals (after dedup):")
    if not events.empty:
        print(events.groupby("tier").size().to_string())
        print(f"\n[summary] originals with >=1 event: {events['original_filename'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
