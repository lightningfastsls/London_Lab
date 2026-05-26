"""Merge per-chunk batch detection JSONs into a per-event CSV in original-file time.

The lab cohort (USV_lab_131204) is processed in 2 s chunks with 0.1 s overlap
because the wild-trained pipeline was sized for short trigger-recorded files
and OOMs on continuous 10-min recordings (see
``scripts/chunk_and_resample_lab_to_300k.py``).

After detection runs on the chunked WAVs, this script unwinds the chunking:

1. Reads each per-chunk detection JSON (output of run_batch_detection.py).
2. Joins against ``chunk_manifest.csv`` to recover ``(original_filename,
   start_s_in_original)`` for every chunk.
3. Translates per-chunk event times to original-file times:

       original_begin_time_s = start_time_s + start_s_in_original
       original_end_time_s   = end_time_s   + start_s_in_original

4. Dedups overlap-region duplicates: an event near a chunk boundary may appear
   in two consecutive chunks because of the 0.1 s overlap. Two events from
   adjacent chunks of the same original file with
   ``|delta_original_begin_time| < tolerance`` are collapsed to one row, keeping
   the higher ``max_probability``.

Output schema mirrors ``scripts/merge_batch_detections.py`` plus original-file
columns. Downstream consumers (Raven export, DeepSqueak bridge, classification,
corpus_facts/lab_131204.json) should use the ``original_*`` columns and the
``original_filename`` as the cohort-level identifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CHUNK_LEVEL_COLS = [
    "chunk_stem",
    "chunk_detection_idx",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "max_probability",
    "mean_probability",
    "start_col",
    "end_col",
]
ORIGINAL_LEVEL_COLS = [
    "original_filename",
    "original_chunk_index",
    "start_s_in_original",
    "original_begin_time_s",
    "original_end_time_s",
]
OUTPUT_COLS = ORIGINAL_LEVEL_COLS + CHUNK_LEVEL_COLS


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    expected = {"chunk_filename", "original_filename", "chunk_index", "start_s_in_original"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    df["chunk_stem"] = df["chunk_filename"].str.replace(r"\.wav$", "", regex=True)
    return df.set_index("chunk_stem")


def explode_chunk_jsons(detections_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    json_files = sorted(detections_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {detections_dir}")

    n_with_events = 0
    n_orphan = 0
    for jp in json_files:
        stem = jp.stem
        if stem not in manifest.index:
            n_orphan += 1
            continue
        m = manifest.loc[stem]
        events = json.loads(jp.read_text())
        if not isinstance(events, list):
            raise ValueError(f"{jp}: expected list at top level, got {type(events).__name__}")
        if events:
            n_with_events += 1
        for det_idx, ev in enumerate(events):
            rows.append({
                "chunk_stem": stem,
                "chunk_detection_idx": det_idx,
                "start_time_s": ev["start_time_s"],
                "end_time_s": ev["end_time_s"],
                "duration_s": ev["duration_s"],
                "max_probability": ev["max_probability"],
                "mean_probability": ev["mean_probability"],
                "start_col": ev.get("start_col"),
                "end_col": ev.get("end_col"),
                "original_filename": m["original_filename"],
                "original_chunk_index": int(m["chunk_index"]),
                "start_s_in_original": float(m["start_s_in_original"]),
                "original_begin_time_s": float(m["start_s_in_original"]) + float(ev["start_time_s"]),
                "original_end_time_s": float(m["start_s_in_original"]) + float(ev["end_time_s"]),
            })

    if n_orphan:
        print(f"[warn] {n_orphan} chunk JSONs had no manifest entry (skipped)", file=sys.stderr)
    print(f"[merge] json_files={len(json_files)}  with_events={n_with_events}  events={len(rows)}")

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLS)
    df = pd.DataFrame(rows)
    return df.sort_values(["original_filename", "original_begin_time_s"]).reset_index(drop=True)


def dedup_overlap_region(
    df: pd.DataFrame,
    tolerance_s: float,
    require_adjacent_chunks: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Collapse duplicates from the chunk overlap region.

    A USV near a chunk boundary may appear once in chunk N and once in chunk N+1
    because the chunker uses 0.1 s overlap. Two events match if:

    * Same ``original_filename``
    * ``|delta_original_begin_time_s| < tolerance_s``
    * If ``require_adjacent_chunks`` is True, ``|delta_chunk_index| == 1``

    The higher ``max_probability`` wins.
    """
    if df.empty:
        return df, 0

    keep = []
    drop_idx: set[int] = set()
    n_dropped = 0

    for orig_name, group in df.groupby("original_filename", sort=False):
        idx_array = group.index.to_numpy()
        for i, row_i in enumerate(group.itertuples(index=True)):
            if row_i.Index in drop_idx:
                continue
            for row_j in group.iloc[i + 1:].itertuples(index=True):
                if row_j.Index in drop_idx:
                    continue
                dt = row_j.original_begin_time_s - row_i.original_begin_time_s
                if dt > tolerance_s:
                    break
                if require_adjacent_chunks and abs(row_j.original_chunk_index - row_i.original_chunk_index) != 1:
                    continue
                if row_i.max_probability >= row_j.max_probability:
                    drop_idx.add(row_j.Index)
                else:
                    drop_idx.add(row_i.Index)
                    break
        kept_in_group = [idx for idx in idx_array if idx not in drop_idx]
        keep.extend(kept_in_group)

    n_dropped = len(df) - len(keep)
    out = df.loc[keep].sort_values(["original_filename", "original_begin_time_s"]).reset_index(drop=True)
    print(f"[dedup] tolerance_s={tolerance_s}  dropped={n_dropped}  kept={len(out)}")
    return out, n_dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--chunk-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dedup-tolerance-s", type=float, default=0.05,
                        help="Overlap-region dedup tolerance (default 0.05 s, half of 0.1 s overlap).")
    parser.add_argument("--no-dedup", action="store_true", help="Skip dedup pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[merge] detections_dir={args.detections_dir}")
    print(f"[merge] chunk_manifest={args.chunk_manifest}")
    print(f"[merge] output={args.output}  dedup_tolerance_s={args.dedup_tolerance_s}  dedup={'OFF' if args.no_dedup else 'ON'}")

    manifest = load_manifest(args.chunk_manifest)
    print(f"[merge] manifest_rows={len(manifest)}  unique_originals={manifest['original_filename'].nunique()}")

    df = explode_chunk_jsons(args.detections_dir, manifest)

    if not args.no_dedup:
        df, _ = dedup_overlap_region(df, tolerance_s=args.dedup_tolerance_s)

    df = df[OUTPUT_COLS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"[merge] wrote {args.output}  rows={len(df)}")
    if len(df) > 0:
        print(f"[merge] events per original (top 5):")
        per_orig = df.groupby("original_filename").size().sort_values(ascending=False)
        print(per_orig.head().to_string())
        print(f"[merge] median events per original: {per_orig.median():.0f}")
        print(f"[merge] originals with no events: {manifest['original_filename'].nunique() - per_orig.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
