"""Stratified-random mining of lab auto_accept events for fine-tune labeling.

Samples N candidates from typical / long_event / borderline strata of the
production lab batch parquet, EXCLUDING events already labeled in the
2026-05-08 audit. Renders PNGs using the same `render_event` function as
the audit so the visual style matches.

Output:
    --out-dir/
        candidates_<seed>.csv   (stable mining IDs min001-minNNN -> event keys)
        renders/
            min001_<chunk>_ev<idx>.png
            min002_*.png
            ...

Usage:
    .venv/bin/python scripts/mine_lab_finetune_candidates.py \
        --n-typical 100 --n-long-event 30 --n-borderline 20 \
        --out-dir data/lab_finetune_v1/mining_candidates_150 \
        --exclude-labels-csv data/lab_finetune_v1/labels_audit_72.csv \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from render_filter_validation_pngs import (  # noqa: E402
    DEFAULT_CORPUS_FACTS,
    render_event,
)
from usv_spectrogram.postprocessing.hysteresis import HysteresisConfig  # noqa: E402

import json

LONG_EVENT_THRESHOLD_S = HysteresisConfig().max_duration_ms / 1000.0
TYPICAL_DUR_MIN_S = 0.05
TYPICAL_DUR_MAX_S = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--events-with-filter", type=Path,
                   default=Path("results/batch_lab_131204_full/merged_events_with_filter.parquet"))
    p.add_argument("--chunks-dir", type=Path,
                   default=Path("USV_lab_131204_chunked_2s_full"))
    p.add_argument("--exclude-labels-csv", type=Path, required=True,
                   help="CSV of already-labeled events (chunk_stem,event_idx columns) to exclude")
    p.add_argument("--n-typical", type=int, default=100)
    p.add_argument("--n-long-event", type=int, default=30)
    p.add_argument("--n-borderline", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--corpus-facts", type=Path, default=DEFAULT_CORPUS_FACTS)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    with args.corpus_facts.open() as fh:
        facts = json.load(fh)
    contrast_db = float(facts["noise_filter"]["contrast_db"])
    print(f"[params] contrast_db={contrast_db}")

    df = pd.read_parquet(args.events_with_filter)
    print(f"[load] {len(df):,} events from {args.events_with_filter.name}")
    aa = df[df["tier"] == "auto_accept"].copy()
    print(f"[filter] {len(aa):,} auto_accept events")

    excluded = set()
    if args.exclude_labels_csv.exists():
        with args.exclude_labels_csv.open() as fh:
            for row in csv.DictReader(fh):
                excluded.add((row["chunk_stem"], int(row["event_idx"])))
        print(f"[exclude] {len(excluded)} already-labeled events removed from sample pool")

    keys = list(zip(aa["chunk_stem"], aa["chunk_detection_idx"].astype(int)))
    keep_mask = [k not in excluded for k in keys]
    aa = aa.loc[keep_mask].reset_index(drop=True)
    print(f"[filter] {len(aa):,} auto_accept events after exclusion")

    sef = aa["stationary_energy_fraction"]
    dur = aa["duration_s"]

    long_event_mask = dur > LONG_EVENT_THRESHOLD_S
    typical_mask = (
        (sef > 0.05) & (sef <= 0.15)
        & (dur >= TYPICAL_DUR_MIN_S) & (dur <= TYPICAL_DUR_MAX_S)
        & ~long_event_mask
    )
    borderline_mask = (
        (sef > 0.20) & (sef <= 0.50)
        & ~long_event_mask
    )

    typical_pool = aa[typical_mask]
    long_event_pool = aa[long_event_mask]
    borderline_pool = aa[borderline_mask]

    print(f"[pool] typical    : {len(typical_pool):>5}  (target sample {args.n_typical})")
    print(f"[pool] long_event : {len(long_event_pool):>5}  (target sample {args.n_long_event})")
    print(f"[pool] borderline : {len(borderline_pool):>5}  (target sample {args.n_borderline})")

    if len(typical_pool) < args.n_typical:
        print(f"[warn] typical pool smaller than requested ({len(typical_pool)} < {args.n_typical})")
    if len(long_event_pool) < args.n_long_event:
        print(f"[warn] long_event pool smaller than requested")
    if len(borderline_pool) < args.n_borderline:
        print(f"[warn] borderline pool smaller than requested")

    samples = {
        "typical": typical_pool.sample(min(args.n_typical, len(typical_pool)),
                                       random_state=args.seed),
        "long_event": long_event_pool.sample(min(args.n_long_event, len(long_event_pool)),
                                             random_state=args.seed),
        "borderline": borderline_pool.sample(min(args.n_borderline, len(borderline_pool)),
                                             random_state=args.seed),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = args.out_dir / "renders"
    renders_dir.mkdir(exist_ok=True)
    csv_path = args.out_dir / f"candidates_seed{args.seed}.csv"

    rows_out = []
    mid = 0
    for stratum, sub in samples.items():
        sub = sub.sort_values(["chunk_stem", "chunk_detection_idx"]).reset_index(drop=True)
        for _, ev in sub.iterrows():
            mid += 1
            stable_id = f"min{mid:03d}"
            chunk_wav = args.chunks_dir / f"{ev['chunk_stem']}.wav"
            if not chunk_wav.exists():
                print(f"[skip] {stable_id}: WAV not found ({chunk_wav})")
                continue
            png_name = f"{stable_id}_{ev['chunk_stem']}_ev{int(ev['chunk_detection_idx']):03d}.png"
            png_path = renders_dir / png_name
            render_event(chunk_wav, ev, png_path, contrast_db,
                         f"MINE-{stratum.upper()}")
            rows_out.append({
                "mining_id": stable_id,
                "stratum": stratum,
                "png_filename": png_name,
                "chunk_stem": ev["chunk_stem"],
                "event_idx": int(ev["chunk_detection_idx"]),
                "sef": float(ev["stationary_energy_fraction"]),
                "duration_ms": int(round(ev["duration_s"] * 1000)),
                "max_probability": float(ev["max_probability"]),
                "verdict": "",
                "reasoning": "",
            })
            if mid % 10 == 0:
                print(f"  rendered {mid}")

    fieldnames = list(rows_out[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n[done] {len(rows_out)} candidates rendered to {renders_dir}/")
    print(f"[done] candidates CSV: {csv_path}")
    print()
    print("=== sample distribution ===")
    from collections import Counter
    strata = Counter(r["stratum"] for r in rows_out)
    for s, n in strata.items():
        print(f"  {s}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
