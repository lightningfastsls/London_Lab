#!/usr/bin/env python3
"""Filter raw lab-batch detection JSONs to match events_clean.parquet membership.

The lab batch ``results/batch_lab_full_softnotch_20260513_1538/detections/``
contains the full 41,563 raw CNN+soft-notch detections. The canonical Phase 2
input ``events_clean.parquet`` (41,061 rows) drops 502 long-duration noise
events via a post-hoc ``<300 ms`` filter (see
``docs/handoffs/2026-05-14_lab_131204_post_labeling.md``).

This script mirrors the parquet membership onto a sibling
``detections_clean/`` directory of JSONs so the downstream Raven exporter
(``scripts/export_raven_tables.py --batch-format``) can run unchanged — the
same path the wild-mouse pipeline (3452, 9252) took on its own detection
JSONs.

Match key: per-stem ``(start_time_s, end_time_s)`` exact float equality.
Verified that parquet and JSON both store full IEEE-754 float64 precision.

Usage::

    .venv/bin/python scripts/filter_lab_detections_by_clean_events.py \\
        --batch-dir results/batch_lab_full_softnotch_20260513_1538 \\
        --events-parquet results/batch_lab_full_softnotch_20260513_1538/events_clean.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Batch result directory containing detections/ and events_clean.parquet",
    )
    parser.add_argument(
        "--events-parquet",
        type=Path,
        default=None,
        help="Override path to events_clean.parquet (default: <batch-dir>/events_clean.parquet)",
    )
    parser.add_argument(
        "--output-subdir",
        default="detections_clean",
        help="Output directory name under batch-dir (default: detections_clean)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=41061,
        help="Expected total events after filtering (default: 41,061 for lab 131204)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    batch_dir = args.batch_dir
    raw_dir = batch_dir / "detections"
    out_dir = batch_dir / args.output_subdir
    parquet_path = args.events_parquet or (batch_dir / "events_clean.parquet")

    if not raw_dir.is_dir():
        log.error("Raw detections dir not found: %s", raw_dir)
        return 1
    if not parquet_path.is_file():
        log.error("Events parquet not found: %s", parquet_path)
        return 1

    log.info("Loading %s", parquet_path)
    events = pd.read_parquet(parquet_path)
    log.info("Parquet: %d events across %d unique stems",
             len(events), events["stem"].nunique())

    # Per-stem set of (start_s, end_s) tuples for O(1) membership tests.
    survivors_by_stem: dict[str, set[tuple[float, float]]] = {
        stem: set(zip(grp["start_s"].tolist(), grp["end_s"].tolist()))
        for stem, grp in events.groupby("stem", observed=True)
    }
    log.info("Built survivor index for %d stems", len(survivors_by_stem))

    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.json"))
    log.info("Scanning %d raw JSONs in %s", len(raw_files), raw_dir)

    kept_total = 0
    dropped_total = 0
    stems_with_data = 0
    orphans_per_stem: dict[str, int] = {}

    for src in raw_files:
        stem = src.stem
        raw_events = json.loads(src.read_text())

        wanted = survivors_by_stem.get(stem, set())
        kept: list[dict] = []
        for ev in raw_events:
            key = (ev["start_time_s"], ev["end_time_s"])
            if key in wanted:
                kept.append(ev)

        # Sanity: every wanted survivor must appear in the raw JSON for this stem.
        kept_keys = {(ev["start_time_s"], ev["end_time_s"]) for ev in kept}
        missing = wanted - kept_keys
        if missing:
            orphans_per_stem[stem] = len(missing)

        # Always write a JSON (empty list if no kept events) so the batch-format
        # exporter can iterate without a missing-file branch.
        (out_dir / src.name).write_text(json.dumps(kept))

        kept_total += len(kept)
        dropped_total += len(raw_events) - len(kept)
        if kept:
            stems_with_data += 1

    log.info("=== Filter summary ===")
    log.info("  Raw JSONs:        %d files", len(raw_files))
    log.info("  Kept events:      %d", kept_total)
    log.info("  Dropped events:   %d", dropped_total)
    log.info("  Non-empty output: %d files", stems_with_data)
    log.info("  Output dir:       %s", out_dir)

    if orphans_per_stem:
        log.error("Parquet rows orphaned in raw JSONs for %d stems (total %d orphans)",
                  len(orphans_per_stem), sum(orphans_per_stem.values()))
        for stem, n in list(orphans_per_stem.items())[:5]:
            log.error("  %s: %d orphans", stem, n)
        return 2

    if kept_total != args.expected_count:
        log.error("Kept count %d != expected %d", kept_total, args.expected_count)
        return 3

    log.info("PARITY OK: %d events kept == expected %d", kept_total, args.expected_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
