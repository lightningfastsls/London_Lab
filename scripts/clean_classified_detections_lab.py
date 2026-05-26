#!/usr/bin/env python3
"""Clean the lab-batch DeepSqueak import output for Phase 2B classifiers.

The Phase 2A import (``classified_detections_lab_131204.csv``) is an outer
join: 40,787 matched + 597 wild-mouse residue + 274 lab detections without
DS features. The downstream Phase 2B classifiers
(``classify_traditional_taxonomy.py``, ``recluster_umap_hdbscan.py``,
``analyze_acoustic_features.py``) assume well-formed inputs with all
acoustic features populated, so this script:

1. Drops outer-join NaN-side rows (rows missing either detection or DS side).
2. Filters to lab stems only (drops 597 wild-mouse 3452 residue).
3. Joins ``tier`` and ``couple`` from ``events_clean.parquet`` on ``wav_stem``.
4. Adds ``couple_keep_set`` boolean: True for the 13 retained couples,
   False for the 4 noise-prone {m1fm1, m1fm2, m1fm4, m3fm3} per the
   2026-05-14 post-labeling handoff.
5. Asserts final count == 40,787 and writes
   ``classified_detections_lab_131204_clean.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NOISE_PRONE_COUPLES = {"m1fm1", "m1fm2", "m1fm4", "m3fm3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("classified_detections_lab_131204.csv"),
        help="Phase 2A import output (outer join CSV)",
    )
    parser.add_argument(
        "--events-parquet",
        type=Path,
        default=Path("results/batch_lab_full_softnotch_20260513_1538/events_clean.parquet"),
        help="Source of tier + couple columns",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("classified_detections_lab_131204_clean.csv"),
        help="Cleaned output CSV (Phase 2B input)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=40787,
        help="Expected final row count after cleaning",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log.info("Loading %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    log.info("Input: %d rows", len(df))

    # 1. Drop outer-join NaN-side rows.
    # det_start_s NaN -> DS-only (wild residue or unmatched DS)
    # principal_freq_hz NaN -> det-only (lab detection without DS match)
    before = len(df)
    df = df.dropna(subset=["det_start_s", "principal_freq_hz"])
    log.info("After dropping NaN-side rows: %d (-%d)", len(df), before - len(df))

    # 2. Filter to lab stems only (defensive — should already be lab-only after #1
    #    since wild-mouse rows have no det_start_s, but keep the check).
    before = len(df)
    df = df[df["wav_stem"].str.startswith("131")].copy()
    if before - len(df) > 0:
        log.info("Dropped %d non-lab-stem rows", before - len(df))

    # 3. Join tier from events_clean.parquet.
    # det_start_s/det_end_s came from JSON via the Phase 2A CSV. Pandas to_csv
    # truncates float64 to ~15 sig-figs, breaking bit-exact match. Round both
    # sides to 6 decimals (1 µs); the 300 kHz sample period is 3.33 µs, so
    # sub-µs precision has no physical meaning anyway.
    log.info("Loading %s", args.events_parquet)
    events = pd.read_parquet(args.events_parquet).copy()
    events["start_s_us"] = events["start_s"].round(6)
    events["end_s_us"] = events["end_s"].round(6)
    df["det_start_s_us"] = df["det_start_s"].round(6)
    df["det_end_s_us"] = df["det_end_s"].round(6)
    df = df.merge(
        events[["stem", "start_s_us", "end_s_us", "tier"]],
        how="left",
        left_on=["wav_stem", "det_start_s_us", "det_end_s_us"],
        right_on=["stem", "start_s_us", "end_s_us"],
    )
    df = df.drop(columns=["stem", "start_s_us", "end_s_us", "det_start_s_us", "det_end_s_us"])

    n_no_tier = df["tier"].isna().sum()
    if n_no_tier > 0:
        log.warning("%d rows did not match parquet on (stem, start_s, end_s)", n_no_tier)

    # Extract couple from wav_stem: e.g. 131204_1400_m1fm1_chunk_022 -> m1fm1
    df["couple"] = df["wav_stem"].str.extract(r"(m\d+fm\d+)", expand=False)
    n_no_couple = df["couple"].isna().sum()
    if n_no_couple > 0:
        log.warning("%d rows could not extract couple from wav_stem", n_no_couple)

    # 4. Add couple_keep_set boolean.
    df["couple_keep_set"] = ~df["couple"].isin(NOISE_PRONE_COUPLES)

    # Add an animal_id column (each couple is its own "animal" for the purposes
    # of repertoire-stats nesting; could be refined later to the male alone).
    df["animal_id"] = df["couple"]

    log.info("Final shape: %d rows × %d cols", *df.shape)
    log.info("Tier breakdown: %s", df["tier"].value_counts(dropna=False).to_dict())
    log.info("Couple breakdown: %s", df["couple"].value_counts(dropna=False).to_dict())
    log.info("couple_keep_set: %s", df["couple_keep_set"].value_counts(dropna=False).to_dict())

    if len(df) != args.expected_count:
        log.error("Final count %d != expected %d", len(df), args.expected_count)
        return 1

    df.to_csv(args.output_csv, index=False)
    log.info("Wrote %d rows to %s", len(df), args.output_csv)
    log.info("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
