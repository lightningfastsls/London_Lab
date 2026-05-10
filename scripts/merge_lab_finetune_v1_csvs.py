#!/usr/bin/env python3
"""Merge lab fine-tune labels into matched_windows_v2 training CSVs.

Phase 3 data prep for lab_finetune_v1. Produces three CSVs in
``data/training/lab_finetune_v1/csv/``:

1. train.csv  — wild train (matched_windows_v2) + lab train (12 sessions)
                with sample_weight: wild=1.0, lab=3.0
2. val.csv    — wild val (matched_windows_v2)  + lab held-out sessions
                (131209_1000 + 131217_1400). No sample_weight.
3. test.csv   — wild test copied unchanged for later evaluation.

Lab→wild schema alignment:
    candidate_id      ←  f"lab_{chunk_stem}_ev{event_idx:03d}"
    source_file       ←  source_recording          (e.g. "131204_1400_m1fm1.wav")
    label             ←  "USV" → "USV"; "NOISE" → "Not USV"
    spectrogram_path  ←  png_path                  (already relative to repo root)
    dataset           ←  "wild" | "lab"            (provenance)
    sample_weight     ←  3.0 (lab) | 1.0 (wild) — train only

Usage:
    .venv/bin/python scripts/merge_lab_finetune_v1_csvs.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

WILD_DIR = REPO_ROOT / "data" / "training" / "matched_windows_v2"
LAB_MANIFEST = REPO_ROOT / "data" / "training" / "lab_finetune_v1" / "manifest.csv"
OUT_DIR = REPO_ROOT / "data" / "training" / "lab_finetune_v1" / "csv"

HELD_OUT_SESSIONS = {"131209_1000", "131217_1400"}
LAB_SAMPLE_WEIGHT = 3.0
WILD_SAMPLE_WEIGHT = 1.0

LABEL_MAP = {"USV": "USV", "NOISE": "Not USV"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def session_of(source_recording: str) -> str:
    """Extract session ID (date_time) from a pair-recording filename.

    "131204_1400_m1fm1.wav" → "131204_1400"
    """
    stem = source_recording.replace(".wav", "")
    return "_".join(stem.split("_")[:-1])


def lab_to_wild_schema(lab: pd.DataFrame) -> pd.DataFrame:
    """Reformat lab manifest rows to wild train/val schema."""
    out = pd.DataFrame()
    out["candidate_id"] = (
        "lab_"
        + lab["chunk_stem"].astype(str)
        + "_ev"
        + lab["event_idx"].astype(int).map(lambda x: f"{x:03d}")
    )
    out["source_file"] = lab["source_recording"]
    out["label"] = lab["label"].map(LABEL_MAP)
    out["spectrogram_path"] = lab["png_path"]

    bad = out["label"].isna()
    if bad.any():
        raise ValueError(
            f"Unexpected lab labels: {lab.loc[bad, 'label'].unique().tolist()}"
        )
    return out


def main() -> int:
    log.info("=" * 64)
    log.info("LAB FINE-TUNE V1 — CSV MERGE")
    log.info("=" * 64)
    log.info("Wild dir:        %s", WILD_DIR)
    log.info("Lab manifest:    %s", LAB_MANIFEST)
    log.info("Output dir:      %s", OUT_DIR)
    log.info("Held-out:        %s", sorted(HELD_OUT_SESSIONS))
    log.info("Sample weights:  lab=%.1f, wild=%.1f", LAB_SAMPLE_WEIGHT, WILD_SAMPLE_WEIGHT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load lab manifest ---
    lab = pd.read_csv(LAB_MANIFEST)
    lab["session"] = lab["source_recording"].apply(session_of)

    in_holdout = lab["session"].isin(HELD_OUT_SESSIONS)
    lab_train = lab[~in_holdout].copy()
    lab_val = lab[in_holdout].copy()

    log.info(
        "Lab events: %d total, %d train (12 sessions), %d val (held-out 2 sessions)",
        len(lab), len(lab_train), len(lab_val),
    )
    log.info(
        "  train USV/NOISE: %d / %d",
        int((lab_train["label"] == "USV").sum()),
        int((lab_train["label"] == "NOISE").sum()),
    )
    log.info(
        "  val   USV/NOISE: %d / %d",
        int((lab_val["label"] == "USV").sum()),
        int((lab_val["label"] == "NOISE").sum()),
    )

    lab_train_wild = lab_to_wild_schema(lab_train)
    lab_val_wild = lab_to_wild_schema(lab_val)

    # --- Load wild splits ---
    wild_train = pd.read_csv(WILD_DIR / "train.csv")
    wild_val = pd.read_csv(WILD_DIR / "val.csv")
    wild_test = pd.read_csv(WILD_DIR / "test.csv")

    keep_cols = ["candidate_id", "source_file", "label", "spectrogram_path"]
    wild_train = wild_train[keep_cols].copy()
    wild_val = wild_val[keep_cols].copy()
    wild_test = wild_test[keep_cols].copy()

    log.info(
        "Wild events: train=%d, val=%d, test=%d",
        len(wild_train), len(wild_val), len(wild_test),
    )

    # --- Build merged train ---
    wild_train["dataset"] = "wild"
    wild_train["sample_weight"] = WILD_SAMPLE_WEIGHT
    lab_train_wild["dataset"] = "lab"
    lab_train_wild["sample_weight"] = LAB_SAMPLE_WEIGHT

    merged_train = pd.concat([wild_train, lab_train_wild], ignore_index=True)

    # --- Build merged val (no sample_weight; val sampling stays uniform) ---
    wild_val["dataset"] = "wild"
    lab_val_wild["dataset"] = "lab"
    merged_val = pd.concat([wild_val, lab_val_wild], ignore_index=True)

    # --- Test stays wild-only for direct comparability ---
    wild_test["dataset"] = "wild"

    # --- Sanity checks ---
    if merged_train["candidate_id"].duplicated().any():
        dups = merged_train[merged_train["candidate_id"].duplicated(keep=False)]
        raise ValueError(
            f"Duplicate candidate_id in merged train: {dups['candidate_id'].head().tolist()}"
        )
    if merged_val["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate_id in merged val")

    # --- Write ---
    merged_train.to_csv(OUT_DIR / "train.csv", index=False)
    merged_val.to_csv(OUT_DIR / "val.csv", index=False)
    wild_test.to_csv(OUT_DIR / "test.csv", index=False)

    # --- Summary ---
    log.info("=" * 64)
    log.info("MERGED OUTPUTS")
    log.info("=" * 64)
    for name, df in [("train", merged_train), ("val", merged_val), ("test", wild_test)]:
        log.info("%-5s: %d rows", name, len(df))
        if "dataset" in df.columns:
            log.info("       by dataset: %s", df["dataset"].value_counts().to_dict())
        log.info("       by label:   %s", df["label"].value_counts().to_dict())
        if "sample_weight" in df.columns:
            log.info(
                "       weighted USV-fraction in batches: %.1f%%",
                100 * (
                    (df["sample_weight"] * (df["label"] == "USV")).sum()
                    / df["sample_weight"].sum()
                ),
            )
            log.info(
                "       weighted lab-fraction in batches: %.1f%%",
                100 * (
                    (df["sample_weight"] * (df["dataset"] == "lab")).sum()
                    / df["sample_weight"].sum()
                ),
            )
    log.info("=" * 64)
    log.info("Wrote: %s", OUT_DIR / "train.csv")
    log.info("Wrote: %s", OUT_DIR / "val.csv")
    log.info("Wrote: %s", OUT_DIR / "test.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
