#!/usr/bin/env python3
"""Extract labeled lab events to CNN training-format PNGs.

Phase 2 of the lab CNN fine-tune pipeline. Reads the two lab-label CSVs
produced during Phases 0+1, joins to the batch-detection parquet for
column timing, and renders 100x256 PNGs that match the production CNN's
training grid exactly.

Drift safety: this script does NOT re-implement spectrogram extraction.
It instantiates ``DatasetAssembler`` (the same class that produced the
wild training set) and calls its private ``_load_global_spectrogram``
and ``_render_window_png`` helpers directly. A dummy
``unified_labels_path`` satisfies ``AssemblyConfig.__post_init__``;
``assemble()`` is never called, so no I/O happens against that path.

Parity is enforced by ``scripts/parity_test_lab_extractor.py`` against an
existing wild hard-negative PNG. Run that test before trusting these
outputs for training.

Usage:
    .venv/bin/python scripts/extract_lab_finetune_pngs.py \
        --output-dir data/training/lab_finetune_v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.dataset.assembler import AssemblyConfig, DatasetAssembler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

WINDOW_COLUMNS = 100
HALF_WINDOW = WINDOW_COLUMNS // 2

LABELS_AUDIT_CSV = REPO_ROOT / "data" / "lab_finetune_v1" / "labels_audit_72.csv"
LABELS_MINING_CSV = (
    REPO_ROOT
    / "data"
    / "lab_finetune_v1"
    / "mining_candidates_500"
    / "candidates_seed42.csv"
)
MERGED_PARQUET = (
    REPO_ROOT
    / "results"
    / "batch_lab_131204_full"
    / "merged_events_with_filter.parquet"
)
WAV_DIR = REPO_ROOT / "USV_lab_131204_chunked_2s_full"


def load_labels() -> pd.DataFrame:
    """Read both label CSVs and return a unified frame.

    Output columns: chunk_stem, event_idx, label, source_csv.
    Verdict is normalized to label in {"USV", "NOISE"}.
    """
    audit = pd.read_csv(LABELS_AUDIT_CSV)
    audit_keep = audit[["chunk_stem", "event_idx", "verdict"]].copy()
    audit_keep["source_csv"] = "audit_72"

    mining = pd.read_csv(LABELS_MINING_CSV)
    mining_keep = mining[["chunk_stem", "event_idx", "verdict"]].copy()
    mining_keep["source_csv"] = "mining_500"

    combined = pd.concat([audit_keep, mining_keep], ignore_index=True)
    combined["label"] = combined["verdict"].str.upper().str.strip()

    bad = combined[~combined["label"].isin({"USV", "NOISE"})]
    if len(bad) > 0:
        raise ValueError(
            f"Unexpected verdict values: {bad['verdict'].unique().tolist()}"
        )

    # bor05 == lng10 in the audit set per the handoff (same event in two
    # buckets). Drop exact duplicates on the join key + label.
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["chunk_stem", "event_idx"], keep="first"
    )
    log.info("Labels: %d rows after dedup (%d duplicates dropped)",
             len(combined), before - len(combined))

    return combined[["chunk_stem", "event_idx", "label", "source_csv"]]


def join_to_parquet(labels: pd.DataFrame) -> pd.DataFrame:
    """Inner-join labels to merged events to recover start_col/end_col.

    Hard-fails if any label row has no parquet match.
    """
    parquet = pd.read_parquet(MERGED_PARQUET)

    keep_cols = [
        "chunk_stem",
        "chunk_detection_idx",
        "start_col",
        "end_col",
        "original_filename",
        "original_chunk_index",
        "duration_s",
    ]
    parquet_subset = parquet[keep_cols].rename(
        columns={"chunk_detection_idx": "event_idx"}
    )

    joined = labels.merge(
        parquet_subset, on=["chunk_stem", "event_idx"], how="left"
    )
    missing = joined[joined["start_col"].isna()]
    if len(missing) > 0:
        log.error("Label rows with no parquet match (first 5):")
        log.error(missing.head().to_string())
        raise ValueError(
            f"{len(missing)} labeled events have no row in {MERGED_PARQUET}"
        )

    joined["start_col"] = joined["start_col"].astype(int)
    joined["end_col"] = joined["end_col"].astype(int)
    joined["original_chunk_index"] = joined["original_chunk_index"].astype(int)
    return joined


def make_assembler() -> DatasetAssembler:
    """Construct an assembler we will use only for its private extraction
    methods. Dummy unified_labels_path satisfies post-init validation; we
    never call .assemble() so no I/O happens against it."""
    cfg = AssemblyConfig(
        unified_labels_path=Path("unused-by-this-script.csv"),
        use_global_mad=True,
        window_columns=WINDOW_COLUMNS,
        # All other defaults are inherited from AssemblyConfig.
    )
    return DatasetAssembler(cfg)


def print_params(label_counts: dict[str, int]) -> None:
    log.info("=" * 64)
    log.info("LAB FINE-TUNE EXTRACTOR — PARAMETERS")
    log.info("=" * 64)
    log.info("Labels (audit_72):  %s", LABELS_AUDIT_CSV)
    log.info("Labels (mining):    %s", LABELS_MINING_CSV)
    log.info("Merged parquet:     %s", MERGED_PARQUET)
    log.info("WAV dir:            %s", WAV_DIR)
    log.info("Window columns:     %d (centered, no jitter)", WINDOW_COLUMNS)
    log.info(
        "Extraction class:   DatasetAssembler "
        "(SAMPLE_RATE=%d, N_FFT=%d, HOP=%d)",
        DatasetAssembler.SAMPLE_RATE,
        DatasetAssembler.N_FFT,
        DatasetAssembler.HOP_LENGTH,
    )
    for k, v in label_counts.items():
        log.info("Labels — %-12s %d", k + ":", v)
    log.info("=" * 64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "training" / "lab_finetune_v1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N labeled events (debug)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = load_labels()
    rows = join_to_parquet(labels)

    label_counts = {
        "total": len(rows),
        "USV": int((rows["label"] == "USV").sum()),
        "NOISE": int((rows["label"] == "NOISE").sum()),
        "audit_72": int((rows["source_csv"] == "audit_72").sum()),
        "mining_500": int((rows["source_csv"] == "mining_500").sum()),
    }
    print_params(label_counts)

    if args.dry_run:
        log.info("Dry run; exiting before extraction.")
        return 0

    if args.limit:
        rows = rows.head(args.limit)
        log.info("Limited to first %d events for debugging", args.limit)

    output_dir = args.output_dir
    (output_dir / "usv").mkdir(parents=True, exist_ok=True)
    (output_dir / "noise").mkdir(parents=True, exist_ok=True)

    assembler = make_assembler()

    spec_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    manifest_rows: list[dict] = []
    skipped_off_edge = 0
    skipped_missing_wav = 0

    for i, (_, row) in enumerate(rows.iterrows(), 1):
        chunk_stem = row["chunk_stem"]
        event_idx = int(row["event_idx"])
        label = row["label"]

        if chunk_stem not in spec_cache:
            wav_path = WAV_DIR / f"{chunk_stem}.wav"
            if not wav_path.exists():
                log.warning("WAV missing: %s", wav_path)
                skipped_missing_wav += 1
                continue
            spec_cache[chunk_stem] = assembler._load_global_spectrogram(wav_path)

        spec_norm, _times_s = spec_cache[chunk_stem]
        n_cols = spec_norm.shape[1]

        center_col = (int(row["start_col"]) + int(row["end_col"])) // 2
        win_start = center_col - HALF_WINDOW
        win_end = win_start + WINDOW_COLUMNS

        # Clip to chunk edges; if the centered window doesn't fit, skip.
        if win_start < 0:
            win_start = 0
            win_end = WINDOW_COLUMNS
        if win_end > n_cols:
            log.warning(
                "Skipping %s ev%03d: window [%d,%d] off-edge (n_cols=%d)",
                chunk_stem, event_idx, win_start, win_end, n_cols,
            )
            skipped_off_edge += 1
            continue

        window_data = spec_norm[:, win_start:win_end]
        if window_data.shape[1] != WINDOW_COLUMNS:
            skipped_off_edge += 1
            continue

        sub_dir = "usv" if label == "USV" else "noise"
        png_path = output_dir / sub_dir / f"{chunk_stem}_ev{event_idx:03d}.png"
        assembler._render_window_png(window_data, png_path)

        manifest_rows.append({
            "png_path": str(png_path.relative_to(REPO_ROOT)),
            "label": label,
            "source_recording": row["original_filename"],
            "original_chunk": int(row["original_chunk_index"]),
            "chunk_stem": chunk_stem,
            "event_idx": event_idx,
            "start_col": int(row["start_col"]),
            "end_col": int(row["end_col"]),
            "duration_s": float(row["duration_s"]),
            "source_csv": row["source_csv"],
        })

        if i % 50 == 0:
            log.info("[%d/%d] extracted %d PNGs (cache=%d chunks)",
                     i, len(rows), len(manifest_rows), len(spec_cache))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    log.info("=" * 64)
    log.info("DONE")
    log.info("Extracted: %d PNGs", len(manifest_df))
    log.info("  USV:   %d", int((manifest_df["label"] == "USV").sum()))
    log.info("  NOISE: %d", int((manifest_df["label"] == "NOISE").sum()))
    log.info("Skipped off-edge: %d", skipped_off_edge)
    log.info("Skipped missing WAV: %d", skipped_missing_wav)
    log.info("Manifest: %s", manifest_path)
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
