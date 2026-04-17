"""Generate a per-dataset empirical-data JSON (``corpus_facts``).

Every USV analysis script that needs "how many calls?", "what is the
median silent gap?", "what's the bout threshold?" should NOT recompute
those numbers on demand and throw them away. This script reads the
existing classified CSVs + sequential-structure outputs and emits
``data/corpus_facts/<dataset>.json`` — a stable, versioned registry that
downstream scripts can cite.

Follows the same parameters-sidecar pattern landed in
``scripts/run_sis_baselines.py`` (Phase 17 module 17.1): every analysis
run must state its inputs, filter rules, and row counts. The JSON
payload doubles as that audit trail.

Usage
-----
    python scripts/audit_corpus.py --dataset 5970 --output data/corpus_facts/5970.json
    python scripts/audit_corpus.py --all      # processes 5970, warns for 3452/9252 if inputs missing

Known sanity-check anchors for 5970 (verified 2026-04-17):
    n_calls_raw = 7921, n_calls_after_dropna_file = 7864
    median_ici_gap_ms ≈ 86.68, median_ioi_ms ≈ 192.99
    q25_ici_gap_ms ≈ 65.14, q75_ici_gap_ms ≈ 209.11
    n_negative_gaps = 10, n_cross_file_pairs_over_10s = 829
    n_bouts = 1238, n_within_bout_pairs = 6350
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Pattern 8 — path bootstrap for both usv_spectrogram (src/) and usv_language (root)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (SRC_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ── Dataset registry ───────────────────────────────────────────────────────
#
# Paths are relative to REPO_ROOT. If any required input is missing for a
# dataset, ``audit_corpus`` prints a warning and skips (per user decision
# 2026-04-17 — 3452/9252 inputs don't exist yet; B1 and 9252 detection in
# flight). Adjust these paths when Phase B1 ships new classified CSVs.

DATASET_REGISTRY: dict[str, dict[str, Path]] = {
    "5970": {
        "classified_csv": REPO_ROOT / "results/traditional_taxonomy/classified_traditional.csv",
        "hdbscan_csv": REPO_ROOT / "results/recluster_umap_hdbscan/reclassified_detections.csv",
        "detection_csv": REPO_ROOT / "results/batch_5970/manual_review_all_detections.csv",
        "ici_gap_npy": REPO_ROOT / "results/sequential_structure/ici_gap.npy",
        "ici_onset_npy": REPO_ROOT / "results/sequential_structure/ici_onset.npy",
        "sequential_summary_csv": REPO_ROOT / "results/sequential_structure/sequential_structure_summary.csv",
    },
    # Phase B1 is scheduled to produce these; the script warns-and-skips until they exist.
    "3452": {
        "classified_csv": REPO_ROOT / "results/traditional_taxonomy/classified_traditional_3452.csv",
        "hdbscan_csv": REPO_ROOT / "results/recluster_umap_hdbscan/reclassified_detections_3452.csv",
        "detection_csv": REPO_ROOT / "results/batch_3452_reviewed/manual_review_all_detections.csv",
        "ici_gap_npy": REPO_ROOT / "results/sequential_structure_3452/ici_gap.npy",
        "ici_onset_npy": REPO_ROOT / "results/sequential_structure_3452/ici_onset.npy",
        "sequential_summary_csv": REPO_ROOT / "results/sequential_structure_3452/sequential_structure_summary.csv",
    },
    "9252": {
        "classified_csv": REPO_ROOT / "results/traditional_taxonomy/classified_traditional_9252.csv",
        "hdbscan_csv": REPO_ROOT / "results/recluster_umap_hdbscan/reclassified_detections_9252.csv",
        "detection_csv": REPO_ROOT / "results/batch_9252/manual_review_all_detections.csv",
        "ici_gap_npy": REPO_ROOT / "results/sequential_structure_9252/ici_gap.npy",
        "ici_onset_npy": REPO_ROOT / "results/sequential_structure_9252/ici_onset.npy",
        "sequential_summary_csv": REPO_ROOT / "results/sequential_structure_9252/sequential_structure_summary.csv",
    },
}


# Required inputs (missing any → skip the dataset). hdbscan_csv is optional —
# analyses older than Phase B can run without it.
REQUIRED_INPUTS = ("classified_csv", "ici_gap_npy", "ici_onset_npy", "sequential_summary_csv")


# Literature references (constants — same across datasets, recorded here
# so any analysis that cites these doesn't have to rediscover them).
LITERATURE_REFERENCES = {
    "median_within_bout_silent_gap_ms": 90,
    "inter_bout_threshold_ms_range": [250, 500],
    "hertz_2020_imsa_sis_bits": 0.22,
    "hertz_2020_ivoice_sis_bits": 0.10,
    "hertz_2020_imupet_sis_bits": 0.13,
}


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-dataset corpus-facts JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/audit_corpus.py --dataset 5970 --output data/corpus_facts/5970.json\n"
            "  python scripts/audit_corpus.py --all\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Process a single dataset (5970, 3452, or 9252).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process every registered dataset; warns and skips any with missing inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (required with --dataset). "
             "With --all, outputs go to data/corpus_facts/<dataset>.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/corpus_facts",
        help="Output directory when --all is used (default: data/corpus_facts/).",
    )
    return parser.parse_args()


# ── Dataset stat computation ──────────────────────────────────────────────

def _inputs_missing(paths: dict[str, Path]) -> list[str]:
    return [key for key in REQUIRED_INPUTS if not paths[key].exists()]


def _compute_counts(classified: pd.DataFrame) -> dict[str, int]:
    n_raw = len(classified)
    deduped = classified.dropna(subset=["file"]).copy()
    return {
        "n_calls_raw": int(n_raw),
        "n_calls_after_dropna_file": int(len(deduped)),
        "n_files": int(deduped["file"].nunique()) if "file" in deduped.columns else 0,
        # A "session" = unique recording date (one YYYY-MM-DD per session).
        # Derived from the filename prefix since that's where the timestamp lives.
        "n_sessions": int(
            deduped["file"].dropna().str.extract(r"^(\d{4}-\d{2}-\d{2})", expand=False).nunique()
        ) if "file" in deduped.columns else 0,
    }


def _compute_timing(
    classified: pd.DataFrame, ici_gap: np.ndarray, ici_onset: np.ndarray
) -> dict[str, float | int]:
    deduped = classified.dropna(subset=["file"]).copy()

    # Median call duration (end_time_s - begin_time_s, in ms)
    if {"end_time_s", "begin_time_s"}.issubset(deduped.columns):
        durations_ms = ((deduped["end_time_s"] - deduped["begin_time_s"]) * 1000.0).dropna()
        median_duration = float(durations_ms.median())
    else:
        median_duration = float("nan")

    return {
        "median_ici_gap_ms": round(float(np.median(ici_gap) * 1000.0), 4),
        "median_ioi_ms": round(float(np.median(ici_onset) * 1000.0), 4),
        "median_call_duration_ms": round(median_duration, 4),
        "q25_ici_gap_ms": round(float(np.quantile(ici_gap, 0.25) * 1000.0), 4),
        "q75_ici_gap_ms": round(float(np.quantile(ici_gap, 0.75) * 1000.0), 4),
        "n_cross_file_pairs_over_10s": int((ici_gap > 10.0).sum()),
        "n_negative_gaps": int((ici_gap < 0).sum()),
        "n_ici_samples": int(len(ici_gap)),
    }


def _compute_bout_stats(seq_summary_csv: Path) -> dict[str, Any]:
    """Extract Phase A2 bout-detection outcomes from sequential-structure summary."""
    df = pd.read_csv(seq_summary_csv)
    # Single-row summary CSV
    if len(df) != 1:
        raise ValueError(
            f"Expected 1-row summary in {seq_summary_csv}, got {len(df)} rows"
        )
    row = df.iloc[0]
    return {
        "threshold_s": float(row["bout_threshold_s"]),
        "derivation": "3 × median(IOI) via scripts/analyze_temporal_dynamics.py",
        "n_bouts": int(row["n_multi_call_bouts"]),
        "n_within_bout_pairs": int(row["n_within_bout_pairs"]),
        "n_cross_bout_pairs_excluded": int(row["n_cross_bout_gaps_excluded"]),
    }


def _compute_labeling_distributions(
    classified: pd.DataFrame, hdbscan_csv: Path | None
) -> dict[str, dict[str, int]]:
    distributions: dict[str, dict[str, int]] = {}

    if "syllable_type" in classified.columns:
        counts = (
            classified.dropna(subset=["file", "syllable_type"])
                      ["syllable_type"].value_counts().to_dict()
        )
        distributions["scattoni_7"] = {str(k): int(v) for k, v in counts.items()}

    # Join on (file, begin_time_s) to pull hdbscan_label from the recluster CSV.
    # Same join strategy as scripts/run_sis_baselines.py:_pick_join_keys.
    if hdbscan_csv is not None and hdbscan_csv.exists():
        hdb = pd.read_csv(hdbscan_csv)
        if {"file", "begin_time_s", "hdbscan_label"}.issubset(hdb.columns):
            dedup_hdb = hdb.dropna(subset=["file", "begin_time_s"]).drop_duplicates(
                subset=["file", "begin_time_s"], keep="first"
            )
            merged = classified.dropna(subset=["file", "begin_time_s"]).merge(
                dedup_hdb[["file", "begin_time_s", "hdbscan_label"]],
                on=["file", "begin_time_s"],
                how="left",
            )
            counts_hdb = (
                merged["hdbscan_label"].dropna().astype(int).value_counts().to_dict()
            )
            distributions["hdbscan"] = {str(k): int(v) for k, v in counts_hdb.items()}

    return distributions


def _build_payload(dataset: str, paths: dict[str, Path]) -> dict[str, Any]:
    classified = pd.read_csv(paths["classified_csv"])
    ici_gap = np.load(paths["ici_gap_npy"])
    ici_onset = np.load(paths["ici_onset_npy"])

    counts = _compute_counts(classified)
    timing = _compute_timing(classified, ici_gap, ici_onset)
    bout_stats = _compute_bout_stats(paths["sequential_summary_csv"])
    distributions = _compute_labeling_distributions(classified, paths.get("hdbscan_csv"))

    return {
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/audit_corpus.py",
        "sources": {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items() if v.exists()},
        "counts": counts,
        "timing": timing,
        "bout_detection_a2": bout_stats,
        "labeling_distributions": distributions,
        "references": LITERATURE_REFERENCES,
    }


# ── Output formatting ──────────────────────────────────────────────────────

def _print_parameters(dataset: str, paths: dict[str, Path], output_path: Path) -> None:
    """Print a parameters block — mirrors scripts/run_sis_baselines.py pattern."""
    print("=" * 66)
    print(f"audit_corpus.py — Parameters (dataset={dataset})")
    print("=" * 66)
    print(f"  timestamp (UTC)        : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"  output                 : {output_path}")
    print()
    print("  [inputs]")
    for key, path in paths.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"  {key:<22} : {path.relative_to(REPO_ROOT)}  [{status}]")
    print()
    print("  [methodology]")
    print(f"  row-drop rule          : dropna(subset=['file']) — matches Phase A1/A2 convention")
    print(f"  ICI gap source         : {paths['ici_gap_npy'].relative_to(REPO_ROOT)} (end-to-start, seconds)")
    print(f"  ICI onset source       : {paths['ici_onset_npy'].relative_to(REPO_ROOT)} (onset-to-onset, seconds)")
    print(f"  bout detection         : Phase A2 threshold = 0.6 s (3× median IOI)")
    print(f"  hdbscan join keys      : (file, begin_time_s), drop duplicates keep='first'")
    print()
    print("  [literature references]")
    for key, value in LITERATURE_REFERENCES.items():
        print(f"  {key:<38}: {value}")
    print("=" * 66)
    print()


def _print_summary(payload: dict[str, Any]) -> None:
    print(f"{'metric':<38} {'value':>16}")
    print("-" * 56)
    for k, v in payload["counts"].items():
        print(f"counts.{k:<31} {v:>16}")
    for k, v in payload["timing"].items():
        print(f"timing.{k:<31} {v:>16}")
    for k, v in payload["bout_detection_a2"].items():
        if isinstance(v, (int, float)):
            print(f"bout.{k:<33} {v:>16}")
    for scheme, dist in payload["labeling_distributions"].items():
        print(f"labeling.{scheme:<29} {sum(dist.values()):>16} ({len(dist)} labels)")


# ── Main ──────────────────────────────────────────────────────────────────

def process_one(
    dataset: str, paths: dict[str, Path], output_path: Path
) -> bool:
    """Returns True on success, False if inputs are missing."""
    missing = _inputs_missing(paths)
    if missing:
        print(
            f"[skip] dataset={dataset}: missing required inputs {missing}",
            file=sys.stderr,
        )
        return False

    _print_parameters(dataset, paths, output_path)
    payload = _build_payload(dataset, paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    print(f"[ok] wrote {output_path}")

    _print_summary(payload)
    return True


def main() -> int:
    args = parse_args()

    if args.dataset:
        if args.output is None:
            print("error: --output is required with --dataset", file=sys.stderr)
            return 1
        paths = DATASET_REGISTRY[args.dataset]
        success = process_one(args.dataset, paths, args.output)
        return 0 if success else 1

    # --all branch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    any_success = False
    for dataset, paths in DATASET_REGISTRY.items():
        output_path = args.output_dir / f"{dataset}.json"
        if process_one(dataset, paths, output_path):
            any_success = True
    return 0 if any_success else 1


if __name__ == "__main__":
    sys.exit(main())
