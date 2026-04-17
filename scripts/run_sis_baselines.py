"""Compute SIS at depth 1 for existing USV labelings (module 17.1 driver).

Reads two CSVs, joins them on a robust composite key (``(file, begin_time_s)``
when available, else ``det_index``), computes ``MI(X_n ; X_{n-1})`` for each
available labeling column (``syllable_type``, ``label``, ``hdbscan_label``),
and writes ``baselines.csv`` + ``baselines.png`` with Hertz 2020 reference
lines.

On real data, pass ``results/traditional_taxonomy/classified_traditional.csv``
as the classified input — that file is a superset of
``classified_detections_full.csv`` with the Scattoni-7 ``syllable_type`` column
added by the traditional-taxonomy pipeline.

Usage
-----
    python scripts/run_sis_baselines.py \\
        --classified-csv results/traditional_taxonomy/classified_traditional.csv \\
        --umap-csv results/recluster_umap_hdbscan/reclassified_detections.csv \\
        --output-dir results/sis_baselines/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — tests run under subprocess without a display
import matplotlib.pyplot as plt
import pandas as pd

# Pattern 8 — path bootstrap. usv_spectrogram lives under src/; usv_language
# is a sibling top-level package at repo root, so both must be importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (SRC_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from usv_spectrogram.classification.sis_baselines import (  # noqa: E402
    SISResult,
    compute_sis_depth_1,
)

# Hertz et al. 2020 published SIS values (bits) at depth 1 on mouse USVs.
HERTZ_REFERENCE = [
    (0.10, "iVoICE"),
    (0.13, "iMUPET"),
    (0.22, "iMSA"),
]

# Ordered (column_name, display_name) pairs for the three baseline labelings.
LABELING_SCHEMES = [
    ("syllable_type", "scattoni-7"),
    ("label", "deepsqueak-27"),
    ("hdbscan_label", "hdbscan-3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SIS at depth 1 for existing USV labelings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--classified-csv",
        required=True,
        type=Path,
        help="Path to classified_detections_full.csv (syllable_type + label columns).",
    )
    parser.add_argument(
        "--umap-csv",
        required=True,
        type=Path,
        help="Path to reclassified_detections.csv (hdbscan_label column).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write baselines.csv + baselines.png.",
    )
    return parser.parse_args()


def _pick_join_keys(classified: pd.DataFrame, umap: pd.DataFrame) -> list[str]:
    """Choose the safest join key present in both frames.

    Prefer ``(file, begin_time_s)`` — it is a natural composite key produced
    by the detection/classification pipelines and is stable across re-runs.
    Fall back to ``det_index`` only when the composite is unavailable; real
    pipelines produce duplicate ``det_index`` values when the 75 ms
    DeepSqueak-match tolerance creates one-to-many matches, which in turn
    drives a cartesian-join blowup (OOM-killer territory on ~8k rows).
    """
    composite = ["file", "begin_time_s"]
    if all(c in classified.columns for c in composite) and all(
        c in umap.columns for c in composite
    ):
        return composite
    if "det_index" in classified.columns and "det_index" in umap.columns:
        return ["det_index"]
    raise ValueError(
        "Neither (file, begin_time_s) nor det_index is present in both CSVs; "
        "cannot determine a join key."
    )


def _load_merged(
    classified_csv: Path, umap_csv: Path
) -> tuple[pd.DataFrame, dict]:
    """Load both CSVs and left-join on the safest available key.

    Rows with NaN in any join column are dropped before merging — these are
    unmatched CNN detections that have no sequence position and cannot
    contribute to depth-1 SIS. Duplicate join-key rows are dropped with a
    warning (``keep='first'``) rather than allowed to cartesian-expand.

    Returns
    -------
    merged: pd.DataFrame
    stats : dict
        Row-count and filter provenance for the parameters audit trail.
        Keys: ``n_classified_raw``, ``n_umap_raw``, ``join_keys``,
        ``n_classified_after_dropna``, ``n_umap_after_dropna``,
        ``n_classified_dup_dropped``, ``n_umap_dup_dropped``,
        ``n_merged``, ``sort_keys``.
    """
    classified = pd.read_csv(classified_csv)
    umap = pd.read_csv(umap_csv)
    n_classified_raw = len(classified)
    n_umap_raw = len(umap)

    join_keys = _pick_join_keys(classified, umap)

    # Unmatched detections (NaN on join keys) cannot be placed in a sequence.
    classified = classified.dropna(subset=join_keys).reset_index(drop=True)
    umap = umap.dropna(subset=join_keys).reset_index(drop=True)
    n_classified_after_dropna = len(classified)
    n_umap_after_dropna = len(umap)

    n_dup_c = int(classified.duplicated(subset=join_keys).sum())
    if n_dup_c:
        print(
            f"[warn] classified_csv has {n_dup_c} duplicate {join_keys} rows "
            f"— dropping all but first occurrence",
            file=sys.stderr,
        )
        classified = classified.drop_duplicates(
            subset=join_keys, keep="first"
        ).reset_index(drop=True)
    n_dup_u = int(umap.duplicated(subset=join_keys).sum())
    if n_dup_u:
        print(
            f"[warn] umap_csv has {n_dup_u} duplicate {join_keys} rows "
            f"— dropping all but first occurrence",
            file=sys.stderr,
        )
        umap = umap.drop_duplicates(
            subset=join_keys, keep="first"
        ).reset_index(drop=True)

    umap_cols = [c for c in umap.columns if c in join_keys or c not in classified.columns]
    merged = classified.merge(umap[umap_cols], on=join_keys, how="left")

    sort_keys = [c for c in ("file", "begin_time_s") if c in merged.columns]
    if not sort_keys:
        print(
            "[warn] neither 'file' nor 'begin_time_s' columns found — "
            "MI will be computed on the raw row order, which may be wrong",
            file=sys.stderr,
        )
    elif len(sort_keys) < 2:
        missing = "begin_time_s" if "file" in sort_keys else "file"
        print(
            f"[warn] sort keys partial (have {sort_keys}, missing '{missing}') "
            f"— MI ordering may be imperfect",
            file=sys.stderr,
        )
        merged = merged.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    else:
        merged = merged.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    stats = {
        "n_classified_raw": n_classified_raw,
        "n_umap_raw": n_umap_raw,
        "join_keys": join_keys,
        "n_classified_after_dropna": n_classified_after_dropna,
        "n_umap_after_dropna": n_umap_after_dropna,
        "n_classified_dup_dropped": n_dup_c,
        "n_umap_dup_dropped": n_dup_u,
        "n_merged": len(merged),
        "sort_keys": sort_keys,
    }
    return merged, stats


def _compute_one(merged: pd.DataFrame, col: str, name: str) -> SISResult | None:
    if col not in merged.columns:
        print(f"[warn] column '{col}' not found — skipping {name}", file=sys.stderr)
        return None

    mask = merged[col].notna()
    if mask.sum() == 0:
        print(f"[warn] column '{col}' has no non-null rows — skipping {name}",
              file=sys.stderr)
        return None

    labels = merged.loc[mask, col].to_numpy()
    return compute_sis_depth_1(labels, name=name)


def _write_plot(results: list[SISResult], output_png: Path) -> None:
    """Render a bar chart of MI at lag 1 with Hertz reference lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r.name for r in results]
    mis = [r.mi_at_lag_1 for r in results]

    ax.bar(names, mis, color="steelblue")
    ax.set_ylabel("MI at lag 1 (bits)")
    ax.set_title("SIS depth-1 baselines — existing USV labelings")

    y_max = max(mis + [0.25])
    ax.set_ylim(0, max(0.3, y_max * 1.1))

    for y, label in HERTZ_REFERENCE:
        ax.axhline(y=y, linestyle="--", color="grey", alpha=0.7)
        ax.text(len(names) - 0.5, y, f"  {label} ({y:.2f})",
                va="center", ha="left", fontsize=8, color="grey")

    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def _write_csv(results: list[SISResult], output_csv: Path) -> None:
    rows = [asdict(r) for r in results]
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def _print_parameters(args: argparse.Namespace, stats: dict) -> None:
    """Print a Parameters block before any analysis output.

    Implements the project-wide rule: every analysis run must state its
    parameters, filter rules, and row counts in its output. A reader should
    be able to reproduce the run from the header alone.
    """
    print("=" * 66)
    print("run_sis_baselines.py — Parameters")
    print("=" * 66)
    print(f"  timestamp (UTC)          : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"  classified_csv           : {args.classified_csv}")
    print(f"  umap_csv                 : {args.umap_csv}")
    print(f"  output_dir               : {args.output_dir}")
    print()
    print("  [methodology]")
    print(f"  MI estimator             : usv_language.analysis.sequence_analysis.mutual_information_at_lag (lag=1)")
    print(f"  sort keys (sequence)     : {stats['sort_keys']} (lexicographic, mergesort stable)")
    print(f"  join keys                : {stats['join_keys']} (auto-selected: composite preferred, det_index fallback)")
    print(f"  bout detection           : NONE — raw consecutive pairs (ROADMAP 17.1 spec)")
    print(f"  file-boundary handling   : NONE — cross-file pairs included (known caveat; see docs/modules/sis-baselines.md)")
    print(f"  labeling columns scanned : {[col for col, _ in LABELING_SCHEMES]}")
    print()
    print("  [row counts & filters]")
    print(f"  classified raw           : {stats['n_classified_raw']}")
    print(f"  classified after dropna  : {stats['n_classified_after_dropna']} (dropped {stats['n_classified_raw'] - stats['n_classified_after_dropna']} with NaN on join keys)")
    print(f"  classified dup dropped   : {stats['n_classified_dup_dropped']} (keep='first')")
    print(f"  umap raw                 : {stats['n_umap_raw']}")
    print(f"  umap after dropna        : {stats['n_umap_after_dropna']} (dropped {stats['n_umap_raw'] - stats['n_umap_after_dropna']} with NaN on join keys)")
    print(f"  umap dup dropped         : {stats['n_umap_dup_dropped']} (keep='first')")
    print(f"  merged rows              : {stats['n_merged']}")
    print()
    print("  [reference values]")
    print(f"  Hertz 2020 published     : iVoICE=0.10, iMUPET=0.13, iMSA=0.22 bits (depth 1, 346K syllables, lab C57BL/6)")
    print(f"  Phase A2 (bout-aware)    : 0.092 bits Scattoni-7 — different methodology, see docs/modules/sis-baselines.md")
    print("=" * 66)
    print()


def _write_parameters_json(
    args: argparse.Namespace, stats: dict, results: list[SISResult], output_path: Path
) -> None:
    """Write a machine-readable parameters+results sidecar.

    Artifacts (baselines.csv/png) become hard to interpret without this
    audit trail — which CSVs went in, what filters ran, what MI came out.
    """
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "run_sis_baselines.py",
        "inputs": {
            "classified_csv": str(args.classified_csv),
            "umap_csv": str(args.umap_csv),
            "output_dir": str(args.output_dir),
        },
        "methodology": {
            "mi_estimator": "usv_language.analysis.sequence_analysis.mutual_information_at_lag",
            "lag": 1,
            "sort_keys": stats["sort_keys"],
            "join_keys": stats["join_keys"],
            "bout_detection": None,
            "file_boundary_handling": None,
            "labeling_columns_scanned": [col for col, _ in LABELING_SCHEMES],
        },
        "row_counts": {
            "classified_raw": stats["n_classified_raw"],
            "classified_after_dropna": stats["n_classified_after_dropna"],
            "classified_dup_dropped": stats["n_classified_dup_dropped"],
            "umap_raw": stats["n_umap_raw"],
            "umap_after_dropna": stats["n_umap_after_dropna"],
            "umap_dup_dropped": stats["n_umap_dup_dropped"],
            "merged": stats["n_merged"],
        },
        "hertz_reference_bits": {name: y for y, name in HERTZ_REFERENCE},
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def _print_summary(results: list[SISResult]) -> None:
    print(f"{'name':<16} {'K':>4} {'MI (bits)':>12} {'H (bits)':>10} {'reduction %':>12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r.name:<16} {r.n_labels:>4} "
            f"{r.mi_at_lag_1:>12.4f} {r.marginal_entropy:>10.4f} "
            f"{r.entropy_reduction_pct:>11.2f}%"
        )


def main() -> int:
    args = parse_args()

    if not args.classified_csv.exists():
        print(f"error: classified-csv not found: {args.classified_csv}", file=sys.stderr)
        return 1
    if not args.umap_csv.exists():
        print(f"error: umap-csv not found: {args.umap_csv}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged, stats = _load_merged(args.classified_csv, args.umap_csv)
    _print_parameters(args, stats)

    results: list[SISResult] = []
    for col, name in LABELING_SCHEMES:
        r = _compute_one(merged, col, name)
        if r is not None:
            results.append(r)

    if not results:
        print("error: no labeling columns found in inputs", file=sys.stderr)
        return 1

    _write_csv(results, args.output_dir / "baselines.csv")
    _write_plot(results, args.output_dir / "baselines.png")
    _write_parameters_json(args, stats, results, args.output_dir / "parameters.json")
    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
