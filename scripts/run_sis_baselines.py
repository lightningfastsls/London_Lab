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
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — tests run under subprocess without a display
import matplotlib.pyplot as plt
import numpy as np
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
    compute_sis_depth_1_bout_aware,
)

# Canonical source of empirical facts (threshold, medians). Reading from
# ``data/corpus_facts/<dataset>.json`` at runtime keeps this module in
# sync with the registry — never hardcode 0.6 s here. See
# ``docs/modules/corpus-constants.md`` (Layer 2).
CORPUS_FACTS_DIR = REPO_ROOT / "data" / "corpus_facts"

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
    parser.add_argument(
        "--dataset",
        default="5970",
        help=(
            "Dataset key for corpus-facts lookup (default: 5970). Used to "
            "auto-load bout_detection_a2.threshold_s from "
            "data/corpus_facts/<dataset>.json when --bout-threshold-s is "
            "not supplied. Pass 'none' to disable corpus lookup entirely."
        ),
    )
    parser.add_argument(
        "--bout-threshold-s",
        type=float,
        default=None,
        help=(
            "Silent-gap threshold (seconds) for bout filtering. Pairs with "
            "gap >= threshold are treated as cross-bout and excluded from "
            "the MI joint-count matrix. Default: read from "
            "data/corpus_facts/<dataset>.json:bout_detection_a2.threshold_s. "
            "Pass a negative value to disable bout filtering entirely."
        ),
    )
    parser.add_argument(
        "--ici-gap-npy",
        type=Path,
        default=None,
        help=(
            "Path to a precomputed per-pair silent-gap array "
            "(length = merged_rows - 1, seconds, end-to-start). When "
            "provided and length-compatible, overrides the inline "
            "computation — this is the canonical path for 5970 so we "
            "exactly match Phase A2's methodology "
            "(results/sequential_structure/ici_gap.npy). Default: "
            "auto-discovered from A2's output dir for the given dataset."
        ),
    )
    return parser.parse_args()


def _default_ici_gap_path(dataset: str) -> Path:
    """Where Phase A2 writes its ici_gap.npy for this dataset.

    For 5970 the path is the legacy layout
    ``results/sequential_structure/ici_gap.npy``; for other datasets,
    Phase B1+ writes to ``results/sequential_structure_<dataset>/...``.
    """
    if dataset.lower() == "5970":
        return REPO_ROOT / "results/sequential_structure/ici_gap.npy"
    return REPO_ROOT / f"results/sequential_structure_{dataset}/ici_gap.npy"


def _resolve_ici_gap(
    dataset: str,
    cli_path: Path | None,
    merged: pd.DataFrame,
) -> tuple[np.ndarray | None, str]:
    """Pick the ICI-gap array and record where it came from.

    Precedence: CLI path > corpus auto-discovery > inline computation.
    Auto-discovery reproduces Phase A2 byte-for-byte on 5970 (its own
    output is the canonical source). Inline computation is a fallback
    that forces cross-file gaps to +inf — slightly more conservative
    than A2's filename-datetime approach, so it can disagree by a
    handful of pairs on real data.

    Returns
    -------
    tuple[np.ndarray | None, str]
        (gap_array, source). ``source`` is one of
        ``cli_override`` | ``corpus_auto`` | ``inline_from_end_time_s`` |
        ``unavailable``.
    """
    n_pairs_expected = max(0, len(merged) - 1)

    def _maybe_load(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        arr = np.load(path)
        if arr.shape[0] != n_pairs_expected:
            print(
                f"[warn] {path.relative_to(REPO_ROOT)} has length "
                f"{arr.shape[0]}, expected {n_pairs_expected} — ignoring",
                file=sys.stderr,
            )
            return None
        return arr.astype(np.float64)

    if cli_path is not None:
        arr = _maybe_load(cli_path)
        if arr is not None:
            return arr, "cli_override"
        return None, "unavailable"

    default_path = _default_ici_gap_path(dataset)
    arr = _maybe_load(default_path)
    if arr is not None:
        return arr, "corpus_auto"

    inline = _compute_ici_gap(merged)
    if inline is not None:
        return inline, "inline_from_end_time_s"
    return None, "unavailable"


def _resolve_bout_threshold(
    dataset: str, cli_value: float | None
) -> tuple[float | None, str]:
    """Pick the bout threshold and record where it came from.

    Precedence: CLI override > corpus-facts lookup > None (no filter). A
    negative CLI value means 'explicitly disable filtering' and returns
    ``(None, 'cli_override_disabled')``.

    Returns
    -------
    tuple[float | None, str]
        (threshold, source). ``source`` is one of
        ``cli_override`` | ``corpus_facts`` | ``cli_override_disabled`` |
        ``missing_corpus_facts`` | ``dataset_none``.
    """
    if cli_value is not None:
        if cli_value < 0:
            return None, "cli_override_disabled"
        return float(cli_value), "cli_override"

    if dataset.lower() == "none":
        return None, "dataset_none"

    facts_path = CORPUS_FACTS_DIR / f"{dataset}.json"
    if not facts_path.exists():
        print(
            f"[warn] corpus facts {facts_path.relative_to(REPO_ROOT)} not "
            f"found — bout filter disabled, computing raw-consecutive MI",
            file=sys.stderr,
        )
        return None, "missing_corpus_facts"

    with open(facts_path) as f:
        facts = json.load(f)
    threshold = facts.get("bout_detection_a2", {}).get("threshold_s")
    if threshold is None:
        print(
            f"[warn] {facts_path.relative_to(REPO_ROOT)} has no "
            f"bout_detection_a2.threshold_s — bout filter disabled",
            file=sys.stderr,
        )
        return None, "missing_corpus_facts"
    return float(threshold), "corpus_facts"


def _compute_ici_gap(merged: pd.DataFrame) -> np.ndarray | None:
    """Compute per-pair silent gap (seconds, end-to-start) from merged df.

    Assumes merged is already sorted by (file, begin_time_s). Returns
    ``None`` if ``end_time_s`` is absent — the script then falls back to
    raw-consecutive MI with a warning.

    Matches Phase A2's definition (gap = start[next] - end[current]),
    except A2 operates on absolute_time (datetime) whereas we operate on
    (file, begin_time_s). Within a file, both give identical gaps. Across
    file boundaries A2 uses the datetime-parsed filename, while here the
    gap degenerates to ``start_next - end_prev`` within begin_time_s —
    i.e. filename-relative times. Cross-file pairs will always have a
    very large (or negative) gap and will be correctly classified as
    cross-bout, so the filter reaches the same conclusion either way.
    """
    if "end_time_s" not in merged.columns or "begin_time_s" not in merged.columns:
        return None

    start = merged["begin_time_s"].to_numpy(dtype=np.float64)
    end = merged["end_time_s"].to_numpy(dtype=np.float64)

    # Within a file the gap is start[i+1] - end[i]. Across files the
    # begin_time_s timelines reset per file, so the naive subtraction is
    # meaningless — force those gaps to +inf so they always exceed any
    # finite bout threshold.
    if "file" in merged.columns:
        file_arr = merged["file"].to_numpy()
        same_file = file_arr[1:] == file_arr[:-1]
        gap = np.where(same_file, start[1:] - end[:-1], np.inf)
    else:
        gap = start[1:] - end[:-1]
    return gap


_FILENAME_TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})"
)


def _compute_absolute_time(merged: pd.DataFrame) -> np.ndarray | None:
    """Parse per-row datetime from the ``file`` column and add begin_time_s.

    Matches ``scripts/analyze_sequential_structure.py:parse_filename_timestamp``
    byte-for-byte: filenames must start with ``YYYY-MM-DD_HH-MM-SS_`` (the
    canonical WAV naming in this repo). Returns ``None`` when any filename
    fails to parse — in that case the caller falls back to lexicographic
    sort and cannot use Phase A2's precomputed ici_gap.npy.

    Returned array is absolute-time in seconds (float64), offset by an
    arbitrary epoch — only relative differences are used downstream, so
    the epoch is irrelevant.
    """
    if "file" not in merged.columns or "begin_time_s" not in merged.columns:
        return None

    files = merged["file"].to_numpy()
    starts = merged["begin_time_s"].to_numpy(dtype=np.float64)

    abs_seconds = np.empty(len(merged), dtype=np.float64)
    for i, fname in enumerate(files):
        m = _FILENAME_TIMESTAMP_RE.match(str(fname))
        if not m:
            return None
        y, mo, d, h, mi, s = (int(x) for x in m.groups())
        file_dt = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
        abs_seconds[i] = file_dt.timestamp() + starts[i]
    return abs_seconds


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
        # Prefer filename-derived absolute_time when every filename has the
        # canonical ``YYYY-MM-DD_HH-MM-SS_...`` prefix. This matches Phase
        # A2's sort (``scripts/analyze_sequential_structure.py`` —
        # ``load_and_enrich``) so Phase A2's precomputed ici_gap.npy is
        # row-aligned with SIS's merged frame. When any filename doesn't
        # parse, fall back to the legacy lexicographic sort — matters for
        # synthetic-CSV tests whose filenames are bare like "rec.wav".
        abs_time = _compute_absolute_time(merged)
        if abs_time is not None:
            merged = merged.assign(_abs_time=abs_time)
            merged = merged.sort_values(
                ["_abs_time", "begin_time_s"], kind="mergesort"
            ).drop(columns="_abs_time").reset_index(drop=True)
            sort_keys = ["file_datetime+begin_time_s"]
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


def _compute_one(
    merged: pd.DataFrame,
    col: str,
    name: str,
    ici_gap_s: np.ndarray | None,
    bout_threshold_s: float | None,
) -> tuple[SISResult, int | None, int | None] | None:
    """Compute SIS for a single labeling.

    When both ``ici_gap_s`` and ``bout_threshold_s`` are provided, runs
    the bout-aware variant and returns per-labeling pair counts
    (n_within, n_excluded) for the parameters audit trail. Otherwise
    falls back to the raw-consecutive variant and returns ``(result,
    None, None)``.

    Rows with NaN on the labeling column are dropped **together** with
    the corresponding ICI gaps — so the returned ici_gap slice stays in
    sync with the filtered labels.
    """
    if col not in merged.columns:
        print(f"[warn] column '{col}' not found — skipping {name}", file=sys.stderr)
        return None

    mask = merged[col].notna().to_numpy()
    if mask.sum() == 0:
        print(f"[warn] column '{col}' has no non-null rows — skipping {name}",
              file=sys.stderr)
        return None

    labels = merged.loc[mask, col].to_numpy()

    if ici_gap_s is None or bout_threshold_s is None:
        return compute_sis_depth_1(labels, name=name), None, None

    # Slice ici_gap_s to match the filtered label order. gap[i] relates
    # calls i and i+1 in the full merged frame; after dropping NaN-label
    # rows, the filtered sequence's pair i relates full-frame calls
    # kept_idx[i] and kept_idx[i+1] — the gap is the end-to-start span
    # of those two. If any intermediate rows were dropped the filtered
    # pair effectively bridges a gap that includes dropped calls; we
    # sum the intermediate gaps + the skipped calls' durations. That's
    # fine for bout detection (the total silence between kept calls is
    # what matters), but it's also an edge case that only triggers when
    # the labeling has NaN — in practice Scattoni/DeepSqueak/HDBSCAN all
    # produce a label for every call, so this path is a safety net.
    kept_idx = np.where(mask)[0]
    if kept_idx.size != len(merged):
        filtered_gap = np.array([
            merged["begin_time_s"].iat[kept_idx[i + 1]]
            - merged["end_time_s"].iat[kept_idx[i]]
            for i in range(len(kept_idx) - 1)
        ], dtype=np.float64)
        if "file" in merged.columns:
            file_arr = merged["file"].to_numpy()
            same_file = file_arr[kept_idx[1:]] == file_arr[kept_idx[:-1]]
            filtered_gap = np.where(same_file, filtered_gap, np.inf)
    else:
        filtered_gap = np.asarray(ici_gap_s, dtype=np.float64)

    result, n_within, n_excluded = compute_sis_depth_1_bout_aware(
        labels, filtered_gap, bout_threshold_s, name=name
    )
    return result, n_within, n_excluded


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


def _print_parameters(
    args: argparse.Namespace,
    stats: dict,
    bout_threshold_s: float | None,
    bout_source: str,
    ici_gap_available: bool,
    ici_source: str,
) -> None:
    """Print a Parameters block before any analysis output.

    Implements the project-wide rule: every analysis run must state its
    parameters, filter rules, and row counts in its output. A reader should
    be able to reproduce the run from the header alone.
    """
    if bout_threshold_s is not None and ici_gap_available:
        bout_line = (
            f"WITHIN-BOUT ONLY (threshold = {bout_threshold_s} s, "
            f"source = {bout_source}, ici_gap_source = {ici_source}) — "
            f"pairs with silent gap >= threshold excluded from MI joint counts"
        )
        file_line = (
            "cross-file pairs excluded implicitly — Phase A2's ici_gap.npy "
            "uses filename-derived absolute_time (corpus_auto mode); inline "
            "fallback forces cross-file gaps to +inf"
        )
    else:
        bout_line = (
            f"NONE — raw consecutive pairs "
            f"(threshold resolution: {bout_source}, "
            f"ici_gap_available: {ici_gap_available})"
        )
        file_line = (
            "NONE — cross-file pairs included "
            "(known caveat; see docs/modules/sis-baselines.md)"
        )

    print("=" * 66)
    print("run_sis_baselines.py — Parameters")
    print("=" * 66)
    print(f"  timestamp (UTC)          : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"  classified_csv           : {args.classified_csv}")
    print(f"  umap_csv                 : {args.umap_csv}")
    print(f"  output_dir               : {args.output_dir}")
    print(f"  dataset (corpus lookup)  : {args.dataset}")
    print()
    print("  [methodology]")
    print(f"  MI estimator             : usv_language.analysis.sequence_analysis.mutual_information_at_lag (lag=1)")
    print(f"  sort keys (sequence)     : {stats['sort_keys']} (lexicographic, mergesort stable)")
    print(f"  join keys                : {stats['join_keys']} (auto-selected: composite preferred, det_index fallback)")
    print(f"  bout detection           : {bout_line}")
    print(f"  file-boundary handling   : {file_line}")
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
    if bout_threshold_s is not None and ici_gap_available:
        print(f"  Phase A2 cross-check     : Scattoni-7 MI ~= 0.0921 bits at threshold {bout_threshold_s}s (same family of methods as Hertz; different data-derived threshold)")
    else:
        print(f"  Phase A2 (bout-aware)    : 0.0921 bits Scattoni-7 — NOT reproduced here; this run is raw-consecutive")
    print("=" * 66)
    print()


def _write_parameters_json(
    args: argparse.Namespace,
    stats: dict,
    results: list[SISResult],
    pair_counts: list[dict],
    bout_threshold_s: float | None,
    bout_source: str,
    ici_gap_available: bool,
    ici_source: str,
    output_path: Path,
) -> None:
    """Write a machine-readable parameters+results sidecar.

    Artifacts (baselines.csv/png) become hard to interpret without this
    audit trail — which CSVs went in, what filters ran, what MI came out.
    """
    bout_block = {
        "threshold_s": bout_threshold_s,
        "threshold_source": bout_source,
        "ici_gap_available": ici_gap_available,
        "ici_gap_source": ici_source,
        "applied": bout_threshold_s is not None and ici_gap_available,
        "semantics": (
            "pairs with ici_gap_s >= threshold_s excluded from MI joint "
            "counts (matches Phase A2 convention; strict '>' inside "
            "segment_into_bouts)"
        ),
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "run_sis_baselines.py",
        "inputs": {
            "classified_csv": str(args.classified_csv),
            "umap_csv": str(args.umap_csv),
            "output_dir": str(args.output_dir),
            "dataset_key": args.dataset,
        },
        "methodology": {
            "mi_estimator": "usv_language.analysis.sequence_analysis.mutual_information_at_lag",
            "lag": 1,
            "sort_keys": stats["sort_keys"],
            "join_keys": stats["join_keys"],
            "bout_detection": bout_block,
            "file_boundary_handling": (
                "cross-file pairs receive gap=+inf and are thus excluded"
                if bout_block["applied"]
                else None
            ),
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
        "per_labeling_pair_counts": pair_counts,
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

    bout_threshold_s, bout_source = _resolve_bout_threshold(
        args.dataset, args.bout_threshold_s
    )
    ici_gap_s, ici_source = _resolve_ici_gap(args.dataset, args.ici_gap_npy, merged)
    ici_gap_available = ici_gap_s is not None
    if bout_threshold_s is not None and not ici_gap_available:
        print(
            "[warn] no ICI gap available (neither precomputed npy nor "
            "end_time_s column); bout filter disabled, falling back to "
            "raw-consecutive MI",
            file=sys.stderr,
        )

    _print_parameters(
        args, stats, bout_threshold_s, bout_source, ici_gap_available, ici_source
    )

    results: list[SISResult] = []
    pair_counts: list[dict] = []
    for col, name in LABELING_SCHEMES:
        outcome = _compute_one(
            merged, col, name,
            ici_gap_s if ici_gap_available else None,
            bout_threshold_s,
        )
        if outcome is None:
            continue
        r, n_within, n_excluded = outcome
        results.append(r)
        pair_counts.append({
            "name": name,
            "column": col,
            "n_within_bout_pairs": n_within,
            "n_excluded_pairs": n_excluded,
            "bout_filter_applied": n_within is not None,
        })

    if not results:
        print("error: no labeling columns found in inputs", file=sys.stderr)
        return 1

    _write_csv(results, args.output_dir / "baselines.csv")
    _write_plot(results, args.output_dir / "baselines.png")
    _write_parameters_json(
        args, stats, results, pair_counts,
        bout_threshold_s, bout_source, ici_gap_available, ici_source,
        args.output_dir / "parameters.json",
    )
    _print_summary(results)

    # Print bout-filter audit block last so it's still visible in captured logs
    # even when the summary table is the most-read section.
    if any(pc["bout_filter_applied"] for pc in pair_counts):
        print()
        print("  [bout filter — per-labeling pair counts]")
        for pc in pair_counts:
            if pc["bout_filter_applied"]:
                print(
                    f"    {pc['name']:<16} within={pc['n_within_bout_pairs']:>6} "
                    f"excluded={pc['n_excluded_pairs']:>6}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
