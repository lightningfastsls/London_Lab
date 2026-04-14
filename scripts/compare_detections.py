#!/usr/bin/env python3
"""Compare CNN detections against DeepSqueak independent detections.

Cross-validates two independent detection systems:
  - Our CNN pipeline (energy detection + CNN classifier + post-processing)
  - DeepSqueak's neural network (independent detection from raw audio)

For each WAV file processed by both systems, matches detections by temporal
overlap and classifies each as:
  - AGREE:    found by both CNN and DeepSqueak (true positives by concordance)
  - CNN_ONLY: found by CNN but not DeepSqueak (potential CNN false positive or DS false negative)
  - DS_ONLY:  found by DeepSqueak but not CNN (potential CNN false negative or DS false positive)

Usage::

    .venv/bin/python scripts/compare_detections.py \\
        --cnn-dir results/batch_5970_v2_full/detections \\
        --ds-csv results/deepsqueak_independent/deepsqueak_independent_detections.csv \\
        --processed-list results/deepsqueak_independent/processed_files.txt \\
        --output results/deepsqueak_independent/comparison_report.csv

Pipeline:
    1. deepsqueak_detect_independent.m  (MATLAB: run DS detection)
    2. compare_detections.py            (Python: this file)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detection from either system."""
    start_s: float
    end_s: float
    score: float = 0.0
    source: str = ""  # "cnn" or "ds"


@dataclass
class MatchResult:
    """Result of matching a single detection pair."""
    wav_stem: str
    cnn_start: float | None
    cnn_end: float | None
    cnn_score: float | None
    ds_start: float | None
    ds_end: float | None
    ds_score: float | None
    overlap_s: float
    iou: float
    category: str  # "agree", "cnn_only", "ds_only"


@dataclass
class ComparisonSummary:
    """Aggregate comparison statistics."""
    files_compared: int = 0
    files_with_cnn_dets: int = 0
    files_with_ds_dets: int = 0
    total_cnn: int = 0
    total_ds: int = 0
    agree: int = 0
    cnn_only: int = 0
    ds_only: int = 0
    cnn_covered: int = 0  # CNN detections with at least one DS match
    mean_iou_agree: float = 0.0
    per_file: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cnn_detections(
    det_dir: Path,
    stems: set[str] | None = None,
) -> dict[str, list[Detection]]:
    """Load CNN batch detections, optionally filtered to specific stems.

    Parameters
    ----------
    det_dir : Path
        Directory with ``<stem>.json`` files from ``run_batch_detection.py``.
    stems : set[str] | None
        If provided, only load these stems. Speeds up loading for large dirs.

    Returns
    -------
    dict mapping wav_stem -> sorted list of Detection objects.
    """
    result: dict[str, list[Detection]] = {}

    for json_path in sorted(det_dir.glob("*.json")):
        stem = json_path.stem
        if stems is not None and stem not in stems:
            continue

        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON: %s", json_path.name)
            continue

        if not isinstance(data, list):
            continue

        dets = []
        for d in data:
            try:
                dets.append(Detection(
                    start_s=d["start_time_s"],
                    end_s=d["end_time_s"],
                    score=d.get("max_probability", 0.0),
                    source="cnn",
                ))
            except KeyError:
                pass

        result[stem] = sorted(dets, key=lambda x: x.start_s)

    return result


def load_ds_detections(csv_path: Path) -> dict[str, list[Detection]]:
    """Load DeepSqueak independent detections from CSV.

    Parameters
    ----------
    csv_path : Path
        CSV exported by ``deepsqueak_detect_independent.m``.

    Returns
    -------
    dict mapping wav_stem -> sorted list of Detection objects.
    """
    df = pd.read_csv(csv_path)

    required = {"wav_stem", "begin_time_s", "end_time_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DS CSV missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    result: dict[str, list[Detection]] = {}
    for stem, group in df.groupby("wav_stem"):
        dets = []
        for _, row in group.iterrows():
            dets.append(Detection(
                start_s=row["begin_time_s"],
                end_s=row["end_time_s"],
                score=row.get("score", 0.0),
                source="ds",
            ))
        result[str(stem)] = sorted(dets, key=lambda x: x.start_s)

    return result


def load_processed_list(path: Path) -> set[str]:
    """Load the list of WAV stems processed by DeepSqueak."""
    stems = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            stems.add(line)
    return stems


# ---------------------------------------------------------------------------
# Matching algorithm
# ---------------------------------------------------------------------------

def temporal_iou(a: Detection, b: Detection) -> float:
    """Compute intersection-over-union in the time domain."""
    overlap_start = max(a.start_s, b.start_s)
    overlap_end = min(a.end_s, b.end_s)
    overlap = max(0.0, overlap_end - overlap_start)

    union = (a.end_s - a.start_s) + (b.end_s - b.start_s) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def overlap_fraction(a: Detection, b: Detection) -> float:
    """Fraction of the shorter detection covered by overlap.

    This is the Szymkiewicz-Simpson coefficient — it handles the case
    where a short DS call sits inside a long merged CNN detection.
    A 30ms DS call fully inside a 500ms CNN detection scores 1.0,
    whereas IoU would score ~0.06.
    """
    overlap_start = max(a.start_s, b.start_s)
    overlap_end = min(a.end_s, b.end_s)
    overlap = max(0.0, overlap_end - overlap_start)

    shorter = min(a.end_s - a.start_s, b.end_s - b.start_s)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def match_detections(
    cnn_dets: list[Detection],
    ds_dets: list[Detection],
    iou_threshold: float = 0.3,
) -> list[tuple[int | None, int | None, float]]:
    """Containment-aware matching of CNN and DS detections.

    Handles the common case where our CNN merges adjacent calls into
    one long detection while DeepSqueak finds them as separate short
    calls. Multiple DS detections can match to a single CNN detection
    (many-to-one) if they fall within its time window.

    Algorithm:
    1. For each DS detection, find the CNN detection with the best
       overlap-fraction (fraction of the shorter detection covered).
    2. If overlap-fraction >= threshold, mark as matched. Multiple DS
       calls can match the same CNN detection.
    3. Leftover CNN detections with no DS match -> CNN_ONLY.
    4. Leftover DS detections with no CNN match -> DS_ONLY.

    Parameters
    ----------
    cnn_dets : list[Detection]
        CNN detections for one WAV, sorted by start time.
    ds_dets : list[Detection]
        DeepSqueak detections for the same WAV, sorted by start time.
    iou_threshold : float
        Minimum overlap fraction for a match (default 0.3).

    Returns
    -------
    List of (cnn_idx | None, ds_idx | None, score) tuples.
    None index means unmatched on that side. Score is the overlap
    fraction for matches, 0.0 for unmatched.
    """
    if not cnn_dets and not ds_dets:
        return []

    matches: list[tuple[int | None, int | None, float]] = []
    matched_cnn: set[int] = set()
    matched_ds: set[int] = set()

    # For each DS detection, find best-overlapping CNN detection
    for di, d in enumerate(ds_dets):
        best_ci = None
        best_score = 0.0

        for ci, c in enumerate(cnn_dets):
            score = overlap_fraction(c, d)
            if score > best_score:
                best_score = score
                best_ci = ci

        if best_ci is not None and best_score >= iou_threshold:
            matches.append((best_ci, di, best_score))
            matched_cnn.add(best_ci)
            matched_ds.add(di)
        else:
            matches.append((None, di, 0.0))

    # CNN detections that no DS detection matched
    for ci in range(len(cnn_dets)):
        if ci not in matched_cnn:
            matches.append((ci, None, 0.0))

    return matches


# ---------------------------------------------------------------------------
# Comparison pipeline
# ---------------------------------------------------------------------------

def compare_all(
    cnn_by_stem: dict[str, list[Detection]],
    ds_by_stem: dict[str, list[Detection]],
    processed_stems: set[str],
    iou_threshold: float = 0.3,
) -> tuple[list[MatchResult], ComparisonSummary]:
    """Compare CNN and DS detections across all processed files.

    Only compares files that DeepSqueak actually processed (from the
    processed_files.txt list). Files not in that list are skipped.

    Parameters
    ----------
    cnn_by_stem : dict
        CNN detections by WAV stem.
    ds_by_stem : dict
        DeepSqueak detections by WAV stem.
    processed_stems : set
        WAV stems that DeepSqueak processed.
    iou_threshold : float
        Minimum temporal IoU for a match.

    Returns
    -------
    (all_results, summary)
    """
    summary = ComparisonSummary()
    all_results: list[MatchResult] = []

    for stem in sorted(processed_stems):
        cnn_dets = cnn_by_stem.get(stem, [])
        ds_dets = ds_by_stem.get(stem, [])

        summary.files_compared += 1
        if cnn_dets:
            summary.files_with_cnn_dets += 1
        if ds_dets:
            summary.files_with_ds_dets += 1

        summary.total_cnn += len(cnn_dets)
        summary.total_ds += len(ds_dets)

        matches = match_detections(cnn_dets, ds_dets, iou_threshold)

        file_agree = 0
        file_cnn_only = 0
        file_ds_only = 0
        file_cnn_covered: set[int] = set()

        for ci, di, iou in matches:
            c = cnn_dets[ci] if ci is not None else None
            d = ds_dets[di] if di is not None else None

            if ci is not None and di is not None:
                category = "agree"
                file_agree += 1
                file_cnn_covered.add(ci)
                overlap_start = max(c.start_s, d.start_s)
                overlap_end = min(c.end_s, d.end_s)
                overlap_s = max(0.0, overlap_end - overlap_start)
            elif ci is not None:
                category = "cnn_only"
                file_cnn_only += 1
                overlap_s = 0.0
            else:
                category = "ds_only"
                file_ds_only += 1
                overlap_s = 0.0

            all_results.append(MatchResult(
                wav_stem=stem,
                cnn_start=c.start_s if c else None,
                cnn_end=c.end_s if c else None,
                cnn_score=c.score if c else None,
                ds_start=d.start_s if d else None,
                ds_end=d.end_s if d else None,
                ds_score=d.score if d else None,
                overlap_s=overlap_s,
                iou=iou,
                category=category,
            ))

        summary.agree += file_agree
        summary.cnn_only += file_cnn_only
        summary.ds_only += file_ds_only
        summary.cnn_covered += len(file_cnn_covered)
        summary.per_file[stem] = {
            "cnn": len(cnn_dets),
            "ds": len(ds_dets),
            "agree": file_agree,
            "cnn_only": file_cnn_only,
            "ds_only": file_ds_only,
        }

    # Mean IoU for agreed detections
    agree_ious = [r.iou for r in all_results if r.category == "agree"]
    if agree_ious:
        summary.mean_iou_agree = sum(agree_ious) / len(agree_ious)

    return all_results, summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(summary: ComparisonSummary) -> None:
    """Print a human-readable comparison report."""
    print()
    print("=" * 65)
    print(" CNN vs DeepSqueak — Independent Cross-Validation Report")
    print("=" * 65)
    print()
    print(f"  Files compared:           {summary.files_compared}")
    print(f"  Files with CNN detections: {summary.files_with_cnn_dets}")
    print(f"  Files with DS detections:  {summary.files_with_ds_dets}")
    print()
    print(f"  Total CNN detections:      {summary.total_cnn}")
    print(f"  Total DS detections:       {summary.total_ds}")
    print()

    print("  --- Detection Agreement ---")
    print()
    print(f"  AGREE (both found):        {summary.agree}")
    print(f"  CNN_ONLY (CNN found, DS missed): {summary.cnn_only}")
    print(f"  DS_ONLY  (DS found, CNN missed): {summary.ds_only}")
    print()

    if summary.total_cnn > 0:
        cnn_coverage = summary.cnn_covered / summary.total_cnn * 100
        print(f"  CNN coverage rate:         {cnn_coverage:.1f}%"
              f"  ({summary.cnn_covered}/{summary.total_cnn} CNN detections overlap with DS)")
    if summary.total_ds > 0:
        ds_confirmed = summary.agree / summary.total_ds * 100
        print(f"  DS confirmation rate:      {ds_confirmed:.1f}%"
              f"  ({summary.agree}/{summary.total_ds} DS detections overlap with CNN)")

    if summary.agree > 0:
        print(f"  Mean IoU (agreed):         {summary.mean_iou_agree:.3f}")

    print()
    print("  --- Interpretation Guide ---")
    print()
    print("  CNN confirmation rate ~90%+  -> CNN detections are reliable")
    print("  CNN confirmation rate ~70-90% -> Some CNN FPs or DS FNs to investigate")
    print("  CNN confirmation rate <70%   -> Significant disagreement, inspect manually")
    print()
    print("  CNN_ONLY detections may be:")
    print("    - CNN false positives (noise classified as USV)")
    print("    - DS false negatives (real USVs that DS missed)")
    print("    - Boundary effects (DS detected but with different boundaries)")
    print()
    print("  DS_ONLY detections may be:")
    print("    - CNN false negatives (real USVs our pipeline missed)")
    print("    - DS false positives (noise DeepSqueak called a USV)")
    print()

    # Show files with biggest disagreements
    if summary.per_file:
        disagreements = []
        for stem, counts in summary.per_file.items():
            total = counts["cnn"] + counts["ds"]
            if total > 0:
                disagree = counts["cnn_only"] + counts["ds_only"]
                disagreements.append((stem, counts, disagree))

        disagreements.sort(key=lambda x: x[2], reverse=True)

        print("  --- Top 10 Disagreement Files ---")
        print()
        print(f"  {'WAV stem':<45} {'CNN':>4} {'DS':>4} {'Agree':>6} {'CNN✓':>5} {'DS✓':>5}")
        print(f"  {'-'*45} {'---':>4} {'---':>4} {'-----':>6} {'----':>5} {'----':>5}")
        for stem, counts, _ in disagreements[:10]:
            print(f"  {stem:<45} {counts['cnn']:>4} {counts['ds']:>4}"
                  f" {counts['agree']:>6} {counts['cnn_only']:>5} {counts['ds_only']:>5}")

    print()
    print("=" * 65)
    print()


def export_results(
    results: list[MatchResult],
    output_path: Path,
) -> None:
    """Export detailed match results to CSV."""
    rows = []
    for r in results:
        rows.append({
            "wav_stem": r.wav_stem,
            "category": r.category,
            "cnn_start_s": r.cnn_start,
            "cnn_end_s": r.cnn_end,
            "cnn_score": r.cnn_score,
            "ds_start_s": r.ds_start,
            "ds_end_s": r.ds_end,
            "ds_score": r.ds_score,
            "overlap_s": r.overlap_s,
            "iou": r.iou,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d match results to %s", len(df), output_path)


def export_summary_json(
    summary: ComparisonSummary,
    output_path: Path,
) -> None:
    """Export summary statistics to JSON."""
    data = {
        "files_compared": summary.files_compared,
        "files_with_cnn_dets": summary.files_with_cnn_dets,
        "files_with_ds_dets": summary.files_with_ds_dets,
        "total_cnn": summary.total_cnn,
        "total_ds": summary.total_ds,
        "agree": summary.agree,
        "cnn_only": summary.cnn_only,
        "ds_only": summary.ds_only,
        "mean_iou_agree": round(summary.mean_iou_agree, 4),
    }
    if summary.total_cnn > 0:
        data["cnn_coverage_rate"] = round(
            summary.cnn_covered / summary.total_cnn, 4
        )
    if summary.total_ds > 0:
        data["ds_confirmation_rate"] = round(
            summary.agree / summary.total_ds, 4
        )

    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Wrote summary to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CNN vs DeepSqueak independent detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cnn-dir",
        type=Path,
        required=True,
        help="Directory with CNN batch detection JSONs (e.g., results/batch_5970_v2_full/detections)",
    )
    parser.add_argument(
        "--ds-csv",
        type=Path,
        required=True,
        help="CSV from deepsqueak_detect_independent.m",
    )
    parser.add_argument(
        "--processed-list",
        type=Path,
        required=True,
        help="Text file listing WAV stems that DeepSqueak processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV for detailed match results (default: next to ds-csv)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum temporal IoU for a match (default: 0.3)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="WAV stems to exclude (e.g., files not in CNN batch)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.cnn_dir.is_dir():
        logger.error("CNN detections directory not found: %s", args.cnn_dir)
        sys.exit(1)
    if not args.ds_csv.is_file():
        logger.error("DeepSqueak CSV not found: %s", args.ds_csv)
        sys.exit(1)
    if not args.processed_list.is_file():
        logger.error("Processed file list not found: %s", args.processed_list)
        sys.exit(1)

    # Default output path
    if args.output is None:
        args.output = args.ds_csv.parent / "comparison_report.csv"

    # Load processed file list
    processed_stems = load_processed_list(args.processed_list)
    if args.exclude:
        excluded = set(args.exclude)
        before = len(processed_stems)
        processed_stems -= excluded
        logger.info(
            "Excluded %d stems (%d -> %d)",
            before - len(processed_stems), before, len(processed_stems),
        )
    logger.info("DeepSqueak processed %d files", len(processed_stems))

    # Load detections (only for stems that DS processed)
    logger.info("Loading CNN detections from %s ...", args.cnn_dir)
    cnn_by_stem = load_cnn_detections(args.cnn_dir, stems=processed_stems)
    cnn_with_dets = sum(1 for v in cnn_by_stem.values() if v)
    logger.info(
        "Loaded CNN detections: %d files (%d with detections)",
        len(cnn_by_stem), cnn_with_dets,
    )

    logger.info("Loading DeepSqueak detections from %s ...", args.ds_csv)
    ds_by_stem = load_ds_detections(args.ds_csv)
    logger.info("Loaded DS detections: %d files", len(ds_by_stem))

    # Compare
    logger.info("Comparing with IoU threshold=%.2f ...", args.iou_threshold)
    results, summary = compare_all(
        cnn_by_stem, ds_by_stem, processed_stems, args.iou_threshold
    )

    # Report
    print_report(summary)

    # Export
    export_results(results, args.output)
    export_summary_json(
        summary,
        args.output.parent / "comparison_summary.json",
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
