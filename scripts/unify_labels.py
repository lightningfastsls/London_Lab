#!/usr/bin/env python3
"""Unify all label sources into a single JSON for the training assembler.

Consolidates:
  1. Primary: data/labels.csv + data/candidates.csv (manually reviewed labels)
  2. Supplementary: USV_Detections/{5970,3452}/*/_saved_tracking.json (accepted detections)
  3. Noise: USV_Detections/noise_labeled_files/*.json (noise-confirmed recordings)

Deduplication: tracking detections that overlap (within tolerance) any primary
label are excluded to avoid double-counting.

WAV resolution: searches --wav-search-dirs recursively to find each recording's
WAV file and stores the resolved path in the output JSON.

Output: data/unified_labels.json

Usage:
    python scripts/unify_labels.py \
        --wav-search-dirs "USV5/usv_lmt_034" "USV_3452_sample_reviewed" \
        --output data/unified_labels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Overlap tolerance in seconds for deduplication
OVERLAP_TOLERANCE_S = 0.020  # 20ms


def build_wav_index(search_dirs: list[Path]) -> dict[str, Path]:
    """Build stem -> WAV path index by searching directories recursively.

    Returns dict mapping recording_stem to resolved WAV path.
    If a stem appears in multiple dirs, the first match wins.
    """
    index: dict[str, Path] = {}
    for search_dir in search_dirs:
        if not search_dir.exists():
            logger.warning("WAV search dir not found: %s", search_dir)
            continue
        for wav_path in search_dir.rglob("*.wav"):
            stem = wav_path.stem
            if stem not in index:
                index[stem] = wav_path
    logger.info("WAV index: %d files across %d search dirs", len(index), len(search_dirs))
    return index


def load_primary_labels(
    labels_csv: Path, candidates_csv: Path
) -> list[dict]:
    """Load manually-reviewed USV labels by joining labels + candidates.

    Only keeps labels with label == "USV".

    Returns list of dicts with recording_stem, start_s, end_s, source.
    """
    labels_df = pd.read_csv(labels_csv)
    candidates_df = pd.read_csv(candidates_csv)

    # Filter to USV-only labels
    usv_labels = labels_df[labels_df["label"] == "USV"].copy()
    logger.info(
        "Primary labels: %d USV out of %d total",
        len(usv_labels), len(labels_df),
    )

    # Join on candidate_id to get timing metadata
    merged = usv_labels.merge(candidates_df, on="candidate_id", how="inner")
    if len(merged) < len(usv_labels):
        logger.warning(
            "Join dropped %d labels (no matching candidate)",
            len(usv_labels) - len(merged),
        )

    positives = []
    for _, row in merged.iterrows():
        source_file = row["source_file"]
        stem = Path(source_file).stem
        positives.append({
            "recording_stem": stem,
            "start_s": round(row["start_ms"] / 1000.0, 6),
            "end_s": round(row["end_ms"] / 1000.0, 6),
            "source": "labels_csv",
        })

    return positives


def load_tracking_detections(
    detections_root: Path, group: str
) -> list[dict]:
    """Load accepted detections from _saved_tracking.json files.

    Parameters
    ----------
    detections_root : Path
        Root of USV_Detections directory.
    group : str
        Subdirectory name (e.g., "5970" or "3452").

    Returns list of dicts with recording_stem, start_s, end_s, source.
    """
    group_dir = detections_root / group
    if not group_dir.exists():
        logger.warning("Tracking directory not found: %s", group_dir)
        return []

    positives = []
    tracking_files = sorted(group_dir.glob("*/_saved_tracking.json"))
    for tracking_path in tracking_files:
        recording_stem = tracking_path.parent.name

        with open(tracking_path, "r", encoding="utf-8") as f:
            detections = json.load(f)

        for det in detections:
            positives.append({
                "recording_stem": recording_stem,
                "start_s": round(det["start_time_s"], 6),
                "end_s": round(det["end_time_s"], 6),
                "source": f"tracking_{group}",
            })

    logger.info(
        "Tracking %s: %d detections from %d recordings",
        group, len(positives), len(tracking_files),
    )
    return positives


def load_noise_recordings(detections_root: Path) -> list[dict]:
    """Load noise-confirmed recordings from noise_labeled_files/.

    Returns list of dicts with recording_stem, duration_s.
    """
    noise_dir = detections_root / "noise_labeled_files"
    if not noise_dir.exists():
        logger.warning("Noise directory not found: %s", noise_dir)
        return []

    recordings = []
    for json_path in sorted(noise_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        file_label = metadata.get("file_label", "")
        if file_label != "noise":
            logger.debug("Skipping non-noise file: %s (label=%s)", json_path.name, file_label)
            continue

        stem = json_path.stem
        duration_s = metadata.get("duration_s", 0.0)

        recordings.append({
            "recording_stem": stem,
            "duration_s": round(duration_s, 6),
        })

    logger.info("Noise recordings: %d files", len(recordings))
    return recordings


def detections_overlap(
    det_start: float, det_end: float,
    ref_start: float, ref_end: float,
    tolerance: float,
) -> bool:
    """Check if two detections overlap within tolerance."""
    return (
        det_start < ref_end + tolerance
        and det_end > ref_start - tolerance
    )


def deduplicate_tracking(
    tracking: list[dict],
    primary: list[dict],
    tolerance: float = OVERLAP_TOLERANCE_S,
) -> list[dict]:
    """Remove tracking detections that overlap any primary label.

    Groups by recording stem for efficiency.
    """
    # Index primary labels by recording stem
    primary_by_stem: dict[str, list[tuple[float, float]]] = {}
    for p in primary:
        stem = p["recording_stem"]
        primary_by_stem.setdefault(stem, []).append((p["start_s"], p["end_s"]))

    kept = []
    removed = 0
    for det in tracking:
        stem = det["recording_stem"]
        ref_intervals = primary_by_stem.get(stem, [])

        is_duplicate = any(
            detections_overlap(det["start_s"], det["end_s"], ref_s, ref_e, tolerance)
            for ref_s, ref_e in ref_intervals
        )

        if is_duplicate:
            removed += 1
        else:
            kept.append(det)

    logger.info(
        "Dedup: removed %d tracking detections overlapping primary labels, kept %d",
        removed, len(kept),
    )
    return kept


def check_noise_positive_overlap(
    noise_recordings: list[dict],
    all_positives: list[dict],
) -> list[str]:
    """Warn about recording stems appearing in both positives and noise."""
    positive_stems = {p["recording_stem"] for p in all_positives}
    noise_stems = {n["recording_stem"] for n in noise_recordings}
    overlap = positive_stems & noise_stems

    warnings = []
    if overlap:
        for stem in sorted(overlap):
            msg = (
                f"Recording '{stem}' appears in both positive detections "
                f"and noise-labeled files — excluding from noise list"
            )
            warnings.append(msg)
            logger.warning(msg)

    return list(overlap)


def resolve_wav_paths(
    positives: list[dict],
    noise_recordings: list[dict],
    wav_index: dict[str, Path],
) -> tuple[int, int]:
    """Add wav_path field to positives and noise recordings using WAV index.

    Returns (n_positives_resolved, n_noise_resolved).
    """
    pos_resolved = 0
    for p in positives:
        wav_path = wav_index.get(p["recording_stem"])
        if wav_path:
            p["wav_path"] = str(wav_path)
            pos_resolved += 1
        else:
            p["wav_path"] = None

    noise_resolved = 0
    for n in noise_recordings:
        wav_path = wav_index.get(n["recording_stem"])
        if wav_path:
            n["wav_path"] = str(wav_path)
            noise_resolved += 1
        else:
            n["wav_path"] = None

    # Count unique recordings with WAVs
    pos_stems_with_wav = len({
        p["recording_stem"] for p in positives if p["wav_path"]
    })
    pos_stems_total = len({p["recording_stem"] for p in positives})

    logger.info(
        "WAV resolution: %d/%d positive recordings, %d/%d noise recordings",
        pos_stems_with_wav, pos_stems_total,
        noise_resolved, len(noise_recordings),
    )

    return pos_resolved, noise_resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unify all label sources into a single JSON."
    )
    parser.add_argument(
        "--detections-root", type=Path, default=Path("USV_Detections"),
        help="Root directory of USV_Detections (default: USV_Detections)",
    )
    parser.add_argument(
        "--labels-csv", type=Path, default=Path("data/labels.csv"),
        help="Path to labels.csv (default: data/labels.csv)",
    )
    parser.add_argument(
        "--candidates-csv", type=Path, default=Path("data/candidates.csv"),
        help="Path to candidates.csv (default: data/candidates.csv)",
    )
    parser.add_argument(
        "--wav-search-dirs", type=Path, nargs="+", default=None,
        help="Directories to search recursively for WAV files (resolves wav_path per recording)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/unified_labels.json"),
        help="Output JSON path (default: data/unified_labels.json)",
    )
    parser.add_argument(
        "--overlap-tolerance-ms", type=float, default=20.0,
        help="Overlap tolerance in ms for deduplication (default: 20.0)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    tolerance_s = args.overlap_tolerance_ms / 1000.0

    # 0. Build WAV index if search dirs provided
    wav_index: dict[str, Path] = {}
    if args.wav_search_dirs:
        wav_index = build_wav_index(args.wav_search_dirs)

    # 1. Load primary labels (labels.csv + candidates.csv)
    primary = load_primary_labels(args.labels_csv, args.candidates_csv)

    # 2. Load tracking detections (5970 + 3452)
    tracking_5970 = load_tracking_detections(args.detections_root, "5970")
    tracking_3452 = load_tracking_detections(args.detections_root, "3452")
    all_tracking = tracking_5970 + tracking_3452

    # 3. Deduplicate tracking against primary
    tracking_deduped = deduplicate_tracking(all_tracking, primary, tolerance_s)

    # 4. Combine all positives
    all_positives = primary + tracking_deduped

    # 5. Load noise recordings
    noise_recordings = load_noise_recordings(args.detections_root)

    # 6. Check for positive/noise overlap and exclude conflicting noise entries
    overlap_stems = check_noise_positive_overlap(noise_recordings, all_positives)
    if overlap_stems:
        noise_recordings = [
            n for n in noise_recordings
            if n["recording_stem"] not in set(overlap_stems)
        ]

    # 7. Resolve WAV paths
    if wav_index:
        resolve_wav_paths(all_positives, noise_recordings, wav_index)

    # 8. Compute stats
    source_counts: dict[str, int] = {}
    for p in all_positives:
        source_counts[p["source"]] = source_counts.get(p["source"], 0) + 1

    unique_recordings = len({p["recording_stem"] for p in all_positives})
    recordings_with_wav = len({
        p["recording_stem"] for p in all_positives
        if p.get("wav_path")
    })

    stats = {
        "total_positives": len(all_positives),
        "by_source": source_counts,
        "unique_recordings_with_positives": unique_recordings,
        "recordings_with_wav": recordings_with_wav,
        "noise_recordings": len(noise_recordings),
        "dedup_tolerance_ms": args.overlap_tolerance_ms,
        "wav_search_dirs": [str(d) for d in (args.wav_search_dirs or [])],
    }

    # 9. Write output
    output = {
        "positives": all_positives,
        "noise_recordings": noise_recordings,
        "stats": stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # 10. Print summary
    print("\n=== Label Unification Summary ===")
    print(f"Primary (labels.csv):    {source_counts.get('labels_csv', 0)}")
    print(f"Tracking 5970:           {source_counts.get('tracking_5970', 0)}")
    print(f"Tracking 3452:           {source_counts.get('tracking_3452', 0)}")
    print(f"Total positives:         {stats['total_positives']}")
    print(f"Unique recordings:       {unique_recordings}")
    if wav_index:
        print(f"Recordings with WAV:     {recordings_with_wav}/{unique_recordings}")
    print(f"Noise recordings:        {stats['noise_recordings']}")
    if overlap_stems:
        print(f"\nWarnings: {len(overlap_stems)} stem(s) in both positives and noise (excluded from noise)")
    print(f"\nOutput: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
