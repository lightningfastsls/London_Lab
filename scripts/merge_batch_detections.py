"""Merge per-file batch detection JSONs into a single per-event CSV.

Batch detection runs (``scripts/run_batch_detection.py``) emit one JSON per
input WAV under ``results/batch_<name>/detections/``. Each JSON is either:

* an empty list ``[]`` — file had no detected events, OR
* a list of event dicts with keys
  ``start_time_s, end_time_s, duration_s, max_probability, mean_probability,
  start_col, end_col`` — one dict per detected USV event.

Downstream steps (Raven export, DeepSqueak bridge, classification, corpus
audit) all want a single CSV keyed by ``(stem, detection_idx)``. This script
builds that CSV with the canonical column order used across 5970/3452/9252
pipelines.

Output schema (matches ``results/batch_5970/manual_review_all_detections.csv``)::

    stem, detection_idx, start_time_s, end_time_s, duration_s,
    max_probability, mean_probability, window_count,
    start_window, end_window

The JSON field ``start_col`` maps to ``start_window`` and ``end_col`` to
``end_window``; ``window_count`` is derived as ``end_col - start_col``
(number of STFT frames spanned by the detection).

Usage
-----
    .venv/bin/python scripts/merge_batch_detections.py \\
        --detections-dir results/batch_9252/detections \\
        --output         results/batch_9252/all_detections.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


# Canonical column order — keep aligned with the 5970 reference CSV.
COLUMNS = [
    "stem",
    "detection_idx",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "max_probability",
    "mean_probability",
    "window_count",
    "start_window",
    "end_window",
]


@dataclass(frozen=True)
class MergeStats:
    json_files_total: int
    json_files_with_events: int
    json_files_empty: int
    json_files_malformed: int
    event_rows: int
    unique_stems: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-file batch detection JSONs into one CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/merge_batch_detections.py \\\n"
            "      --detections-dir results/batch_9252/detections \\\n"
            "      --output         results/batch_9252/all_detections.csv\n"
        ),
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        required=True,
        help="Directory containing per-file detection JSONs (one JSON per WAV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (will be created/overwritten).",
    )
    return parser.parse_args()


def _event_rows_from_json(stem: str, payload: Any) -> list[dict[str, Any]]:
    """Convert a parsed JSON payload into a list of event rows.

    The batch detection format is a flat list of event dicts. We defensively
    handle two quirks seen in the wild:

    * top-level dict wrapping a ``"detections"`` or ``"events"`` list —
      reject with a malformed signal (hasn't happened for 9252 but cheap
      to guard).
    * an event missing ``start_col``/``end_col`` — emit NaN rather than
      crashing; the MATLAB/Raven side doesn't need these fields.
    """
    if not isinstance(payload, list):
        raise ValueError(f"expected list at top level, got {type(payload).__name__}")

    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(payload):
        if not isinstance(event, dict):
            raise ValueError(f"event #{idx} is not a dict ({type(event).__name__})")
        start_col = event.get("start_col")
        end_col = event.get("end_col")
        if start_col is not None and end_col is not None:
            window_count = int(end_col) - int(start_col)
        else:
            window_count = None
        rows.append(
            {
                "stem": stem,
                "detection_idx": idx,
                "start_time_s": event.get("start_time_s"),
                "end_time_s": event.get("end_time_s"),
                "duration_s": event.get("duration_s"),
                "max_probability": event.get("max_probability"),
                "mean_probability": event.get("mean_probability"),
                "window_count": window_count,
                "start_window": start_col,
                "end_window": end_col,
            }
        )
    return rows


def merge(detections_dir: Path) -> tuple[pd.DataFrame, MergeStats]:
    if not detections_dir.is_dir():
        raise SystemExit(f"[error] detections dir not found: {detections_dir}")

    json_files = sorted(detections_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"[error] no .json files in {detections_dir}")

    all_rows: list[dict[str, Any]] = []
    n_empty = 0
    n_with_events = 0
    n_malformed = 0

    for path in json_files:
        stem = path.stem
        try:
            raw = path.read_text()
            if not raw.strip():
                # Zero-byte file — treat as empty detection result.
                n_empty += 1
                continue
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(
                f"[warn] malformed JSON: {path.name}: {exc}",
                file=sys.stderr,
            )
            n_malformed += 1
            continue

        try:
            rows = _event_rows_from_json(stem, payload)
        except ValueError as exc:
            print(
                f"[warn] unexpected structure in {path.name}: {exc}",
                file=sys.stderr,
            )
            n_malformed += 1
            continue

        if rows:
            n_with_events += 1
            all_rows.extend(rows)
        else:
            n_empty += 1

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    stats = MergeStats(
        json_files_total=len(json_files),
        json_files_with_events=n_with_events,
        json_files_empty=n_empty,
        json_files_malformed=n_malformed,
        event_rows=len(df),
        unique_stems=int(df["stem"].nunique()) if len(df) else 0,
    )
    return df, stats


def _print_parameters(detections_dir: Path, output: Path) -> None:
    print("=" * 66)
    print("merge_batch_detections.py — Parameters")
    print("=" * 66)
    print(f"  timestamp (UTC)        : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"  detections-dir         : {detections_dir}")
    print(f"  output                 : {output}")
    print(f"  schema                 : {', '.join(COLUMNS)}")
    print("  field mapping          : start_col→start_window, end_col→end_window, "
          "window_count = end_col - start_col")
    print("=" * 66)
    print()


def _print_summary(stats: MergeStats) -> None:
    print(f"{'metric':<32} {'value':>12}")
    print("-" * 46)
    print(f"{'json_files_total':<32} {stats.json_files_total:>12}")
    print(f"{'json_files_with_events':<32} {stats.json_files_with_events:>12}")
    print(f"{'json_files_empty':<32} {stats.json_files_empty:>12}")
    print(f"{'json_files_malformed':<32} {stats.json_files_malformed:>12}")
    print(f"{'event_rows':<32} {stats.event_rows:>12}")
    print(f"{'unique_stems':<32} {stats.unique_stems:>12}")


def main() -> int:
    args = parse_args()

    _print_parameters(args.detections_dir, args.output)
    df, stats = merge(args.detections_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    _print_summary(stats)
    print(f"\n[ok] wrote {args.output} ({stats.event_rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
