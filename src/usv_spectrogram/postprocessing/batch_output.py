"""Batch output for USV detection pipeline.

Writes triage results as:
- ``summary.parquet`` — one row per recording with QC metrics (fast column queries)
- ``detections/<stem>.json`` — per-recording event list in ADR-010 format
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .triage import RecordingResult


# ADR-010 columns for the parquet summary
_PARQUET_COLUMNS = [
    "filepath",
    "tier",
    "n_events",
    "max_confidence",
    "mean_event_confidence",
    "total_usv_duration_ms",
    "noise_floor_p90",
    "confidence_score",
]


def _event_to_adr010_dict(event, hop_px: int = 10) -> dict:
    """Convert a USVEvent to an ADR-010 compatible detection dict.

    Includes ``start_col``/``end_col`` computed from window indices
    and ``hop_px`` for compatibility with the desktop app's
    ``label_storage.py`` which requires these fields.
    """
    return {
        "start_time_s": event.start_time_s,
        "end_time_s": event.end_time_s,
        "duration_s": event.end_time_s - event.start_time_s,
        "start_col": int(event.start_window * hop_px),
        "end_col": int(event.end_window * hop_px),
        "max_probability": event.peak_probability,
        "mean_probability": event.mean_probability,
    }


def write_batch_results(
    results: List[RecordingResult],
    output_dir: Path,
    write_parquet: bool = True,
    write_per_recording_json: bool = True,
) -> None:
    """Write batch triage results to disk.

    Parameters
    ----------
    results : list[RecordingResult]
        Triage results from ``triage_recording()``.
    output_dir : Path
        Root directory for output files.
    write_parquet : bool
        If True, write ``summary.parquet`` with one row per recording.
    write_per_recording_json : bool
        If True, write ``detections/<stem>.json`` for each recording.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if write_parquet:
        _write_parquet(results, output_dir)

    if write_per_recording_json:
        _write_per_recording_jsons(results, output_dir)


def _write_parquet(results: List[RecordingResult], output_dir: Path) -> None:
    """Write summary.parquet — one row per recording."""
    import pandas as pd  # noqa: PLC0415

    if not results:
        # Empty DataFrame with the correct schema
        df = pd.DataFrame(columns=_PARQUET_COLUMNS)
    else:
        rows = [
            {
                "filepath": r.filepath,
                "tier": r.tier,
                "n_events": r.n_events,
                "max_confidence": r.max_confidence,
                "mean_event_confidence": r.mean_event_confidence,
                "total_usv_duration_ms": r.total_usv_duration_ms,
                "noise_floor_p90": r.noise_floor_p90,
                "confidence_score": r.confidence_score,
            }
            for r in results
        ]
        df = pd.DataFrame(rows, columns=_PARQUET_COLUMNS)

    df.to_parquet(output_dir / "summary.parquet", index=False)


def _write_per_recording_jsons(
    results: List[RecordingResult], output_dir: Path
) -> None:
    """Write detections/<stem>.json for each recording."""
    detections_dir = output_dir / "detections"
    detections_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        stem = Path(r.filepath).stem
        detections = [_event_to_adr010_dict(e) for e in r.events]
        json_path = detections_dir / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=2)
