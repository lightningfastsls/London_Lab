"""Import DeepSqueak classification results and merge with detection metadata.

DeepSqueak exports per-call classification labels and acoustic features as Excel
files. This module reads those files, normalizes column names to snake_case,
and re-merges them with our detection JSONs using timestamp proximity matching.

Round-trip pipeline::

    Our detections -> Raven tables -> DeepSqueak (MATLAB) -> Excel -> THIS -> merged CSV

Usage::

    from usv_spectrogram.classification.deepsqueak_import import (
        DeepSqueakImportConfig, import_deepsqueak_results,
    )

    config = DeepSqueakImportConfig(
        results_dir=Path("deepsqueak_results"),
        detections_dir=Path("USV_Detections"),
        output_path=Path("classified_detections.csv"),
    )
    summary = import_deepsqueak_results(config)
"""

# VAULT: DeepSqueak import previously required exact subdirectory name matches while Raven export already supported prefix matches creating a silent asymmetric round-trip,
#        timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations
# Run `/kcheck` before modifying this file.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .raven_export import _is_detection_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSqueak Excel column normalization
# ---------------------------------------------------------------------------

# Maps raw DeepSqueak Excel column names to clean snake_case names.
# DeepSqueak exports 16 columns (15 features + ID).
_COLUMN_MAP: dict[str, str] = {
    "ID": "id",
    "Label": "label",
    "Type": "label",  # Some DS versions use "Type" instead of "Label"
    "Begin Time (s)": "begin_time_s",
    "End Time (s)": "end_time_s",
    "Call Length (s)": "call_length_s",
    "Principal Frequency (kHz)": "principal_freq_hz",
    "Low Freq (kHz)": "low_freq_hz",
    "High Freq (kHz)": "high_freq_hz",
    "Delta Freq (kHz)": "bandwidth_hz",
    "Frequency Standard Deviation (kHz)": "freq_std_dev_hz",
    "Slope (kHz/s)": "slope",
    "Sinuosity": "sinuosity",
    "Mean Power (dB/Hz)": "mean_power_db",
    "Tonality": "tonality",
    "Peak Frequency (kHz)": "peak_freq_hz",
}

# The 16 canonical output columns in order (after normalization).
_DS_OUTPUT_COLUMNS = [
    "id",
    "label",
    "begin_time_s",
    "end_time_s",
    "call_length_s",
    "principal_freq_hz",
    "low_freq_hz",
    "high_freq_hz",
    "bandwidth_hz",
    "freq_std_dev_hz",
    "slope",
    "sinuosity",
    "mean_power_db",
    "tonality",
    "peak_freq_hz",
]

# Known DeepSqueak filename suffixes to strip when extracting WAV stem.
_DS_SUFFIXES = ("_Detections", "_detections", "_calls", "_Calls", "_classified")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepSqueakImportConfig:
    """Configuration for importing DeepSqueak classification results.

    Parameters
    ----------
    results_dir : Path
        Directory containing DeepSqueak Excel output files (.xlsx).
    detections_dir : Path
        Root of the detection tree (e.g. ``USV_Detections/``).
    output_path : Path
        Where to write the merged CSV file.
    tolerance_ms : float
        Maximum time difference (in ms) for timestamp matching.
        Default 5.0 ms is conservative — timestamps should be near-identical
        since DeepSqueak reads our Raven selection tables.
    """

    results_dir: Path
    detections_dir: Path
    output_path: Path
    tolerance_ms: float = 5.0

    def __post_init__(self) -> None:
        # Allow string paths for convenience (same pattern as RavenExportConfig).
        for attr in ("results_dir", "detections_dir", "output_path"):
            val = getattr(self, attr)
            if not isinstance(val, Path):
                object.__setattr__(self, attr, Path(val))

        if self.tolerance_ms <= 0:
            raise ValueError(
                f"tolerance_ms must be positive, got {self.tolerance_ms}"
            )


@dataclass
class ImportSummary:
    """Accumulates statistics during an import+merge run."""

    total_ds_calls: int = 0
    total_detections: int = 0
    matched: int = 0
    unmatched_ds: int = 0
    unmatched_det: int = 0
    files_processed: int = 0
    per_file_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializable dictionary for summary output."""
        return {
            "total_ds_calls": self.total_ds_calls,
            "total_detections": self.total_detections,
            "matched": self.matched,
            "unmatched_ds": self.unmatched_ds,
            "unmatched_det": self.unmatched_det,
            "files_processed": self.files_processed,
            "match_rate": (
                round(self.matched / self.total_ds_calls, 3)
                if self.total_ds_calls > 0
                else 0.0
            ),
            "per_file_counts": self.per_file_counts,
        }


# ---------------------------------------------------------------------------
# Excel loading
# ---------------------------------------------------------------------------

def _extract_wav_stem(filename: str) -> str:
    """Extract the WAV stem from a DeepSqueak output filename.

    DeepSqueak names its output after the input WAV file, sometimes
    appending suffixes like ``_Detections`` or ``_calls``.

    Parameters
    ----------
    filename : str
        Excel filename (e.g. ``rec_001.xlsx`` or ``rec_001_Detections.xlsx``).

    Returns
    -------
    str
        The WAV stem (e.g. ``rec_001``).
    """
    stem = Path(filename).stem
    for suffix in _DS_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DeepSqueak Excel columns to snake_case using ``_COLUMN_MAP``.

    Columns not in the map are passed through unchanged with basic
    normalization (lowercase, spaces to underscores).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from a DeepSqueak Excel file.

    Returns
    -------
    pd.DataFrame
        Same data with normalized column names.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in _COLUMN_MAP:
            rename_map[col] = _COLUMN_MAP[col_str]
        else:
            # Fallback normalization for unknown columns.
            rename_map[col] = (
                col_str.lower()
                .replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("/", "_").replace("-", "_").replace(".", "_")
            )
    return df.rename(columns=rename_map)


def load_deepsqueak_excel(path: Path) -> pd.DataFrame:
    """Load a single DeepSqueak Excel file and normalize columns.

    Parameters
    ----------
    path : Path
        Path to an ``.xlsx`` file exported by DeepSqueak.

    Returns
    -------
    pd.DataFrame
        Normalized columns, with ``source_file`` and ``wav_stem`` added.

    Raises
    ------
    ValueError
        If the file is corrupt, unreadable, or has zero data rows.
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Cannot read Excel file {path.name}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Empty Excel file: {path.name}")

    df = _normalize_columns(df)
    df["source_file"] = path.name
    df["wav_stem"] = _extract_wav_stem(path.name)
    return df


def load_all_deepsqueak_results(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate all DeepSqueak Excel files from a directory.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``.xlsx`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all files with ``source_file`` column.

    Raises
    ------
    FileNotFoundError
        If *results_dir* does not exist or contains no ``.xlsx`` files.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    xlsx_paths = sorted(results_dir.glob("*.xlsx"))
    if not xlsx_paths:
        raise FileNotFoundError(f"No .xlsx files found in {results_dir}")

    frames: list[pd.DataFrame] = []
    for path in xlsx_paths:
        try:
            df = load_deepsqueak_excel(path)
            frames.append(df)
            logger.info("Loaded %d rows from %s", len(df), path.name)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)

    if not frames:
        raise ValueError("No valid Excel files could be loaded")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Detection JSON loading (extended for merge)
# ---------------------------------------------------------------------------

def _load_detection_json_extended(json_path: Path) -> dict:
    """Load a detection JSON with all fields needed for merge.

    Extends the basic ``load_detection_json`` from raven_export by also
    extracting probability scores, detection index, and user action.

    Parameters
    ----------
    json_path : Path
        Path to a detection JSON file.

    Returns
    -------
    dict
        Keys: ``start_s``, ``end_s``, ``duration_ms``, ``detection_index``,
        ``prob_max``, ``prob_mean``, ``user_action``, ``json_path``.

    Raises
    ------
    ValueError
        If the JSON is malformed or missing ``core_time``.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {json_path}: {exc}") from exc

    if "core_time" not in data:
        raise ValueError(f"Missing 'core_time' in {json_path}")

    ct = data["core_time"]
    probs = data.get("probabilities", {})

    return {
        "start_s": ct["start_s"],
        "end_s": ct["end_s"],
        "duration_ms": ct["duration_ms"],
        "detection_index": data.get("detection_index", -1),
        "prob_max": probs.get("max", None),
        "prob_mean": probs.get("mean", None),
        "user_action": data.get("user_action", None),
        "json_path": str(json_path),
    }


def load_detections_for_merge(
    detections_dir: Path,
) -> dict[str, list[dict]]:
    """Load all detection JSONs grouped by WAV stem for merge.

    Parameters
    ----------
    detections_dir : Path
        Root detection directory (e.g. ``USV_Detections/``).

    Returns
    -------
    dict[str, list[dict]]
        ``{wav_stem: [detection_dicts]}`` sorted by start time within
        each stem.

    Raises
    ------
    FileNotFoundError
        If *detections_dir* does not exist.
    """
    if not detections_dir.is_dir():
        raise FileNotFoundError(
            f"Detections directory not found: {detections_dir}"
        )

    result: dict[str, list[dict]] = {}

    for subdir in sorted(detections_dir.iterdir()):
        if not subdir.is_dir():
            continue

        wav_stem = subdir.name
        detections: list[dict] = []

        for json_path in sorted(subdir.iterdir()):
            if not _is_detection_json(json_path):
                continue
            try:
                det = _load_detection_json_extended(json_path)
                detections.append(det)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", json_path.name, exc)

        if detections:
            # Sort by start time for efficient matching.
            detections.sort(key=lambda d: d["start_s"])
            result[wav_stem] = detections

    return result


# ---------------------------------------------------------------------------
# Timestamp proximity matching
# ---------------------------------------------------------------------------

def merge_with_detections(
    ds_df: pd.DataFrame,
    detections_by_stem: dict[str, list[dict]],
    tolerance_ms: float = 5.0,
) -> tuple[pd.DataFrame, ImportSummary]:
    """Merge DeepSqueak results with detection JSONs by timestamp proximity.

    Algorithm:
    1. Group DS rows by ``wav_stem``.
    2. For each group, perform greedy 1:1 nearest-neighbor matching against
       the detection list for that WAV.
    3. Matches within ``tolerance_ms`` are accepted; others are marked
       unmatched.

    Parameters
    ----------
    ds_df : pd.DataFrame
        Concatenated DeepSqueak results (must have ``wav_stem`` and
        ``begin_time_s`` columns).
    detections_by_stem : dict[str, list[dict]]
        Detection JSONs grouped by WAV stem (from :func:`load_detections_for_merge`).
    tolerance_ms : float
        Maximum allowed time difference in milliseconds.

    Returns
    -------
    tuple[pd.DataFrame, ImportSummary]
        Merged DataFrame and summary statistics.
    """
    summary = ImportSummary()
    summary.total_ds_calls = len(ds_df)

    # Count total detections across all stems.
    for dets in detections_by_stem.values():
        summary.total_detections += len(dets)

    # Track per-file counts.
    if "source_file" in ds_df.columns:
        summary.per_file_counts = ds_df["source_file"].value_counts().to_dict()
        summary.files_processed = len(summary.per_file_counts)

    tolerance_s = tolerance_ms / 1000.0
    merged_rows: list[dict] = []
    ds_stems = [str(stem) for stem in ds_df["wav_stem"].dropna().unique()]
    ds_to_detection_stem = _resolve_detection_stem_mapping(
        ds_stems, set(detections_by_stem)
    )
    matched_detection_stems: set[str] = set()

    for wav_stem, group_df in ds_df.groupby("wav_stem"):
        detection_stem = ds_to_detection_stem.get(str(wav_stem))
        available_dets = detections_by_stem.get(detection_stem, [])
        if detection_stem is not None:
            matched_detection_stems.add(detection_stem)

        if not available_dets:
            # All DS calls for this stem are unmatched.
            for _, row in group_df.iterrows():
                merged_row = row.to_dict()
                merged_row.update(_empty_detection_columns())
                merged_row["match_quality"] = "unmatched_ds"
                merged_row["match_distance_ms"] = None
                merged_rows.append(merged_row)
                summary.unmatched_ds += 1
            continue

        # Build index of available detections for this stem.
        # We'll remove matched ones as we go (greedy 1:1).
        available_indices = set(range(len(available_dets)))

        # Sort DS calls by begin_time for consistent matching order.
        sorted_group = group_df.sort_values("begin_time_s")

        for _, ds_row in sorted_group.iterrows():
            ds_time = ds_row["begin_time_s"]
            best_idx = None
            best_dist = float("inf")

            # Find nearest available detection.
            for idx in available_indices:
                det = available_dets[idx]
                dist = abs(ds_time - det["start_s"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None and best_dist <= tolerance_s:
                # Match found.
                det = available_dets[best_idx]
                available_indices.discard(best_idx)

                merged_row = ds_row.to_dict()
                merged_row.update(_detection_columns(det))

                if best_dist == 0.0:
                    merged_row["match_quality"] = "exact"
                else:
                    merged_row["match_quality"] = "fuzzy"
                merged_row["match_distance_ms"] = round(best_dist * 1000, 3)

                merged_rows.append(merged_row)
                summary.matched += 1
            else:
                # No match within tolerance.
                merged_row = ds_row.to_dict()
                merged_row.update(_empty_detection_columns())
                merged_row["match_quality"] = "unmatched_ds"
                merged_row["match_distance_ms"] = (
                    round(best_dist * 1000, 3) if best_dist != float("inf") else None
                )
                merged_rows.append(merged_row)
                summary.unmatched_ds += 1

        # Remaining unmatched detections for this stem.
        for idx in sorted(available_indices):
            det = available_dets[idx]
            merged_row = _empty_ds_columns()
            merged_row["wav_stem"] = wav_stem
            merged_row.update(_detection_columns(det))
            merged_row["match_quality"] = "unmatched_det"
            merged_row["match_distance_ms"] = None
            merged_rows.append(merged_row)
            summary.unmatched_det += 1

    # Handle detections for stems with no DS results at all.
    for wav_stem, dets in detections_by_stem.items():
        if wav_stem not in matched_detection_stems:
            for det in dets:
                merged_row = _empty_ds_columns()
                merged_row["wav_stem"] = wav_stem
                merged_row.update(_detection_columns(det))
                merged_row["match_quality"] = "unmatched_det"
                merged_row["match_distance_ms"] = None
                merged_rows.append(merged_row)
                summary.unmatched_det += 1

    result_df = pd.DataFrame(merged_rows)
    return result_df, summary


def _resolve_detection_stem_mapping(
    ds_stems: list[str],
    detection_stems: set[str],
) -> dict[str, str]:
    """Resolve DS wav stems to detection stems using Raven-export semantics.

    Exact matches win first. Remaining stems may match a detection directory
    whose name starts with the DS wav stem, mirroring Raven export's prefix
    matching so round-trip imports work with suffixed detection folders.
    """
    resolved: dict[str, str] = {}
    unmatched_detection_stems = set(detection_stems)

    for ds_stem in sorted(ds_stems, key=len, reverse=True):
        if ds_stem in unmatched_detection_stems:
            resolved[ds_stem] = ds_stem
            unmatched_detection_stems.remove(ds_stem)

    for ds_stem in sorted(ds_stems, key=len, reverse=True):
        if ds_stem in resolved:
            continue
        candidates = [
            detection_stem
            for detection_stem in unmatched_detection_stems
            if detection_stem.startswith(ds_stem)
        ]
        if not candidates:
            continue
        matched_stem = min(candidates, key=len)
        resolved[ds_stem] = matched_stem
        unmatched_detection_stems.remove(matched_stem)

    return resolved


def _detection_columns(det: dict) -> dict:
    """Extract detection fields with ``det_`` prefix for merge output."""
    return {
        "det_start_s": det["start_s"],
        "det_end_s": det["end_s"],
        "det_duration_ms": det["duration_ms"],
        "det_index": det["detection_index"],
        "det_prob_max": det["prob_max"],
        "det_prob_mean": det["prob_mean"],
        "det_user_action": det["user_action"],
        "det_json_path": det["json_path"],
    }


def _empty_detection_columns() -> dict:
    """Return empty detection columns for unmatched DS calls."""
    return {
        "det_start_s": None,
        "det_end_s": None,
        "det_duration_ms": None,
        "det_index": None,
        "det_prob_max": None,
        "det_prob_mean": None,
        "det_user_action": None,
        "det_json_path": None,
    }


def _empty_ds_columns() -> dict:
    """Return empty DS columns for unmatched detections."""
    result: dict = {}
    for col in _DS_OUTPUT_COLUMNS:
        result[col] = None
    result["source_file"] = None
    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_classified_detections(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write merged results to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from :func:`merge_with_detections`.
    output_path : Path
        Destination CSV path.

    Returns
    -------
    Path
        The written file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def import_deepsqueak_results(config: DeepSqueakImportConfig) -> ImportSummary:
    """Run the full import+merge pipeline.

    1. Load all DeepSqueak Excel files from ``results_dir``.
    2. Load all detection JSONs from ``detections_dir``.
    3. Merge by timestamp proximity.
    4. Export to CSV.

    Parameters
    ----------
    config : DeepSqueakImportConfig
        Validated import configuration.

    Returns
    -------
    ImportSummary
        Statistics about the merge operation.
    """
    # Step 1: Load DeepSqueak results.
    ds_df = load_all_deepsqueak_results(config.results_dir)
    logger.info(
        "Loaded %d DeepSqueak calls from %d files",
        len(ds_df),
        ds_df["source_file"].nunique(),
    )

    # Step 2: Load detection JSONs.
    detections_by_stem = load_detections_for_merge(config.detections_dir)
    total_dets = sum(len(v) for v in detections_by_stem.values())
    logger.info(
        "Loaded %d detections across %d WAV stems",
        total_dets,
        len(detections_by_stem),
    )

    # Step 3: Merge.
    merged_df, summary = merge_with_detections(
        ds_df, detections_by_stem, tolerance_ms=config.tolerance_ms
    )

    # Step 4: Export.
    export_classified_detections(merged_df, config.output_path)

    # Write summary JSON alongside the CSV.
    summary_path = config.output_path.parent / "import_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)
    logger.info("Import summary -> %s", summary_path)

    return summary
