"""Convert USV detection JSONs to Raven Pro selection table format.

Raven selection tables are tab-delimited text files with a standard header,
used as the interchange format across bioacoustics tools (Raven Pro, DeepSqueak,
Audacity). This module reads individual detection JSON files produced by the
detection_exporter and writes one `.Table.1.selections.txt` per source WAV.

Usage::

    from usv_spectrogram.classification.raven_export import (
        RavenExportConfig, export_raven_tables,
    )

    config = RavenExportConfig(
        detections_dir=Path("USV_Detections"),
        wav_dir=Path("5970 USV"),
        output_dir=Path("raven_tables"),
    )
    paths = export_raven_tables(config)
"""

# VAULT: Raven selection table format is the standard interchange format between bioacoustic analysis tools,
#        DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries
# Run `/kcheck` before modifying this file.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Files inside detection subdirectories that are NOT individual detection JSONs.
_SKIP_FILENAMES = {"_saved_tracking.json", "detections_summary.csv"}
_SKIP_SUFFIXES = {".png", ".csv"}

# Raven column spec — order matters for TSV output.
_RAVEN_COLUMNS = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RavenExportConfig:
    """Configuration for a Raven selection-table export run.

    Parameters
    ----------
    detections_dir : Path
        Root of the detection tree (e.g. ``USV_Detections/``).
        Each immediate subdirectory corresponds to one WAV file.
    wav_dir : Path
        Directory containing the source ``.wav`` files.  May be ``None``
        when ``batch_format=True`` (batch JSONs encode their own stems).
    output_dir : Path
        Where to write the ``.Table.1.selections.txt`` files.
    low_freq_hz : float
        Lower frequency bound written into every selection row.
    high_freq_hz : float
        Upper frequency bound written into every selection row.
    batch_format : bool
        When True, *detections_dir* contains flat JSON files (one per WAV,
        each holding a list of detection dicts) as produced by
        ``run_batch_detection.py``.  The WAV stem is the JSON filename.
    """

    detections_dir: Path
    wav_dir: Path | None = None
    output_dir: Path = Path("raven_tables")
    low_freq_hz: float = 25_000.0
    high_freq_hz: float = 125_000.0
    batch_format: bool = False

    def __post_init__(self) -> None:
        # Allow string paths for convenience.
        for attr in ("detections_dir", "output_dir"):
            val = getattr(self, attr)
            if not isinstance(val, Path):
                object.__setattr__(self, attr, Path(val))
        if self.wav_dir is not None and not isinstance(self.wav_dir, Path):
            object.__setattr__(self, "wav_dir", Path(self.wav_dir))

        if not self.batch_format and self.wav_dir is None:
            raise ValueError(
                "wav_dir is required when batch_format is False"
            )

        if self.low_freq_hz < 0 or self.high_freq_hz < 0:
            raise ValueError(
                f"Frequencies must be non-negative, "
                f"got low={self.low_freq_hz}, high={self.high_freq_hz}"
            )
        if self.low_freq_hz >= self.high_freq_hz:
            raise ValueError(
                f"low_freq_hz ({self.low_freq_hz}) must be less than "
                f"high_freq_hz ({self.high_freq_hz})"
            )


@dataclass
class ExportSummary:
    """Accumulates statistics during an export run."""

    total_wav_files: int = 0
    total_detections: int = 0
    total_tables_written: int = 0
    unmapped_dirs: list[str] = field(default_factory=list)
    empty_detection_dirs: list[str] = field(default_factory=list)
    per_wav_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializable dictionary for ``export_summary.json``."""
        return {
            "total_wav_files": self.total_wav_files,
            "total_detections": self.total_detections,
            "total_tables_written": self.total_tables_written,
            "unmapped_count": len(self.unmapped_dirs),
            "unmapped_dirs": self.unmapped_dirs,
            "empty_detection_dirs": self.empty_detection_dirs,
            "per_wav_counts": self.per_wav_counts,
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_detection_json(json_path: Path) -> dict:
    """Load a single detection JSON and extract core timing fields.

    Callers are responsible for pre-filtering non-detection files
    (e.g. ``_saved_tracking.json``); see :func:`_is_detection_json`.

    Parameters
    ----------
    json_path : Path
        Path to a detection JSON file written by ``detection_exporter``.

    Returns
    -------
    dict
        ``{"start_s": float, "end_s": float, "duration_ms": float}``

    Raises
    ------
    ValueError
        If the file is not valid JSON or lacks the ``core_time`` key.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {json_path}: {exc}") from exc

    if "core_time" not in data:
        raise ValueError(f"Missing 'core_time' in {json_path}")

    ct = data["core_time"]
    return {
        "start_s": ct["start_s"],
        "end_s": ct["end_s"],
        "duration_ms": ct["duration_ms"],
    }


def _is_detection_json(path: Path) -> bool:
    """Return True if *path* looks like an individual detection JSON."""
    if path.name in _SKIP_FILENAMES:
        return False
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return False
    return path.suffix.lower() == ".json"


def discover_wav_detection_mapping(
    detections_dir: Path,
    wav_dir: Path,
) -> dict[str, list[Path]]:
    """Map WAV file stems to their sorted lists of detection JSON paths.

    The convention is: each immediate subdirectory of *detections_dir* has
    a name that matches (or starts with) a ``.wav`` stem in *wav_dir*.

    Parameters
    ----------
    detections_dir : Path
        Root detection directory (e.g. ``USV_Detections/``).
    wav_dir : Path
        Directory containing ``.wav`` files.

    Returns
    -------
    dict[str, list[Path]]
        ``{wav_stem: [sorted json paths]}`` for every matched WAV.
        Unmatched detection directories are logged as warnings.
    """
    if not detections_dir.is_dir():
        raise FileNotFoundError(f"Detections directory not found: {detections_dir}")
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    # Build a set of WAV stems for fast lookup.
    wav_stems: set[str] = set()
    for wav_path in wav_dir.iterdir():
        if wav_path.suffix.lower() == ".wav":
            wav_stems.add(wav_path.stem)

    mapping: dict[str, list[Path]] = {}

    for subdir in sorted(detections_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Match directory name to a WAV stem.
        matched_stem: str | None = None
        if subdir.name in wav_stems:
            matched_stem = subdir.name
        else:
            # Try prefix match: detection dir may have a session suffix.
            # Sort by descending length so the longest (most specific) stem wins.
            for stem in sorted(wav_stems, key=len, reverse=True):
                if subdir.name.startswith(stem):
                    matched_stem = stem
                    break

        if matched_stem is None:
            logger.warning("Unmapped detection directory: %s", subdir.name)
            continue

        # Collect detection JSONs from this directory.
        json_paths = sorted(
            p for p in subdir.iterdir() if _is_detection_json(p)
        )

        if not json_paths:
            continue

        mapping.setdefault(matched_stem, []).extend(json_paths)

    # Re-sort after potential multi-directory merges.
    for stem in mapping:
        mapping[stem] = sorted(mapping[stem])

    return mapping


def discover_batch_detection_mapping(
    detections_dir: Path,
) -> dict[str, list[dict]]:
    """Load batch-format detection JSONs (one flat file per WAV).

    Batch detection files are produced by ``run_batch_detection.py``.  Each
    file is named ``<wav_stem>.json`` and contains a JSON list of detection
    dicts with keys ``start_time_s``, ``end_time_s``, ``duration_s``,
    ``max_probability``, ``mean_probability``.

    Parameters
    ----------
    detections_dir : Path
        Directory of flat ``.json`` files (one per WAV).

    Returns
    -------
    dict[str, list[dict]]
        ``{wav_stem: [detection dicts]}`` for every WAV with ≥1 detection.
        Each detection dict is normalized to ``{start_s, end_s, duration_ms}``.
    """
    if not detections_dir.is_dir():
        raise FileNotFoundError(
            f"Detections directory not found: {detections_dir}"
        )

    mapping: dict[str, list[dict]] = {}

    for json_path in sorted(detections_dir.glob("*.json")):
        wav_stem = json_path.stem

        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s: malformed JSON: %s", json_path.name, exc)
            continue

        if not isinstance(data, list):
            logger.warning("Skipping %s: expected list, got %s", json_path.name, type(data).__name__)
            continue

        if not data:
            continue  # Empty list = no detections for this WAV.

        detections: list[dict] = []
        for det_raw in data:
            try:
                detections.append({
                    "start_s": det_raw["start_time_s"],
                    "end_s": det_raw["end_time_s"],
                    "duration_ms": det_raw.get("duration_s", 0) * 1000,
                })
            except KeyError as exc:
                logger.warning(
                    "Skipping detection in %s: missing key %s",
                    json_path.name, exc,
                )

        if detections:
            mapping[wav_stem] = sorted(detections, key=lambda d: d["start_s"])

    return mapping


def detections_to_raven_table(
    detections: list[dict],
    low_freq_hz: float = 25_000.0,
    high_freq_hz: float = 125_000.0,
) -> pd.DataFrame:
    """Build a Raven-format DataFrame from loaded detection dicts.

    Parameters
    ----------
    detections : list[dict]
        Each dict must contain ``start_s`` and ``end_s`` (as returned by
        :func:`load_detection_json`).
    low_freq_hz, high_freq_hz : float
        Frequency bounds written into every row (mouse USV band defaults).

    Returns
    -------
    pd.DataFrame
        Columns match :data:`_RAVEN_COLUMNS`, sorted by Begin Time,
        with 1-indexed Selection numbers.
    """
    # Sort by start time.
    sorted_dets = sorted(detections, key=lambda d: d["start_s"])

    rows: list[dict] = []
    for idx, det in enumerate(sorted_dets, start=1):
        rows.append(
            {
                "Selection": idx,
                "View": "Spectrogram 1",
                "Channel": 1,
                "Begin Time (s)": round(det["start_s"], 4),
                "End Time (s)": round(det["end_s"], 4),
                "Low Freq (Hz)": int(low_freq_hz),
                "High Freq (Hz)": int(high_freq_hz),
            }
        )

    return pd.DataFrame(rows, columns=_RAVEN_COLUMNS)


def export_raven_tables(config: RavenExportConfig) -> list[Path]:
    """Run the full export pipeline.

    1. Discover WAV ↔ detection JSON mapping.
    2. Convert each WAV's detections into a Raven selection table.
    3. Write TSV files and an ``export_summary.json``.

    Parameters
    ----------
    config : RavenExportConfig
        Validated export configuration.

    Returns
    -------
    list[Path]
        Paths of the created ``.Table.1.selections.txt`` files.
    """
    if not config.detections_dir.is_dir():
        raise FileNotFoundError(
            f"Detections directory not found: {config.detections_dir}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.batch_format:
        # Batch path: flat JSON files, no WAV directory needed.
        batch_mapping = discover_batch_detection_mapping(config.detections_dir)
        return _export_from_preloaded(config, batch_mapping)

    # Per-detection subdirectory path (original).
    if config.wav_dir is None or not config.wav_dir.is_dir():
        raise FileNotFoundError(
            f"WAV directory not found: {config.wav_dir}"
        )

    # -- Step 1: discover --
    mapping = discover_wav_detection_mapping(
        config.detections_dir, config.wav_dir
    )

    # Classify detection subdirectories into three buckets:
    #   - mapped: has a matching WAV AND detection JSONs (in `mapping`)
    #   - empty_matched: has a matching WAV but zero detection JSONs
    #   - unmapped: no matching WAV file exists at all
    wav_stems = {
        p.stem for p in config.wav_dir.iterdir()
        if p.suffix.lower() == ".wav"
    }
    all_subdirs = sorted(
        d.name for d in config.detections_dir.iterdir() if d.is_dir()
    )

    mapped_dir_names: set[str] = set()
    for stem in mapping:
        for jp in mapping[stem]:
            mapped_dir_names.add(jp.parent.name)

    unmapped: list[str] = []
    empty_matched: list[str] = []
    for name in all_subdirs:
        if name in mapped_dir_names:
            continue  # Has detections and a matching WAV — all good.
        # Check if this dir matches a WAV (exact or prefix).
        has_wav = name in wav_stems or any(
            name.startswith(stem) for stem in wav_stems
        )
        if has_wav:
            empty_matched.append(name)
        else:
            unmapped.append(name)

    summary = ExportSummary(
        total_wav_files=len(mapping),
        unmapped_dirs=unmapped,
        empty_detection_dirs=empty_matched,
    )

    created_paths: list[Path] = []

    # -- Step 2–3: convert & write --
    for wav_stem, json_paths in sorted(mapping.items()):
        detections: list[dict] = []
        for jp in json_paths:
            try:
                det = load_detection_json(jp)
                detections.append(det)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", jp.name, exc)

        if not detections:
            continue

        df = detections_to_raven_table(
            detections,
            low_freq_hz=config.low_freq_hz,
            high_freq_hz=config.high_freq_hz,
        )

        out_path = config.output_dir / f"{wav_stem}.Table.1.selections.txt"
        df.to_csv(out_path, sep="\t", index=False, lineterminator="\n")

        created_paths.append(out_path)
        summary.total_detections += len(detections)
        summary.total_tables_written += 1
        summary.per_wav_counts[wav_stem] = len(detections)

        logger.info(
            "Wrote %d selections -> %s", len(detections), out_path.name
        )

    # -- Step 4: summary file --
    summary_path = config.output_dir / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)
    logger.info("Export summary -> %s", summary_path)

    return created_paths


def _export_from_preloaded(
    config: RavenExportConfig,
    detections_by_stem: dict[str, list[dict]],
) -> list[Path]:
    """Write Raven tables from pre-loaded detection dicts.

    Shared write path for batch format (dicts already loaded) vs
    per-detection format (which loads from individual JSON files).
    """
    summary = ExportSummary(total_wav_files=len(detections_by_stem))
    created_paths: list[Path] = []

    for wav_stem, detections in sorted(detections_by_stem.items()):
        if not detections:
            continue

        df = detections_to_raven_table(
            detections,
            low_freq_hz=config.low_freq_hz,
            high_freq_hz=config.high_freq_hz,
        )

        out_path = config.output_dir / f"{wav_stem}.Table.1.selections.txt"
        df.to_csv(out_path, sep="\t", index=False, lineterminator="\n")

        created_paths.append(out_path)
        summary.total_detections += len(detections)
        summary.total_tables_written += 1
        summary.per_wav_counts[wav_stem] = len(detections)

        logger.info(
            "Wrote %d selections -> %s", len(detections), out_path.name
        )

    summary_path = config.output_dir / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)
    logger.info("Export summary -> %s", summary_path)

    return created_paths
