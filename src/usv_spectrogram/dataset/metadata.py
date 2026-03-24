"""Recording metadata management for dataset preparation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RecordingMetadata:
    """Metadata for a single recording."""

    source_file: str
    mouse_id: str
    population: str = "unknown"
    notes: str = ""


def extract_source_file_from_candidate_id(candidate_id: str) -> str:
    """Extract source WAV filename from candidate_id.

    Examples:
        '2024-09-30_11-18-17_0000001_00000764' -> '2024-09-30_11-18-17_0000001.wav'
        '2024-09-30_11-18-17_0000001_noise_abc123' -> '2024-09-30_11-18-17_0000001.wav'
    """
    # Pattern: date_time_mouseid_framenum or date_time_mouseid_noise_hash
    # We want everything before the last underscore-number or underscore-noise
    parts = candidate_id.split("_")

    # Find where the recording identifier ends
    # Format: YYYY-MM-DD_HH-MM-SS_MOUSEID
    # That's 3 parts: date, time, mouseid
    if len(parts) >= 4:
        # Check if this is a noise sample
        if "noise" in parts:
            noise_idx = parts.index("noise")
            source_parts = parts[:noise_idx]
        else:
            # Regular candidate: date_time_mouseid_framenum
            source_parts = parts[:3]

        source_file = "_".join(source_parts) + ".wav"
        return source_file

    # Fallback: return as-is with .wav
    return candidate_id + ".wav"


def extract_mouse_id_from_source_file(source_file: str) -> str:
    """Extract mouse_id from source filename.

    Example: '2024-09-30_11-18-17_0000001.wav' -> '0000001'
    """
    stem = Path(source_file).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


def generate_metadata_csv(
    labels_csv: Path,
    noise_samples_csv: Path,
    output_csv: Path,
) -> list[RecordingMetadata]:
    """Generate metadata CSV template from labels and noise samples.

    Parameters
    ----------
    labels_csv : Path
        Path to labels.csv with candidate_id,label,labeled_at columns.
    noise_samples_csv : Path
        Path to noise_samples_final.csv with source_file column.
    output_csv : Path
        Path to write the metadata template.

    Returns
    -------
    List of RecordingMetadata objects for all unique recordings.
    """
    source_files = set()

    # Extract source files from labels.csv
    labels_csv = Path(labels_csv)
    if labels_csv.exists():
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidate_id = row.get("candidate_id", "")
                if candidate_id:
                    source_file = extract_source_file_from_candidate_id(candidate_id)
                    source_files.add(source_file)

    # Extract source files from noise samples CSV
    noise_samples_csv = Path(noise_samples_csv)
    if noise_samples_csv.exists():
        with open(noise_samples_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_file = row.get("source_file", "")
                if source_file:
                    # Ensure .wav extension
                    if not source_file.endswith(".wav"):
                        source_file = source_file + ".wav"
                    source_files.add(source_file)

    # Create metadata objects
    metadata_list = []
    for source_file in sorted(source_files):
        mouse_id = extract_mouse_id_from_source_file(source_file)
        metadata_list.append(
            RecordingMetadata(
                source_file=source_file,
                mouse_id=mouse_id,
                population="unknown",
                notes="",
            )
        )

    # Write CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_file", "mouse_id", "population", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for meta in metadata_list:
            writer.writerow({
                "source_file": meta.source_file,
                "mouse_id": meta.mouse_id,
                "population": meta.population,
                "notes": meta.notes,
            })

    return metadata_list


def load_metadata(metadata_csv: Path) -> dict[str, RecordingMetadata]:
    """Load recording metadata from CSV.

    Returns a dict mapping source_file -> RecordingMetadata.
    """
    metadata_csv = Path(metadata_csv)
    result = {}

    if not metadata_csv.exists():
        return result

    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_file = row.get("source_file", "")
            if source_file:
                result[source_file] = RecordingMetadata(
                    source_file=source_file,
                    mouse_id=row.get("mouse_id", "unknown"),
                    population=row.get("population", "unknown"),
                    notes=row.get("notes", ""),
                )

    return result
