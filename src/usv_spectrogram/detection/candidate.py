"""Candidate USV segment dataclass.

A Candidate represents a detected region that may contain a USV.
Includes full metadata for traceability through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Candidate:
    """A detected candidate USV segment.

    Includes full metadata for traceability (see Section 4.8 of reference doc).
    Storage is cheap; missing metadata is expensive later.
    """

    # Source identification
    source_file: Path  # Path to original WAV file
    candidate_id: str  # Unique ID: "{source_stem}_{start_ms:08.0f}"

    # Temporal location (in milliseconds)
    start_ms: float  # Start of detected USV region
    end_ms: float  # End of detected USV region
    duration_ms: float  # end_ms - start_ms

    # Context window for spectrogram extraction
    context_start_ms: float  # Start of context window
    context_end_ms: float  # End of context window

    # Spectral characteristics
    peak_freq_hz: float  # Frequency with maximum energy
    peak_energy_db: float  # Maximum energy in dB

    # Quality flags
    interference_flag: bool = False  # True if peak_freq near known interference

    # Optional: spectrogram path (populated after extraction)
    spectrogram_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Convert source_file to Path if string."""
        if isinstance(self.source_file, str):
            object.__setattr__(self, "source_file", Path(self.source_file))

    @classmethod
    def create(
        cls,
        source_file: Path,
        start_ms: float,
        end_ms: float,
        peak_freq_hz: float,
        peak_energy_db: float,
        context_before_ms: float = 50.0,
        context_after_ms: float = 50.0,
        interference_flag: bool = False,
    ) -> "Candidate":
        """Factory method to create a Candidate with auto-generated ID."""
        duration_ms = end_ms - start_ms
        context_start_ms = max(0.0, start_ms - context_before_ms)
        context_end_ms = end_ms + context_after_ms
        candidate_id = f"{source_file.stem}_{start_ms:08.0f}"

        return cls(
            source_file=source_file,
            candidate_id=candidate_id,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            context_start_ms=context_start_ms,
            context_end_ms=context_end_ms,
            peak_freq_hz=peak_freq_hz,
            peak_energy_db=peak_energy_db,
            interference_flag=interference_flag,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "candidate_id": self.candidate_id,
            "source_file": str(self.source_file.name),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "context_start_ms": self.context_start_ms,
            "context_end_ms": self.context_end_ms,
            "peak_freq_hz": self.peak_freq_hz,
            "peak_energy_db": self.peak_energy_db,
            "interference_flag": self.interference_flag,
            "spectrogram_path": str(self.spectrogram_path) if self.spectrogram_path else "",
        }

    @classmethod
    def from_dict(cls, d: dict, base_path: Optional[Path] = None) -> "Candidate":
        """Create Candidate from dictionary (e.g., from CSV row)."""
        source_file = Path(d["source_file"])
        if base_path and not source_file.is_absolute():
            source_file = base_path / source_file

        spec_path = d.get("spectrogram_path", "")
        spectrogram_path = Path(spec_path) if spec_path else None

        return cls(
            source_file=source_file,
            candidate_id=d["candidate_id"],
            start_ms=float(d["start_ms"]),
            end_ms=float(d["end_ms"]),
            duration_ms=float(d["duration_ms"]),
            context_start_ms=float(d["context_start_ms"]),
            context_end_ms=float(d["context_end_ms"]),
            peak_freq_hz=float(d["peak_freq_hz"]),
            peak_energy_db=float(d["peak_energy_db"]),
            interference_flag=str(d.get("interference_flag", "false")).lower() == "true",
            spectrogram_path=spectrogram_path,
        )
