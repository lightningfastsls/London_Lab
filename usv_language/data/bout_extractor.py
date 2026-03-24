"""Bout extraction from USV detection results.

Groups individual USV detections into bouts based on temporal proximity,
adds context padding, and handles splitting of over-long bouts.

References:
    ADR-014 (bout grouping parameters)
    ADR-010 (detection JSON format)
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoutExtractionConfig:
    """Configuration for bout extraction from USV detections.

    All time thresholds are in milliseconds for human-readable config,
    converted to seconds internally when needed.
    """

    bout_gap_threshold_ms: float = 500.0
    context_padding_ms: float = 200.0
    min_bout_duration_ms: float = 50.0
    max_bout_duration_ms: float = 10000.0

    def __post_init__(self) -> None:
        if self.bout_gap_threshold_ms <= 0:
            raise ValueError(
                f"bout_gap_threshold_ms must be positive, got {self.bout_gap_threshold_ms}"
            )
        if self.context_padding_ms < 0:
            raise ValueError(
                f"context_padding_ms must be non-negative, got {self.context_padding_ms}"
            )
        if self.min_bout_duration_ms <= 0:
            raise ValueError(
                f"min_bout_duration_ms must be positive, got {self.min_bout_duration_ms}"
            )
        if self.max_bout_duration_ms <= self.min_bout_duration_ms:
            raise ValueError(
                f"max_bout_duration_ms ({self.max_bout_duration_ms}) must exceed "
                f"min_bout_duration_ms ({self.min_bout_duration_ms})"
            )


@dataclass
class Bout:
    """A temporal group of USV detections with context padding."""

    source_file: str
    start_time_s: float
    end_time_s: float
    usv_count: int
    usv_times: list[tuple[float, float]] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000.0


class BoutExtractor:
    """Extract bouts from USV detection results.

    Supports two input formats:
    1. Detection directories with _saved_tracking.json or individual detection JSONs
    2. CSV files with columns: wav_file, start_time_s, end_time_s
    """

    def __init__(self, config: BoutExtractionConfig | None = None) -> None:
        self.config = config or BoutExtractionConfig()
        self._wav_index: dict[str, Path] | None = None

    def extract_from_csv(
        self,
        csv_path: str | Path,
        wav_dir: str | Path | None = None,
    ) -> list[Bout]:
        """Extract bouts from a CSV file.

        CSV must have columns: wav_file, start_time_s, end_time_s.
        wav_file can be a stem (resolved via wav_dir) or full path.
        """
        csv_path = Path(csv_path)
        detections: dict[str, list[tuple[float, float]]] = {}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_file = row["wav_file"].strip()
                start = float(row["start_time_s"])
                end = float(row["end_time_s"])
                detections.setdefault(wav_file, []).append((start, end))

        all_bouts: list[Bout] = []
        for wav_file, times in detections.items():
            source = self._resolve_wav_path(wav_file, wav_dir)
            bouts = self._group_into_bouts(source, times)
            all_bouts.extend(bouts)

        return all_bouts

    def extract_from_detection_dir(
        self,
        detection_dir: str | Path,
        wav_dir: str | Path | None = None,
    ) -> list[Bout]:
        """Extract bouts from a detection output directory.

        Looks for detections_summary.csv first, then falls back to
        _saved_tracking.json, then individual detection JSONs.
        """
        detection_dir = Path(detection_dir)
        all_bouts: list[Bout] = []

        # Walk subdirectories (each represents one WAV file's detections)
        subdirs = sorted(
            p for p in detection_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
            and p.name not in ("rejected_detections", "noise_labeled_files",
                               "__pycache__")
        )

        for subdir in subdirs:
            wav_stem = subdir.name
            source = self._resolve_wav_path(wav_stem, wav_dir)
            times = self._load_detections_from_subdir(subdir)
            if times:
                bouts = self._group_into_bouts(source, times)
                all_bouts.extend(bouts)

        return all_bouts

    def extract_from_tracking_json(
        self,
        json_path: str | Path,
        source_file: str = "",
    ) -> list[Bout]:
        """Extract bouts from a _saved_tracking.json file.

        Filters out records where user_action == "deleted_by_user".
        """
        json_path = Path(json_path)
        with open(json_path) as f:
            records = json.load(f)

        if not records:
            return []

        # Filter out user-rejected detections
        valid_records = [
            r for r in records
            if r.get("user_action") != "deleted_by_user"
        ]
        if not valid_records:
            return []

        # Infer source_file from output_path if not provided
        if not source_file and valid_records:
            source_file = str(json_path.parent.name)

        times = [
            (r["start_time_s"], r["end_time_s"])
            for r in valid_records
        ]
        return self._group_into_bouts(source_file, times)

    def _build_wav_index(self, wav_dir: Path) -> dict[str, Path]:
        """Build a stem -> path index by recursively scanning wav_dir.

        Called lazily on first cache miss from _resolve_wav_path.
        """
        index: dict[str, Path] = {}
        for p in wav_dir.rglob("*.wav"):
            index[p.stem] = p
        logger.debug("Built WAV index: %d files from %s", len(index), wav_dir)
        return index

    def _resolve_wav_path(
        self,
        wav_file: str,
        wav_dir: str | Path | None,
    ) -> str:
        """Resolve a WAV file reference to a usable path string.

        Fast path: direct join (flat directory).
        Slow path: recursive index lookup (nested directories, lazy-built).
        """
        if wav_dir is not None:
            wav_dir = Path(wav_dir)
            # Fast path: direct flat lookup
            candidate = wav_dir / wav_file
            if candidate.suffix != ".wav":
                candidate = candidate.with_suffix(".wav")
            if candidate.exists():
                return str(candidate)

            # Slow path: build recursive index on first miss
            stem = Path(wav_file).stem
            if self._wav_index is None:
                self._wav_index = self._build_wav_index(wav_dir)
            if stem in self._wav_index:
                return str(self._wav_index[stem])

            # Fall back to the original flat path (may not exist)
            return str(candidate)
        return wav_file

    def _load_detections_from_subdir(
        self,
        subdir: Path,
    ) -> list[tuple[float, float]]:
        """Load detection times from a recording's detection subdirectory."""
        # Try detections_summary.csv first (most complete)
        csv_path = subdir / "detections_summary.csv"
        if csv_path.exists():
            return self._parse_summary_csv(csv_path)

        # Try _saved_tracking.json
        tracking_path = subdir / "_saved_tracking.json"
        if tracking_path.exists():
            return self._parse_tracking_json(tracking_path)

        # Fall back to individual detection JSONs
        return self._parse_individual_jsons(subdir)

    def _parse_summary_csv(self, csv_path: Path) -> list[tuple[float, float]]:
        """Parse detections from a detections_summary.csv file."""
        times: list[tuple[float, float]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = float(row["start_time_s"])
                end = float(row["end_time_s"])
                times.append((start, end))
        return times

    def _parse_tracking_json(self, json_path: Path) -> list[tuple[float, float]]:
        """Parse detections from a _saved_tracking.json file.

        Filters out records where user_action == "deleted_by_user" to prevent
        rejected detections from leaking into training data.
        """
        with open(json_path) as f:
            records = json.load(f)
        return [
            (r["start_time_s"], r["end_time_s"])
            for r in records
            if r.get("user_action") != "deleted_by_user"
        ]

    def _parse_individual_jsons(self, subdir: Path) -> list[tuple[float, float]]:
        """Parse detections from individual detection_*.json files."""
        times: list[tuple[float, float]] = []
        for json_path in sorted(subdir.glob("detection_*.json")):
            with open(json_path) as f:
                data = json.load(f)
            core = data.get("core_time", {})
            if "start_s" in core and "end_s" in core:
                times.append((core["start_s"], core["end_s"]))
        return times

    def _group_into_bouts(
        self,
        source_file: str,
        usv_times: Sequence[tuple[float, float]],
    ) -> list[Bout]:
        """Group USV times into bouts based on gap threshold.

        Algorithm:
        1. Sort USVs by start time
        2. Walk list, start new bout when gap > threshold
        3. Add context padding, clamp to [0, inf) (duration unknown here)
        4. Split bouts exceeding max duration
        5. Discard bouts shorter than min duration
        """
        if not usv_times:
            return []

        sorted_times = sorted(usv_times, key=lambda t: t[0])
        gap_threshold_s = self.config.bout_gap_threshold_ms / 1000.0
        padding_s = self.config.context_padding_ms / 1000.0

        # Step 1-2: Group into raw bouts
        raw_groups: list[list[tuple[float, float]]] = []
        current_group: list[tuple[float, float]] = [sorted_times[0]]

        for i in range(1, len(sorted_times)):
            prev_end = sorted_times[i - 1][1]
            curr_start = sorted_times[i][0]
            gap = curr_start - prev_end

            if gap > gap_threshold_s:
                raw_groups.append(current_group)
                current_group = [sorted_times[i]]
            else:
                current_group.append(sorted_times[i])

        raw_groups.append(current_group)

        # Step 3-5: Build Bout objects with padding, splitting, filtering
        bouts: list[Bout] = []
        for group in raw_groups:
            group_bouts = self._process_bout_group(
                source_file, group, padding_s,
            )
            bouts.extend(group_bouts)

        return bouts

    def _process_bout_group(
        self,
        source_file: str,
        usv_group: list[tuple[float, float]],
        padding_s: float,
    ) -> list[Bout]:
        """Process a single bout group: add padding, split if needed, filter."""
        max_duration_s = self.config.max_bout_duration_ms / 1000.0
        min_duration_s = self.config.min_bout_duration_ms / 1000.0

        bout_start = max(0.0, usv_group[0][0] - padding_s)
        bout_end = usv_group[-1][1] + padding_s
        duration = bout_end - bout_start

        if duration <= max_duration_s:
            if duration >= min_duration_s:
                return [Bout(
                    source_file=source_file,
                    start_time_s=bout_start,
                    end_time_s=bout_end,
                    usv_count=len(usv_group),
                    usv_times=list(usv_group),
                )]
            return []

        # Split oversized bouts
        return self._split_bout(
            source_file, usv_group, padding_s, max_duration_s, min_duration_s,
        )

    def _split_bout(
        self,
        source_file: str,
        usv_group: list[tuple[float, float]],
        padding_s: float,
        max_duration_s: float,
        min_duration_s: float,
    ) -> list[Bout]:
        """Split an oversized bout at the largest internal gap.

        If all gaps < 100ms, split at the midpoint USV instead.
        """
        if len(usv_group) <= 1:
            # Single USV that's somehow > max_duration (shouldn't happen with
            # reasonable padding, but handle gracefully)
            bout_start = max(0.0, usv_group[0][0] - padding_s)
            bout_end = usv_group[0][1] + padding_s
            if bout_end - bout_start >= min_duration_s:
                return [Bout(
                    source_file=source_file,
                    start_time_s=bout_start,
                    end_time_s=bout_end,
                    usv_count=1,
                    usv_times=list(usv_group),
                )]
            return []

        # Find largest internal gap
        gaps = []
        for i in range(len(usv_group) - 1):
            gap = usv_group[i + 1][0] - usv_group[i][1]
            gaps.append((gap, i))

        max_gap, max_gap_idx = max(gaps, key=lambda g: g[0])

        if max_gap >= 0.1:  # 100ms threshold
            split_idx = max_gap_idx + 1
        else:
            # Split at midpoint
            split_idx = len(usv_group) // 2

        left = usv_group[:split_idx]
        right = usv_group[split_idx:]

        result: list[Bout] = []
        for sub_group in (left, right):
            if not sub_group:
                continue
            result.extend(
                self._process_bout_group(source_file, sub_group, padding_s)
            )

        return result
