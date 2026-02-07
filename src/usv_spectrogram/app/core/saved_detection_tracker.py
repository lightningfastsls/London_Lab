"""Track saved detections to avoid duplicates."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

from usv_spectrogram.app.core.detection_logic import DetectedUSV


@dataclass
class SavedDetectionRecord:
    """Record of a saved detection for duplicate checking."""
    start_time_s: float
    end_time_s: float
    save_timestamp: str  # ISO format
    output_path: str
    threshold_preset: str | None = None
    threshold_high: float | None = None
    threshold_low: float | None = None
    session_id: str | None = None
    user_action: str | None = None  # None (CNN), "added_manually", or "deleted_by_user"


class SavedDetectionTracker:
    """Track saved detections by time range (without context) to avoid duplicates."""

    def __init__(self, wav_filename: str, output_dir: Path):
        """Initialize tracker for a specific WAV file.

        Args:
            wav_filename: Name of WAV file (without extension)
            output_dir: Base output directory for all saved detections
        """
        self.wav_filename = wav_filename
        self.output_dir = output_dir
        self.saved_detections: List[SavedDetectionRecord] = []
        self._load_tracking_file()

    def _get_tracking_file_path(self) -> Path:
        """Get path to tracking JSON file."""
        return self.output_dir / self.wav_filename / "_saved_tracking.json"

    def _load_tracking_file(self):
        """Load saved detection records from JSON."""
        tracking_file = self._get_tracking_file_path()

        if not tracking_file.exists():
            return

        try:
            with open(tracking_file, 'r') as f:
                data = json.load(f)

            self.saved_detections = []
            for record in data:
                # Handle old tracking files missing new optional fields
                filtered = {
                    k: v for k, v in record.items()
                    if k in SavedDetectionRecord.__dataclass_fields__
                }
                self.saved_detections.append(SavedDetectionRecord(**filtered))
        except Exception as e:
            print(f"Warning: Could not load tracking file {tracking_file}: {e}")
            self.saved_detections = []

    def _save_tracking_file(self):
        """Persist tracking records to JSON."""
        tracking_file = self._get_tracking_file_path()
        tracking_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = [asdict(record) for record in self.saved_detections]
            with open(tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving tracking file {tracking_file}: {e}")

    def is_saved(self, detection: DetectedUSV) -> bool:
        """Check if detection already saved (by core time range overlap).

        Args:
            detection: Detection to check

        Returns:
            True if this detection overlaps with any saved detection
        """
        for record in self.saved_detections:
            if self._time_ranges_overlap(
                detection.start_time_s, detection.end_time_s,
                record.start_time_s, record.end_time_s
            ):
                return True
        return False

    def mark_saved(
        self,
        detection: DetectedUSV,
        output_path: str,
        threshold_preset: str | None = None,
        threshold_high: float | None = None,
        threshold_low: float | None = None,
        session_id: str | None = None,
        user_action: str | None = None
    ):
        """Mark detection as saved.

        Args:
            detection: Detection that was saved
            output_path: Path where detection was saved
            threshold_preset: Name of threshold preset used (if any)
            threshold_high: High threshold value at time of save
            threshold_low: Low threshold value at time of save
            session_id: Session UUID for tracking labeling sessions
            user_action: User action type (None, "added_manually", "deleted_by_user")
        """
        record = SavedDetectionRecord(
            start_time_s=detection.start_time_s,
            end_time_s=detection.end_time_s,
            save_timestamp=datetime.now().isoformat(),
            output_path=output_path,
            threshold_preset=threshold_preset,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            session_id=session_id,
            user_action=user_action
        )
        self.saved_detections.append(record)
        self._save_tracking_file()

    def get_unsaved_detections(self, all_detections: List[DetectedUSV]) -> List[DetectedUSV]:
        """Filter to only unsaved detections.

        Args:
            all_detections: List of all detections

        Returns:
            List of detections that have not been saved
        """
        return [d for d in all_detections if not self.is_saved(d)]

    def _time_ranges_overlap(self, start1: float, end1: float,
                            start2: float, end2: float) -> bool:
        """Check if two time ranges overlap.

        Uses core detection time (not including context) for overlap checking.

        Args:
            start1, end1: First time range
            start2, end2: Second time range

        Returns:
            True if ranges overlap
        """
        return not (end1 <= start2 or end2 <= start1)

    def get_saved_count(self) -> int:
        """Get number of saved detections."""
        return len(self.saved_detections)
