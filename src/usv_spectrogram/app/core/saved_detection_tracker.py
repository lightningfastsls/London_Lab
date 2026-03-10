"""Track saved detections to avoid duplicates."""

# VAULT: saved-previous ghost detections current editable and saved-current form three aligned detection state tiers in the app
# Run `/kcheck` before modifying this file.

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

from usv_spectrogram.app.core.detection_logic import DetectedUSV


_MATCH_TOLERANCE_S = 0.001


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
        """Check if detection already saved.

        Detections explicitly marked `save_state="saved_current"` are treated
        as saved even if tracker records are unavailable, which matters when
        labels are reloaded from JSON without reconstructing historical tracker
        state first.

        For manual detections (user_action="added_manually"), checks the
        detection's own save_state since they are new by definition and
        should not be matched against tracker records by time overlap.

        For user-adjusted detections (user_adjusted=True, save_state="unsaved"),
        returns False since they need re-saving with new boundaries.

        For CNN detections, checks tracker records by time range overlap.

        Args:
            detection: Detection to check

        Returns:
            True if this detection is already saved
        """
        if detection.save_state == "saved_current":
            return True

        # Manual detections are new — only trust their own save_state
        if detection.user_action == "added_manually":
            return detection.save_state != "unsaved"

        # Boundary-adjusted detections that haven't been re-saved need saving
        if detection.user_adjusted and detection.save_state == "unsaved":
            return False

        # CNN detections: check tracker by boundary identity
        for record in self.saved_detections:
            if self.matches_record(
                detection.start_time_s,
                detection.end_time_s,
                record,
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

    def matches_record(
        self,
        start_time_s: float,
        end_time_s: float,
        record: SavedDetectionRecord,
    ) -> bool:
        """Return True when boundaries match an existing saved record.

        Uses core detection time (not including context) and requires both
        boundaries to match within a small tolerance. Overlap alone is not
        enough, because adjacent or partially overlapping detections may be
        distinct user-reviewed events.

        Args:
            start_time_s: Candidate start time
            end_time_s: Candidate end time
            record: Previously saved detection record

        Returns:
            True if both boundaries match within tolerance
        """
        return (
            abs(start_time_s - record.start_time_s) <= _MATCH_TOLERANCE_S
            and abs(end_time_s - record.end_time_s) <= _MATCH_TOLERANCE_S
        )

    def get_saved_count(self) -> int:
        """Get number of saved detections."""
        return len(self.saved_detections)

    def clear_non_deleted_records(self) -> None:
        """Remove accepted/manual saved records while preserving deletions."""
        self.saved_detections = [
            record for record in self.saved_detections
            if record.user_action == "deleted_by_user"
        ]
        self._save_tracking_file()

    def remove_records_matching(
        self,
        start_time_s: float,
        end_time_s: float,
        *,
        include_deleted: bool = False,
    ) -> List[SavedDetectionRecord]:
        """Remove and return records matching the given boundaries.

        Args:
            start_time_s: Start time to match
            end_time_s: End time to match
            include_deleted: If False, keep deletion-history records intact

        Returns:
            Removed records in their original order
        """
        removed: List[SavedDetectionRecord] = []
        kept: List[SavedDetectionRecord] = []

        for record in self.saved_detections:
            is_deleted = record.user_action == "deleted_by_user"
            if (not include_deleted and is_deleted) or not self.matches_record(
                start_time_s,
                end_time_s,
                record,
            ):
                kept.append(record)
                continue
            removed.append(record)

        if removed:
            self.saved_detections = kept
            self._save_tracking_file()

        return removed

    def clear_records(self) -> None:
        """Remove all tracked detections and persist the cleared state."""
        self.saved_detections.clear()
        self._save_tracking_file()
