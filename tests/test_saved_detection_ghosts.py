"""Tests for ghost overlay construction from saved detection records."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.detection_logic import DetectedUSV
from usv_spectrogram.app.core.saved_detection_ghosts import (
    build_saved_previous_detections,
)
from usv_spectrogram.app.core.saved_detection_tracker import (
    SavedDetectionRecord,
    SavedDetectionTracker,
)


def _make_detection(
    start: float,
    end: float,
    *,
    save_state: str = "unsaved",
    user_action: str | None = None,
) -> DetectedUSV:
    return DetectedUSV(
        start_time_s=start,
        end_time_s=end,
        start_col=0,
        end_col=1,
        max_probability=0.9,
        mean_probability=0.8,
        save_state=save_state,
        user_action=user_action,
    )


def _make_record(
    start: float,
    end: float,
    *,
    user_action: str | None = None,
) -> SavedDetectionRecord:
    return SavedDetectionRecord(
        start_time_s=start,
        end_time_s=end,
        save_timestamp="2026-03-07T12:00:00",
        output_path="saved.png",
        user_action=user_action,
    )


class TestBuildSavedPreviousDetections:
    def test_deleted_detection_record_does_not_become_ghost(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        times = np.linspace(0.0, 5.0, 501)

        ghosts = build_saved_previous_detections(
            [_make_record(1.0, 1.1, user_action="deleted_by_user")],
            current_detections=[],
            times=times,
            matches_record=tracker.matches_record,
        )

        assert ghosts == []

    def test_saved_current_detection_suppresses_duplicate_ghost(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        times = np.linspace(0.0, 5.0, 501)
        current = [_make_detection(1.0, 1.1, save_state="saved_current")]

        ghosts = build_saved_previous_detections(
            [_make_record(1.0, 1.1)],
            current_detections=current,
            times=times,
            matches_record=tracker.matches_record,
        )

        assert ghosts == []

    def test_unmatched_saved_record_becomes_saved_previous_ghost(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        times = np.linspace(0.0, 5.0, 501)

        ghosts = build_saved_previous_detections(
            [_make_record(2.0, 2.2)],
            current_detections=[],
            times=times,
            matches_record=tracker.matches_record,
        )

        assert len(ghosts) == 1
        assert ghosts[0].start_time_s == 2.0
        assert ghosts[0].end_time_s == 2.2
        assert ghosts[0].start_col == 200
        assert ghosts[0].end_col == 220
        assert ghosts[0].save_state == "saved_previous"
