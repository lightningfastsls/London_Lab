"""Tests for saved detection deduplication semantics in the app."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.detection_logic import DetectedUSV
from usv_spectrogram.app.core.saved_detection_tracker import SavedDetectionTracker


def _make_detection(
    start_time_s: float,
    end_time_s: float,
    *,
    user_action: str | None = None,
    user_adjusted: bool = False,
    save_state: str = "unsaved",
) -> DetectedUSV:
    return DetectedUSV(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        start_col=0,
        end_col=1,
        max_probability=0.9,
        mean_probability=0.8,
        user_adjusted=user_adjusted,
        save_state=save_state,
        user_action=user_action,
    )


class TestSavedDetectionTracker:
    def test_exact_saved_detection_is_recognized(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        saved = _make_detection(1.000, 1.100)
        tracker.mark_saved(saved, "saved.png")

        assert tracker.is_saved(_make_detection(1.000, 1.100)) is True

    def test_small_boundary_drift_within_tolerance_is_recognized(
        self, tmp_path: Path
    ) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        tracker.mark_saved(_make_detection(1.0000, 1.1000), "saved.png")

        assert tracker.is_saved(_make_detection(1.0008, 1.1007)) is True

    def test_partial_overlap_is_not_treated_as_saved(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        tracker.mark_saved(_make_detection(1.000, 1.100), "saved.png")

        assert tracker.is_saved(_make_detection(1.090, 1.140)) is False

    def test_deleted_record_does_not_hide_distinct_overlap(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        tracker.mark_saved(
            _make_detection(1.000, 1.100, user_action="deleted_by_user"),
            "deleted.png",
            user_action="deleted_by_user",
        )

        assert tracker.is_saved(_make_detection(1.090, 1.140)) is False
        assert tracker.is_saved(_make_detection(1.000, 1.100)) is True

    def test_unsaved_adjusted_detection_stays_unsaved(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        tracker.mark_saved(_make_detection(1.000, 1.100), "saved.png")

        adjusted = _make_detection(
            1.000,
            1.100,
            user_adjusted=True,
            save_state="unsaved",
        )
        assert tracker.is_saved(adjusted) is False

    def test_manual_detection_uses_own_save_state(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        tracker.mark_saved(_make_detection(1.000, 1.100), "saved.png")

        manual_unsaved = _make_detection(
            1.000,
            1.100,
            user_action="added_manually",
            save_state="unsaved",
        )
        manual_saved = _make_detection(
            1.000,
            1.100,
            user_action="added_manually",
            save_state="saved_current",
        )

        assert tracker.is_saved(manual_unsaved) is False
        assert tracker.is_saved(manual_saved) is True

    def test_loaded_saved_current_detection_counts_as_saved_without_tracker_record(
        self, tmp_path: Path
    ) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)

        loaded_saved = _make_detection(
            1.000,
            1.100,
            save_state="saved_current",
        )

        assert tracker.is_saved(loaded_saved) is True

    def test_loaded_adjusted_saved_current_detection_counts_as_saved_without_tracker_record(
        self, tmp_path: Path
    ) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)

        loaded_adjusted_saved = _make_detection(
            1.000,
            1.120,
            user_adjusted=True,
            save_state="saved_current",
        )

        assert tracker.is_saved(loaded_adjusted_saved) is True

    def test_clear_records_persists_empty_tracking_state(self, tmp_path: Path) -> None:
        tracker = SavedDetectionTracker("rec_001", tmp_path)
        tracker.mark_saved(_make_detection(1.000, 1.100), "saved.png")

        tracker.clear_records()

        reloaded = SavedDetectionTracker("rec_001", tmp_path)
        assert reloaded.get_saved_count() == 0
