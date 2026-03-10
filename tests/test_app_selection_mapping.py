"""Tests for mapping displayed app selections back to editable detections."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.app.core.detection_logic import DetectedUSV
from usv_spectrogram.app.core.selection_mapping import resolve_current_detection_selection


def _make_detection(start: float, end: float, *, save_state: str = "unsaved") -> DetectedUSV:
    return DetectedUSV(
        start_time_s=start,
        end_time_s=end,
        start_col=0,
        end_col=1,
        max_probability=0.9,
        mean_probability=0.8,
        save_state=save_state,
    )


class TestResolveCurrentDetectionSelection:
    def test_returns_current_detection_index_for_current_entry(self) -> None:
        current = [_make_detection(1.0, 1.1), _make_detection(2.0, 2.1)]

        idx, detection = resolve_current_detection_selection(1, current, current)

        assert idx == 1
        assert detection is current[1]

    def test_returns_ghost_detection_without_current_index(self) -> None:
        current = [_make_detection(1.0, 1.1)]
        ghost = _make_detection(3.0, 3.1, save_state="saved_previous")
        displayed = current + [ghost]

        idx, detection = resolve_current_detection_selection(1, displayed, current)

        assert idx is None
        assert detection is ghost

    def test_out_of_range_selection_returns_none(self) -> None:
        current = [_make_detection(1.0, 1.1)]

        idx, detection = resolve_current_detection_selection(5, current, current)

        assert idx is None
        assert detection is None
