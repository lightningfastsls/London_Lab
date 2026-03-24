"""Build read-only ghost detections from saved tracker records."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from usv_spectrogram.app.core.detection_logic import DetectedUSV
from usv_spectrogram.app.core.saved_detection_tracker import SavedDetectionRecord


def build_saved_previous_detections(
    saved_records: Iterable[SavedDetectionRecord],
    current_detections: list[DetectedUSV],
    times: np.ndarray,
    matches_record: Callable[[float, float, SavedDetectionRecord], bool],
) -> list[DetectedUSV]:
    """Convert saved tracker records into read-only ghost detections.

    Deleted detections are intentionally excluded. They are review actions,
    not prior accepted detections, so showing them as gray ghost overlays
    re-introduces detections the user explicitly removed.
    """
    ghosts: list[DetectedUSV] = []

    for record in saved_records:
        if record.user_action == "deleted_by_user":
            continue

        is_current = any(
            det.save_state == "saved_current"
            and matches_record(det.start_time_s, det.end_time_s, record)
            for det in current_detections
        )
        if is_current:
            continue

        start_col = int(np.searchsorted(times, record.start_time_s))
        end_col = int(np.searchsorted(times, record.end_time_s))

        ghosts.append(
            DetectedUSV(
                start_time_s=record.start_time_s,
                end_time_s=record.end_time_s,
                start_col=start_col,
                end_col=end_col,
                max_probability=0.0,
                mean_probability=0.0,
                save_state="saved_previous",
                user_action=record.user_action,
            )
        )

    return ghosts
