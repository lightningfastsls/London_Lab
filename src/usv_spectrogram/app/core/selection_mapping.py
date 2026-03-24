"""Helpers for mapping displayed app selections to editable detections."""

from __future__ import annotations

from usv_spectrogram.app.core.detection_logic import DetectedUSV


def resolve_current_detection_selection(
    selected_idx: int | None,
    displayed_detections: list[DetectedUSV],
    current_detections: list[DetectedUSV],
) -> tuple[int | None, DetectedUSV | None]:
    """Map a canvas selection index back to an editable current detection."""
    if selected_idx is None or not (0 <= selected_idx < len(displayed_detections)):
        return None, None

    selected_detection = displayed_detections[selected_idx]
    if selected_detection.save_state == "saved_previous":
        return None, selected_detection

    try:
        current_idx = current_detections.index(selected_detection)
    except ValueError:
        return None, selected_detection

    return current_idx, selected_detection
