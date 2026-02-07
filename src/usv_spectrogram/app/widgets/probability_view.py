"""Probability curve visualization widget."""

from __future__ import annotations

from typing import Optional, List

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QPointF, pyqtSignal

from ..core.detection_logic import DetectedUSV


class ProbabilityCanvas(QWidget):
    """Canvas for drawing probability curve with thresholds."""

    def __init__(self):
        super().__init__()
        self.times: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.column_indices: Optional[np.ndarray] = None  # NEW: for pixel-perfect alignment
        self.total_columns: int = 0  # NEW: total width of spectrogram
        self.high_threshold: float = 0.40
        self.low_threshold: float = 0.28
        self.detections: List[DetectedUSV] = []
        self._scroll_offset: int = 0  # Horizontal scroll offset for sticky Y-axis labels

        self.setMinimumHeight(200)
        self.setMinimumWidth(800)

    def set_scroll_offset(self, offset: int):
        """Update scroll offset for Y-axis label positioning.

        Args:
            offset: Horizontal scroll position in pixels
        """
        self._scroll_offset = offset
        self.update()

    def set_data(
        self,
        times: np.ndarray,
        probabilities: np.ndarray,
        high_threshold: float,
        low_threshold: float,
        detections: List[DetectedUSV],
        column_indices: np.ndarray | None = None,  # NEW: for alignment
        target_width: int | None = None
    ):
        """Set probability curve data.

        Args:
            times: Time values in seconds
            probabilities: Probability values in [0, 1]
            high_threshold: High threshold value
            low_threshold: Low threshold value
            detections: List of detected USV events
            column_indices: Column indices for each probability (for pixel-perfect alignment)
            target_width: Target width in pixels (total spectrogram width)
        """
        self.times = times
        self.probabilities = probabilities
        self.column_indices = column_indices
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.detections = detections

        # Set width to match spectrogram
        if target_width is not None:
            self.total_columns = target_width
            self.setFixedWidth(target_width)
        elif len(times) > 0:
            duration = times[-1] - times[0]
            pixels_per_second = 100
            width = max(800, int(duration * pixels_per_second))
            self.total_columns = width
            self.setFixedWidth(width)

        self.update()

    def paintEvent(self, event):
        """Paint the probability curve."""
        if self.times is None or self.probabilities is None:
            return

        if len(self.times) == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get drawing area - use total_columns for horizontal to match spectrogram pixel-for-pixel
        width = self.total_columns if self.total_columns > 0 else self.width()
        height = self.height()
        margin_left = 0  # No left margin for horizontal alignment with spectrogram
        margin_right = 0  # No right margin
        margin_top = 30  # Top margin for probability labels
        margin_bottom = 25  # Bottom margin for time labels

        draw_width = width  # Full width, no horizontal margins
        draw_height = height - margin_top - margin_bottom

        # Draw background
        painter.fillRect(0, 0, width, height, QColor(255, 255, 255))

        # Draw detection regions (shaded) - color based on save_state
        for usv in self.detections:
            start_x = self._col_to_x(usv.start_col, margin_left, draw_width)
            end_x = self._col_to_x(usv.end_col, margin_left, draw_width)

            if usv.save_state == "saved_current":
                color = QColor(0, 100, 255, 40)  # Blue
            elif usv.save_state == "saved_previous":
                color = QColor(128, 128, 128, 25)  # Gray
            else:
                color = QColor(0, 255, 0, 30)  # Green (unsaved)

            painter.fillRect(
                int(start_x),
                margin_top,
                int(end_x - start_x),
                draw_height,
                color
            )

        # Draw threshold lines (use actual widget width for full-width lines)
        actual_width = self.width()
        high_y = self._prob_to_y(self.high_threshold, margin_top, draw_height)
        low_y = self._prob_to_y(self.low_threshold, margin_top, draw_height)

        pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(0, int(high_y), actual_width, int(high_y))

        pen = QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(0, int(low_y), actual_width, int(low_y))

        # Draw probability curve
        if len(self.probabilities) > 1:
            path = QPainterPath()

            # Use column indices for pixel-perfect alignment if available
            if self.column_indices is not None and len(self.column_indices) > 0:
                # Start path
                x0 = self._col_to_x(self.column_indices[0], margin_left, draw_width)
                y0 = self._prob_to_y(self.probabilities[0], margin_top, draw_height)
                path.moveTo(x0, y0)

                # Add line segments
                for col, p in zip(self.column_indices[1:], self.probabilities[1:]):
                    x = self._col_to_x(col, margin_left, draw_width)
                    y = self._prob_to_y(p, margin_top, draw_height)
                    path.lineTo(x, y)
            else:
                # Fallback to time-based mapping
                x0 = self._time_to_x(self.times[0], margin_left, draw_width)
                y0 = self._prob_to_y(self.probabilities[0], margin_top, draw_height)
                path.moveTo(x0, y0)

                for t, p in zip(self.times[1:], self.probabilities[1:]):
                    x = self._time_to_x(t, margin_left, draw_width)
                    y = self._prob_to_y(p, margin_top, draw_height)
                    path.lineTo(x, y)

            pen = QPen(QColor(0, 0, 255), 2)
            painter.setPen(pen)
            painter.drawPath(path)

        # Draw axis labels
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)

        # Y-axis labels (sticky - adjust for scroll position)
        for p in [0.0, 0.5, 1.0]:
            y = self._prob_to_y(p, margin_top, draw_height)
            painter.drawText(self._scroll_offset + 5, int(y) + 5, f"{p:.1f}")

        # X-axis labels (time) - only show at edges to avoid clutter
        t_min, t_max = self.times[0], self.times[-1]
        painter.drawText(margin_left + 5, height - 5, f"{t_min:.1f}s")
        x_max = self._time_to_x(t_max, margin_left, draw_width)
        painter.drawText(int(x_max) - 40, height - 5, f"{t_max:.1f}s")

        painter.end()

    def _time_to_x(self, time_s: float, margin_left: int, draw_width: int) -> float:
        """Convert time to x-coordinate.

        For detections (using times), map proportionally across full width.

        Args:
            time_s: Time in seconds
            margin_left: Left margin in pixels
            draw_width: Drawing area width

        Returns:
            X-coordinate
        """
        # Map time to column index first, then to x-coordinate
        # This ensures alignment with spectrogram
        if self.times is not None and len(self.times) > 1:
            t_min, t_max = self.times[0], self.times[-1]
            if t_max > t_min and self.total_columns > 0:
                # Estimate column index for this time
                fraction = (time_s - t_min) / (t_max - t_min)
                estimated_col = fraction * self.total_columns
                return margin_left + (estimated_col / self.total_columns) * draw_width
        return margin_left

    def _col_to_x(self, col_idx: int, margin_left: int, draw_width: int) -> float:
        """Convert column index to x-coordinate (pixel-perfect mapping).

        Args:
            col_idx: Column index in spectrogram
            margin_left: Left margin in pixels
            draw_width: Drawing area width

        Returns:
            X-coordinate
        """
        if self.total_columns > 0:
            return margin_left + (col_idx / self.total_columns) * draw_width
        return margin_left

    def _prob_to_y(self, prob: float, margin_top: int, draw_height: int) -> float:
        """Convert probability to y-coordinate.

        Args:
            prob: Probability value in [0, 1]
            margin_top: Top margin in pixels
            draw_height: Drawing area height

        Returns:
            Y-coordinate (flipped, 0 is top)
        """
        # Flip y-axis (higher prob at top)
        return margin_top + draw_height * (1.0 - prob)


class ProbabilityView(QWidget):
    """Probability curve view with scrolling support."""

    # Signal emitted when horizontal scroll position changes (normalized 0.0-1.0)
    scroll_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area (scrollbar hidden - controlled by spectrogram view)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Hidden - one scrollbar controls both
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.canvas = ProbabilityCanvas()
        self.scroll_area.setWidget(self.canvas)

        layout.addWidget(self.scroll_area)

        self.setMinimumHeight(200)

        # Connect scroll to update Y-axis label position
        self.scroll_area.horizontalScrollBar().valueChanged.connect(
            self.canvas.set_scroll_offset
        )

    def set_data(
        self,
        times: np.ndarray,
        probabilities: np.ndarray,
        high_threshold: float,
        low_threshold: float,
        detections: List[DetectedUSV],
        column_indices: np.ndarray | None = None,
        target_width: int | None = None
    ):
        """Set probability curve data."""
        self.canvas.set_data(
            times,
            probabilities,
            high_threshold,
            low_threshold,
            detections,
            column_indices,
            target_width
        )

    def set_scroll_position(self, normalized_pos: float):
        """Set horizontal scroll position from normalized position (0.0-1.0).

        Args:
            normalized_pos: Normalized scroll position (0.0 = start, 1.0 = end)
        """
        scrollbar = self.scroll_area.horizontalScrollBar()
        scroll_range = scrollbar.maximum() - scrollbar.minimum()

        if scroll_range <= 0:
            return

        target_value = scrollbar.minimum() + int(normalized_pos * scroll_range)

        # Don't block signals - let Qt handle the scroll event properly
        scrollbar.setValue(target_value)
