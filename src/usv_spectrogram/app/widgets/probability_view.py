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
        self.high_threshold: float = 0.40
        self.low_threshold: float = 0.28
        self.detections: List[DetectedUSV] = []

        self.setMinimumHeight(200)
        self.setMinimumWidth(800)

    def set_data(
        self,
        times: np.ndarray,
        probabilities: np.ndarray,
        high_threshold: float,
        low_threshold: float,
        detections: List[DetectedUSV]
    ):
        """Set probability curve data.

        Args:
            times: Time values in seconds
            probabilities: Probability values in [0, 1]
            high_threshold: High threshold value
            low_threshold: Low threshold value
            detections: List of detected USV events
        """
        self.times = times
        self.probabilities = probabilities
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.detections = detections

        # Set fixed width based on time range
        if len(times) > 0:
            duration = times[-1] - times[0]
            pixels_per_second = 100
            width = max(800, int(duration * pixels_per_second))
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

        # Get drawing area
        width = self.width()
        height = self.height()
        margin = 40

        draw_width = width - 2 * margin
        draw_height = height - 2 * margin

        # Draw background
        painter.fillRect(0, 0, width, height, QColor(255, 255, 255))

        # Draw detection regions (shaded)
        for usv in self.detections:
            start_x = self._time_to_x(usv.start_time_s, margin, draw_width)
            end_x = self._time_to_x(usv.end_time_s, margin, draw_width)

            painter.fillRect(
                int(start_x),
                margin,
                int(end_x - start_x),
                draw_height,
                QColor(0, 255, 0, 30)
            )

        # Draw threshold lines
        high_y = self._prob_to_y(self.high_threshold, margin, draw_height)
        low_y = self._prob_to_y(self.low_threshold, margin, draw_height)

        pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(margin, int(high_y), width - margin, int(high_y))

        pen = QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(margin, int(low_y), width - margin, int(low_y))

        # Draw probability curve
        if len(self.times) > 1:
            path = QPainterPath()

            # Start path
            x0 = self._time_to_x(self.times[0], margin, draw_width)
            y0 = self._prob_to_y(self.probabilities[0], margin, draw_height)
            path.moveTo(x0, y0)

            # Add line segments
            for t, p in zip(self.times[1:], self.probabilities[1:]):
                x = self._time_to_x(t, margin, draw_width)
                y = self._prob_to_y(p, margin, draw_height)
                path.lineTo(x, y)

            pen = QPen(QColor(0, 0, 255), 2)
            painter.setPen(pen)
            painter.drawPath(path)

        # Draw axis labels
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)

        # Y-axis labels
        for p in [0.0, 0.5, 1.0]:
            y = self._prob_to_y(p, margin, draw_height)
            painter.drawText(5, int(y) + 5, f"{p:.1f}")

        # X-axis labels (time)
        t_min, t_max = self.times[0], self.times[-1]
        for t in np.linspace(t_min, t_max, 5):
            x = self._time_to_x(t, margin, draw_width)
            painter.drawText(int(x) - 20, height - 10, f"{t:.1f}s")

        painter.end()

    def _time_to_x(self, time_s: float, margin: int, draw_width: int) -> float:
        """Convert time to x-coordinate.

        Args:
            time_s: Time in seconds
            margin: Left margin in pixels
            draw_width: Drawing area width

        Returns:
            X-coordinate
        """
        t_min, t_max = self.times[0], self.times[-1]
        if t_max > t_min:
            fraction = (time_s - t_min) / (t_max - t_min)
            return margin + fraction * draw_width
        return margin

    def _prob_to_y(self, prob: float, margin: int, draw_height: int) -> float:
        """Convert probability to y-coordinate.

        Args:
            prob: Probability value in [0, 1]
            margin: Top margin in pixels
            draw_height: Drawing area height

        Returns:
            Y-coordinate (flipped, 0 is top)
        """
        # Flip y-axis (higher prob at top)
        return margin + draw_height * (1.0 - prob)


class ProbabilityView(QWidget):
    """Probability curve view with scrolling support."""

    # Signal emitted when horizontal scroll position changes
    scroll_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.canvas = ProbabilityCanvas()
        self.scroll_area.setWidget(self.canvas)

        layout.addWidget(self.scroll_area)

        self.setMinimumHeight(200)

        # Connect scroll bar to emit signal
        self.scroll_area.horizontalScrollBar().valueChanged.connect(
            lambda value: self.scroll_changed.emit(value)
        )

    def set_data(
        self,
        times: np.ndarray,
        probabilities: np.ndarray,
        high_threshold: float,
        low_threshold: float,
        detections: List[DetectedUSV]
    ):
        """Set probability curve data."""
        self.canvas.set_data(
            times,
            probabilities,
            high_threshold,
            low_threshold,
            detections
        )

    def set_scroll_position(self, value: int):
        """Set horizontal scroll position (for synchronization).

        Args:
            value: Scroll position value
        """
        # Block signals to prevent feedback loop
        self.scroll_area.horizontalScrollBar().blockSignals(True)
        self.scroll_area.horizontalScrollBar().setValue(value)
        self.scroll_area.horizontalScrollBar().blockSignals(False)
