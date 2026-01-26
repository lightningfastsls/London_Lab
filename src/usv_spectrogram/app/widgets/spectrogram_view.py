"""Spectrogram visualization widget."""

from __future__ import annotations

from typing import Optional, List

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QPixmap, QImage, QPen, QColor
from PyQt6.QtCore import Qt, QRect

from ..core.detection_logic import DetectedUSV


class SpectrogramCanvas(QWidget):
    """Canvas for drawing spectrogram with detection overlays."""

    def __init__(self):
        super().__init__()
        self.pixmap: Optional[QPixmap] = None
        self.detections: List[DetectedUSV] = []
        self.times: Optional[np.ndarray] = None
        self.setMinimumHeight(400)

    def set_data(
        self,
        spectrogram_db: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray
    ):
        """Set spectrogram data and render to pixmap.

        Args:
            spectrogram_db: Spectrogram in dB, shape (freqs, times)
            times: Time values in seconds
            frequencies: Frequency values in Hz
        """
        self.times = times

        # Convert spectrogram to RGB image
        img = self._spectrogram_to_image(spectrogram_db)

        # Convert to QPixmap
        height, width, channels = img.shape
        bytes_per_line = channels * width
        q_img = QImage(
            img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        self.pixmap = QPixmap.fromImage(q_img)
        self.setFixedSize(self.pixmap.size())
        self.update()

    def set_detections(self, detections: List[DetectedUSV]):
        """Set detection results to overlay on spectrogram.

        Args:
            detections: List of detected USV events
        """
        self.detections = detections
        self.update()

    def _spectrogram_to_image(self, spec_db: np.ndarray) -> np.ndarray:
        """Convert spectrogram to RGB image using magma colormap.

        Args:
            spec_db: Spectrogram in dB, shape (freqs, times)

        Returns:
            RGB image, shape (height, width, 3), uint8
        """
        # Normalize to [0, 1]
        vmin, vmax = spec_db.min(), spec_db.max()
        if vmax > vmin:
            spec_norm = (spec_db - vmin) / (vmax - vmin)
        else:
            spec_norm = np.zeros_like(spec_db)

        # Apply colormap (simple grayscale for MVP, can add magma later)
        img = (spec_norm * 255).astype(np.uint8)

        # Convert to RGB (flip vertically for display)
        img = np.flip(img, axis=0)
        img_rgb = np.stack([img, img, img], axis=-1)

        return img_rgb

    def paintEvent(self, event):
        """Paint the spectrogram with detection overlays."""
        if self.pixmap is None:
            return

        painter = QPainter(self)

        # Draw spectrogram
        painter.drawPixmap(0, 0, self.pixmap)

        # Draw detection boundaries
        if self.detections and self.times is not None:
            for usv in self.detections:
                # Convert time to pixel coordinate
                start_x = self._time_to_pixel(usv.start_time_s)
                end_x = self._time_to_pixel(usv.end_time_s)

                # Draw start line (green)
                pen = QPen(QColor(0, 255, 0), 2)
                painter.setPen(pen)
                painter.drawLine(start_x, 0, start_x, self.height())

                # Draw end line (red)
                pen = QPen(QColor(255, 0, 0), 2)
                painter.setPen(pen)
                painter.drawLine(end_x, 0, end_x, self.height())

        painter.end()

    def _time_to_pixel(self, time_s: float) -> int:
        """Convert time in seconds to pixel x-coordinate.

        Args:
            time_s: Time in seconds

        Returns:
            Pixel x-coordinate
        """
        if self.times is None or len(self.times) == 0:
            return 0

        # Linear interpolation
        t_min, t_max = self.times[0], self.times[-1]
        if t_max > t_min:
            fraction = (time_s - t_min) / (t_max - t_min)
            return int(fraction * self.width())

        return 0


class SpectrogramView(QWidget):
    """Spectrogram view with scrolling support."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = SpectrogramCanvas()
        layout.addWidget(self.canvas)

        self.setMinimumHeight(400)

    def set_data(
        self,
        spectrogram_db: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray
    ):
        """Set spectrogram data."""
        self.canvas.set_data(spectrogram_db, times, frequencies)

    def set_detections(self, detections: List[DetectedUSV]):
        """Set detection overlays."""
        self.canvas.set_detections(detections)
