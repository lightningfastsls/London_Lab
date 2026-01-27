"""Spectrogram visualization widget."""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtGui import QPainter, QPixmap, QImage, QPen, QColor
from PyQt6.QtCore import Qt, QRect, pyqtSignal

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

        # Convert to QPixmap (ensure contiguous array for PyQt6)
        height, width, channels = img.shape

        # Make sure array is contiguous and convert to bytes
        img_contiguous = np.ascontiguousarray(img)
        bytes_per_line = channels * width

        q_img = QImage(
            img_contiguous.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        self.pixmap = QPixmap.fromImage(q_img.copy())  # Copy to avoid data lifetime issues
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

        # Apply magma colormap
        magma = plt.cm.magma
        img_rgba = magma(spec_norm)  # Returns RGBA in [0, 1]

        # Convert to RGB uint8 (flip vertically for display)
        img_rgb = (img_rgba[:, :, :3] * 255).astype(np.uint8)
        img_rgb = np.flip(img_rgb, axis=0)

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

        self.canvas = SpectrogramCanvas()
        self.scroll_area.setWidget(self.canvas)

        layout.addWidget(self.scroll_area)

        self.setMinimumHeight(400)

        # Connect scroll bar to emit signal
        self.scroll_area.horizontalScrollBar().valueChanged.connect(
            lambda value: self.scroll_changed.emit(value)
        )

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

    def set_scroll_position(self, value: int):
        """Set horizontal scroll position (for synchronization).

        Args:
            value: Scroll position value
        """
        # Block signals to prevent feedback loop
        self.scroll_area.horizontalScrollBar().blockSignals(True)
        self.scroll_area.horizontalScrollBar().setValue(value)
        self.scroll_area.horizontalScrollBar().blockSignals(False)
