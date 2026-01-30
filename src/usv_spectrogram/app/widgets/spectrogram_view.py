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
        self.setMinimumHeight(150)  # Reduced for compact layout

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

        Uses MAD-based dynamic range normalization for better contrast,
        matching the labeling app's visualization.

        Args:
            spec_db: Spectrogram in dB, shape (freqs, times)

        Returns:
            RGB image, shape (height, width, 3), uint8
        """
        # Use MAD (median absolute deviation) based normalization
        # This gives much better contrast than min/max
        median = np.median(spec_db)
        mad = np.median(np.abs(spec_db - median))

        # MAD scale factors from ExtractionConfig
        mad_vmin_scale = 2.0
        mad_vmax_scale = 4.0

        vmin = median - mad_vmin_scale * mad
        vmax = median + mad_vmax_scale * mad

        # Normalize to [0, 1] and clip
        if vmax > vmin:
            spec_norm = (spec_db - vmin) / (vmax - vmin)
            spec_norm = np.clip(spec_norm, 0, 1)
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

                # Draw start line (bright green, solid)
                pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.drawLine(start_x, 0, start_x, self.height())

                # Draw end line (cyan for visibility on magma, dashed)
                pen = QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine)
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

    # Signal emitted when horizontal scroll position changes (normalized 0.0-1.0)
    scroll_changed = pyqtSignal(float)

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

        self.setMinimumHeight(200)  # Reduced for compact layout

        # Connect scroll bar to emit normalized position
        self.scroll_area.horizontalScrollBar().valueChanged.connect(
            self._on_scroll_changed
        )

    def _on_scroll_changed(self, value: int):
        """Emit normalized scroll position (0.0 to 1.0)."""
        scrollbar = self.scroll_area.horizontalScrollBar()
        scroll_range = scrollbar.maximum() - scrollbar.minimum()
        if scroll_range > 0:
            normalized_pos = (value - scrollbar.minimum()) / scroll_range
        else:
            normalized_pos = 0.0
        self.scroll_changed.emit(normalized_pos)

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

    def get_canvas_width(self) -> int:
        """Get the width of the spectrogram canvas in pixels."""
        if self.canvas.pixmap is not None:
            return self.canvas.pixmap.width()
        return 800  # Default fallback

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

        # No need to block signals - we only have one-way connection now
        scrollbar.setValue(target_value)
