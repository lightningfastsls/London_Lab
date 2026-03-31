"""Sonic-range (0-30 kHz) spectrogram visualization widget.

Simplified spectrogram view for the audible/sonic frequency range.
No detection overlays or mouse interaction — display-only.
Uses MAD-based normalization + magma colormap, matching the USV view.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PyQt6.QtGui import QPainter, QPixmap, QImage, QWheelEvent
from PyQt6.QtCore import Qt, pyqtSignal


class SonicCanvas(QWidget):
    """Canvas for drawing sonic-range spectrogram."""

    def __init__(self):
        super().__init__()
        self.pixmap: Optional[QPixmap] = None
        self.setMinimumHeight(80)

    def set_data(
        self,
        spectrogram_db: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray,
        target_width: int | None = None,
    ):
        """Set sonic spectrogram data and render to pixmap.

        Height is native resolution (1 pixel per freq bin) for crispness.
        Width is scaled to match the USV canvas for time-axis alignment.

        Args:
            spectrogram_db: Spectrogram in dB, shape (freqs, times)
            times: Time values in seconds
            frequencies: Frequency values in Hz
            target_width: Target pixel width to match USV canvas
        """
        img = self._spectrogram_to_image(spectrogram_db)

        height, width, channels = img.shape
        img_contiguous = np.ascontiguousarray(img)
        bytes_per_line = channels * width

        q_img = QImage(
            img_contiguous.tobytes(),
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Scale vertically by 3x using nearest-neighbor to keep crispness.
        # Sonic range has few freq bins (~102 for 0-30kHz) so native height
        # is too small to see detail.
        scaled_height = height * 3
        q_img = q_img.scaled(
            width,
            scaled_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,  # nearest-neighbor
        )

        self.pixmap = QPixmap.fromImage(q_img)
        self.setFixedSize(self.pixmap.size())
        self.update()

    def clear(self):
        """Clear the canvas."""
        self.pixmap = None
        self.update()

    def _spectrogram_to_image(self, spec_db: np.ndarray) -> np.ndarray:
        """Convert spectrogram to RGB image using MAD normalization + magma.

        Args:
            spec_db: Spectrogram in dB, shape (freqs, times)

        Returns:
            RGB image, shape (height, width, 3), uint8
        """
        median = np.median(spec_db)
        mad = np.median(np.abs(spec_db - median))

        vmin = median - 2.0 * mad
        vmax = median + 4.0 * mad

        if vmax > vmin:
            spec_norm = (spec_db - vmin) / (vmax - vmin)
            spec_norm = np.clip(spec_norm, 0, 1)
        else:
            spec_norm = np.zeros_like(spec_db)

        magma = plt.cm.magma
        img_rgba = magma(spec_norm)

        img_rgb = (img_rgba[:, :, :3] * 255).astype(np.uint8)
        img_rgb = np.flip(img_rgb, axis=0)

        return img_rgb

    def paintEvent(self, event):
        """Paint the sonic spectrogram."""
        if self.pixmap is None:
            return
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.end()


class SonicSpectrogramView(QWidget):
    """Sonic spectrogram view with scroll synchronization."""

    scroll_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._syncing = False  # Guard against infinite scroll loops

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        # Scrollbar hidden — controlled by spectrogram view
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Emit scroll_changed when scrollbar moves (for reverse sync)
        self.scroll_area.horizontalScrollBar().valueChanged.connect(
            self._on_scroll_changed
        )

        self.canvas = SonicCanvas()
        self.scroll_area.setWidget(self.canvas)

        layout.addWidget(self.scroll_area)

        self.setMinimumHeight(120)

    def set_data(
        self,
        spectrogram_db: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray,
        target_width: int | None = None,
    ):
        """Set sonic spectrogram data."""
        self.canvas.set_data(spectrogram_db, times, frequencies, target_width)

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

        self._syncing = True
        scrollbar.setValue(target_value)
        self._syncing = False

    def clear(self):
        """Clear the sonic view."""
        self.canvas.clear()

    def _on_scroll_changed(self, value: int):
        """Emit normalized scroll position when scrollbar moves."""
        if self._syncing:
            return
        scrollbar = self.scroll_area.horizontalScrollBar()
        scroll_range = scrollbar.maximum() - scrollbar.minimum()
        if scroll_range <= 0:
            return
        normalized_pos = (value - scrollbar.minimum()) / scroll_range
        self.scroll_changed.emit(normalized_pos)

    def wheelEvent(self, event: QWheelEvent):
        """Redirect mouse wheel to horizontal scrollbar."""
        scrollbar = self.scroll_area.horizontalScrollBar()
        delta = event.angleDelta().y()
        scrollbar.setValue(scrollbar.value() - delta)
        event.accept()
