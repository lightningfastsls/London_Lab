"""Spectrogram visualization widget."""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtGui import QPainter, QPixmap, QImage, QPen, QColor, QMouseEvent, QKeyEvent
from PyQt6.QtCore import Qt, QRect, pyqtSignal

from ..core.detection_logic import DetectedUSV


class SpectrogramCanvas(QWidget):
    """Canvas for drawing spectrogram with detection overlays."""

    # Signal emitted when user adjusts detection boundary
    boundary_adjusted = pyqtSignal(object, float, float)  # (detection, new_start, new_end)

    # Signal emitted when user creates new detection (right-click-drag)
    detection_created = pyqtSignal(float, float)  # (start_time_s, end_time_s)

    def __init__(self):
        super().__init__()
        self.pixmap: Optional[QPixmap] = None
        self.detections: List[DetectedUSV] = []
        self.times: Optional[np.ndarray] = None
        self.total_columns: int = 0  # Total spectrogram columns for pixel-perfect alignment
        self.setMinimumHeight(150)  # Reduced for compact layout

        # Enable mouse tracking for hover cursor feedback
        self.setMouseTracking(True)

        # Enable keyboard focus for Escape key handling
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Detection boundary adjustment state
        self._selected_detection: Optional[DetectedUSV] = None
        self._selected_detection_idx: Optional[int] = None  # Track by index, not reference
        self._dragging_edge: Optional[str] = None  # 'start' or 'end'
        self._original_time: Optional[float] = None  # For cancel/undo
        self._is_dragging: bool = False

        # Detection creation state (right-click-drag)
        self._creating_detection: bool = False
        self._creation_start_x: int = 0
        self._creation_current_x: int = 0

    def set_data(
        self,
        spectrogram_db: np.ndarray,
        times: np.ndarray,
        frequencies: np.ndarray,
        total_columns: int = 0
    ):
        """Set spectrogram data and render to pixmap.

        Args:
            spectrogram_db: Spectrogram in dB, shape (freqs, times)
            times: Time values in seconds
            frequencies: Frequency values in Hz
            total_columns: Total spectrogram columns for coordinate mapping
        """
        self.times = times
        self.total_columns = total_columns if total_columns > 0 else spectrogram_db.shape[1]

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

        # If currently dragging, update the selected detection reference
        # to point to the updated object at the same index
        if self._is_dragging and self._selected_detection_idx is not None:
            if 0 <= self._selected_detection_idx < len(self.detections):
                self._selected_detection = self.detections[self._selected_detection_idx]

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
        if self.detections and self.total_columns > 0:
            for idx, usv in enumerate(self.detections):
                # Skip selected detection - it will be drawn differently below
                if self._selected_detection_idx is not None and idx == self._selected_detection_idx:
                    continue

                # Skip ghost (previously saved) detections - too cluttered
                if usv.save_state == "saved_previous":
                    continue

                # Convert column index to pixel coordinate (pixel-perfect alignment)
                start_x = self._col_to_pixel(usv.start_col)
                end_x = self._col_to_pixel(usv.end_col)

                # Draw start line (bright green, solid)
                pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.drawLine(start_x, 0, start_x, self.height())

                # Draw end line (cyan for visibility on magma, dashed)
                pen = QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(end_x, 0, end_x, self.height())

        # Draw selected detection with highlights
        if self._selected_detection_idx is not None and self.total_columns > 0:
            # Bounds check before accessing by index
            if 0 <= self._selected_detection_idx < len(self.detections):
                selected_usv = self.detections[self._selected_detection_idx]
                start_x = self._col_to_pixel(selected_usv.start_col)
                end_x = self._col_to_pixel(selected_usv.end_col)

                # Draw the boundary being adjusted in yellow (highlighted)
                if self._dragging_edge == 'start':
                    painter.setPen(QPen(QColor(255, 255, 0), 3, Qt.PenStyle.SolidLine))
                    painter.drawLine(start_x, 0, start_x, self.height())
                    # Draw the other boundary normally
                    painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine))
                    painter.drawLine(end_x, 0, end_x, self.height())
                else:  # dragging end
                    # Draw the other boundary normally
                    painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
                    painter.drawLine(start_x, 0, start_x, self.height())
                    # Draw the boundary being adjusted in yellow (highlighted)
                    painter.setPen(QPen(QColor(255, 255, 0), 3, Qt.PenStyle.DashLine))
                    painter.drawLine(end_x, 0, end_x, self.height())

        # Draw creation preview box (right-click-drag)
        if self._creating_detection:
            start_x = min(self._creation_start_x, self._creation_current_x)
            end_x = max(self._creation_start_x, self._creation_current_x)

            # Draw semi-transparent yellow rectangle
            painter.fillRect(start_x, 0, end_x - start_x, self.height(),
                           QColor(255, 255, 0, 50))

            # Draw boundary lines
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.SolidLine))
            painter.drawLine(start_x, 0, start_x, self.height())
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

    def _pixel_to_time(self, x: int) -> float:
        """Convert pixel x-coordinate to time in seconds.

        Args:
            x: Pixel x-coordinate

        Returns:
            Time in seconds
        """
        if self.times is None or len(self.times) == 0:
            return 0.0

        t_min, t_max = self.times[0], self.times[-1]

        if t_max <= t_min:
            return t_min

        width = self.width()
        if width == 0:
            return t_min

        fraction = x / width
        return t_min + fraction * (t_max - t_min)

    def _col_to_pixel(self, col_idx: int) -> int:
        """Convert column index to pixel x-coordinate (pixel-perfect alignment).

        Args:
            col_idx: Column index in spectrogram

        Returns:
            Pixel x-coordinate
        """
        if self.total_columns <= 0:
            return 0
        return int((col_idx / self.total_columns) * self.width())

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse click to select detection boundary or start creating new detection."""
        click_x = int(event.position().x())

        # Right-click: start creating new detection
        if event.button() == Qt.MouseButton.RightButton:
            self._creating_detection = True
            self._creation_start_x = click_x
            self._creation_current_x = click_x
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
            return

        # Left-click: boundary adjustment (existing behavior)
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.detections is None:
            return

        # Check if click is near any detection boundary (5px tolerance)
        for usv in self.detections:
            start_x = self._col_to_pixel(usv.start_col)
            end_x = self._col_to_pixel(usv.end_col)

            if abs(click_x - start_x) < 5:
                # Find index of this detection in the list
                try:
                    detection_idx = self.detections.index(usv)
                except ValueError:
                    return  # Detection not in list

                self._selected_detection = usv
                self._selected_detection_idx = detection_idx  # Store index
                self._dragging_edge = 'start'
                self._original_time = usv.start_time_s
                self._is_dragging = True
                self.setFocus()  # Ensure keyboard events are received
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.update()
                return
            elif abs(click_x - end_x) < 5:
                # Find index of this detection in the list
                try:
                    detection_idx = self.detections.index(usv)
                except ValueError:
                    return  # Detection not in list

                self._selected_detection = usv
                self._selected_detection_idx = detection_idx  # Store index
                self._dragging_edge = 'end'
                self._original_time = usv.end_time_s
                self._is_dragging = True
                self.setFocus()  # Ensure keyboard events are received
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.update()
                return

        # Click not near any boundary - clear selection
        self._selected_detection = None
        self._selected_detection_idx = None
        self._dragging_edge = None
        self.unsetCursor()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle drag to adjust boundary position or update creation preview."""
        # Handle detection creation mode
        if self._creating_detection:
            self._creation_current_x = int(event.position().x())
            self.update()  # Repaint to show preview box
            return

        if not self._is_dragging or self._selected_detection is None:
            # Not dragging - check if hovering near boundary for cursor feedback
            if self.detections is not None:
                x = event.position().x()
                near_boundary = False
                for usv in self.detections:
                    start_x = self._col_to_pixel(usv.start_col)
                    end_x = self._col_to_pixel(usv.end_col)
                    if abs(x - start_x) < 5 or abs(x - end_x) < 5:
                        near_boundary = True
                        break

                if near_boundary:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.unsetCursor()
            return

        # Calculate new time from mouse position
        new_time = self._pixel_to_time(int(event.position().x()))

        # Validation: prevent start >= end
        if self._dragging_edge == 'start':
            if new_time >= self._selected_detection.end_time_s:
                new_time = self._selected_detection.end_time_s - 0.001  # Min 1ms duration
        else:  # dragging end
            if new_time <= self._selected_detection.start_time_s:
                new_time = self._selected_detection.start_time_s + 0.001

        # Clamp to valid time range
        if self.times is not None and len(self.times) > 0:
            new_time = max(self.times[0], min(self.times[-1], new_time))

        # Emit signal for MainWindow to update data
        # (Don't modify detection here - MainWindow owns the data)
        if self._dragging_edge == 'start':
            self.boundary_adjusted.emit(
                self._selected_detection,
                new_time,
                self._selected_detection.end_time_s
            )
        else:
            self.boundary_adjusted.emit(
                self._selected_detection,
                self._selected_detection.start_time_s,
                new_time
            )

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Finalize boundary adjustment or detection creation."""
        # Handle detection creation completion
        if event.button() == Qt.MouseButton.RightButton and self._creating_detection:
            self._creating_detection = False
            self.unsetCursor()

            # Convert pixel positions to times
            start_x = min(self._creation_start_x, self._creation_current_x)
            end_x = max(self._creation_start_x, self._creation_current_x)

            # Minimum width check (at least 10px)
            if (end_x - start_x) < 10:
                self.update()
                return

            start_time = self._pixel_to_time(start_x)
            end_time = self._pixel_to_time(end_x)

            # Emit signal for MainWindow to create detection
            self.detection_created.emit(start_time, end_time)
            self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            # Keep index tracking and selection - only clear on Escape or new selection
            # User can press Escape to clear selection
            self.update()  # Redraw to show selection persists

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts for boundary adjustment and creation cancellation."""
        if event.key() == Qt.Key.Key_Escape:
            # Cancel creation mode
            if self._creating_detection:
                self._creating_detection = False
                self.unsetCursor()
                self.update()
                event.accept()
                return

            if self._selected_detection is not None:
                # Cancel adjustment - restore original
                if self._original_time is not None:
                    if self._dragging_edge == 'start':
                        self.boundary_adjusted.emit(
                            self._selected_detection,
                            self._original_time,
                            self._selected_detection.end_time_s
                        )
                    else:
                        self.boundary_adjusted.emit(
                            self._selected_detection,
                            self._selected_detection.start_time_s,
                            self._original_time
                        )

                # Clear selection
                self._is_dragging = False
                self._selected_detection = None
                self._selected_detection_idx = None  # Clear index tracking
                self._dragging_edge = None
                self._original_time = None
                self.unsetCursor()
                self.update()
                event.accept()
                return

        super().keyPressEvent(event)


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
        frequencies: np.ndarray,
        total_columns: int = 0
    ):
        """Set spectrogram data."""
        self.canvas.set_data(spectrogram_db, times, frequencies, total_columns)

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
