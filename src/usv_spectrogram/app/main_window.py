"""Main window for USV Detection App."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QScrollArea,
    QSplitter
)
from PyQt6.QtGui import QAction, QKeyEvent, QShortcut, QKeySequence

from .core.audio_loader import AudioLoader, AudioData
from .core.sliding_inference import SlidingInference, InferenceResult
from .core.detection_logic import HysteresisDetector, DetectionResult
from .core.label_storage import LabelStorage
from .widgets.spectrogram_view import SpectrogramView
from .widgets.probability_view import ProbabilityView


class InferenceWorker(QThread):
    """Background worker for running CNN inference."""

    finished = pyqtSignal(object)  # InferenceResult
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        model_path: Path,
        audio_data: AudioData
    ):
        super().__init__()
        self.model_path = model_path
        self.audio_data = audio_data

    def run(self):
        """Run inference in background thread."""
        try:
            self.progress.emit("Loading CNN model...")
            inference = SlidingInference(
                model_path=self.model_path,
                window_width_px=150,
                hop_px=10,
                batch_size=32
            )

            self.progress.emit("Running inference...")
            result = inference.infer(
                self.audio_data.spectrogram_db,
                self.audio_data.times
            )

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, default_model_path: Path | None = None):
        super().__init__()

        self.default_model_path = default_model_path
        self.current_wav_path: Optional[Path] = None
        self.audio_data: Optional[AudioData] = None
        self.inference_result: Optional[InferenceResult] = None
        self.detection_result: Optional[DetectionResult] = None

        # Settings
        self.settings = QSettings("USV Lab", "USV Detection")

        # Detection parameters (load from settings)
        self.high_threshold = self.settings.value("high_threshold", 0.40, type=float)
        self.low_threshold = self.settings.value("low_threshold", 0.28, type=float)

        self._init_ui()
        self._load_window_geometry()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("USV Detection")
        self.setGeometry(100, 100, 1400, 800)

        # Create menu bar
        self._create_menu_bar()

        # Create main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        # Splitter for views
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Spectrogram and Probability views (stacked vertically)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.spectrogram_view = SpectrogramView()
        self.probability_view = ProbabilityView()

        left_layout.addWidget(QLabel("Spectrogram:"), 0)
        left_layout.addWidget(self.spectrogram_view, 3)
        left_layout.addWidget(QLabel("Probability:"), 0)
        left_layout.addWidget(self.probability_view, 1)

        # Right side: Threshold control
        right_widget = self._create_threshold_panel()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # Connect scroll synchronization
        self.spectrogram_view.scroll_changed.connect(self.probability_view.set_scroll_position)
        self.probability_view.scroll_changed.connect(self.spectrogram_view.set_scroll_position)

        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open WAV...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_wav)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Labels...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_labels)
        file_menu.addAction(save_action)

        export_action = QAction("&Export Image...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_image)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _create_control_panel(self) -> QWidget:
        """Create control panel with action buttons."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        self.load_btn = QPushButton("Open WAV File")
        self.load_btn.clicked.connect(self._open_wav)
        layout.addWidget(self.load_btn)

        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self._run_detection)
        self.detect_btn.setEnabled(False)
        layout.addWidget(self.detect_btn)

        layout.addStretch()

        return panel

    def _create_threshold_panel(self) -> QWidget:
        """Create threshold adjustment panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("<b>Threshold Settings</b>"))

        # High threshold slider
        layout.addWidget(QLabel("High Threshold:"))
        self.high_threshold_label = QLabel(f"{self.high_threshold:.2f}")
        layout.addWidget(self.high_threshold_label)

        self.high_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_threshold_slider.setMinimum(0)
        self.high_threshold_slider.setMaximum(100)
        self.high_threshold_slider.setValue(int(self.high_threshold * 100))
        self.high_threshold_slider.valueChanged.connect(self._on_high_threshold_changed)
        layout.addWidget(self.high_threshold_slider)

        layout.addSpacing(20)

        # Low threshold slider
        layout.addWidget(QLabel("Low Threshold:"))
        self.low_threshold_label = QLabel(f"{self.low_threshold:.2f}")
        layout.addWidget(self.low_threshold_label)

        self.low_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_threshold_slider.setMinimum(0)
        self.low_threshold_slider.setMaximum(100)
        self.low_threshold_slider.setValue(int(self.low_threshold * 100))
        self.low_threshold_slider.valueChanged.connect(self._on_low_threshold_changed)
        layout.addWidget(self.low_threshold_slider)

        layout.addSpacing(20)

        # Apply button
        self.apply_btn = QPushButton("Apply Thresholds")
        self.apply_btn.clicked.connect(self._apply_thresholds)
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)

        layout.addSpacing(20)

        # Detection info
        self.detection_info_label = QLabel("No detections")
        layout.addWidget(self.detection_info_label)

        layout.addStretch()

        return panel

    def _on_high_threshold_changed(self, value: int):
        """Handle high threshold slider change."""
        self.high_threshold = value / 100.0
        self.high_threshold_label.setText(f"{self.high_threshold:.2f}")

    def _on_low_threshold_changed(self, value: int):
        """Handle low threshold slider change."""
        self.low_threshold = value / 100.0
        self.low_threshold_label.setText(f"{self.low_threshold:.2f}")

    def _open_wav(self):
        """Open WAV file dialog and load audio."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open WAV File",
            "",
            "WAV Files (*.wav);;All Files (*)"
        )

        if file_path:
            self._load_wav_file(Path(file_path))

    def _load_wav_file(self, wav_path: Path):
        """Load WAV file and compute spectrogram."""
        try:
            self.statusBar.showMessage(f"Loading {wav_path.name}...")

            audio_loader = AudioLoader()
            self.audio_data = audio_loader.load(wav_path)
            self.current_wav_path = wav_path

            # Display spectrogram
            self.spectrogram_view.set_data(
                self.audio_data.spectrogram_db,
                self.audio_data.times,
                self.audio_data.frequencies
            )

            # Enable detection button
            self.detect_btn.setEnabled(True)

            self.statusBar.showMessage(
                f"Loaded {wav_path.name} ({self.audio_data.duration_s:.2f}s)",
                3000
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading WAV",
                f"Failed to load WAV file:\n{str(e)}"
            )
            self.statusBar.showMessage("Error loading WAV", 3000)

    def _run_detection(self):
        """Run CNN inference and detection."""
        if self.audio_data is None or self.default_model_path is None:
            return

        self.statusBar.showMessage("Running detection...")
        self.detect_btn.setEnabled(False)

        # Start worker thread
        self.worker = InferenceWorker(self.default_model_path, self.audio_data)
        self.worker.finished.connect(self._on_inference_finished)
        self.worker.error.connect(self._on_inference_error)
        self.worker.progress.connect(self.statusBar.showMessage)
        self.worker.start()

    def _on_inference_finished(self, result: InferenceResult):
        """Handle inference completion."""
        self.inference_result = result

        # Apply detection
        self._apply_thresholds()

        self.detect_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self.statusBar.showMessage("Detection complete", 3000)

    def _on_inference_error(self, error_msg: str):
        """Handle inference error."""
        QMessageBox.critical(
            self,
            "Inference Error",
            f"Failed to run inference:\n{error_msg}"
        )
        self.detect_btn.setEnabled(True)
        self.statusBar.showMessage("Inference failed", 3000)

    def _apply_thresholds(self):
        """Apply current thresholds to detection."""
        if self.inference_result is None:
            return

        # Run hysteresis detection
        detector = HysteresisDetector(
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
            merge_gap_columns=3
        )

        self.detection_result = detector.detect(
            self.inference_result.probabilities,
            self.inference_result.column_indices,
            self.inference_result.times
        )

        # Update views
        self.probability_view.set_data(
            self.detection_result.times,
            self.detection_result.probabilities,
            self.high_threshold,
            self.low_threshold,
            self.detection_result.usvs
        )

        self.spectrogram_view.set_detections(self.detection_result.usvs)

        # Update info label
        n_detections = len(self.detection_result.usvs)
        self.detection_info_label.setText(f"Detected: {n_detections} USVs")

    def _save_labels(self):
        """Save detection results to JSON."""
        if self.detection_result is None or self.audio_data is None:
            QMessageBox.warning(
                self,
                "No Detections",
                "Run detection first before saving."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Labels",
            str(self.current_wav_path.with_suffix('.json')),
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                LabelStorage.save(
                    output_path=file_path,
                    audio_data=self.audio_data,
                    detection_result=self.detection_result,
                    wav_path=self.current_wav_path,
                    model_path=self.default_model_path,
                    high_threshold=self.high_threshold,
                    low_threshold=self.low_threshold
                )

                QMessageBox.information(
                    self,
                    "Success",
                    f"Labels saved to {Path(file_path).name}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save labels:\n{str(e)}"
                )

    def _export_image(self):
        """Export annotated spectrogram image."""
        if self.detection_result is None or self.audio_data is None:
            QMessageBox.warning(
                self,
                "No Detections",
                "Run detection first before exporting."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            str(self.current_wav_path.with_suffix('.png')),
            "PNG Files (*.png);;All Files (*)"
        )

        if file_path:
            try:
                LabelStorage.export_annotated_image(
                    output_path=file_path,
                    audio_data=self.audio_data,
                    detection_result=self.detection_result,
                    high_threshold=self.high_threshold,
                    low_threshold=self.low_threshold
                )

                QMessageBox.information(
                    self,
                    "Success",
                    f"Image exported to {Path(file_path).name}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export image:\n{str(e)}"
                )

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for threshold adjustment."""
        # Arrow up/down for high threshold
        up_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        up_shortcut.activated.connect(self._increase_high_threshold)

        down_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        down_shortcut.activated.connect(self._decrease_high_threshold)

        # Left/right for low threshold
        left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        left_shortcut.activated.connect(self._decrease_low_threshold)

        right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        right_shortcut.activated.connect(self._increase_low_threshold)

        # Space for run detection
        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.activated.connect(self._run_detection)

    def _increase_high_threshold(self):
        """Increase high threshold by 0.01."""
        new_value = min(100, self.high_threshold_slider.value() + 1)
        self.high_threshold_slider.setValue(new_value)
        if self.inference_result is not None:
            self._apply_thresholds()

    def _decrease_high_threshold(self):
        """Decrease high threshold by 0.01."""
        new_value = max(0, self.high_threshold_slider.value() - 1)
        self.high_threshold_slider.setValue(new_value)
        if self.inference_result is not None:
            self._apply_thresholds()

    def _increase_low_threshold(self):
        """Increase low threshold by 0.01."""
        new_value = min(100, self.low_threshold_slider.value() + 1)
        self.low_threshold_slider.setValue(new_value)
        if self.inference_result is not None:
            self._apply_thresholds()

    def _decrease_low_threshold(self):
        """Decrease low threshold by 0.01."""
        new_value = max(0, self.low_threshold_slider.value() - 1)
        self.low_threshold_slider.setValue(new_value)
        if self.inference_result is not None:
            self._apply_thresholds()

    def _load_window_geometry(self):
        """Load window geometry from settings."""
        geometry = self.settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)

        state = self.settings.value("window_state")
        if state:
            self.restoreState(state)

    def _save_settings(self):
        """Save window geometry and threshold settings."""
        self.settings.setValue("window_geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
        self.settings.setValue("high_threshold", self.high_threshold)
        self.settings.setValue("low_threshold", self.low_threshold)

    def closeEvent(self, event):
        """Handle window close event to save settings."""
        self._save_settings()
        event.accept()
