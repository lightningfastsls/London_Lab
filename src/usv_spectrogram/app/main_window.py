"""Main window for USV Detection App."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
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
    QSplitter,
    QProgressDialog
)
from PyQt6.QtGui import QAction, QKeyEvent, QShortcut, QKeySequence

from .core.audio_loader import AudioLoader, AudioData
from .core.preset_config import PresetManager
from .core.sliding_inference import SlidingInference, InferenceResult
from .core.detection_logic import HysteresisDetector, DetectionResult, DetectedUSV
from .core.label_storage import LabelStorage
from .core.saved_detection_tracker import SavedDetectionTracker
from .core.detection_exporter import DetectionExporter
from .widgets.spectrogram_view import SpectrogramView
from .widgets.probability_view import ProbabilityView

# App version for settings migration
# Increment when default settings change
APP_VERSION = "1.1.0"  # Bumped for threshold changes (2026-02-06)


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
                window_width_px=100,  # ~43ms, matches median training USV duration
                hop_px=10,            # 90% overlap for robust detection
                batch_size=32,
                energy_threshold=0.35,  # Skip windows with max < 0.35 (quiet noise regions)
                enable_per_window_norm=False  # Training only uses per-image norm after colormap
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

        # Settings migration: Reset specific settings when version changes
        last_version = self.settings.value("app_version", "0.0.0", type=str)

        if last_version != APP_VERSION:
            print(f"[Settings Migration] Version changed: {last_version} → {APP_VERSION}")
            print("[Settings Migration] Resetting detection parameters...")

            # Reset detection thresholds to new defaults
            self.settings.remove("high_threshold")      # Forces reload: 0.04
            self.settings.remove("low_threshold")       # Forces reload: 0.03
            self.settings.remove("min_sustained_prob")  # Forces reload: 0.0

            # Reset output directory to project folder
            self.settings.remove("detection_output_dir")

            # Preserve window geometry (user customization)
            # DO NOT remove: window_geometry, window_state

            # Update version marker
            self.settings.setValue("app_version", APP_VERSION)
            print("[Settings Migration] Complete. Using new defaults.")

        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.current_preset: str | None = None

        # Preset manager
        self.preset_manager = PresetManager()

        # Detection saving
        self.detection_exporter: Optional[DetectionExporter] = None
        self.saved_tracker: Optional[SavedDetectionTracker] = None
        # Default output dir: project folder instead of home directory
        repo_root = Path(__file__).resolve().parents[3]
        default_output = str(repo_root / "USV_Detections")
        self.output_dir = Path(self.settings.value("detection_output_dir", default_output))

        # Detection parameters (load from settings)
        # Updated defaults per user request (2026-02-06)
        # Lower thresholds to catch more USVs, continuity disabled
        self.high_threshold = self.settings.value("high_threshold", 0.04, type=float)
        self.low_threshold = self.settings.value("low_threshold", 0.03, type=float)
        self.min_sustained_prob = self.settings.value("min_sustained_prob", 0.0, type=float)  # Continuity disabled
        self.exclude_start_sec = self.settings.value("exclude_start_sec", 0.5, type=float)
        self.exclude_end_sec = self.settings.value("exclude_end_sec", 0.5, type=float)

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

        # Add persistent file info label
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("QLabel { color: #555; margin-right: 10px; }")
        self.statusBar.addPermanentWidget(self.file_info_label)

        # Connect scroll synchronization - one scrollbar controls both views
        # Only spectrogram scrollbar is visible, it controls probability view
        self.spectrogram_view.scroll_changed.connect(self.probability_view.set_scroll_position)

        # Connect boundary adjustment signal
        self.spectrogram_view.canvas.boundary_adjusted.connect(self._on_boundary_adjusted)

        # Connect detection creation signal
        self.spectrogram_view.canvas.detection_created.connect(self._add_detection)

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
        save_action.setStatusTip("Save all detections to single JSON with adjustment history")
        file_menu.addAction(save_action)

        export_action = QAction("&Export Image...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_image)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        output_dir_action = QAction("Set Output &Directory...", self)
        output_dir_action.triggered.connect(self._change_output_directory)
        file_menu.addAction(output_dir_action)

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

        # Save buttons
        self.save_current_btn = QPushButton("Export Current View")
        self.save_current_btn.clicked.connect(self._save_current_view)
        self.save_current_btn.setEnabled(False)
        self.save_current_btn.setToolTip(
            "Export visible detections as individual PNG/JSON files (visualization format)"
        )
        layout.addWidget(self.save_current_btn)

        self.save_all_btn = QPushButton("Export All as PNGs")
        self.save_all_btn.clicked.connect(self._save_all_detections)
        self.save_all_btn.setEnabled(False)
        self.save_all_btn.setToolTip(
            "Export all detections as individual PNG/JSON files (visualization format)"
        )
        layout.addWidget(self.save_all_btn)

        layout.addSpacing(20)

        # Manual editing buttons
        self.remove_detection_btn = QPushButton("Remove Detection")
        self.remove_detection_btn.clicked.connect(self._remove_detection)
        self.remove_detection_btn.setEnabled(False)
        self.remove_detection_btn.setToolTip(
            "Remove selected detection (click a boundary line first, or press Del)"
        )
        layout.addWidget(self.remove_detection_btn)

        layout.addStretch()

        return panel

    def _create_threshold_panel(self) -> QWidget:
        """Create threshold adjustment panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("<b>Threshold Settings</b>"))

        # Preset buttons
        preset_layout = QHBoxLayout()
        for preset in self.preset_manager.presets:
            btn = QPushButton(preset.name)
            btn.setToolTip(f"High: {preset.high_threshold:.2f}, Low: {preset.low_threshold:.2f}")
            btn.clicked.connect(lambda checked, p=preset: self._apply_preset(p))
            preset_layout.addWidget(btn)
        layout.addLayout(preset_layout)

        layout.addSpacing(10)

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

        # Min sustained probability slider
        layout.addWidget(QLabel("Min Sustained Probability:"))
        self.min_sustained_prob_label = QLabel(f"{self.min_sustained_prob:.2f}")
        layout.addWidget(self.min_sustained_prob_label)

        self.min_sustained_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_sustained_prob_slider.setMinimum(0)
        self.min_sustained_prob_slider.setMaximum(100)
        self.min_sustained_prob_slider.setValue(int(self.min_sustained_prob * 100))
        self.min_sustained_prob_slider.valueChanged.connect(self._on_min_sustained_prob_changed)
        layout.addWidget(self.min_sustained_prob_slider)

        layout.addSpacing(20)

        # Exclude start seconds slider
        layout.addWidget(QLabel("Exclude Start (seconds):"))
        self.exclude_start_label = QLabel(f"{self.exclude_start_sec:.1f}s")
        layout.addWidget(self.exclude_start_label)

        self.exclude_start_slider = QSlider(Qt.Orientation.Horizontal)
        self.exclude_start_slider.setMinimum(0)
        self.exclude_start_slider.setMaximum(200)  # 0.0 to 2.0 seconds (step 0.01)
        self.exclude_start_slider.setValue(int(self.exclude_start_sec * 100))
        self.exclude_start_slider.valueChanged.connect(self._on_exclude_start_changed)
        layout.addWidget(self.exclude_start_slider)

        layout.addSpacing(20)

        # Exclude end seconds slider
        layout.addWidget(QLabel("Exclude End (seconds):"))
        self.exclude_end_label = QLabel(f"{self.exclude_end_sec:.1f}s")
        layout.addWidget(self.exclude_end_label)

        self.exclude_end_slider = QSlider(Qt.Orientation.Horizontal)
        self.exclude_end_slider.setMinimum(0)
        self.exclude_end_slider.setMaximum(200)  # 0.0 to 2.0 seconds (step 0.01)
        self.exclude_end_slider.setValue(int(self.exclude_end_sec * 100))
        self.exclude_end_slider.valueChanged.connect(self._on_exclude_end_changed)
        layout.addWidget(self.exclude_end_slider)

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
        new_high = value / 100.0

        # Validate: high threshold must be >= low threshold
        if new_high < self.low_threshold:
            QMessageBox.warning(
                self,
                "Invalid Threshold",
                f"High threshold ({new_high:.2f}) must be ≥ low threshold ({self.low_threshold:.2f}).\n\n"
                "Please adjust low threshold first."
            )
            # Revert slider to previous valid value
            self.high_threshold_slider.setValue(int(self.high_threshold * 100))
            return

        self.high_threshold = new_high
        self.high_threshold_label.setText(f"{self.high_threshold:.2f}")

    def _on_low_threshold_changed(self, value: int):
        """Handle low threshold slider change."""
        new_low = value / 100.0

        # Validate: low threshold must be <= high threshold
        if new_low > self.high_threshold:
            QMessageBox.warning(
                self,
                "Invalid Threshold",
                f"Low threshold ({new_low:.2f}) must be ≤ high threshold ({self.high_threshold:.2f}).\n\n"
                "Please adjust high threshold first."
            )
            # Revert slider to previous valid value
            self.low_threshold_slider.setValue(int(self.low_threshold * 100))
            return

        self.low_threshold = new_low
        self.low_threshold_label.setText(f"{self.low_threshold:.2f}")

    def _on_min_sustained_prob_changed(self, value: int):
        """Handle min sustained probability slider change."""
        self.min_sustained_prob = value / 100.0
        self.min_sustained_prob_label.setText(f"{self.min_sustained_prob:.2f}")

    def _on_exclude_start_changed(self, value: int):
        """Handle exclude start slider change."""
        self.exclude_start_sec = value / 100.0
        self.exclude_start_label.setText(f"{self.exclude_start_sec:.1f}s")

    def _on_exclude_end_changed(self, value: int):
        """Handle exclude end slider change."""
        self.exclude_end_sec = value / 100.0
        self.exclude_end_label.setText(f"{self.exclude_end_sec:.1f}s")

    def _apply_preset(self, preset):
        """Apply a threshold preset to sliders.

        Sets high slider first, then low, to avoid validation rejection.
        Uses blockSignals to batch the update.

        Args:
            preset: ThresholdPreset to apply
        """
        self.current_preset = preset.name

        # Block signals to prevent validation popups during batch update
        self.high_threshold_slider.blockSignals(True)
        self.low_threshold_slider.blockSignals(True)

        # Set high first (must be >= low)
        self.high_threshold = preset.high_threshold
        self.high_threshold_slider.setValue(int(preset.high_threshold * 100))
        self.high_threshold_label.setText(f"{preset.high_threshold:.2f}")

        # Then set low
        self.low_threshold = preset.low_threshold
        self.low_threshold_slider.setValue(int(preset.low_threshold * 100))
        self.low_threshold_label.setText(f"{preset.low_threshold:.2f}")

        # Re-enable signals
        self.high_threshold_slider.blockSignals(False)
        self.low_threshold_slider.blockSignals(False)

        # Auto-apply if inference has run
        if self.inference_result is not None:
            self._apply_thresholds()

        self.statusBar.showMessage(f"Applied preset: {preset.name}", 3000)

    def _open_wav(self):
        """Open WAV file dialog and load audio."""
        # Check for unsaved detections before switching files
        if not self._check_unsaved_detections():
            return  # User cancelled

        # Show loading state
        self.file_info_label.setText("Loading...")

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

            # Update file info display
            self.file_info_label.setText(f"File: {wav_path.name}")

            # Display spectrogram with total_columns for pixel-perfect alignment
            total_columns = self.audio_data.spectrogram_db.shape[1]
            self.spectrogram_view.set_data(
                self.audio_data.spectrogram_db,
                self.audio_data.times,
                self.audio_data.frequencies,
                total_columns=total_columns
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

        # Initialize detection saving components
        wav_name = self.current_wav_path.stem
        self.saved_tracker = SavedDetectionTracker(wav_name, self.output_dir)
        self.detection_exporter = DetectionExporter(self.output_dir, context_ms=20.0)

        self.detect_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self.save_current_btn.setEnabled(True)
        self.save_all_btn.setEnabled(True)
        self.remove_detection_btn.setEnabled(True)
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
            merge_gap_columns=3,
            min_duration_ms=10.0,  # Reject events < 10ms (noise artifacts)
            max_duration_ms=500.0,  # Reject events > 500ms (non-USV vocalizations)
            min_sustained_prob=self.min_sustained_prob,  # Reject events with brief probability dips
            exclude_start_sec=self.exclude_start_sec,  # Reject detections near file start
            exclude_end_sec=self.exclude_end_sec  # Reject detections near file end
        )

        self.detection_result = detector.detect(
            self.inference_result.probabilities,
            self.inference_result.column_indices,
            self.inference_result.times
        )

        # Mark save state for current detections
        if self.saved_tracker is not None:
            for det in self.detection_result.usvs:
                if self.saved_tracker.is_saved(det):
                    det.save_state = "saved_current"

        # Load ghost detections from previous sessions
        ghost_detections = self._load_previous_saved_detections()

        # Combine current + ghost for views
        all_detections = self.detection_result.usvs + ghost_detections

        # Update views - pass spectrogram width AND column indices for pixel-perfect alignment
        spectrogram_width = self.spectrogram_view.get_canvas_width()
        self.probability_view.set_data(
            self.detection_result.times,
            self.detection_result.probabilities,
            self.high_threshold,
            self.low_threshold,
            all_detections,
            column_indices=self.detection_result.column_indices,  # For pixel-perfect alignment
            target_width=spectrogram_width
        )

        self.spectrogram_view.set_detections(all_detections)

        # Update info label
        n_detections = len(self.detection_result.usvs)
        n_ghosts = len(ghost_detections)
        info_text = f"Detected: {n_detections} USVs"
        if n_ghosts > 0:
            info_text += f" ({n_ghosts} previously saved)"
        self.detection_info_label.setText(info_text)

    def _load_previous_saved_detections(self) -> list:
        """Load previously saved detections as ghost DetectedUSV objects.

        Returns:
            List of DetectedUSV with save_state="saved_previous"
        """
        if self.saved_tracker is None or self.audio_data is None:
            return []

        ghosts = []
        times = self.audio_data.times

        for record in self.saved_tracker.saved_detections:
            # Check if this record overlaps with any current detection
            # (those are already marked "saved_current", skip to avoid duplicates)
            is_current = False
            if self.detection_result is not None:
                for det in self.detection_result.usvs:
                    if det.save_state == "saved_current":
                        # Check overlap
                        if not (det.end_time_s <= record.start_time_s or
                                record.end_time_s <= det.start_time_s):
                            is_current = True
                            break

            if is_current:
                continue

            # Estimate column indices via searchsorted
            start_col = int(np.searchsorted(times, record.start_time_s))
            end_col = int(np.searchsorted(times, record.end_time_s))

            ghost = DetectedUSV(
                start_time_s=record.start_time_s,
                end_time_s=record.end_time_s,
                start_col=start_col,
                end_col=end_col,
                max_probability=0.0,
                mean_probability=0.0,
                save_state="saved_previous"
            )
            ghosts.append(ghost)

        return ghosts

    def _on_boundary_adjusted(self, original_detection, new_start: float, new_end: float):
        """Handle boundary adjustment from SpectrogramCanvas.

        Args:
            original_detection: The DetectedUSV being adjusted
            new_start: New start time in seconds
            new_end: New end time in seconds
        """
        if self.detection_result is None or self.inference_result is None:
            return

        # Find index of original detection
        try:
            idx = self.detection_result.usvs.index(original_detection)
        except ValueError:
            return  # Detection not found

        # Convert times to column indices
        # Use times from detection result (matches spectrogram columns)
        times = self.detection_result.times

        def time_to_col(time_s: float) -> int:
            """Convert time to spectrogram column index."""
            if len(times) == 0:
                return 0
            # Find nearest probability array index, then map to spectrogram column
            prob_idx = np.argmin(np.abs(times - time_s))
            return int(self.detection_result.column_indices[prob_idx])

        new_start_col = time_to_col(new_start)
        new_end_col = time_to_col(new_end)

        # Create new DetectedUSV with adjusted boundaries
        # Preserve probability metadata from original
        # Track adjustment history
        adjusted_usv = DetectedUSV(
            start_time_s=new_start,
            end_time_s=new_end,
            start_col=new_start_col,
            end_col=new_end_col,
            max_probability=original_detection.max_probability,
            mean_probability=original_detection.mean_probability,
            user_adjusted=True,
            original_start_time_s=(
                original_detection.start_time_s
                if not original_detection.user_adjusted
                else original_detection.original_start_time_s
            ),
            original_end_time_s=(
                original_detection.end_time_s
                if not original_detection.user_adjusted
                else original_detection.original_end_time_s
            )
        )

        # Replace in detection result
        self.detection_result.usvs[idx] = adjusted_usv

        # Redraw both views with updated detections (including ghosts)
        self._refresh_detection_views()

        # Show status message with updated duration
        duration_ms = (new_end - new_start) * 1000
        self.statusBar.showMessage(
            f"Adjusted detection: {new_start:.3f}s - {new_end:.3f}s "
            f"(duration: {duration_ms:.1f}ms)",
            3000
        )

    def _remove_detection(self):
        """Remove selected detection from detection list.

        User must first select a detection by clicking a boundary line.
        Ghost detections (saved_previous) cannot be removed.
        """
        if self.detection_result is None:
            return

        # Get selected detection index from spectrogram canvas
        selected_idx = self.spectrogram_view.canvas._selected_detection_idx

        if selected_idx is None:
            QMessageBox.information(
                self,
                "No Selection",
                "Please select a detection first by clicking on a boundary line."
            )
            return

        # Bounds check
        if not (0 <= selected_idx < len(self.detection_result.usvs)):
            return

        selected_detection = self.detection_result.usvs[selected_idx]

        # Don't allow removing ghost detections (read-only)
        if selected_detection.save_state == "saved_previous":
            QMessageBox.warning(
                self,
                "Cannot Remove",
                "Previously saved detections cannot be removed.\n\n"
                "These are shown in gray and represent detections from earlier sessions."
            )
            return

        # Confirm removal
        duration_ms = (selected_detection.end_time_s - selected_detection.start_time_s) * 1000
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove detection at {selected_detection.start_time_s:.3f}s - "
            f"{selected_detection.end_time_s:.3f}s ({duration_ms:.1f}ms)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Export to rejected_detections/ before deleting (for training data)
        if self.detection_exporter is not None and self.audio_data is not None:
            try:
                # Create modified detection for export
                deleted_detection = DetectedUSV(
                    start_time_s=selected_detection.start_time_s,
                    end_time_s=selected_detection.end_time_s,
                    start_col=selected_detection.start_col,
                    end_col=selected_detection.end_col,
                    max_probability=selected_detection.max_probability,
                    mean_probability=selected_detection.mean_probability,
                    user_adjusted=selected_detection.user_adjusted,
                    original_start_time_s=selected_detection.original_start_time_s,
                    original_end_time_s=selected_detection.original_end_time_s,
                    save_state=selected_detection.save_state,
                    user_action="deleted_by_user",
                    original_cnn_probability=selected_detection.max_probability  # Preserve CNN's prediction
                )

                # Export to rejected_detections subfolder
                rejected_output_dir = self.output_dir / "rejected_detections"
                rejected_exporter = DetectionExporter(rejected_output_dir, context_ms=20.0)

                png, json_file, csv = rejected_exporter.export_detection(
                    detection=deleted_detection,
                    audio_data=self.audio_data,
                    wav_filename=self.current_wav_path.stem,
                    detection_index=selected_idx,
                    session_id=self.session_id,
                    threshold_preset=self.current_preset,
                    threshold_high=self.high_threshold,
                    threshold_low=self.low_threshold
                )

                # Also track in saved_tracker for rejected detections
                if self.saved_tracker is not None:
                    self.saved_tracker.mark_saved(
                        deleted_detection, str(png),
                        threshold_preset=self.current_preset,
                        threshold_high=self.high_threshold,
                        threshold_low=self.low_threshold,
                        session_id=self.session_id,
                        user_action="deleted_by_user"
                    )

            except Exception as e:
                print(f"Warning: Could not export deleted detection: {e}")
                # Continue with deletion even if export fails

        # Remove from detection list
        del self.detection_result.usvs[selected_idx]

        # Clear selection in canvas
        self.spectrogram_view.canvas._selected_detection = None
        self.spectrogram_view.canvas._selected_detection_idx = None

        # Refresh views
        self._refresh_detection_views()

        # Update info label
        n_detections = len(self.detection_result.usvs)
        ghost_detections = self._load_previous_saved_detections()
        n_ghosts = len(ghost_detections)
        info_text = f"Detected: {n_detections} USVs"
        if n_ghosts > 0:
            info_text += f" ({n_ghosts} previously saved)"
        self.detection_info_label.setText(info_text)

        self.statusBar.showMessage("Detection removed", 2000)

    def _add_detection(self, start_time_s: float, end_time_s: float):
        """Add manually created detection.

        Args:
            start_time_s: Start time in seconds
            end_time_s: End time in seconds
        """
        if self.detection_result is None or self.audio_data is None:
            return

        # Convert times to column indices
        times = self.audio_data.times

        def time_to_col(time_s: float) -> int:
            """Convert time to column index."""
            if len(times) == 0:
                return 0
            col_idx = np.searchsorted(times, time_s)
            return int(col_idx)

        start_col = time_to_col(start_time_s)
        end_col = time_to_col(end_time_s)

        # Create new DetectedUSV (manual, no CNN probabilities)
        new_detection = DetectedUSV(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            start_col=start_col,
            end_col=end_col,
            max_probability=0.0,
            mean_probability=0.0,
            user_adjusted=False,
            original_start_time_s=0.0,
            original_end_time_s=0.0,
            save_state="unsaved",
            user_action="added_manually",
            original_cnn_probability=None
        )

        # Add to detection list
        self.detection_result.usvs.append(new_detection)

        # Sort by start time (keep detections ordered)
        self.detection_result.usvs.sort(key=lambda d: d.start_time_s)

        # Refresh views
        self._refresh_detection_views()

        # Update info label
        n_detections = len(self.detection_result.usvs)
        ghost_detections = self._load_previous_saved_detections()
        n_ghosts = len(ghost_detections)
        info_text = f"Detected: {n_detections} USVs"
        if n_ghosts > 0:
            info_text += f" ({n_ghosts} previously saved)"
        self.detection_info_label.setText(info_text)

        duration_ms = (end_time_s - start_time_s) * 1000
        self.statusBar.showMessage(
            f"Added detection: {start_time_s:.3f}s - {end_time_s:.3f}s ({duration_ms:.1f}ms)",
            3000
        )

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

        # Delete for remove detection
        delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        delete_shortcut.activated.connect(self._remove_detection)

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
        self.settings.setValue("min_sustained_prob", self.min_sustained_prob)
        self.settings.setValue("exclude_start_sec", self.exclude_start_sec)
        self.settings.setValue("exclude_end_sec", self.exclude_end_sec)
        self.settings.setValue("detection_output_dir", str(self.output_dir))

    def closeEvent(self, event):
        """Handle window close event to save settings and check unsaved detections."""
        # Check for unsaved detections
        if not self._check_unsaved_detections():
            event.ignore()  # User cancelled close
            return

        self._save_settings()
        event.accept()

    def _save_current_view(self):
        """Save detections visible in current viewport."""
        if self.detection_result is None:
            return

        # 1. Get visible detection indices
        visible_detections = self._get_visible_detections()

        if not visible_detections:
            QMessageBox.information(self, "No Detections",
                                    "No detections in current view.")
            return

        # 2. Filter out already-saved detections
        unsaved = [d for d in visible_detections if not self.saved_tracker.is_saved(d)]

        if not unsaved:
            QMessageBox.information(self, "Already Saved",
                                    "All detections in view already saved.")
            return

        # 3. Confirm save with user (since multiple detections might be visible)
        reply = QMessageBox.question(
            self, "Confirm Save",
            f"Save {len(unsaved)} detection(s) from current view?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # 4. Save each unsaved detection
        # Show progress for multiple detections
        if len(unsaved) > 1:
            progress = QProgressDialog("Saving detections...", "Cancel", 0, len(unsaved), self)
            progress.setWindowTitle("Save Current View")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
        else:
            progress = None

        saved_count = 0
        for i, detection in enumerate(unsaved):
            if progress and progress.wasCanceled():
                break

            idx = self.detection_result.usvs.index(detection)
            try:
                png, json_file, csv = self.detection_exporter.export_detection(
                    detection=detection,
                    audio_data=self.audio_data,
                    wav_filename=self.current_wav_path.stem,
                    detection_index=idx,
                    session_id=self.session_id,
                    threshold_preset=self.current_preset,
                    threshold_high=self.high_threshold,
                    threshold_low=self.low_threshold
                )
                self.saved_tracker.mark_saved(
                    detection, str(png),
                    threshold_preset=self.current_preset,
                    threshold_high=self.high_threshold,
                    threshold_low=self.low_threshold,
                    session_id=self.session_id,
                    user_action=detection.user_action
                )
                detection.save_state = "saved_current"
                saved_count += 1
            except Exception as e:
                print(f"Error saving detection {idx}: {e}")

            if progress:
                progress.setValue(i + 1)

        if progress:
            progress.close()

        # 5. Refresh views to show updated save states
        self._refresh_detection_views()

        # 6. Show success message
        self.statusBar.showMessage(f"Saved {saved_count} detection(s)", 3000)

    def _save_all_detections(self):
        """Save all detections in current file."""
        if self.detection_result is None:
            return

        all_detections = self.detection_result.usvs

        if not all_detections:
            QMessageBox.information(self, "No Detections", "No detections to save.")
            return

        # Filter out already-saved
        unsaved = self.saved_tracker.get_unsaved_detections(all_detections)

        if not unsaved:
            QMessageBox.information(self, "Already Saved", "All detections already saved.")
            return

        # Confirm batch save
        reply = QMessageBox.question(
            self, "Confirm Save All",
            f"Save {len(unsaved)} unsaved detection(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Progress dialog
        progress = QProgressDialog("Saving detections...", "Cancel", 0, len(unsaved), self)
        progress.setWindowTitle("Save All Detections")
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        saved_count = 0
        for i, detection in enumerate(unsaved):
            if progress.wasCanceled():
                break

            idx = self.detection_result.usvs.index(detection)
            try:
                png, json_file, csv = self.detection_exporter.export_detection(
                    detection=detection,
                    audio_data=self.audio_data,
                    wav_filename=self.current_wav_path.stem,
                    detection_index=idx,
                    session_id=self.session_id,
                    threshold_preset=self.current_preset,
                    threshold_high=self.high_threshold,
                    threshold_low=self.low_threshold
                )
                self.saved_tracker.mark_saved(
                    detection, str(png),
                    threshold_preset=self.current_preset,
                    threshold_high=self.high_threshold,
                    threshold_low=self.low_threshold,
                    session_id=self.session_id,
                    user_action=detection.user_action
                )
                detection.save_state = "saved_current"
                saved_count += 1
            except Exception as e:
                print(f"Error saving detection {idx}: {e}")

            progress.setValue(i + 1)

        progress.close()

        # Refresh views to show updated save states
        self._refresh_detection_views()

        self.statusBar.showMessage(f"Saved {saved_count} detection(s)", 3000)

    def _refresh_detection_views(self):
        """Refresh both views with current detections + ghosts."""
        if self.detection_result is None:
            return

        ghost_detections = self._load_previous_saved_detections()
        all_detections = self.detection_result.usvs + ghost_detections

        spectrogram_width = self.spectrogram_view.get_canvas_width()
        self.probability_view.set_data(
            self.detection_result.times,
            self.detection_result.probabilities,
            self.high_threshold,
            self.low_threshold,
            all_detections,
            column_indices=self.detection_result.column_indices,
            target_width=spectrogram_width
        )
        self.spectrogram_view.set_detections(all_detections)

    def _get_visible_detections(self):
        """Get detections currently visible in viewport.

        Returns:
            List of DetectedUSV objects that are visible in current viewport
        """
        if self.detection_result is None:
            return []

        scrollbar = self.spectrogram_view.scroll_area.horizontalScrollBar()
        scroll_value = scrollbar.value()
        viewport_width = self.spectrogram_view.scroll_area.viewport().width()
        canvas_width = self.spectrogram_view.canvas.width()

        # Calculate visible time range
        t_min = self.audio_data.times[0]
        t_max = self.audio_data.times[-1]

        visible_start_frac = scroll_value / canvas_width if canvas_width > 0 else 0
        visible_end_frac = (scroll_value + viewport_width) / canvas_width if canvas_width > 0 else 1

        visible_start_time = t_min + visible_start_frac * (t_max - t_min)
        visible_end_time = t_min + visible_end_frac * (t_max - t_min)

        # Filter detections in visible range (detection overlaps visible window)
        visible = [
            d for d in self.detection_result.usvs
            if not (d.end_time_s < visible_start_time or d.start_time_s > visible_end_time)
        ]

        return visible

    def _check_unsaved_detections(self) -> bool:
        """Check for unsaved detections and prompt user.

        Returns:
            True if OK to proceed (no unsaved or user confirmed)
            False if user cancelled
        """
        if self.detection_result is None or self.saved_tracker is None:
            return True

        unsaved = self.saved_tracker.get_unsaved_detections(self.detection_result.usvs)

        if not unsaved:
            return True

        # Show warning dialog
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Unsaved Detections")
        msg.setText(f"You have {len(unsaved)} unsaved detection(s).")
        msg.setInformativeText("What would you like to do?")

        review_btn = msg.addButton("Review Unsaved", QMessageBox.ButtonRole.ActionRole)
        discard_btn = msg.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == review_btn:
            # Scroll to first unsaved detection
            first_unsaved = unsaved[0]
            self._scroll_to_detection(first_unsaved)
            return False  # Don't proceed with file switch
        elif clicked == discard_btn:
            return True  # Proceed, discard unsaved
        else:  # Cancel
            return False

    def _scroll_to_detection(self, detection):
        """Scroll viewport to show the given detection.

        Args:
            detection: DetectedUSV object to scroll to
        """
        # Calculate pixel position of detection center
        t_min = self.audio_data.times[0]
        t_max = self.audio_data.times[-1]

        detection_center_time = (detection.start_time_s + detection.end_time_s) / 2
        center_fraction = (detection_center_time - t_min) / (t_max - t_min)

        canvas_width = self.spectrogram_view.canvas.width()
        viewport_width = self.spectrogram_view.scroll_area.viewport().width()

        # Center the detection in viewport
        target_scroll = int(center_fraction * canvas_width - viewport_width / 2)
        target_scroll = max(0, min(target_scroll, canvas_width - viewport_width))

        scrollbar = self.spectrogram_view.scroll_area.horizontalScrollBar()
        scrollbar.setValue(target_scroll)

    def _change_output_directory(self):
        """Allow user to change detection output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Detection Output Directory",
            str(self.output_dir)
        )
        if directory:
            self.output_dir = Path(directory)
            self.settings.setValue("detection_output_dir", str(self.output_dir))
            self.statusBar.showMessage(f"Output directory: {directory}", 3000)
