# USV Detection App - Implementation Plan

## Overview

Build a desktop application for USV detection and visualization, similar to Audacity/DeepSqueak. The app loads full WAV files, displays spectrograms, runs CNN inference with sliding window, and allows interactive threshold adjustment for USV boundary detection.

**Technology:** PyQt6 (desktop application framework)

**Reference documents:**
- `usv_signal_processing_reference.md` — Signal processing background
- `CNN_IMPLEMENTATION_INSTRUCTIONS.md` — CNN model details

---

## App Interface Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  File   Edit   View   Help                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Open WAV]  [Run Detection]  [Save Labels]  [Export Image]                  │
├───────────────────────────────────────────────────────────────────────┬─────┤
│                                                                       │     │
│  ┌─────────────────────────────────────────────────────────────────┐ │  T  │
│  │                                                                 │ │  H  │
│  │                    SPECTROGRAM VIEW                             │ │  R  │
│  │                                                                 │ │  E  │
│  │         |                               |                       │ │  S  │
│  │         |  USV 1                        |  USV 2                 │ │  H  │
│  │         |                               |                       │ │  O  │
│  │      (vertical lines at USV boundaries)                         │ │  L  │
│  │                                                                 │ │  D  │
│  └─────────────────────────────────────────────────────────────────┘ │     │
│                                                                       │ [=] │
│  ┌─────────────────────────────────────────────────────────────────┐ │  │  │
│  │                                                                 │ │  │  │
│  │  ▁▂▃▅███▅▃▂▁▁▁▂▄███████▄▂▁▁▁▁▂▃▅██▅▃▁▁▁▂▅████▅▂▁               │ │  │  │
│  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ threshold line ─ ─ ─ ─ ─  │ │  │  │
│  │                    PROBABILITY CURVE                            │ │  │  │
│  │                                                                 │ │  │  │
│  └─────────────────────────────────────────────────────────────────┘ │  │  │
│                                                                       │ 0.0 │
│  ◄━━━━━━━━━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━► │     │
│                    TIME SLIDER                                        │     │
├───────────────────────────────────────────────────────────────────────┴─────┤
│  Time: 00:42.350 / 03:24.100  │  Threshold: 0.65  │  USVs detected: 47      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Elements

1. **Spectrogram View** — Shows portion of spectrogram currently in view, with vertical lines at detected USV start/end points
2. **Probability Curve** — Shows CNN output probability over time, synced with spectrogram view
3. **Time Slider (horizontal)** — Scrub through entire recording
4. **Threshold Slider (vertical)** — Adjust detection threshold (0.0 to 1.0)
5. **Status Bar** — Current time, threshold value, USV count
6. **Toolbar** — Open, Run Detection, Save, Export

---

## Architecture

```
src/usv_spectrogram/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Application entry point
│   ├── main_window.py             # Main window layout
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── spectrogram_view.py    # Spectrogram display widget
│   │   ├── probability_view.py    # Probability curve widget
│   │   ├── time_slider.py         # Horizontal time navigation
│   │   └── threshold_slider.py    # Vertical threshold control
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio_loader.py        # WAV loading and spectrogram computation
│   │   ├── sliding_inference.py   # CNN sliding window inference
│   │   ├── detection_logic.py     # Hysteresis-based USV detection
│   │   └── label_storage.py       # Save/load labels and state
│   └── utils/
│       ├── __init__.py
│       └── drawing.py             # Draw vertical lines on spectrogram
scripts/
└── run_app.py                     # Launch script
```

---

## Implementation Phases

### Phase 1: Core Infrastructure

#### 1.1 Audio Loader and Full Spectrogram Generation

**File:** `src/usv_spectrogram/app/core/audio_loader.py`

```python
"""
Load WAV files and compute full spectrograms.

The spectrogram is computed once when file is loaded and cached.
Only the visible portion is rendered to the screen.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
import scipy.io.wavfile as wav
import scipy.signal as signal


@dataclass
class AudioData:
    """Container for loaded audio and its spectrogram."""
    filepath: Path
    sample_rate: int
    duration_seconds: float
    
    # Raw audio (may be large - consider memory mapping for very long files)
    audio: np.ndarray
    
    # Full spectrogram (frequency x time)
    spectrogram_db: np.ndarray
    
    # Time and frequency axes
    times: np.ndarray      # Time in seconds for each spectrogram column
    frequencies: np.ndarray  # Frequency in Hz for each spectrogram row
    
    # Spectrogram parameters used (for reproducibility)
    n_fft: int
    hop_length: int
    

class AudioLoader:
    """Load audio and compute spectrograms."""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        freq_min_hz: int = 20000,
        freq_max_hz: int = 120000
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_min_hz = freq_min_hz
        self.freq_max_hz = freq_max_hz
    
    def load(self, wav_path: Path) -> AudioData:
        """
        Load WAV file and compute full spectrogram.
        
        This may take a few seconds for long files.
        Consider showing a progress bar in the UI.
        """
        # 1. Load audio
        sample_rate, audio = wav.read(wav_path)
        
        # 2. Convert to float if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # 3. Compute spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            audio,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            scaling='spectrum'
        )
        
        # 4. Convert to dB
        spectrogram_db = 10 * np.log10(spectrogram + 1e-10)
        
        # 5. Crop to frequency range of interest
        freq_mask = (frequencies >= self.freq_min_hz) & (frequencies <= self.freq_max_hz)
        spectrogram_db = spectrogram_db[freq_mask, :]
        frequencies = frequencies[freq_mask]
        
        # 6. Normalize for display (dynamic range based on this file)
        vmin = np.mean(spectrogram_db) - 2 * np.std(spectrogram_db)
        vmax = np.mean(spectrogram_db) + 3 * np.std(spectrogram_db)
        spectrogram_db = np.clip(spectrogram_db, vmin, vmax)
        spectrogram_db = (spectrogram_db - vmin) / (vmax - vmin)  # Normalize to [0, 1]
        
        return AudioData(
            filepath=wav_path,
            sample_rate=sample_rate,
            duration_seconds=len(audio) / sample_rate,
            audio=audio,
            spectrogram_db=spectrogram_db,
            times=times,
            frequencies=frequencies,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def get_time_for_column(self, audio_data: AudioData, column_index: int) -> float:
        """Convert spectrogram column index to time in seconds."""
        return audio_data.times[column_index]
    
    def get_column_for_time(self, audio_data: AudioData, time_seconds: float) -> int:
        """Convert time in seconds to spectrogram column index."""
        return np.searchsorted(audio_data.times, time_seconds)
```

#### 1.2 Sliding Window CNN Inference

**File:** `src/usv_spectrogram/app/core/sliding_inference.py`

```python
"""
Run CNN inference with sliding window across full spectrogram.

Output: probability value for each position (every N pixels).
"""

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass

# Import your existing CNN model
from usv_spectrogram.models.cnn_classifier import USVClassifierCNN


@dataclass
class InferenceResult:
    """Container for sliding window inference results."""
    
    # Probability at each sampled position
    probabilities: np.ndarray  # Shape: (num_positions,)
    
    # Column indices corresponding to each probability
    column_indices: np.ndarray  # Shape: (num_positions,)
    
    # Time in seconds for each position
    times: np.ndarray  # Shape: (num_positions,)
    
    # Parameters used
    window_width_pixels: int
    hop_pixels: int


class SlidingInference:
    """
    Run CNN with sliding window across spectrogram.
    """
    
    def __init__(
        self,
        model_path: Path,
        window_width_pixels: int = 150,  # Match training window size
        hop_pixels: int = 10,            # 10-pixel hop as discussed
        device: str = 'auto'
    ):
        self.window_width_pixels = window_width_pixels
        self.hop_pixels = hop_pixels
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path: Path) -> USVClassifierCNN:
        """Load trained CNN model."""
        model = USVClassifierCNN()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model
    
    def run(self, audio_data, progress_callback=None) -> InferenceResult:
        """
        Run sliding window inference on full spectrogram.
        
        Args:
            audio_data: AudioData object with spectrogram_db
            progress_callback: Optional function(percent) for progress updates
        
        Returns:
            InferenceResult with probabilities for each position
        """
        spectrogram = audio_data.spectrogram_db
        num_columns = spectrogram.shape[1]
        
        # Calculate positions to sample
        positions = list(range(0, num_columns - self.window_width_pixels, self.hop_pixels))
        num_positions = len(positions)
        
        probabilities = np.zeros(num_positions)
        
        # Batch inference for efficiency
        batch_size = 32
        
        with torch.no_grad():
            for batch_start in range(0, num_positions, batch_size):
                batch_end = min(batch_start + batch_size, num_positions)
                batch_positions = positions[batch_start:batch_end]
                
                # Extract windows
                windows = []
                for pos in batch_positions:
                    window = spectrogram[:, pos:pos + self.window_width_pixels]
                    windows.append(window)
                
                # Stack into batch tensor
                batch = np.stack(windows)
                batch = torch.from_numpy(batch).float().unsqueeze(1)  # Add channel dim
                batch = batch.to(self.device)
                
                # Run inference
                outputs = self.model(batch).squeeze().cpu().numpy()
                
                # Handle single-item batch
                if batch_end - batch_start == 1:
                    outputs = np.array([outputs])
                
                probabilities[batch_start:batch_end] = outputs
                
                # Progress callback
                if progress_callback:
                    progress_callback(int(100 * batch_end / num_positions))
        
        # Convert positions to times
        column_indices = np.array(positions) + self.window_width_pixels // 2  # Center of window
        times = audio_data.times[column_indices]
        
        return InferenceResult(
            probabilities=probabilities,
            column_indices=column_indices,
            times=times,
            window_width_pixels=self.window_width_pixels,
            hop_pixels=self.hop_pixels
        )
    
    def interpolate_to_full_resolution(
        self, 
        inference_result: InferenceResult, 
        num_columns: int
    ) -> np.ndarray:
        """
        Interpolate sparse probabilities to full spectrogram resolution.
        
        Returns array of shape (num_columns,) with probability for each column.
        """
        return np.interp(
            np.arange(num_columns),
            inference_result.column_indices,
            inference_result.probabilities
        )
```

#### 1.3 Hysteresis-Based Detection Logic

**File:** `src/usv_spectrogram/app/core/detection_logic.py`

```python
"""
Detect USV boundaries using hysteresis thresholding.

Hysteresis prevents flickering: 
- USV starts when probability rises above HIGH threshold
- USV ends when probability falls below LOW threshold
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DetectedUSV:
    """A single detected USV."""
    start_column: int      # Spectrogram column where USV starts
    end_column: int        # Spectrogram column where USV ends
    start_time: float      # Time in seconds
    end_time: float        # Time in seconds
    duration_ms: float     # Duration in milliseconds
    peak_probability: float  # Maximum probability within USV
    peak_column: int       # Column with peak probability


@dataclass 
class DetectionResult:
    """Container for all detected USVs."""
    usv_list: list[DetectedUSV]
    threshold_high: float
    threshold_low: float
    total_count: int
    
    def to_dict_list(self) -> list[dict]:
        """Convert to list of dicts for saving."""
        return [
            {
                'start_column': usv.start_column,
                'end_column': usv.end_column,
                'start_time': usv.start_time,
                'end_time': usv.end_time,
                'duration_ms': usv.duration_ms,
                'peak_probability': usv.peak_probability,
                'peak_column': usv.peak_column
            }
            for usv in self.usv_list
        ]


class HysteresisDetector:
    """
    Detect USVs using hysteresis thresholding.
    
    This is similar to the DeepSqueak approach:
    - USV starts when P > threshold_high
    - USV ends when P < threshold_low
    - threshold_low < threshold_high prevents rapid on/off flickering
    """
    
    def __init__(
        self,
        threshold_high: float = 0.6,
        threshold_low: float = 0.4,
        min_duration_columns: int = 5,  # Minimum USV length in spectrogram columns
        max_gap_columns: int = 3        # Max gap to merge adjacent detections
    ):
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.min_duration_columns = min_duration_columns
        self.max_gap_columns = max_gap_columns
    
    def detect(
        self, 
        probabilities: np.ndarray, 
        times: np.ndarray
    ) -> DetectionResult:
        """
        Detect USVs from probability curve.
        
        Args:
            probabilities: Array of probabilities (one per spectrogram column or interpolated)
            times: Array of times corresponding to each probability
        
        Returns:
            DetectionResult with list of detected USVs
        """
        usv_list = []
        
        in_usv = False
        usv_start = None
        usv_probs = []
        
        for i, prob in enumerate(probabilities):
            if not in_usv:
                # Looking for USV start
                if prob >= self.threshold_high:
                    in_usv = True
                    usv_start = i
                    usv_probs = [prob]
            else:
                # In USV, looking for end
                usv_probs.append(prob)
                if prob < self.threshold_low:
                    # USV ended
                    usv_end = i
                    
                    # Check minimum duration
                    if usv_end - usv_start >= self.min_duration_columns:
                        peak_prob = max(usv_probs)
                        peak_idx = usv_start + usv_probs.index(peak_prob)
                        
                        usv = DetectedUSV(
                            start_column=usv_start,
                            end_column=usv_end,
                            start_time=times[usv_start],
                            end_time=times[usv_end],
                            duration_ms=(times[usv_end] - times[usv_start]) * 1000,
                            peak_probability=peak_prob,
                            peak_column=peak_idx
                        )
                        usv_list.append(usv)
                    
                    in_usv = False
                    usv_start = None
                    usv_probs = []
        
        # Handle USV that extends to end of file
        if in_usv and len(probabilities) - usv_start >= self.min_duration_columns:
            usv_end = len(probabilities) - 1
            peak_prob = max(usv_probs)
            peak_idx = usv_start + usv_probs.index(peak_prob)
            
            usv = DetectedUSV(
                start_column=usv_start,
                end_column=usv_end,
                start_time=times[usv_start],
                end_time=times[usv_end],
                duration_ms=(times[usv_end] - times[usv_start]) * 1000,
                peak_probability=peak_prob,
                peak_column=peak_idx
            )
            usv_list.append(usv)
        
        # Merge nearby detections
        usv_list = self._merge_nearby(usv_list, times)
        
        return DetectionResult(
            usv_list=usv_list,
            threshold_high=self.threshold_high,
            threshold_low=self.threshold_low,
            total_count=len(usv_list)
        )
    
    def _merge_nearby(
        self, 
        usv_list: list[DetectedUSV], 
        times: np.ndarray
    ) -> list[DetectedUSV]:
        """Merge USVs that are separated by small gaps."""
        if len(usv_list) < 2:
            return usv_list
        
        merged = [usv_list[0]]
        
        for usv in usv_list[1:]:
            prev = merged[-1]
            gap = usv.start_column - prev.end_column
            
            if gap <= self.max_gap_columns:
                # Merge with previous
                merged[-1] = DetectedUSV(
                    start_column=prev.start_column,
                    end_column=usv.end_column,
                    start_time=prev.start_time,
                    end_time=usv.end_time,
                    duration_ms=(usv.end_time - prev.start_time) * 1000,
                    peak_probability=max(prev.peak_probability, usv.peak_probability),
                    peak_column=prev.peak_column if prev.peak_probability > usv.peak_probability else usv.peak_column
                )
            else:
                merged.append(usv)
        
        return merged
    
    def update_thresholds(self, threshold_high: float, threshold_low: float = None):
        """
        Update thresholds for re-detection.
        
        If threshold_low not specified, set it to 0.8 * threshold_high.
        """
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low if threshold_low else 0.8 * threshold_high
```

#### 1.4 Label Storage

**File:** `src/usv_spectrogram/app/core/label_storage.py`

```python
"""
Save and load detection results, thresholds, and app state.

Saves:
- Detected USVs (start/end times, probabilities)
- Threshold values used
- Source file information
- Spectrogram image with overlays (optional export)
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt


class LabelStorage:
    """Save and load USV detection labels."""
    
    def save_labels(
        self,
        filepath: Path,
        source_wav: Path,
        detection_result,  # DetectionResult
        inference_result,  # InferenceResult (optional, for probability curve)
        threshold_high: float,
        threshold_low: float,
        notes: str = ""
    ):
        """
        Save detection labels to JSON file.
        """
        data = {
            'metadata': {
                'source_wav': str(source_wav),
                'created_at': datetime.now().isoformat(),
                'threshold_high': threshold_high,
                'threshold_low': threshold_low,
                'total_usvs': detection_result.total_count,
                'notes': notes
            },
            'usvs': detection_result.to_dict_list(),
            'probability_curve': {
                'values': inference_result.probabilities.tolist() if inference_result else None,
                'times': inference_result.times.tolist() if inference_result else None,
                'hop_pixels': inference_result.hop_pixels if inference_result else None
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_labels(self, filepath: Path) -> dict:
        """
        Load detection labels from JSON file.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def export_image(
        self,
        filepath: Path,
        spectrogram_db: np.ndarray,
        detection_result,
        times: np.ndarray,
        frequencies: np.ndarray,
        probabilities: np.ndarray = None
    ):
        """
        Export spectrogram image with USV boundaries marked.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])
        
        # Spectrogram
        ax1 = axes[0]
        ax1.imshow(
            spectrogram_db,
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], frequencies[0]/1000, frequencies[-1]/1000],
            cmap='magma'
        )
        
        # Draw USV boundaries as vertical lines
        for usv in detection_result.usv_list:
            ax1.axvline(x=usv.start_time, color='green', linewidth=1, alpha=0.8)
            ax1.axvline(x=usv.end_time, color='red', linewidth=1, alpha=0.8)
        
        ax1.set_ylabel('Frequency (kHz)')
        ax1.set_title(f'USV Detection - {detection_result.total_count} USVs found')
        
        # Probability curve
        if probabilities is not None:
            ax2 = axes[1]
            ax2.plot(times[:len(probabilities)], probabilities, 'b-', linewidth=0.5)
            ax2.axhline(y=detection_result.threshold_high, color='g', linestyle='--', label='High threshold')
            ax2.axhline(y=detection_result.threshold_low, color='r', linestyle='--', label='Low threshold')
            ax2.set_ylabel('P(USV)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylim(0, 1)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
```

---

### Phase 2: PyQt6 UI Components

#### 2.1 Main Window

**File:** `src/usv_spectrogram/app/main_window.py`

```python
"""
Main application window.

Layout:
- Menu bar (File, Edit, View, Help)
- Toolbar (Open, Run Detection, Save, Export)
- Central widget with spectrogram view, probability view, sliders
- Status bar
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QStatusBar, QFileDialog, QSlider, QLabel,
    QProgressDialog, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from pathlib import Path

from .widgets.spectrogram_view import SpectrogramView
from .widgets.probability_view import ProbabilityView
from .core.audio_loader import AudioLoader
from .core.sliding_inference import SlidingInference
from .core.detection_logic import HysteresisDetector
from .core.label_storage import LabelStorage


class InferenceWorker(QThread):
    """Background thread for CNN inference."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    
    def __init__(self, inference_engine, audio_data):
        super().__init__()
        self.inference_engine = inference_engine
        self.audio_data = audio_data
    
    def run(self):
        result = self.inference_engine.run(
            self.audio_data,
            progress_callback=self.progress.emit
        )
        self.finished.emit(result)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, model_path: Path):
        super().__init__()
        
        self.model_path = model_path
        self.audio_data = None
        self.inference_result = None
        self.detection_result = None
        self.full_probabilities = None
        
        # Core components
        self.audio_loader = AudioLoader()
        self.inference_engine = SlidingInference(model_path)
        self.detector = HysteresisDetector()
        self.label_storage = LabelStorage()
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("USV Detection App")
        self.setMinimumSize(1200, 800)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: spectrogram and probability views with time slider
        left_layout = QVBoxLayout()
        
        # Splitter for spectrogram and probability views
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.spectrogram_view = SpectrogramView()
        self.probability_view = ProbabilityView()
        
        splitter.addWidget(self.spectrogram_view)
        splitter.addWidget(self.probability_view)
        splitter.setSizes([600, 200])  # Initial sizes
        
        left_layout.addWidget(splitter)
        
        # Time slider
        time_slider_layout = QHBoxLayout()
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)  # Will be updated when file loads
        self.time_slider.valueChanged.connect(self._on_time_slider_changed)
        
        self.time_label = QLabel("00:00.000 / 00:00.000")
        
        time_slider_layout.addWidget(self.time_slider)
        time_slider_layout.addWidget(self.time_label)
        
        left_layout.addLayout(time_slider_layout)
        
        main_layout.addLayout(left_layout, stretch=1)
        
        # Right side: threshold slider
        right_layout = QVBoxLayout()
        
        threshold_label = QLabel("Threshold")
        threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Vertical)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(60)  # Default 0.6
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        right_layout.addWidget(self.threshold_slider)
        
        self.threshold_value_label = QLabel("0.60")
        self.threshold_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.threshold_value_label)
        
        main_layout.addLayout(right_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Open a WAV file to begin")
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        open_action = QAction("&Open WAV...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Labels...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_labels)
        file_menu.addAction(save_action)
        
        export_action = QAction("&Export Image...", self)
        export_action.triggered.connect(self._export_image)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
    
    def _create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        open_action = QAction("Open WAV", self)
        open_action.triggered.connect(self._open_file)
        toolbar.addAction(open_action)
        
        self.run_action = QAction("Run Detection", self)
        self.run_action.triggered.connect(self._run_detection)
        self.run_action.setEnabled(False)
        toolbar.addAction(self.run_action)
        
        toolbar.addSeparator()
        
        self.save_action = QAction("Save Labels", self)
        self.save_action.triggered.connect(self._save_labels)
        self.save_action.setEnabled(False)
        toolbar.addAction(self.save_action)
        
        self.export_action = QAction("Export Image", self)
        self.export_action.triggered.connect(self._export_image)
        self.export_action.setEnabled(False)
        toolbar.addAction(self.export_action)
    
    def _open_file(self):
        """Open WAV file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open WAV File",
            "",
            "WAV Files (*.wav);;All Files (*)"
        )
        
        if filepath:
            self._load_audio(Path(filepath))
    
    def _load_audio(self, filepath: Path):
        """Load audio file and display spectrogram."""
        self.status_bar.showMessage(f"Loading {filepath.name}...")
        
        try:
            self.audio_data = self.audio_loader.load(filepath)
            
            # Update spectrogram view
            self.spectrogram_view.set_spectrogram(
                self.audio_data.spectrogram_db,
                self.audio_data.times,
                self.audio_data.frequencies
            )
            
            # Update time slider
            self.time_slider.setMaximum(int(self.audio_data.duration_seconds * 1000))
            self._update_time_label()
            
            # Enable run detection
            self.run_action.setEnabled(True)
            
            # Clear previous results
            self.inference_result = None
            self.detection_result = None
            self.probability_view.clear()
            
            self.status_bar.showMessage(
                f"Loaded: {filepath.name} ({self.audio_data.duration_seconds:.1f}s)"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
    
    def _run_detection(self):
        """Run CNN inference on loaded audio."""
        if self.audio_data is None:
            return
        
        # Show progress dialog
        progress = QProgressDialog("Running detection...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        # Run inference in background thread
        self.worker = InferenceWorker(self.inference_engine, self.audio_data)
        self.worker.progress.connect(progress.setValue)
        self.worker.finished.connect(lambda result: self._on_inference_complete(result, progress))
        self.worker.start()
    
    def _on_inference_complete(self, result, progress):
        """Handle completed inference."""
        progress.close()
        
        self.inference_result = result
        
        # Interpolate to full resolution
        self.full_probabilities = self.inference_engine.interpolate_to_full_resolution(
            result, 
            self.audio_data.spectrogram_db.shape[1]
        )
        
        # Run detection with current threshold
        self._update_detection()
        
        # Update probability view
        self.probability_view.set_probabilities(
            self.full_probabilities,
            self.audio_data.times,
            self.detector.threshold_high,
            self.detector.threshold_low
        )
        
        # Enable save/export
        self.save_action.setEnabled(True)
        self.export_action.setEnabled(True)
        
        self.status_bar.showMessage(
            f"Detection complete: {self.detection_result.total_count} USVs found"
        )
    
    def _update_detection(self):
        """Re-run detection with current threshold."""
        if self.full_probabilities is None:
            return
        
        self.detection_result = self.detector.detect(
            self.full_probabilities,
            self.audio_data.times
        )
        
        # Update spectrogram view with USV boundaries
        self.spectrogram_view.set_usv_boundaries(self.detection_result.usv_list)
        
        # Update probability view threshold lines
        self.probability_view.set_thresholds(
            self.detector.threshold_high,
            self.detector.threshold_low
        )
        
        self.status_bar.showMessage(
            f"USVs detected: {self.detection_result.total_count} | "
            f"Threshold: {self.detector.threshold_high:.2f}"
        )
    
    def _on_threshold_changed(self, value):
        """Handle threshold slider change."""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        
        # Update detector thresholds
        self.detector.update_thresholds(threshold)
        
        # Re-run detection
        self._update_detection()
    
    def _on_time_slider_changed(self, value):
        """Handle time slider change."""
        time_seconds = value / 1000.0
        self._update_time_label()
        
        # Scroll spectrogram and probability views to this time
        self.spectrogram_view.scroll_to_time(time_seconds)
        self.probability_view.scroll_to_time(time_seconds)
    
    def _update_time_label(self):
        """Update time display label."""
        current = self.time_slider.value() / 1000.0
        total = self.audio_data.duration_seconds if self.audio_data else 0
        
        current_str = f"{int(current // 60):02d}:{current % 60:06.3f}"
        total_str = f"{int(total // 60):02d}:{total % 60:06.3f}"
        
        self.time_label.setText(f"{current_str} / {total_str}")
    
    def _save_labels(self):
        """Save detection labels."""
        if self.detection_result is None:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Labels",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            self.label_storage.save_labels(
                Path(filepath),
                self.audio_data.filepath,
                self.detection_result,
                self.inference_result,
                self.detector.threshold_high,
                self.detector.threshold_low
            )
            self.status_bar.showMessage(f"Labels saved to {filepath}")
    
    def _export_image(self):
        """Export spectrogram image with annotations."""
        if self.detection_result is None:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            "",
            "PNG Files (*.png);;All Files (*)"
        )
        
        if filepath:
            self.label_storage.export_image(
                Path(filepath),
                self.audio_data.spectrogram_db,
                self.detection_result,
                self.audio_data.times,
                self.audio_data.frequencies,
                self.full_probabilities
            )
            self.status_bar.showMessage(f"Image exported to {filepath}")
```

#### 2.2 Spectrogram View Widget

**File:** `src/usv_spectrogram/app/widgets/spectrogram_view.py`

```python
"""
Spectrogram display widget.

Features:
- Displays portion of spectrogram currently in view
- Draws vertical lines at USV start (green) and end (red) points
- Syncs with time slider for navigation
- Efficient rendering using QPixmap caching
"""

from PyQt6.QtWidgets import QWidget, QScrollArea, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPixmap, QPen, QColor
import numpy as np


class SpectrogramCanvas(QWidget):
    """Canvas that actually draws the spectrogram."""
    
    def __init__(self):
        super().__init__()
        
        self.spectrogram_pixmap = None
        self.usv_boundaries = []
        self.times = None
        self.view_start_time = 0
        self.view_end_time = 10
        self.pixels_per_second = 100
    
    def set_spectrogram(self, spectrogram_db: np.ndarray, times: np.ndarray, frequencies: np.ndarray):
        """Set spectrogram data and create pixmap."""
        self.times = times
        
        # Convert normalized spectrogram to image
        # spectrogram_db should be [0, 1] normalized
        img_data = (spectrogram_db * 255).astype(np.uint8)
        
        # Apply colormap (magma-like)
        # For simplicity, using grayscale here - can enhance with proper colormap
        img_data = np.flip(img_data, axis=0)  # Flip so low freq at bottom
        
        height, width = img_data.shape
        
        # Create QImage
        bytes_per_line = width
        image = QImage(img_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        # Scale to reasonable display size
        self.spectrogram_pixmap = QPixmap.fromImage(image)
        
        # Set widget size based on spectrogram
        self.pixels_per_second = width / (times[-1] - times[0])
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)
        
        self.update()
    
    def set_usv_boundaries(self, usv_list):
        """Set USV boundaries to draw."""
        self.usv_boundaries = usv_list
        self.update()
    
    def paintEvent(self, event):
        """Paint the spectrogram and USV boundaries."""
        if self.spectrogram_pixmap is None:
            return
        
        painter = QPainter(self)
        
        # Draw spectrogram
        painter.drawPixmap(0, 0, self.spectrogram_pixmap)
        
        # Draw USV boundaries
        if self.times is not None:
            for usv in self.usv_boundaries:
                # Start line (green)
                start_x = int(usv.start_column)
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawLine(start_x, 0, start_x, self.height())
                
                # End line (red)
                end_x = int(usv.end_column)
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(end_x, 0, end_x, self.height())
        
        painter.end()


class SpectrogramView(QScrollArea):
    """Scrollable spectrogram view."""
    
    time_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        
        self.canvas = SpectrogramCanvas()
        self.setWidget(self.canvas)
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    
    def set_spectrogram(self, spectrogram_db, times, frequencies):
        """Set spectrogram data."""
        self.canvas.set_spectrogram(spectrogram_db, times, frequencies)
    
    def set_usv_boundaries(self, usv_list):
        """Set USV boundaries to draw."""
        self.canvas.set_usv_boundaries(usv_list)
    
    def scroll_to_time(self, time_seconds):
        """Scroll view to center on given time."""
        if self.canvas.times is None:
            return
        
        # Convert time to pixel position
        x = int(time_seconds * self.canvas.pixels_per_second)
        
        # Center in view
        viewport_width = self.viewport().width()
        scroll_x = x - viewport_width // 2
        
        self.horizontalScrollBar().setValue(max(0, scroll_x))
```

#### 2.3 Probability View Widget

**File:** `src/usv_spectrogram/app/widgets/probability_view.py`

```python
"""
Probability curve display widget.

Shows CNN output probability over time with threshold lines.
Synced with spectrogram view for navigation.
"""

from PyQt6.QtWidgets import QWidget, QScrollArea
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
import numpy as np


class ProbabilityCanvas(QWidget):
    """Canvas for probability curve."""
    
    def __init__(self):
        super().__init__()
        
        self.probabilities = None
        self.times = None
        self.threshold_high = 0.6
        self.threshold_low = 0.4
        self.pixels_per_second = 100
        
        self.setMinimumHeight(100)
    
    def set_probabilities(self, probabilities, times, threshold_high, threshold_low):
        """Set probability data."""
        self.probabilities = probabilities
        self.times = times
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        
        # Match width to spectrogram
        width = len(probabilities)
        self.pixels_per_second = width / (times[-1] - times[0])
        self.setMinimumSize(width, 100)
        self.setMaximumSize(width, 200)
        
        self.update()
    
    def set_thresholds(self, threshold_high, threshold_low):
        """Update threshold lines."""
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.update()
    
    def clear(self):
        """Clear display."""
        self.probabilities = None
        self.update()
    
    def paintEvent(self, event):
        """Paint probability curve and thresholds."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))  # Dark background
        
        if self.probabilities is None:
            painter.end()
            return
        
        height = self.height()
        width = len(self.probabilities)
        
        # Draw threshold lines
        # High threshold (green dashed)
        painter.setPen(QPen(QColor(0, 255, 0), 1, Qt.PenStyle.DashLine))
        y_high = int(height * (1 - self.threshold_high))
        painter.drawLine(0, y_high, width, y_high)
        
        # Low threshold (red dashed)
        painter.setPen(QPen(QColor(255, 0, 0), 1, Qt.PenStyle.DashLine))
        y_low = int(height * (1 - self.threshold_low))
        painter.drawLine(0, y_low, width, y_low)
        
        # Draw probability curve
        painter.setPen(QPen(QColor(100, 150, 255), 1))
        
        path = QPainterPath()
        for i, prob in enumerate(self.probabilities):
            y = height * (1 - prob)  # Flip so 1.0 is at top
            if i == 0:
                path.moveTo(i, y)
            else:
                path.lineTo(i, y)
        
        painter.drawPath(path)
        painter.end()


class ProbabilityView(QScrollArea):
    """Scrollable probability view."""
    
    def __init__(self):
        super().__init__()
        
        self.canvas = ProbabilityCanvas()
        self.setWidget(self.canvas)
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    
    def set_probabilities(self, probabilities, times, threshold_high, threshold_low):
        """Set probability data."""
        self.canvas.set_probabilities(probabilities, times, threshold_high, threshold_low)
    
    def set_thresholds(self, threshold_high, threshold_low):
        """Update threshold lines."""
        self.canvas.set_thresholds(threshold_high, threshold_low)
    
    def scroll_to_time(self, time_seconds):
        """Scroll to given time."""
        if self.canvas.times is None:
            return
        
        x = int(time_seconds * self.canvas.pixels_per_second)
        viewport_width = self.viewport().width()
        scroll_x = x - viewport_width // 2
        
        self.horizontalScrollBar().setValue(max(0, scroll_x))
    
    def clear(self):
        """Clear display."""
        self.canvas.clear()
```

#### 2.4 Application Entry Point

**File:** `src/usv_spectrogram/app/main.py`

```python
"""
Application entry point.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from .main_window import MainWindow


def main():
    # Default model path - can be overridden with command line arg
    # Using latest trained model from Jan 24, 2026
    model_path = Path("checkpoints/best_model.pt")

    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Usage: python -m usv_spectrogram.app.main [model_path]")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setApplicationName("USV Detection App")
    
    window = MainWindow(model_path)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

**File:** `scripts/run_app.py`

```python
"""Launch script for USV Detection App."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.app.main import main

if __name__ == "__main__":
    main()
```

---

### Phase 3: Enhancements

After basic functionality works, add these improvements:

#### 3.1 Colormap for Spectrogram

Replace grayscale with magma colormap using matplotlib's colormap:

```python
from matplotlib import cm

def apply_colormap(spectrogram_normalized):
    """Apply magma colormap to normalized spectrogram."""
    colormap = cm.get_cmap('magma')
    colored = colormap(spectrogram_normalized)
    return (colored[:, :, :3] * 255).astype(np.uint8)
```

#### 3.2 Keyboard Shortcuts

Add in MainWindow:

```python
def _setup_shortcuts(self):
    """Setup keyboard shortcuts."""
    # Space: play/pause audio (future feature)
    # Left/Right arrows: move time slider
    # Up/Down arrows: adjust threshold
    # S: save labels
    # E: export image
```

#### 3.3 Sync Scrolling

Link spectrogram and probability view scroll positions:

```python
def _sync_scroll(self):
    """Sync scroll position between views."""
    self.spectrogram_view.horizontalScrollBar().valueChanged.connect(
        self.probability_view.horizontalScrollBar().setValue
    )
    self.probability_view.horizontalScrollBar().valueChanged.connect(
        self.spectrogram_view.horizontalScrollBar().setValue
    )
```

#### 3.4 Audio Playback (Optional Future Feature)

```python
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# Pitch-shifted playback for audible review of USVs
```

---

## Dependencies

Add to requirements.txt:

```
PyQt6>=6.4.0
numpy
scipy
torch
matplotlib
```

---

## Implementation Order

1. **Phase 1.1:** Audio loader - can load WAV and compute spectrogram
2. **Phase 1.2:** Sliding inference - can run CNN across spectrogram
3. **Phase 1.3:** Detection logic - hysteresis thresholding works
4. **Phase 1.4:** Label storage - can save/load JSON
5. **Phase 2.1:** Main window skeleton - opens, has layout
6. **Phase 2.2:** Spectrogram view - displays spectrogram, scrollable
7. **Phase 2.3:** Probability view - displays curve, threshold lines
8. **Phase 2.4:** Wire everything together - sliders work, detection updates
9. **Phase 3:** Enhancements - colormap, shortcuts, sync scrolling

---

## Testing Commands

```powershell
# Install PyQt6
pip install PyQt6

# Run the app (uses latest trained model from Jan 24, 2026)
python scripts/run_app.py checkpoints/best_model.pt

# Or as module
python -m usv_spectrogram.app.main checkpoints/best_model.pt

# Or let it use default path
python scripts/run_app.py
```

---

## Deliverables

When complete, the app should:

1. ✓ Load any WAV file and display full spectrogram
2. ✓ Run CNN inference with progress indicator
3. ✓ Display probability curve below spectrogram
4. ✓ Allow threshold adjustment with immediate visual feedback
5. ✓ Draw vertical lines at USV start (green) and end (red)
6. ✓ Scroll through recording with time slider
7. ✓ Save labels (USV times, threshold, probabilities) to JSON
8. ✓ Export annotated spectrogram image
