# Architecture Patterns

Reusable patterns established across the USV Detection codebase.
Reference these when building new modules — consistency matters more than novelty.

---

## 1. Config Dataclass Pattern

All configurable modules use frozen dataclasses with defaults and `__post_init__` validation.

**Key files:**
- `src/usv_spectrogram/config.py` — `SpectrogramConfig`
- `src/usv_spectrogram/detection/config.py` — `DetectionConfig`
- `src/usv_spectrogram/detection/extraction_config.py` — `ExtractionConfig`
- `src/usv_spectrogram/models/config.py` — `TrainingConfig`

**Example** (from `detection/config.py`):

```python
@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for energy-based USV candidate detection."""

    # STFT parameters
    sample_rate: int = 300_000  # Must be >= 2 * max_freq (Nyquist)
    n_fft: int = 512            # ~586 Hz freq resolution at 300 kHz
    hop_length: int = 128       # 75% overlap

    # Frequency band
    freq_min_hz: int = 25_000
    freq_max_hz: int = 110_000

    # Energy threshold — deliberately LOW for high recall
    energy_threshold_db: float = -60.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError("freq_min_hz must be < freq_max_hz")

    def hop_ms(self) -> float:
        return (self.hop_length / self.sample_rate) * 1000.0
```

**Rules:**
- `frozen=True` — immutable after creation
- Sensible defaults for every field
- Numeric field suffixes encode units: `_hz`, `_ms`, `_db`, `_px`, `_s`, `_windows`
- `__post_init__` validates interdependent constraints
- Convenience methods for derived values (e.g. `hop_ms()`, `freq_resolution_hz()`)

---

## 2. Candidate Data Flow

Candidates flow through the pipeline as dataclass objects with full traceability metadata:

```
WAV file
  -> load_wav_mono()           -> numpy array (float32, mono)
  -> EnergyDetector.detect()   -> list[Candidate]
  -> SpectrogramExtractor      -> PNG spectrogram patches
  -> CNN.predict_proba()       -> probability float [0, 1]
  -> LabelStorage.save()       -> JSON file with metadata
```

**Key classes:**

| Class | File | Role |
|-------|------|------|
| `Candidate` | `detection/candidate.py` | Detected segment with timing + spectral metadata |
| `DetectedUSV` | `app/core/detection_logic.py` | CNN-scored detection with user interaction tracking |
| `DetectionResult` | `app/core/detection_logic.py` | Container for all detections from one file |

**Rules:**
- `Candidate` uses milliseconds for timing; `DetectedUSV` uses seconds
- Auto-generate candidate IDs as `{source_stem}_{start_ms:08.0f}`
- Always include source file path for traceability
- Use factory methods (`Candidate.create()`) to auto-compute derived fields

---

## 3. Test Fixture Pattern

Tests use synthetic WAV data (never real recordings), `yield` for cleanup, and factory fixtures for parameterized creation.

**Key file:** `tests/conftest.py`

**Simple fixture with cleanup:**

```python
@pytest.fixture
def sample_wav_path() -> Path:
    """Create a temporary WAV with a synthetic USV-like signal."""
    sample_rate_hz = 250_000
    duration_s = 0.1
    n_samples = int(sample_rate_hz * duration_s)

    t = np.arange(n_samples) / sample_rate_hz
    noise = 0.01 * np.random.randn(n_samples).astype(np.float32)
    tone = 0.1 * np.sin(2.0 * np.pi * 60_000.0 * t).astype(np.float32)
    signal = noise + tone

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    sf.write(wav_path, signal, sample_rate_hz, subtype="FLOAT")

    yield wav_path

    if wav_path.exists():
        wav_path.unlink()
```

**Factory fixture (returns callable):**

```python
@pytest.fixture
def create_tone_wav():
    """Factory: create WAV files with configurable tones."""
    created_paths = []

    def _create(freq_hz: float, duration_ms: float, ...) -> Path:
        # Generate signal, write WAV, track path
        created_paths.append(wav_path)
        return wav_path

    yield _create

    for p in created_paths:
        if p.exists():
            p.unlink()
```

**Rules:**
- Use `yield` for fixtures that create files (enables cleanup)
- Track created temp files in a list for batch cleanup
- Synthetic signals: noise + pure tone at known frequency
- Use `tmp_path` (pytest builtin) for directory-based temp files
- Never depend on real WAV recordings in tests

---

## 4. Script CLI Pattern

Scripts in `scripts/` follow a consistent argparse structure with path bootstrapping.

**Example** (from `scripts/run_detection.py`):

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: add src/ to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.detection.config import DetectionConfig
from usv_spectrogram.detection.energy_detector import EnergyDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect USV candidates in WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detection.py --input recording.wav --output candidates.csv
        """,
    )
    parser.add_argument("--input", required=True, help="...")
    parser.add_argument("--output", required=True, help="...")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # ... do work ...
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Rules:**
- Path bootstrap: `REPO_ROOT = Path(__file__).resolve().parents[1]` for `scripts/` directory
- Use `parents[2]` for scripts nested deeper (e.g. `usv_language/scripts/`)
- Separate `parse_args()` function for testability
- Return exit codes: 0 success, 1 error
- Include usage examples in epilog

---

## 5. PyQt6 Widget Structure (Model-View Separation)

The detection app separates logic from UI. Core modules handle data/computation; widgets handle rendering.

**Directory layout:**

```
src/usv_spectrogram/app/
  core/                          # Business logic (no Qt imports)
    audio_loader.py              # AudioData, load and cache audio
    detection_logic.py           # DetectedUSV, DetectionResult, hysteresis
    label_storage.py             # JSON persistence
    detection_exporter.py        # CSV/NPZ export
    preset_config.py             # ThresholdPreset management
    saved_detection_tracker.py   # Track save state
    sliding_inference.py         # CNN sliding window inference
  widgets/                       # Qt rendering (no business logic)
    spectrogram_view.py          # SpectrogramCanvas
    probability_view.py          # ProbabilityCanvas
  main_window.py                 # Orchestration (connects core to widgets)
```

**Communication pattern:**

```python
class SpectrogramCanvas(QWidget):
    # Signals for parent to connect
    boundary_adjusted = pyqtSignal(object, float, float)
    detection_created = pyqtSignal(float, float)

    def set_data(self, spectrogram_db, times, frequencies, ...):
        """Set data — rendering only, no business logic."""
        ...
```

**Rules:**
- Core modules must not import from `widgets/`
- Widgets must not contain business logic
- Use `pyqtSignal` for widget-to-parent communication
- `main_window.py` connects signals between core and widgets

---

## 6. Label Storage Pattern (JSON with Rich Metadata)

Detection results are persisted as JSON with full metadata for auditability.

**Structure:**

```json
{
  "metadata": {
    "wav_file": "/absolute/path/to/recording.wav",
    "model_file": "/absolute/path/to/model.pt",
    "created_at": "2024-01-15T10:30:45.123456",
    "duration_s": 120.5,
    "sample_rate": 300000,
    "n_detections": 42,
    "file_label": null
  },
  "detection_params": {
    "high_threshold": 0.08,
    "low_threshold": 0.05
  },
  "detections": [
    {
      "start_time_s": 1.234,
      "end_time_s": 1.567,
      "max_probability": 0.95,
      "user_adjusted": false,
      "user_action": null
    }
  ],
  "probability_curve": {
    "times": [0.0, 0.01, ...],
    "probabilities": [0.02, 0.03, ...],
    "column_indices": [0, 1, ...]
  }
}
```

**Rules:**
- Absolute file paths in metadata for cross-machine reproducibility
- ISO 8601 timestamps for sorting
- Preserve original CNN outputs alongside user-adjusted values
- Track `user_action` field for audit trail
- `indent=2` for human-readable JSON

---

## 7. STFT Core Pattern (Shared Computation)

STFT computation is factored into reusable functions in `_stft_core.py` to ensure consistency between detection and training.

**File:** `src/usv_spectrogram/_stft_core.py`

```python
def extract_frames(
    samples: np.ndarray,
    window_length: int,
    hop_length: int,
) -> np.ndarray:
    """Extract overlapping frames from a 1D signal.
    Returns shape (n_frames, window_length).
    """
    n_frames = 1 + (len(samples) - window_length) // hop_length
    if n_frames <= 0:
        return np.empty((0, window_length), dtype=samples.dtype)
    frame_starts = np.arange(n_frames) * hop_length
    return np.stack(
        [samples[start:start + window_length] for start in frame_starts],
        axis=0,
    )


def compute_stft_frames_db(
    frames: np.ndarray,
    window: np.ndarray,
    n_fft: int,
    band_mask: np.ndarray,
    eps: float,
    normalize_magnitude: bool = False,
) -> np.ndarray:
    """Compute dB spectrogram from windowed frames.
    Returns shape (n_freq_bins, n_frames).
    """
    windowed = frames * window
    stft = np.fft.rfft(windowed, n=n_fft, axis=1)
    magnitude = np.abs(stft)
    if normalize_magnitude:
        magnitude = magnitude / (np.max(magnitude) + eps)
    spec_db = 20.0 * np.log10(magnitude + eps)
    return spec_db[:, band_mask].T
```

**Usage pattern:**

```python
from usv_spectrogram._stft_core import extract_frames, compute_stft_frames_db

frames = extract_frames(samples, cfg.n_fft, cfg.hop_length)
window = scipy.signal.get_window("hann", cfg.n_fft)
band_mask = (freqs_hz >= cfg.freq_min_hz) & (freqs_hz < cfg.freq_max_hz)
spec_db = compute_stft_frames_db(frames, window, cfg.n_fft, band_mask, eps=1e-12)
```

**Rules:**
- Keep as standalone functions (no classes)
- Use `np.fft.rfft` for real signals (half-spectrum is sufficient)
- Apply window before FFT
- Return transposed output: shape `(n_freq_bins, n_frames)` — frequency on rows
- Use `band_mask` for frequency filtering before returning

---

## 8. Import Bootstrap Pattern

All scripts and tests add `src/` to `sys.path` before importing project modules.

**For scripts in `scripts/`:**

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

**For scripts in `usv_language/scripts/` (nested deeper):**

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

**For tests in `tests/conftest.py`:**

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

**Rules:**
- Guard with `if str(SRC_ROOT) not in sys.path` to avoid duplicates
- Use `parents[N]` where N is the directory depth from the repo root
- Place before any project imports
