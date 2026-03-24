# Energy Detector

**Phase:** 1 (Detection Pipeline)
**ADRs:** ADR-001 (Sample Rate), ADR-002 (STFT Parameters), ADR-003 (Detection Threshold)
**Tests:** `tests/test_energy_detector.py` — 42 tests across 17 test classes

## Purpose

The EnergyDetector is a high-recall candidate USV detection system that identifies potential ultrasonic vocalizations in WAV files by thresholding on energy in the ultrasonic frequency band. It is intentionally permissive — precision is handled downstream by CNN classification or human labeling.

## Public Interface

### `EnergyDetector`

```python
class EnergyDetector:
    def __init__(self, config: Optional[DetectionConfig] = None):
        """Create detector. Uses DetectionConfig defaults if not provided."""

    def detect(self, wav_path: Path) -> list[Candidate]:
        """Detect all candidate USVs in a WAV file.
        Returns candidates sorted by start time with full metadata."""

    def detect_batch(self, wav_dir: Path, pattern: str = "*.wav") -> Iterator[Candidate]:
        """Yield candidates from all matching WAV files in a directory.
        Memory-efficient: processes one file at a time."""

    def save_candidates_csv(self, candidates: list[Candidate], output_path: Path) -> None:
        """Export candidates to CSV with headers matching Candidate.to_dict() keys."""
```

### Module-Level Helpers

```python
def analyze_threshold_sensitivity(
    wav_path: Path,
    config: Optional[DetectionConfig] = None,
    threshold_range: tuple[float, float] = (-60.0, -20.0),
    threshold_step: float = 5.0,
) -> dict[float, int]:
    """Returns dict mapping threshold_db -> candidate_count."""

def verify_detection_coverage(
    wav_path: Path,
    candidates: list[Candidate],
    manual_usv_times_ms: list[float],
    tolerance_ms: float = 50.0,
) -> dict:
    """Returns dict with keys: detected, missed, coverage_rate, total_manual, total_candidates."""
```

## Data Model

### `Candidate` (from `detection/candidate.py`)

| Field | Type | Description |
|-------|------|-------------|
| `source_file` | `Path` | Path to original WAV file |
| `candidate_id` | `str` | Unique ID: `"{source_stem}_{start_ms:08.0f}"` |
| `start_ms` | `float` | Start of detected region (ms) |
| `end_ms` | `float` | End of detected region (ms) |
| `duration_ms` | `float` | `end_ms - start_ms` |
| `context_start_ms` | `float` | Start of context window |
| `context_end_ms` | `float` | End of context window |
| `peak_freq_hz` | `float` | Frequency with maximum energy |
| `peak_energy_db` | `float` | Maximum energy in dB |
| `interference_flag` | `bool` | True if peak_freq near known interference (60 kHz, etc.) |
| `spectrogram_path` | `Optional[Path]` | Path to extracted spectrogram (populated post-detection) |

Key methods: `Candidate.create()` (factory), `to_dict()`, `from_dict()`.

### `DetectionConfig` (from `detection/config.py`)

Core parameters (see file for full list of 30+ fields):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sample_rate` | 300,000 | STFT sample rate (Hz) |
| `n_fft` | 512 | FFT window size (~1.7 ms) |
| `hop_length` | 128 | Hop size (75% overlap) |
| `auto_sample_rate` | True | Use WAV file's actual sample rate |
| `freq_min_hz` | 25,000 | Lower frequency bound |
| `freq_max_hz` | 110,000 | Upper frequency bound |
| `energy_threshold_db` | -60.0 | Relative to max energy (low = more candidates) |
| `energy_mode` | "peak" | "peak" or "mean" energy per frame |
| `max_bandwidth_hz` | 20,000 | Reject broadband noise |
| `min_duration_ms` | 10.0 | Minimum candidate duration |
| `max_duration_ms` | 500.0 | Maximum candidate duration |
| `merge_gap_ms` | 3.0 | Merge segments closer than this |
| `segment_continuity_enabled` | True | Enable continuity-based extension |
| `segment_continuity_max_gap_ms` | 5.0 | Max gap to bridge via continuity |

## Usage Examples

### Basic detection

```python
from usv_spectrogram.detection.energy_detector import EnergyDetector
from usv_spectrogram.detection.config import DetectionConfig

# Default configuration
detector = EnergyDetector()
candidates = detector.detect(Path("recording.wav"))
print(f"Found {len(candidates)} candidates")

# Custom configuration
config = DetectionConfig(
    energy_threshold_db=-50.0,   # More selective
    max_bandwidth_hz=15_000,     # Stricter bandwidth filter
)
detector = EnergyDetector(config)
candidates = detector.detect(Path("recording.wav"))
```

### Batch processing

```python
detector = EnergyDetector()

# Memory-efficient directory processing
for candidate in detector.detect_batch(Path("wav_directory/")):
    print(f"{candidate.candidate_id}: {candidate.start_ms:.1f}-{candidate.end_ms:.1f} ms")

# Export to CSV
all_candidates = list(detector.detect_batch(Path("wav_directory/")))
detector.save_candidates_csv(all_candidates, Path("candidates.csv"))
```

### Threshold tuning

```python
from usv_spectrogram.detection.energy_detector import analyze_threshold_sensitivity

results = analyze_threshold_sensitivity(
    Path("recording.wav"),
    threshold_range=(-60.0, -20.0),
    threshold_step=5.0,
)
# {-60.0: 150, -55.0: 120, -50.0: 80, ..., -20.0: 5}
# Look for the "knee" where candidate count rises sharply
```

## Detection Algorithm

1. **Load audio** via `load_wav_mono()` (auto-detects or enforces sample rate)
2. **Compute band-limited spectrogram** — STFT with Hann window, keep `freq_min_hz` to `freq_max_hz`
3. **Compute energy per frame** — peak mode: `max(spec_db, axis=0)`, mean mode: `mean(spec_db, axis=0)`
4. **Threshold** relative to max energy — `active = energy >= max_energy + threshold_db`
5. **Group** contiguous active frames into segments
6. **Merge** segments separated by < `merge_gap_ms`
7. **Extend** by continuity (optional) — bridge gaps where peak frequency and energy match
8. **Filter** by duration — keep `min_duration_ms` <= duration <= `max_duration_ms`
9. **Create Candidates** — extract peak frequency, check bandwidth, flag interference
10. **Return** sorted by `start_ms`

## Key Decisions

1. **High recall, low precision** — the detector deliberately uses a low threshold (-60 dB) to avoid missing any USVs. False positives are filtered by downstream CNN or human review. (ADR-003)
2. **Peak energy mode** — "peak" mode uses max energy per frequency bin per frame, which is better for detecting narrow-band USVs than "mean" mode.
3. **Segment continuity** — an optional stage bridges small gaps (< 5 ms) where spectral properties are consistent, preventing a single USV from being split into fragments.
4. **Bandwidth filter** — rejects broadband noise artifacts (> 20 kHz bandwidth) that pass the energy threshold but aren't narrow-band USVs.
5. **Interference flagging** — marks candidates near known electrical interference frequencies (50, 60, 100, 120 kHz) without auto-rejecting them.

## Integration Points

### Consumes
- `io_wav.load_wav_mono()` — WAV file loading
- `scipy.signal` — Hann window, convolution for continuity smoothing
- `numpy.fft.rfft` — STFT computation

### Consumed By
- `scripts/run_detection.py` — CLI batch detection
- `SpectrogramExtractor` — extracts spectrogram patches from candidates
- CNN classifier pipeline — candidates are scored by CNN
- Detection app (`app/core/`) — uses candidates indirectly via CNN sliding inference
- Labeling tools (`labeling/`) — presents candidates for human review

### Data Flow

```
WAV file -> EnergyDetector.detect() -> list[Candidate]
                                         |
                                         +-> save_candidates_csv() -> CSV
                                         +-> SpectrogramExtractor -> PNG patches
                                         +-> CNN scoring -> probability [0, 1]
                                         +-> LabelStorage -> JSON with metadata
```
