x# Phase 2: Spectrogram Extraction - Implementation Plan

## Overview

Convert the 490 candidate segments from Phase 1 into standardized spectrogram PNG images for human labeling and CNN training.

## Key Design Decisions

### 1. Use Detection-Matched STFT Parameters
- `sample_rate=300_000` (actual recordings, not 250 kHz)
- `n_fft=512`, `hop_length=128` (same as detection)
- Ensures consistency between detection and extracted images

### 2. Two Render Modes
- **"review"** - Matplotlib with axes/labels for human labeling
- **"training"** - Raw images for CNN (future)

### 3. File Structure
```
src/usv_spectrogram/detection/
    extraction_config.py   # NEW: ExtractionConfig dataclass
    spectrogram_extractor.py  # NEW: SpectrogramExtractor class

scripts/
    extract_spectrograms.py  # NEW: CLI for batch extraction

tests/
    test_spectrogram_extractor.py  # NEW: Unit tests
```

## Implementation Steps

### Step 2.1: Create ExtractionConfig dataclass
- Location: `src/usv_spectrogram/detection/extraction_config.py`
- STFT parameters matching detection
- Image output parameters (dimensions, color scale, colormap)
- Validation in `__post_init__`

### Step 2.2: Implement SpectrogramExtractor core
- Location: `src/usv_spectrogram/detection/spectrogram_extractor.py`
- `extract_single()` - single candidate extraction
- `_compute_spectrogram()` - STFT computation
- `_render_image()` - PNG rendering (both modes)

### Step 2.3: Implement batch extraction
- `extract_batch()` - process entire CSV
- CSV update with spectrogram_path column
- Progress reporting and error handling

### Step 2.4: Create CLI script
- Location: `scripts/extract_spectrograms.py`
- Follow `run_detection.py` pattern
- Arguments: --candidates, --wav-dir, --output-dir, --render-mode

### Step 2.5: Write tests and document
- Unit tests for config, extraction, rendering
- Update IMPLEMENTATION_PROGRESS.md with locked parameters

## ExtractionConfig Parameters

```python
@dataclass(frozen=True)
class ExtractionConfig:
    # STFT (match detection)
    sample_rate: int = 300_000
    n_fft: int = 512
    hop_length: int = 128
    window: str = "hann"

    # Frequency range
    freq_min_hz: int = 20_000   # Below USV band for context
    freq_max_hz: int = 120_000  # Above USV band for context

    # Image dimensions
    image_height_px: int = 256
    pixels_per_ms: float = 2.0
    min_width_px: int = 128
    max_width_px: int = 512

    # Color scale
    db_floor: float = -80.0
    db_ceiling: float = 0.0
    colormap: str = "magma"  # Default for review PNGs in labeling app
```

## Reuse Strategy

| Component | Location | Action |
|-----------|----------|--------|
| `load_wav_segment_mono()` | io_wav.py | Direct reuse |
| STFT helpers | _stft_core.py | Direct reuse |
| `render_png()` pattern | render_tiles.py | Adapt |
| `Candidate.from_dict()` | candidate.py | Direct reuse |

## Deliverables

- [ ] ExtractionConfig dataclass
- [ ] SpectrogramExtractor class
- [ ] CLI script (extract_spectrograms.py)
- [ ] Unit tests
- [ ] 490 PNG spectrograms in output directory
- [ ] Updated IMPLEMENTATION_PROGRESS.md

## Subagents to Use

| Step | Agent | Purpose |
|------|-------|---------|
| Config & Core implementation | (main) | Write code |
| Testing | `test-writer` | Write test cases |
| DSP validation | `dsp-reviewer` | Verify STFT correctness |
| Final review | `pr-reviewer` | Quality check |
