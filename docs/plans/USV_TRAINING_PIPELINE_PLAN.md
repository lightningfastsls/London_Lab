# USV Detection Pipeline: Implementation Plan

## Overview

This document provides implementation guidance for building a USV (Ultrasonic Vocalization) detection pipeline for mouse courtship behavior research. The pipeline will generate training data for an ML classifier that distinguishes USVs from noise.

**Research context:** Comparing courtship behavior of wild mice vs. lab mice. Recordings will have 2 mice per cage, meaning overlapping calls are possible.

**Reference document:** See `usv_signal_processing_reference.md` for detailed explanations of all technical subtleties, trade-offs, and the reasoning behind each decision in this plan.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  STAGE 1: Candidate Generation                                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
│  │  WAV files  │───▶│ Energy detector  │───▶│ Candidate segments    │  │
│  └─────────────┘    │ (high recall)    │    │ with timestamps       │  │
│                     └──────────────────┘    └───────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 2: Spectrogram Extraction                                        │
│  ┌─────────────────────┐    ┌───────────────────────────────────────┐  │
│  │ Candidate segments  │───▶│ Spectrogram images (PNG)              │  │
│  │                     │    │ + metadata CSV                        │  │
│  └─────────────────────┘    └───────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 3: Labeling Interface                                            │
│  ┌─────────────────────┐    ┌───────────────────────────────────────┐  │
│  │ Spectrogram images  │───▶│ Human review: USV / Not USV / Uncertain│  │
│  │                     │    │ Labels stored in CSV                  │  │
│  └─────────────────────┘    └───────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 4: Dataset Preparation                                           │
│  ┌─────────────────────┐    ┌───────────────────────────────────────┐  │
│  │ Labeled data        │───▶│ Train/Val/Test splits (by recording) │  │
│  │                     │    │ + augmentation (training only)        │  │
│  └─────────────────────┘    └───────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 5: Model Training (future)                                       │
│  ┌─────────────────────┐    ┌───────────────────────────────────────┐  │
│  │ Prepared dataset    │───▶│ CNN binary classifier                 │  │
│  └─────────────────────┘    └───────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
usv_detection/
├── src/
│   └── usv_spectrogram/
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── energy_detector.py      # Stage 1: Candidate generation
│       │   ├── spectrogram_extractor.py # Stage 2: Image extraction
│       │   ├── feature_filters.py      # Optional: bandwidth, flatness filters
│       │   └── config.py               # Detection parameters
│       ├── labeling/
│       │   ├── __init__.py
│       │   ├── labeling_app.py         # Stage 3: Streamlit or GUI labeling tool
│       │   └── label_storage.py        # CSV read/write utilities
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── splits.py               # Stage 4: Train/val/test splitting
│       │   ├── augmentation.py         # Stage 4: Data augmentation
│       │   └── quality_checks.py       # Pre-training validation
│       └── existing modules...
├── data/
│   ├── raw/
│   │   └── recordings/                 # Original WAV files
│   ├── candidates/
│   │   ├── spectrograms/               # Extracted PNG images
│   │   └── candidates.csv              # Metadata for all candidates
│   ├── labeled/
│   │   └── labels.csv                  # Human labels
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── docs/
│   ├── usv_signal_processing_reference.md  # Technical reference (READ THIS)
│   └── labeling_guide.md               # Visual examples for consistent labeling
├── tests/
│   └── ...
└── scripts/
    ├── run_detection.py                # CLI for Stage 1
    ├── extract_spectrograms.py         # CLI for Stage 2
    ├── run_labeling.py                 # CLI for Stage 3
    └── prepare_dataset.py              # CLI for Stage 4
```

---

## Stage 1: Candidate Generation (PRIMARY FOCUS)

### Purpose
Generate candidate USV segments from WAV files with HIGH RECALL. It's acceptable to have many false positives (50-70%); it's NOT acceptable to miss real USVs.

### Why High Recall Matters
See `usv_signal_processing_reference.md` Section 3.1 and 3.2:
- Missing USV types at this stage means the ML model can never learn them
- False positives are filtered out during labeling
- Systematic blind spots (e.g., quiet calls, unusual frequencies) propagate through entire pipeline

### Implementation: `src/usv_spectrogram/detection/energy_detector.py`
``
```python
"""
Energy-based USV candidate detector.

Design principles (see usv_signal_processing_reference.md):
- Optimize for RECALL over precision
- Use conservative (low) energy threshold
- Add duration filters to reject obvious artifacts
- Output candidates with full metadata for traceability
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Iterator

@dataclass
class DetectionConfig:
    """
    Configuration for energy-based detection.
    
    See usv_signal_processing_reference.md Section 1.1 for parameter trade-offs.
    """
    # STFT parameters
    sample_rate: int = 250_000          # Must be >= 2 * max_freq (Nyquist)
    n_fft: int = 512                    # ~488 Hz freq resolution, ~2 ms time resolution at 250 kHz
    hop_length: int = 128               # 75% overlap
    
    # Frequency band for USV detection
    freq_min_hz: int = 25_000           # High-pass: remove sub-ultrasonic noise
    freq_max_hz: int = 110_000          # Upper bound of mouse USV range
    
    # Energy threshold - deliberately LOW for high recall
    # See Section 3.2: threshold bias creates systematic blind spots
    energy_threshold_db: float = -50.0  # Relative to max; tune based on your recordings
    
    # Duration filters - reject obvious non-USVs
    # See Section 2.4: minimum duration filtering
    min_duration_ms: float = 10.0       # USVs are >= 10 ms
    max_duration_ms: float = 500.0      # USVs are <= 300 ms, allow margin
    
    # Merging nearby detections
    merge_gap_ms: float = 5.0           # If two detections are < 5 ms apart, merge them
    
    # Context window for extraction
    context_before_ms: float = 50.0     # Include 50 ms before detection
    context_after_ms: float = 50.0      # Include 50 ms after detection


@dataclass 
class Candidate:
    """
    A detected candidate USV segment.
    
    Includes full metadata for traceability (see Section 4.8).
    """
    source_file: Path
    start_ms: float
    end_ms: float
    peak_freq_hz: float
    peak_energy_db: float
    duration_ms: float
    
    # Context window (for spectrogram extraction)
    context_start_ms: float
    context_end_ms: float
    
    # Unique identifier
    candidate_id: str  # Format: "{source_stem}_{start_ms:08d}"


class EnergyDetector:
    """
    Detect candidate USV segments using energy thresholding.
    
    This is intentionally a simple detector optimized for recall.
    Precision is handled downstream by human labeling.
    """
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def detect(self, wav_path: Path) -> list[Candidate]:
        """
        Detect all candidate USVs in a WAV file.
        
        Returns candidates sorted by start time.
        """
        # Implementation steps:
        # 1. Load audio
        # 2. Apply high-pass filter (freq_min_hz)
        # 3. Compute spectrogram (STFT)
        # 4. Sum energy in USV frequency band per frame
        # 5. Threshold to get candidate frames
        # 6. Group adjacent frames into segments
        # 7. Merge segments separated by < merge_gap_ms
        # 8. Apply duration filters
        # 9. Extract peak frequency for each candidate
        # 10. Create Candidate objects with full metadata
        
        raise NotImplementedError("Implement this")
    
    def detect_batch(self, wav_dir: Path) -> Iterator[Candidate]:
        """
        Detect candidates across all WAV files in a directory.
        
        Yields candidates one at a time for memory efficiency.
        """
        raise NotImplementedError("Implement this")
```

### Critical Implementation Notes

#### 1. Threshold Tuning Strategy
```python
# DON'T: Pick a threshold and hope it works
# DO: Implement threshold analysis

def analyze_threshold_sensitivity(wav_path: Path, config: DetectionConfig) -> dict:
    """
    Analyze how detection count changes with threshold.
    
    Returns dict mapping threshold_db -> candidate_count.
    
    Use this to find the "knee" where lowering threshold
    rapidly increases candidates (likely adding noise).
    
    Start BELOW the knee for high recall.
    """
    pass
```

#### 2. Verification: Check for Systematic Blind Spots
```python
def verify_detection_coverage(
    wav_path: Path, 
    candidates: list[Candidate],
    manual_usv_times_ms: list[float]  # A few manually identified USVs
) -> dict:
    """
    Check that known USVs were detected.
    
    If manual USVs are NOT in candidates, threshold is too high
    or there's a bug in detection logic.
    
    See Section 3.2: this prevents training data bias.
    """
    pass
```

#### 3. Handle Known Interference
```python
# See Section 2.3: 60 kHz and harmonics are often electrical interference

def is_likely_interference(candidate: Candidate) -> bool:
    """
    Flag candidates that match known interference patterns.
    
    Characteristics of interference:
    - Frequency exactly at 50/60 Hz harmonics (50 kHz, 60 kHz, etc.)
    - Duration >> 300 ms
    - Perfectly stable frequency (no modulation)
    
    Don't auto-reject; flag for review. Some real USVs 
    happen to be near these frequencies.
    """
    pass
```

### Output Format: `candidates.csv`

```csv
candidate_id,source_file,start_ms,end_ms,duration_ms,context_start_ms,context_end_ms,peak_freq_hz,peak_energy_db,interference_flag
lab_mouse_01_00014200,lab_mouse_01.wav,142.0,198.0,56.0,92.0,248.0,52340.5,-32.1,false
lab_mouse_01_00058700,lab_mouse_01.wav,587.0,612.0,25.0,537.0,662.0,61002.3,-41.8,true
```

Include all metadata. Storage is cheap; missing metadata is expensive later.

### Testing Stage 1

```python
# tests/test_energy_detector.py

def test_minimum_duration_filter():
    """Candidates shorter than min_duration_ms are rejected."""
    pass

def test_maximum_duration_filter():
    """Candidates longer than max_duration_ms are rejected."""
    pass

def test_frequency_band_filter():
    """Only energy in freq_min to freq_max is considered."""
    pass

def test_merge_nearby_detections():
    """Detections < merge_gap_ms apart are merged into one candidate."""
    pass

def test_known_usv_detected():
    """
    A manually verified USV is detected.
    Use a test WAV file with a known USV at a known timestamp.
    """
    pass

def test_candidate_metadata_complete():
    """All metadata fields are populated."""
    pass
```

---

## Stage 2: Spectrogram Extraction

### Purpose
Convert each candidate segment into a spectrogram image suitable for human review and later CNN training.

### Implementation: `src/usv_spectrogram/detection/spectrogram_extractor.py`

```python
"""
Extract spectrogram images from candidate segments.

Key decisions (see usv_signal_processing_reference.md Section 1):
- n_fft/hop_length determine time-frequency resolution trade-off
- Frequency range should show full USV band (25-110 kHz)
- Color scale should make faint calls visible
"""

@dataclass
class SpectrogramConfig:
    """Configuration for spectrogram generation."""
    
    # STFT parameters (should match detection config)
    n_fft: int = 512
    hop_length: int = 128
    sample_rate: int = 250_000
    
    # Display frequency range
    freq_min_hz: int = 20_000           # Show a bit below USV range for context
    freq_max_hz: int = 120_000          # Show a bit above USV range for context
    
    # Image parameters
    image_height_px: int = 256
    image_width_px: int = 256           # Will vary based on duration; this is target
    
    # Color scale
    db_min: float = -80.0               # Floor for color scale
    db_max: float = 0.0                 # Ceiling for color scale (relative to max)
    colormap: str = "magma"             # Good for seeing faint details


class SpectrogramExtractor:
    """Extract spectrogram images from candidates."""
    
    def __init__(self, config: SpectrogramConfig):
        self.config = config
    
    def extract(self, wav_path: Path, candidate: Candidate, output_dir: Path) -> Path:
        """
        Extract spectrogram image for a single candidate.
        
        Uses candidate.context_start_ms and context_end_ms for the window.
        
        Returns path to saved PNG.
        """
        # Implementation steps:
        # 1. Load audio segment (context window)
        # 2. Compute spectrogram
        # 3. Crop to frequency range
        # 4. Convert to dB scale
        # 5. Normalize to color scale
        # 6. Save as PNG
        # 7. Return path
        
        raise NotImplementedError("Implement this")
    
    def extract_batch(self, candidates_csv: Path, output_dir: Path) -> None:
        """
        Extract spectrograms for all candidates in a CSV.
        
        Updates candidates CSV with spectrogram_path column.
        """
        raise NotImplementedError("Implement this")
```

### Critical Note: Consistent Parameters
The spectrogram parameters used for extraction MUST match what you'll use for the CNN. If you extract at n_fft=512 but later train on n_fft=1024 images, you'll need to re-extract everything.

Document your parameters and don't change them mid-project.

---

## Stage 3: Labeling Interface

### Purpose
Provide an efficient interface for human review of candidates. Each candidate is labeled as: USV / Not USV / Uncertain.

### Key Requirements

See `usv_signal_processing_reference.md` Section 4.5 (labeler drift) and 4.6 (high uncertainty diagnostic):

1. **Keyboard shortcuts** - Essential for speed
   - `Y` or `1` = USV
   - `N` or `0` = Not USV  
   - `U` or `2` = Uncertain
   - `→` = Next candidate
   - `←` = Previous candidate (allow corrections)
   - `Space` = Play audio (pitch-shifted to audible range)

2. **Visual display**
   - Large, clear spectrogram
   - Frequency axis labeled
   - Time axis labeled
   - Indication of detected region within context window

3. **Progress tracking**
   - "142 / 500 labeled"
   - "Session time: 47 minutes"
   - "Uncertain rate: 8%" (flag if > 15%)

4. **Session management**
   - Auto-save after each label
   - Resume from where you left off
   - Session breaks encouraged (see Section 4.5: labeler drift)

### Implementation Options

**Option A: Streamlit app** (recommended for consistency with existing Parameter Lab)
```python
# src/usv_spectrogram/labeling/labeling_app.py
# Run with: streamlit run labeling_app.py

import streamlit as st

def main():
    st.title("USV Labeling Tool")
    
    # Load candidates that haven't been labeled yet
    # Display spectrogram
    # Capture keyboard input
    # Save label to CSV
    # Show progress stats
    
    pass
```

**Option B: Fix existing GUI 10.py**
- Add actual label saving (currently incomplete)
- Add keyboard shortcuts
- Add progress tracking

### Label Storage Format

```csv
candidate_id,label,labeled_at,session_id,notes
lab_mouse_01_00014200,usv,2025-01-15T10:23:00,session_001,clear upward sweep
lab_mouse_01_00058700,noise,2025-01-15T10:23:15,session_001,electrical interference at 60kHz
lab_mouse_01_00092300,uncertain,2025-01-15T10:23:22,session_001,very faint - might be USV
```

### Labeling Guide Document

Create `docs/labeling_guide.md` with visual examples:

```markdown
# USV Labeling Guide

## Clear USV Examples
[Image 1: Upward sweep, 50-70 kHz, 45 ms]
[Image 2: Flat call, 55 kHz, 80 ms]
[Image 3: Chevron, 60-75-60 kHz, 35 ms]

## Clear Not-USV Examples  
[Image 4: Broadband noise - vertical smear]
[Image 5: Electrical interference - perfect horizontal line at 60 kHz]
[Image 6: Click artifact - very short broadband spike]

## Uncertain - Mark These for Review
[Image 7: Very faint but USV-shaped]
[Image 8: Partial call at window edge]
[Image 9: Two overlapping calls]

## Decision Rules
1. If you can see clear frequency structure in 30-110 kHz range → USV
2. If it's broadband (vertical smear) → Not USV
3. If duration < 10 ms → Not USV (artifact)
4. If frequency perfectly stable for > 300 ms → Not USV (interference)
5. If genuinely can't tell → Uncertain (but aim for < 10% uncertain rate)
```

---

## Stage 4: Dataset Preparation

### Purpose
Convert labeled data into train/validation/test splits ready for model training.

### Critical Rules

See `usv_signal_processing_reference.md` Sections 4.2, 4.3, 4.4, 4.7:

1. **Split by RECORDING, not by candidate**
   - All candidates from a recording stay in the same split
   - Prevents data leakage from correlated candidates

2. **Stratify by population**
   - Both lab and wild mice should appear in test set
   - Prioritize minority class (wild mice) in test set

3. **Handle class imbalance**
   - Target 1:1 to 1:3 ratio (USV : not-USV)
   - Undersample majority or oversample minority

4. **Augmentation on training set only**
   - Never augment validation or test
   - Keep augmentation moderate (2-5×)

### Implementation: `src/usv_spectrogram/dataset/splits.py`

```python
"""
Create train/validation/test splits.

CRITICAL: Split by recording, not by candidate.
See usv_signal_processing_reference.md Section 4.2.
"""

@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Stratification
    stratify_by_population: bool = True  # lab vs wild
    ensure_minority_in_test: bool = True  # wild mice must be in test


def create_splits(
    labels_csv: Path,
    recordings_metadata: Path,  # Maps recording -> population (lab/wild)
    output_dir: Path,
    config: SplitConfig
) -> dict[str, Path]:
    """
    Create stratified train/val/test splits.
    
    Returns paths to train.csv, val.csv, test.csv.
    
    Algorithm:
    1. Group candidates by source recording
    2. Group recordings by population (lab/wild)
    3. For each population, randomly assign recordings to splits
    4. Ensure at least one recording from each population in test
    5. Write split CSVs
    """
    raise NotImplementedError("Implement this")
```

### Implementation: `src/usv_spectrogram/dataset/augmentation.py`

```python
"""
Data augmentation for training set.

Apply ONLY to training data. See Section 4.4: augmentation overfitting trap.
"""

@dataclass
class AugmentationConfig:
    # How many augmented versions per original
    augmentation_factor: int = 3  # Conservative; 2-5 is safe range
    
    # Augmentation types
    time_shift_ms: float = 20.0      # Shift call left/right within window
    freq_shift_hz: float = 2000.0    # Shift up/down by up to 2 kHz
    noise_factor: float = 0.1        # Add random noise
    intensity_scale: tuple = (0.8, 1.2)  # Scale intensity by 0.8-1.2×


def augment_training_set(
    train_csv: Path,
    spectrograms_dir: Path,
    output_dir: Path,
    config: AugmentationConfig
) -> Path:
    """
    Create augmented versions of training spectrograms.
    
    Returns path to augmented_train.csv with additional entries.
    
    IMPORTANT: Keep track of which augmented examples came from
    which original. If original is later found to be mislabeled,
    all its augmentations should be removed.
    """
    raise NotImplementedError("Implement this")
```

### Implementation: `src/usv_spectrogram/dataset/quality_checks.py`

```python
"""
Pre-training quality checks.

Run these before starting model training. See Section 4.11.
"""

def run_all_checks(
    train_csv: Path,
    val_csv: Path, 
    test_csv: Path,
    spectrograms_dir: Path
) -> dict:
    """
    Run all quality checks. Returns dict of check_name -> pass/fail.
    
    Checks:
    1. No duplicate candidates across splits
    2. All spectrogram files exist
    3. Class balance is reasonable (not worse than 1:5)
    4. Both populations in test set
    5. No recording in multiple splits
    6. No uncertain labels in final splits
    """
    raise NotImplementedError("Implement this")


def check_class_balance(csv_path: Path) -> dict:
    """
    Report class distribution.
    
    Returns: {
        'usv_count': int,
        'noise_count': int,
        'ratio': float,
        'warning': str or None
    }
    """
    raise NotImplementedError("Implement this")


def check_recording_leakage(train_csv: Path, val_csv: Path, test_csv: Path) -> list[str]:
    """
    Check if any recording appears in multiple splits.
    
    Returns list of recordings that appear in multiple splits (should be empty).
    """
    raise NotImplementedError("Implement this")
```

---

## Stage 5: Model Training (Future)

Defer detailed planning until Stages 1-4 are complete and you have labeled data. High-level notes:

- Start with simple CNN (e.g., ResNet-18 pretrained, fine-tuned)
- Use class weights if imbalance persists after balancing
- Monitor validation accuracy, not training accuracy
- Evaluate separately on lab mice and wild mice
- If performance differs significantly between populations, you have a generalization problem

---

## Implementation Order

### Phase 1: Candidate Generation (Weeks 1-2)

1. **Set up project structure** as defined above
2. **Implement DetectionConfig and Candidate dataclasses**
3. **Implement EnergyDetector.detect()** for single file
4. **Write tests** for duration filters, frequency band, merging
5. **Run on sample WAV files** and manually verify candidates
6. **Tune threshold** using analyze_threshold_sensitivity()
7. **Verify no systematic blind spots** with verify_detection_coverage()
8. **Implement batch detection** across directory

**Deliverable:** Can run `python scripts/run_detection.py --input data/raw/recordings/ --output data/candidates/` and get candidates.csv

### Phase 2: Spectrogram Extraction (Week 2)

1. **Implement SpectrogramExtractor**
2. **Test on a few candidates** - verify images look correct
3. **Batch extract** all candidates
4. **Document parameters** used (they're locked in now)

**Deliverable:** All candidates have corresponding PNG spectrograms

### Phase 3: Labeling Tool (Week 3)

1. **Create labeling_guide.md** with visual examples
2. **Implement labeling interface** (Streamlit recommended)
3. **Test workflow** on 20-30 candidates
4. **Label full dataset** (expect 1-2 hours for 500 candidates)
5. **Monitor uncertain rate** - investigate if > 15%
6. **Calibration check** - re-label 50 candidates, compare to originals

**Deliverable:** labels.csv with all candidates labeled

### Phase 4: Dataset Preparation (Week 3-4)

1. **Create recordings_metadata.csv** mapping recording -> population
2. **Implement create_splits()** with stratification
3. **Run quality checks**
4. **Implement augmentation** (optional, if class imbalance is severe)
5. **Final quality checks**

**Deliverable:** train.csv, val.csv, test.csv ready for model training

### Phase 5: Model Training (Week 4+)

Defer planning until Phase 4 complete.

---

## Files to Migrate from Lab Code

| Original File | Action | New Location |
|---------------|--------|--------------|
| Extract USV smaples.py | Rewrite using this plan | src/usv_spectrogram/detection/energy_detector.py |
| GUI 10.py | Rewrite as Streamlit or fix and move | src/usv_spectrogram/labeling/labeling_app.py |
| Show_training_examples.py | Adapt for verification | scripts/verify_candidates.py |

**Do not copy-paste old code.** Use it as reference for what the original author intended, but implement fresh following this plan.

---

## Environment Configuration

### Required Environment Variables

```bash
# Path to WAV recordings
USV_WAV_DIR=/path/to/data/raw/recordings

# Path to output directory
USV_OUTPUT_DIR=/path/to/data
```

### Dependencies

```
numpy
scipy
librosa  # or use existing STFT implementation
matplotlib
pandas
streamlit  # for labeling tool
pytest
```

---

## Quick Reference: Key Parameters

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Sample rate | 250,000 Hz | Section 2.6 |
| n_fft | 512 | Section 1.1 |
| hop_length | 128 | Section 1.4 |
| Frequency range | 25-110 kHz | Section 2 intro |
| Min USV duration | 10 ms | Section 2.4 |
| Max USV duration | 500 ms | Section 2.3 |
| Energy threshold | Tune for recall | Section 3.1, 3.2 |
| Class balance target | 1:1 to 1:3 | Section 4.1 |
| Augmentation factor | 2-5× | Section 4.4 |
| Uncertain rate target | < 10% | Section 4.6 |

---

## Reference Documents

- **`usv_signal_processing_reference.md`** - Detailed explanations of all technical decisions
- **`docs/labeling_guide.md`** - Visual examples for consistent labeling (create during Phase 3)

---

*Implementation plan created based on signal processing curriculum. See usv_signal_processing_reference.md for theoretical foundations.*
