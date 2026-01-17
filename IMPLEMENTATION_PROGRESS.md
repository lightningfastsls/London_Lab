# USV Detection Pipeline - Implementation Progress

**Started:** 2026-01-16
**Plan Document:** USV_DETECTION_IMPLEMENTATION_PLAN.md
**Reference:** usv_signal_processing_reference.md

---

## Current Status: Phase 2 - Spectrogram Extraction (COMPLETE)

### Phase 1 Steps (from plan)

- [x] **Step 1.1** - Set up project structure (detection, labeling, dataset modules)
- [x] **Step 1.2** - Implement DetectionConfig dataclass
- [x] **Step 1.3** - Implement Candidate dataclass
- [x] **Step 1.4** - Implement EnergyDetector.detect() for single file
- [x] **Step 1.5** - Write tests for duration filters, frequency band, merging
- [ ] **Step 1.6** - Run on sample WAV files and manually verify candidates ← **NEXT**
- [x] **Step 1.7** - Implement analyze_threshold_sensitivity()
- [x] **Step 1.8** - Implement verify_detection_coverage()
- [x] **Step 1.9** - Implement batch detection across directory
- [x] **Step 1.10** - Create CLI script (run_detection.py)

### Phase 2 Steps (Spectrogram Extraction) - COMPLETE

- [x] **Step 2.1** - Implement SpectrogramExtractor
- [x] **Step 2.2** - Test on a few candidates (via automated tests)
- [x] **Step 2.3** - Batch extract all candidates
- [x] **Step 2.4** - Document parameters used

### Phase 3 Steps (Labeling Tool)

- [ ] **Step 3.1** - Create labeling_guide.md with visual examples
- [ ] **Step 3.2** - Implement labeling interface (Streamlit)
- [ ] **Step 3.3** - Test workflow on 20-30 candidates
- [ ] **Step 3.4** - (User task) Label full dataset

### Phase 4 Steps (Dataset Preparation)

- [ ] **Step 4.1** - Create recordings_metadata.csv mapping recording -> population
- [ ] **Step 4.2** - Implement create_splits() with stratification
- [ ] **Step 4.3** - Run quality checks
- [ ] **Step 4.4** - Implement augmentation
- [ ] **Step 4.5** - Final quality checks

### Phase 5 (Model Training)

- Deferred until Phases 1-4 complete

---

## Implementation Log

### 2026-01-16

**Session started** - Beginning Phase 1 implementation

**Completed:**
- [x] Step 1.1 - Set up project structure (detection module created)
- [x] Step 1.2 - Implement DetectionConfig dataclass
- [x] Step 1.3 - Implement Candidate dataclass
- [x] Step 1.4 - Implement EnergyDetector.detect() for single file

**In Progress:**
- [ ] Step 1.5 - Write tests for energy detector (NEXT TASK)

**Files created:**
- `src/usv_spectrogram/detection/__init__.py`
- `src/usv_spectrogram/detection/config.py`
- `src/usv_spectrogram/detection/candidate.py`
- `src/usv_spectrogram/detection/energy_detector.py`

**Session ended** - User switching computers

---

### 2026-01-16 (Session 2)

**Session started** - Continuing Phase 1 implementation

**Completed:**
- [x] Step 1.5 - Write tests for energy detector (25 tests, all passing)
- [x] Step 1.10 - Create CLI script (run_detection.py)

**Files created:**
- `tests/test_energy_detector.py` - 25 comprehensive tests for energy detector
- `scripts/run_detection.py` - CLI for batch USV detection

**Files modified:**
- `tests/conftest.py` - Added detection fixtures (create_tone_wav, create_multi_tone_wav)
- `IMPLEMENTATION_PROGRESS.md` - Updated progress

**Test Results:**
- 77 tests passing (all modules)
- Detection tests cover: duration filters, frequency band, merging, interference flags, config validation, edge cases

**Next Steps:**
- Step 1.6 - Manual verification on sample WAV files using Parameter Lab

**Session ended**

---

### 2026-01-16 (Session 3)

**Session started** - Improving detection algorithm based on manual verification

**Problem identified:**
User manually verified candidates and found issues:
1. File 1: Missed USV at 768ms (first detection was at 899ms)
2. File 2: False positive (entire file detected as one candidate)
3. File 3: False positive at detected location, real USVs missed at 1707ms

**Root cause analysis:**
- MEAN energy per frame was missing narrow-band USVs (concentrated energy at single frequency)
- Bandwidth filter was measuring bandwidth across entire segment (picking up noisy frames)

**Solution implemented:**
1. Added `energy_mode` parameter ("peak" vs "mean") - peak mode uses max energy per frame
2. Added `max_bandwidth_hz` parameter to reject broadband noise candidates
3. Fixed bandwidth calculation to only check at peak frame, not across whole segment
4. Updated CLI with `--energy-mode` and `--max-bandwidth` options

**Results after improvements:**
- File 1: Now detects USV at 764ms (near user-reported 768ms)
- File 2: 0 candidates (correctly rejecting false positive)
- File 3: 19 candidates detected
- Total: 490 candidates across 50 files (was 256 with mean mode)

**Optimal parameters for this dataset:**
- `--threshold -20` (relative to max peak energy)
- `--sample-rate 300000` (recordings are 300kHz, not 250kHz)
- `--energy-mode peak` (better for narrow-band USVs)
- `--max-bandwidth 20000` (reject broadband noise)

**Files modified:**
- `src/usv_spectrogram/detection/config.py` - Added energy_mode, max_bandwidth_hz params
- `src/usv_spectrogram/detection/energy_detector.py` - Implemented peak energy mode and bandwidth filter
- `scripts/run_detection.py` - Added --energy-mode and --max-bandwidth CLI options

**Test Results:**
- All 25 detection tests still passing

**Output files:**
- `candidates_optimized.csv` - 490 candidates with improved detection

**Session status:** In progress

---

### 2026-01-17 (Session 4)

**Session started** - Implementing Phase 2: Spectrogram Extraction

**Completed:**
- [x] Step 2.1 - Implement SpectrogramExtractor (ExtractionConfig + SpectrogramExtractor + CLI)
- [x] Step 2.2 - Comprehensive test suite for spectrogram extractor (27 tests)

**Files created:**
- `src/usv_spectrogram/detection/extraction_config.py` - ExtractionConfig dataclass
- `src/usv_spectrogram/detection/spectrogram_extractor.py` - SpectrogramExtractor class
- `scripts/extract_spectrograms.py` - CLI for batch spectrogram extraction
- `tests/test_spectrogram_extractor.py` - 27 comprehensive tests

**ExtractionConfig Parameters (locked for Phase 2):**
```python
# STFT (match detection for consistency)
sample_rate: int = 300_000  # Actual recording sample rate
n_fft: int = 512            # ~586 Hz freq resolution
hop_length: int = 128       # 75% overlap
window: str = "hann"

# Frequency range (wider than detection for visual context)
freq_min_hz: int = 20_000   # Below USV band
freq_max_hz: int = 120_000  # Above USV band

# Image dimensions
image_height_px: int = 256  # Fixed height for CNN input
pixels_per_ms: float = 2.0  # Temporal resolution
min_width_px: int = 128     # Minimum width
max_width_px: int = 512     # Maximum width

# Color scale
db_floor: float = -80.0     # Black level
db_ceiling: float = 0.0     # White level
colormap: str = "magma"

# Render modes
# "review" - Matplotlib with axes/labels for human labeling
# "training" - Raw images for CNN (no axes, exact dimensions)
```

**Test Coverage (27 tests):**
- ExtractionConfig validation (13 tests)
- SpectrogramExtractor single extraction (5 tests)
- Spectrogram computation (3 tests)
- Batch extraction (3 tests)
- Edge cases (3 tests)

**Test Results:**
- 111 tests passing (27 new + 84 existing)

**Subagents Used:**
- `test-writer` - Created comprehensive test suite
- `dsp-reviewer` - Validated STFT implementation (PASS, 1 critical issue found and fixed)
- `pr-reviewer` - Final quality check (APPROVED with minor suggestions)

**DSP Review Notes:**
- Added sample rate validation in extract_single() (critical fix)
- STFT implementation matches energy_detector.py pattern
- dB conversion correct (20*log10 for amplitude spectra)
- Frequency band masking correct

**CLI Usage:**
```powershell
# Extract spectrograms for all candidates in review mode
python scripts/extract_spectrograms.py --candidates candidates_optimized.csv --wav-dir "5970 USV" --output-dir spectrograms/ --mode review -v

# Extract in training mode (raw images for CNN)
python scripts/extract_spectrograms.py --candidates candidates_optimized.csv --wav-dir "5970 USV" --output-dir spectrograms_training/ --mode training
```

**Step 2.3 - Batch Extraction Complete:**
- Command: `python scripts/extract_spectrograms.py --candidates candidates_optimized.csv --wav-dir "5970 USV" --output-dir spectrograms_review --mode review -v`
- Output: 490 spectrogram PNGs in `spectrograms_review/`
- Mode: review (with axes/labels for human labeling)
- Verified: Sample spectrogram shows correct frequency range (20-120 kHz), candidate markers, dB colorbar

**Step 2.4 - Documentation Complete:**
- Added Module 5 to `usv_signal_processing_reference.md`
- Documented STFT parameters, frequency range, image dimensions, color scale, render modes
- Recorded final detection and extraction parameters used

**Phase 2 COMPLETE**

**Next Steps:**
- Phase 3 - Labeling Tool (Step 3.1: Create labeling_guide.md)

**Session status:** Phase 2 complete, ready for Phase 3

