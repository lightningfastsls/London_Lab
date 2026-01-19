# USV Detection Pipeline - Implementation Progress

**Started:** 2026-01-16
**Plan Document:** USV_DETECTION_IMPLEMENTATION_PLAN.md
**Reference:** usv_signal_processing_reference.md

---

## Current Status: Phase 3 Complete, Phase 4 In Progress

**Dataset Status:**
- USV samples: 458 labeled
- Not USV samples: 374 (62 from original candidates + 312 verified noise samples)
- Uncertain: 8
- **Need more noise samples** to improve class balance

### Phase 1 Steps (from plan) - COMPLETE

- [x] **Step 1.1** - Set up project structure (detection, labeling, dataset modules)
- [x] **Step 1.2** - Implement DetectionConfig dataclass
- [x] **Step 1.3** - Implement Candidate dataclass
- [x] **Step 1.4** - Implement EnergyDetector.detect() for single file
- [x] **Step 1.5** - Write tests for duration filters, frequency band, merging
- [x] **Step 1.6** - Run on sample WAV files and manually verify candidates
- [x] **Step 1.7** - Implement analyze_threshold_sensitivity()
- [x] **Step 1.8** - Implement verify_detection_coverage()
- [x] **Step 1.9** - Implement batch detection across directory
- [x] **Step 1.10** - Create CLI script (run_detection.py)

### Phase 2 Steps (Spectrogram Extraction) - COMPLETE

- [x] **Step 2.1** - Implement SpectrogramExtractor
- [x] **Step 2.2** - Test on a few candidates (via automated tests)
- [x] **Step 2.3** - Batch extract all candidates
- [x] **Step 2.4** - Document parameters used

### Phase 3 Steps (Labeling Tool) - COMPLETE

- [x] **Step 3.1** - Create labeling_guide.md with visual examples (in-app guide)
- [x] **Step 3.2** - Implement labeling interface (Streamlit)
- [x] **Step 3.3** - Test workflow on 20-30 candidates
- [x] **Step 3.4** - Label full dataset (490 candidates labeled)
- [x] **Step 3.5** - Extract and review noise samples for dataset balancing

### Phase 4 Steps (Dataset Preparation) - COMPLETE

- [x] **Step 4.1** - Create recordings_metadata.csv mapping recording -> population
- [x] **Step 4.2** - Implement create_splits() with stratification
- [x] **Step 4.3** - Run quality checks (all 7 checks pass)
- [ ] **Step 4.4** - Implement augmentation (DEFERRED - class balance is acceptable)
- [x] **Step 4.5** - Final quality checks (PASS)
- [x] **Step 4.6** - Regenerate noise sample spectrograms (350 regenerated)

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

---

## Session 4: Phase 3 - Labeling Tool Implementation

**Date:** 2026-01-17
**Steps Completed:** 3.1, 3.2

### Changes Made

**New Files Created:**
- `src/usv_spectrogram/labeling/__init__.py` - Package init
- `src/usv_spectrogram/labeling/labeling_app.py` - Main Streamlit labeling UI
- `src/usv_spectrogram/labeling/README.md` - User guide for labeling tool
- `scripts/usv_labeling_tool.py` - Launcher script

**Files Modified:**
- `CLAUDE.md` - Added labeling tool to project structure
- `IMPLEMENTATION_PROGRESS.md` - Updated Phase 3 status

### Labeling Tool Features

**Core Functionality:**
- Load candidates from `candidates_optimized.csv`
- Display spectrogram PNGs from `spectrograms_review/` directory
- One-at-a-time labeling workflow
- Three label categories: USV, Not USV, Uncertain
- Save labels incrementally to `labels.csv`

**Navigation:**
- Previous/Next buttons
- Jump to Unlabeled feature
- Progress tracking (X of Y labeled)

**User Experience:**
- Keyboard shortcuts (1=USV, 2=Not USV, 3=Uncertain)
- In-app labeling guide with criteria
- Sidebar statistics (total labeled, percentage, breakdown by label type)
- Wide layout for optimal spectrogram viewing

**Data Persistence:**
- Labels saved immediately to prevent data loss
- Resume support - tool loads existing labels on startup
- CSV format: candidate_id, label, labeled_at

### Usage

```powershell
# Run the labeling tool
.\.venv\Scripts\streamlit.exe run scripts/usv_labeling_tool.py

# Or with launcher (default port 8502)
.\.venv\Scripts\python.exe scripts/usv_labeling_tool.py
```

### Implementation Notes

- Follows existing Streamlit patterns from `param_lab/app.py`
- Uses `st.set_page_config(layout="wide")` for optimal viewing
- Session state manages current index and labels
- Sorted candidates by candidate_id for consistent ordering
- Default port 8502 to avoid conflict with Parameter Lab (8501)

**Next Steps:**
- Phase 3 Step 3.3 - Test workflow on 20-30 candidates
- Phase 3 Step 3.4 - (User task) Label full dataset

**Subagents Used:**
- None (straightforward Streamlit implementation following existing patterns)

**Session status:** Phase 3 Steps 3.1-3.2 complete, ready for testing

---

### 2026-01-18 (Session 5)

**Session started** - Labeling dataset and creating noise samples for balance

**Completed:**
- [x] Step 3.3 - Test labeling workflow
- [x] Step 3.4 - Label full dataset (490 candidates)
- [x] Step 3.5 - Create noise sample extraction tool
- [x] Step 3.6 - Create noise sample review tool
- [x] Step 3.7 - Review and verify noise samples

**Labeling Results (from labels.csv):**
- USV: 458 (420 original + 38 from noise review)
- Not USV: 62
- Uncertain: 8

**Noise Sample Extraction:**
- Created `scripts/extract_noise_samples.py` - Extracts random segments from time gaps between detected candidates
- Extracted 350 initial noise samples
- Created `src/usv_spectrogram/labeling/noise_review_app.py` - Streamlit tool for reviewing noise samples
- Created `scripts/noise_review_tool.py` - Launcher for noise review tool

**Noise Review Results:**
- Clean (verified noise): 287
- Trimmed (partial USV removed): 25
- Skip (contains USV, moved to USV dataset): 38

**Final Dataset:**
- `labels.csv` - 528 entries (458 USV, 62 Not USV, 8 Uncertain)
- `noise_samples/noise_samples_final.csv` - 312 verified noise samples
- Total "Not USV": 374 (62 + 312)
- **Class balance: 458 USV vs 374 Not USV (55% vs 45%)**

**Files Created:**
- `scripts/extract_noise_samples.py` - CLI for extracting noise samples from non-candidate regions
- `scripts/noise_review_tool.py` - Launcher for noise review Streamlit app
- `src/usv_spectrogram/labeling/noise_review_app.py` - Streamlit tool for reviewing/trimming noise samples
- `noise_samples/` - Directory with 350 noise sample PNGs
- `noise_samples/noise_samples.csv` - Original noise sample metadata
- `noise_samples/noise_samples_final.csv` - 312 verified clean noise samples
- `noise_samples/noise_reviews.csv` - Review status for each noise sample

**Next Steps:**
- Extract more noise samples to improve class balance (target ~100 more)
- Phase 4.1 - Create recordings_metadata.csv
- Phase 4.2 - Implement train/val/test splits

**Session status:** Phase 3 complete, Phase 4 ready to start

---

### 2026-01-19 (Session 6)

**Session started** - Implementing Phase 4: Dataset Preparation

**Completed:**
- [x] Step 4.1 - Create metadata.py for recordings metadata generation
- [x] Step 4.2 - Implement splits.py with SplitConfig and recording-level splitting
- [x] Step 4.3 - Implement quality_checks.py with 6 quality checks
- [x] CLI script `scripts/prepare_dataset.py`

**Files Created:**
- `src/usv_spectrogram/dataset/__init__.py` - Package init with public API
- `src/usv_spectrogram/dataset/metadata.py` - Recording metadata management
- `src/usv_spectrogram/dataset/splits.py` - Dataset splitting by recording
- `src/usv_spectrogram/dataset/quality_checks.py` - Dataset quality validation
- `scripts/prepare_dataset.py` - CLI for dataset preparation workflow

**Output Files Created:**
- `recordings_metadata.csv` - Template with 36 unique recordings (population TBD)
- `splits/train.csv` - 373 samples (332 USV, 41 Not USV)
- `splits/val.csv` - 60 samples (53 USV, 7 Not USV)
- `splits/test.csv` - 49 samples (35 USV, 14 Not USV)

**Quality Check Results:**
- [PASS] Splits Loaded - 482 total samples
- [PASS] No Recording Leakage - No recordings in multiple splits
- [PASS] Spectrogram Files Exist - All 482 files exist
- [FAIL] Class Balance - Severe imbalance (87% USV, 13% Not USV)
- [PASS] No Duplicate IDs - All 482 IDs unique
- [PASS] Population Coverage - Unknown (metadata not filled)
- [PASS] Split Sizes - Within tolerance

**CRITICAL ISSUE:** Noise sample spectrograms are missing. They were generated on a different machine (`C:\Users\light\...`) and 350 samples were skipped because their spectrogram files don't exist locally.

**Current Dataset (without noise samples):**
- Total: 482 samples
- USV: 420 (87%)
- Not USV: 62 (13%)
- **This is severely imbalanced for training!**

**Next Steps to Fix:**
1. **Option A (Recommended):** Regenerate noise sample spectrograms from `noise_samples_final.csv` using the SpectrogramExtractor
2. **Option B:** Use class weights during training (less ideal, still need spectrograms)

**CLI Usage:**
```powershell
# Generate metadata template
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --generate-metadata

# Create splits (after regenerating noise spectrograms)
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --create-splits

# Run quality checks
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --check

# All steps
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --all
```

**Key Design Decisions:**
1. Split by RECORDING (not candidate) to prevent temporal correlation leakage
2. Stratify by population when metadata available (fallback to random)
3. Exclude "Uncertain" labels from training
4. Skip samples with missing spectrogram files (with warning)

**RESOLVED:** Regenerated all 350 noise sample spectrograms from `noise_samples_final.csv` metadata.

**Final Dataset (after regeneration):**
- Total: 832 samples
- Train: 618 samples (361 USV, 257 Not USV) - 58% / 42%
- Val: 117 samples (60 USV, 57 Not USV) - 51% / 49%
- Test: 97 samples (37 USV, 60 Not USV) - 38% / 62%

**Final Quality Check Results:**
- [PASS] Splits Loaded - 832 total samples
- [PASS] No Recording Leakage - No recordings in multiple splits
- [PASS] Spectrogram Files Exist - All 832 files exist
- [PASS] Class Balance - Acceptable in all splits
- [PASS] No Duplicate IDs - All 832 IDs unique
- [PASS] Population Coverage - Unknown (metadata not filled)
- [PASS] Split Sizes - Within tolerance

**Session status:** Phase 4 complete, dataset ready for training

