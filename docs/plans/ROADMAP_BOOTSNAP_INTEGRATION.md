# Phase 15: BootSnap Pretrained Classifier Integration — Implementation Roadmap

> **Purpose:** Integrate the BootSnap pretrained CNN ensemble classifier (Abbasi et al., 2022, PLOS Comp Bio)
> with the existing USV detection pipeline. BootSnap classifies detected USVs into ~12 syllable categories
> using Gammatone spectrogram input. This is the **fastest path** to syllable-typed data for the May presentation.
>
> **Approach:** Use BootSnap's **pretrained** model (not retrain) on USVs already detected by our energy
> detector + CNN pipeline. This avoids training from scratch and leverages BootSnap's validation on both
> wild-derived and lab mice — exactly our comparison axis.
>
> **Source:** `bootsnap_integration_plan.md` (web Claude conversation output)
> **Repo:** https://github.com/ReyhanehAbbasi/BootSnap (Apache-2.0)
> **Paper:** Abbasi et al., 2022, PLOS Computational Biology

---

## 15.1 Repository Setup & Dependency Environment

**What:** Clone the BootSnap repo, set up an isolated Python environment with its dependencies, locate pretrained model weights, and document the repo structure. This is the foundation step — everything else depends on having a working BootSnap installation.
**Status:** FUTURE
**Review Tier:** 1
**Depends on:** None

/implement BootSnap Repository Setup & Dependency Environment (Phase 15.1)

Set up the BootSnap pretrained classifier repository and verify all dependencies work. This is a prerequisite for all subsequent phases.

**Context:** BootSnap (https://github.com/ReyhanehAbbasi/BootSnap) is a pretrained CNN ensemble for mouse USV syllable classification, published in PLOS Comp Bio (Abbasi et al., 2022). It classifies USVs into ~12 categories: 'c' (complex), 'h' (complex+harmonics), 'c2' (2 jumps), 'c3' (3+ jumps), 'up', 'd' (down), 'ui' (up-inverted), 's' (short), 'f' (flat), 'u' (unstructured), 'us' (ultra-short), 'composite', and 'FP' (false positive). The key directory is `applying_classifier_on_EV_data/` which contains code for applying pretrained models to new data.

**Action items:**

1. Clone the repo into a `vendor/` or `external/` directory (NOT into the main src tree)
   ```bash
   git clone https://github.com/ReyhanehAbbasi/BootSnap.git external/BootSnap
   ```

2. Explore the repo structure:
   - Read the README.md
   - List all files in `applying_classifier_on_EV_data/` (this is the inference pipeline)
   - List all files in `train_classifier/` (reference only, not needed now)
   - Identify the main entry-point script for inference

3. Locate pretrained model weights:
   - Search for `.h5`, `.pt`, `.pkl`, `.pth`, `.keras`, `.onnx` files
   - If weights are missing from the repo, document what's needed and where to request them
   - Check if weights need downloading from an external URL

4. Identify dependencies:
   - Check for `requirements.txt`, `setup.py`, `pyproject.toml`, or imports at top of scripts
   - Determine the ML framework (likely TensorFlow/Keras or PyTorch)
   - Check Python version requirements

5. Create an isolated environment setup script:
   ```
   scripts/setup_bootsnap_env.py (or .ps1)
   ```
   That handles: venv/conda creation, dependency installation, model weight verification

6. Document findings in `docs/bootsnap_setup_report.md`:
   - Repo structure diagram
   - List of dependencies with versions
   - Location of pretrained weights
   - Any compatibility issues with our Python 3.12.1 / Windows environment
   - Entry point script path for inference

**Exit criteria:**
- [ ] BootSnap repo cloned and accessible
- [ ] All dependencies identified and documented
- [ ] Pretrained model weights located (or download instructions documented)
- [ ] Can import the main inference module without errors
- [ ] Setup report written at `docs/bootsnap_setup_report.md`

---

## 15.2 Gammatone Preprocessing & Input Format Analysis

**What:** Understand and implement BootSnap's expected input format — Gammatone spectrograms with specific parameters. This is the critical bridge component: our pipeline produces STFT spectrograms, but BootSnap requires Gammatone spectrograms computed from raw audio segments.
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 15.1

**Key design decisions:**
- BootSnap uses Gammatone filter bank (~128 filters), NOT standard STFT
- Center frequency midpoint optimized at 68 kHz
- Frequency range: 20–120 kHz
- Input is time-windowed audio around each detected USV, then converted to Gammatone spectrogram

/implement Gammatone Preprocessing & Input Format Analysis (Phase 15.2)

Reverse-engineer BootSnap's preprocessing pipeline and build a Gammatone spectrogram generator that converts our detected USV audio segments into BootSnap-compatible input tensors.

**Context:** BootSnap does NOT use standard STFT spectrograms — it uses Gammatone filter bank spectrograms (inspired by the mammalian auditory system). The Gammatone filterbank uses ~128 filters spanning 20–120 kHz with center frequency at ~68 kHz. Our detection pipeline (Phase 1) uses 512-point STFT at 300 kHz sample rate (ADR-002), but BootSnap needs raw audio re-processed through its own spectrogram pipeline. The key code to analyze is in BootSnap's `applying_classifier_on_EV_data/` directory.

**Files to create:**

1. `src/usv_spectrogram/classification/gammatone_config.py` (NEW) — Configuration

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class GammatoneConfig:
    """Configuration for Gammatone spectrogram generation (BootSnap-compatible)."""
    n_filters: int = 128              # Number of Gammatone filters
    freq_min_hz: int = 20_000         # Lower frequency bound
    freq_max_hz: int = 120_000        # Upper frequency bound
    center_freq_hz: int = 68_000      # Center frequency (BootSnap optimized)
    sr: int = 300_000                 # ADR-001: always explicit
    # Padding around detected USV segment
    padding_ms: float = 10.0          # Padding each side for context
    # Output tensor shape (verify from BootSnap code)
    output_height: int = 128          # TBD: read from BootSnap preprocessing
    output_width: int = 128           # TBD: read from BootSnap preprocessing
    output_channels: int = 1          # TBD: 1 (grayscale) or 3 (RGB)
```

2. `src/usv_spectrogram/classification/gammatone_extractor.py` (NEW) — Extraction logic

```python
class GammatoneExtractor:
    """Extract Gammatone spectrograms from audio segments for BootSnap classification."""

    def __init__(self, config: GammatoneConfig): ...

    def extract_segment(self, wav_path: Path, start_s: float, end_s: float) -> np.ndarray:
        """
        Extract Gammatone spectrogram for a single USV segment.

        Steps:
        1. Load audio [start - padding, end + padding] from WAV at 300 kHz
        2. Apply Gammatone filterbank (128 filters, 20-120 kHz)
        3. Compute envelope of each filter output
        4. Apply any normalization BootSnap expects (read from their code)
        5. Resize to BootSnap's expected input dimensions
        6. Return as numpy array ready for classifier input
        """
        ...

    def extract_batch(self, detections: list[dict], wav_dir: Path) -> list[np.ndarray]:
        """Extract Gammatone spectrograms for a batch of detections."""
        ...
```

3. `tests/test_gammatone_extractor.py` (NEW)

**Test plan:**
```
1. Gammatone filterbank produces correct number of output channels (128)
2. Frequency range covers 20-120 kHz (not outside)
3. Output tensor matches BootSnap's expected input shape
4. Padding correctly extends segment by configured padding_ms
5. Segments near WAV file boundaries are clamped gracefully
6. Very short USVs (<5 ms) produce valid output (not empty)
7. Output normalization matches BootSnap's expectations
```

**Action items:**
- Read BootSnap's preprocessing code line by line — document every transformation
- Identify exact input tensor shape, dtype, and normalization
- Verify whether BootSnap uses `gammatone` pip package or custom implementation
- Check if A-MUD detection format is expected or if any timestamp format works
- Document all parameters in the config dataclass

**Exit criteria:**
- [ ] BootSnap's preprocessing fully understood and documented
- [ ] GammatoneConfig captures all parameters with correct defaults
- [ ] GammatoneExtractor produces output that matches BootSnap's expected input shape
- [ ] Can generate a Gammatone spectrogram from a real WAV segment without errors
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## 15.3 Detection-to-BootSnap Bridge Adapter

**What:** Build the core bridge module that takes our detection pipeline output (Candidate CSV format), extracts audio segments, generates Gammatone spectrograms, feeds them to the BootSnap classifier, and produces enriched output with syllable predictions. This is the main coding deliverable.
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 15.1, Phase 15.2

**Key design decisions:**
- Input: our detection CSV format (columns: `wav_file`, `start_time_s`, `end_time_s`, `duration_ms`, etc.)
- Output: same CSV enriched with `predicted_class`, `confidence`, `classifier_version` columns
- BootSnap's 'FP' (false positive) class doubles as a denoising signal
- Confidence threshold configurable — low-confidence predictions can be flagged

/implement Detection-to-BootSnap Bridge Adapter (Phase 15.3)

Build the adapter that connects our USV detection pipeline output to BootSnap's pretrained classifier and produces syllable-typed results.

**Context:** Our detection pipeline outputs CSV files in `USV_Detections/` with columns including `wav_file`, `start_time_s`, `end_time_s`, `duration_ms`. The Candidate dataclass (`src/usv_spectrogram/detection/candidate.py`) has fields: `candidate_id`, `start_ms`, `end_ms`, `duration_ms`, `peak_freq_hz`, `peak_energy_db`, `spectrogram_path`. BootSnap classifies into ~12 categories plus 'FP' (false positive). The bridge must: (1) read our detection output, (2) for each USV extract audio + compute Gammatone spectrogram using Phase 15.2's extractor, (3) run BootSnap inference, (4) produce enriched CSV with syllable labels. Sample rate is always 300 kHz (ADR-001).

**Files to create:**

1. `src/usv_spectrogram/classification/bootsnap_config.py` (NEW)

```python
from dataclasses import dataclass, field
from pathlib import Path

# BootSnap syllable categories (Abbasi et al., 2022)
BOOTSNAP_CLASSES = [
    "c",          # complex
    "h",          # complex + harmonics
    "c2",         # 2 frequency jumps
    "c3",         # 3+ frequency jumps
    "up",         # upward sweep
    "d",          # downward sweep
    "ui",         # up-inverted
    "s",          # short
    "f",          # flat
    "u",          # unstructured
    "us",         # ultra-short
    "composite",  # composite call
    "FP",         # false positive (denoising signal)
]

@dataclass(frozen=True)
class BootSnapConfig:
    """Configuration for BootSnap classifier bridge."""
    model_dir: Path = Path("external/BootSnap/applying_classifier_on_EV_data")
    weights_path: Path | None = None  # Path to pretrained weights (found in 15.1)
    n_classes: int = 13
    class_names: tuple[str, ...] = tuple(BOOTSNAP_CLASSES)

    # Inference
    batch_size: int = 32
    confidence_threshold: float = 0.5  # Flag predictions below this
    use_ensemble: bool = True          # BootSnap snapshot ensemble

    # Input/output
    detections_csv: Path = Path("USV_Detections/detections_summary.csv")
    wav_dir: Path = Path("5970 USV")
    output_dir: Path = Path("USV_Detections/classified")
```

2. `src/usv_spectrogram/classification/bootsnap_bridge.py` (NEW) — Core adapter

```python
class BootSnapBridge:
    """
    Bridge between USV detection pipeline and BootSnap pretrained classifier.

    Pipeline: Detection CSV → Audio extraction → Gammatone spectrogram → BootSnap inference → Enriched CSV
    """

    def __init__(self, config: BootSnapConfig, gammatone_config: GammatoneConfig): ...

    def load_model(self) -> None:
        """Load BootSnap pretrained model (or ensemble of snapshots)."""
        ...

    def classify_detections(self, detections_csv: Path) -> pd.DataFrame:
        """
        Main entry point: classify all detections from a CSV file.

        Steps:
        1. Read detection CSV (our format: wav_file, start_time_s, end_time_s, ...)
        2. For each detection, extract audio segment from source WAV
        3. Generate Gammatone spectrogram via GammatoneExtractor
        4. Batch predictions through BootSnap model
        5. Return DataFrame with original columns + predicted_class, confidence, low_confidence_flag
        """
        ...

    def classify_single(self, wav_path: Path, start_s: float, end_s: float) -> dict:
        """Classify a single USV segment. Returns {class, confidence, probabilities}."""
        ...

    def _batch_predict(self, spectrograms: list[np.ndarray]) -> list[dict]:
        """Run batch inference through BootSnap. Returns per-item predictions."""
        ...

    def export_results(self, results: pd.DataFrame, output_path: Path) -> None:
        """
        Export classified results as enriched CSV.

        Output columns: all original detection columns +
        predicted_class, confidence, low_confidence_flag, classifier_version, fp_flagged
        """
        ...
```

3. `scripts/classify_usv_bootsnap.py` (NEW) — CLI entry point

```
Usage:
  .\.venv\Scripts\python.exe scripts/classify_usv_bootsnap.py \
      --detections USV_Detections/detections_summary.csv \
      --wav-dir "5970 USV" \
      --output-dir USV_Detections/classified \
      --confidence-threshold 0.5 \
      --ensemble
```

Output structure:
```
USV_Detections/classified/
├── classified_detections.csv    # Original columns + predicted_class, confidence
├── classification_report.json   # Class distribution, confidence stats, FP count
└── low_confidence_flagged.csv   # Detections below confidence threshold
```

4. `tests/test_bootsnap_bridge.py` (NEW)

**Test plan:**
```
1. Bridge reads detection CSV and produces output with correct additional columns
2. classify_single returns dict with 'class', 'confidence', 'probabilities' keys
3. FP class detections are correctly flagged in output
4. Low confidence detections (below threshold) are flagged
5. Batch prediction handles empty input gracefully
6. Output CSV preserves all original detection columns
7. Edge case: detection with no corresponding WAV file raises clear error
8. Edge case: very short detection (<5 ms) is handled
```

**Exit criteria:**
- [ ] Bridge reads our detection CSV format and produces enriched output
- [ ] BootSnap model loads and produces predictions
- [ ] All 13 syllable categories appear in predictions (or documented why not)
- [ ] FP class flags false positives from our detection pipeline
- [ ] CLI script runs end-to-end on a single detection
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## 15.4 Validation on Test Batch

**What:** Run the bridge on a small curated batch (~100 USVs from both wild and lab mice), manually verify classifications against spectrograms, and check for systematic issues before committing to the full dataset run.
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 15.3

/implement BootSnap Validation on Test Batch (Phase 15.4)

Run the BootSnap bridge on ~100 curated USV detections from both wild and lab mice, generate diagnostic outputs for manual verification, and check for systematic classification issues.

**Context:** Before running on the full dataset (~1000+ detections), we need to validate that the BootSnap integration produces sensible results on our specific recordings. BootSnap was trained on CBA/CaJ and wild-derived Mus musculus musculus mice — performance on our specific populations may differ. Key things to check: (1) category distribution looks plausible (not all one class), (2) FP class catches actual noise detections, (3) confidence scores are calibrated, (4) visual spot-check of ~20 classified spectrograms.

**Files to create:**

1. `scripts/validate_bootsnap_batch.py` (NEW) — Validation runner

```python
"""
Run BootSnap on a curated test batch and generate diagnostic outputs.

Usage:
  .\.venv\Scripts\python.exe scripts/validate_bootsnap_batch.py \
      --detections USV_Detections/detections_summary.csv \
      --wav-dir "5970 USV" \
      --n-samples 100 \
      --output-dir analysis/bootsnap_validation \
      --include-wild --include-lab
"""
```

Script should:
1. Sample ~50 wild + ~50 lab detections (stratified by recording)
2. Run BootSnap classification on the sample
3. Generate summary table: usv_id, predicted_class, confidence, source_file, population (wild/lab)
4. Generate spectrogram + label gallery (grid of classified spectrograms for visual review)
5. Compute basic stats: class distribution, mean confidence per class, FP rate
6. Flag systematic issues: >50% in one class, mean confidence <0.3, zero FP detections

Output structure:
```
analysis/bootsnap_validation/
├── validation_results.csv          # Per-USV: id, class, confidence, population
├── class_distribution.png          # Bar chart: class counts (wild vs lab stacked)
├── confidence_histogram.png        # Histogram of confidence scores
├── spectrogram_gallery/            # Grid PNGs of classified spectrograms
│   ├── gallery_page_01.png         # 5x4 grid with labels
│   └── ...
├── issues_flagged.md               # Any systematic problems found
└── validation_summary.json         # Stats: n_processed, class_counts, mean_confidence
```

2. `tests/test_bootsnap_validation.py` (NEW)

**Test plan:**
```
1. Stratified sampling selects from both wild and lab recordings
2. Gallery generation creates valid PNG images
3. Confidence histogram uses correct bin ranges [0, 1]
4. Issue detection flags when >50% predictions are one class
5. Summary JSON contains all required fields
```

**Exit criteria:**
- [ ] 100 USVs classified without errors
- [ ] Category distribution looks plausible (not all one class)
- [ ] Mean confidence > 0.3 (if lower, BootSnap may not work on our data)
- [ ] FP class catches at least some noise detections (if any noise in sample)
- [ ] Visual spot-check of ~20 spectrograms: labels mostly make sense
- [ ] No systematic issues flagged (or issues documented with workaround)
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## 15.5 Full Dataset Classification & Repertoire Comparison

**What:** Run BootSnap on ALL detected USVs, generate the master classified dataset, compute wild vs. lab repertoire comparison statistics, and document limitations. This produces the deliverable for the May presentation.
**Status:** FUTURE
**Review Tier:** 2
**Depends on:** Phase 15.4

**Key design decisions:**
- Master CSV with all original detection columns + syllable classification
- Repertoire statistics: per-group syllable proportions, complexity ratios
- BootSnap was trained on specific strains — document generalization caveats
- Confidence-based filtering: report results both with and without low-confidence exclusions

/implement Full Dataset Classification & Repertoire Comparison (Phase 15.5)

Run BootSnap classifier on the full USV dataset, generate the master classified CSV, compute wild vs. lab syllable repertoire comparison statistics, and document limitations for the May presentation.

**Context:** This phase produces the final deliverable: syllable-typed USV data comparing wild field mice vs. laboratory mice. BootSnap (Abbasi et al., 2022) found that classifiers trained on lab mice generalize poorly to wild mice and vice versa — this is a relevant caveat for our results since we're using their pretrained model. The 'FP' class provides additional denoising beyond our CNN pipeline. Report statistics both with and without low-confidence exclusions to show robustness. Key metrics: syllable repertoire distribution per group, complexity ratio (simple vs. multi-jump calls), and per-syllable-type proportions.

**Files to create:**

1. `scripts/classify_full_dataset.py` (NEW) — Batch processing

```python
"""
Run BootSnap classification on the full USV detection dataset.

Usage:
  .\.venv\Scripts\python.exe scripts/classify_full_dataset.py \
      --detections USV_Detections/detections_summary.csv \
      --wav-dir "5970 USV" \
      --output-dir USV_Detections/classified_full \
      --confidence-threshold 0.5 \
      --metadata analysis/metadata.csv
"""
```

Output: master CSV with columns:
```
wav_file, detection_index, start_time_s, end_time_s, duration_ms,
max_prob, mean_prob, peak_freq_hz,
predicted_class, confidence, low_confidence_flag, fp_flagged,
population (wild/lab), animal_id (if available)
```

2. `scripts/compare_repertoires.py` (NEW) — Statistical comparison

```python
"""
Compare syllable repertoires between wild and lab mouse populations.

Usage:
  .\.venv\Scripts\python.exe scripts/compare_repertoires.py \
      --classified USV_Detections/classified_full/master_classified.csv \
      --output-dir analysis/repertoire_comparison
"""
```

Analysis outputs:
```
analysis/repertoire_comparison/
├── repertoire_distribution.png      # Stacked bar: syllable proportions (wild vs lab)
├── complexity_comparison.png        # Simple vs complex call ratios
├── syllable_counts.csv             # Raw counts per class per population
├── proportions_table.csv           # Proportions with 95% CI
├── chi_squared_test.json           # Chi-squared test on syllable distributions
├── fp_analysis.md                  # How many FPs per population, what they look like
├── confidence_analysis.png         # Confidence by class and population
├── presentation_summary.md         # Key findings formatted for May presentation
└── limitations.md                  # Documented caveats (strain specificity, etc.)
```

3. `tests/test_repertoire_comparison.py` (NEW)

**Test plan:**
```
1. Master CSV contains all original detections plus classification columns
2. Syllable proportions sum to 1.0 per population (excluding FP)
3. Chi-squared test returns p-value in [0, 1]
4. Complexity ratio computed correctly (simple = f+s+us+up+d, complex = c+h+c2+c3+composite)
5. Confidence filtering produces fewer rows than unfiltered
6. Presentation summary includes wild vs lab comparison
```

**Limitations to document (in `limitations.md`):**
- BootSnap trained on CBA/CaJ and wild-derived Mus musculus musculus — our strains may differ
- Pretrained model NOT fine-tuned on our recording setup/conditions (microphone, room acoustics)
- The 'FP' predictions flag potential false positives but aren't ground truth
- Low-confidence predictions should be interpreted cautiously
- Cross-strain generalization is a known weak point (BootSnap's own finding)
- Classification confidence ≠ classification correctness

**Exit criteria:**
- [ ] All detected USVs classified (master CSV complete)
- [ ] Repertoire statistics computed for wild vs. lab populations
- [ ] At least 3 visualization plots generated for the presentation
- [ ] Chi-squared test (or equivalent) computed on syllable distributions
- [ ] FP analysis completed — documents how many noise detections flagged
- [ ] Limitations documented thoroughly
- [ ] Presentation summary written with key findings
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

## Phase 15 Dependencies

```
Phase 1 (detection) ──────────────────→ Phase 15.1 (setup)
                                              │
                                              ▼
                                        Phase 15.2
                                     (Gammatone preprocessing)
                                              │
                                              ▼
                                        Phase 15.3
                                     (bridge adapter)
                                              │
                                              ▼
                                        Phase 15.4
                                     (validation batch)
                                              │
                                              ▼
                                        Phase 15.5
                                  (full dataset + comparison)
                                              │
                                              ▼
                                    May Presentation
```

## Phase 15 External Dependencies

```
tensorflow / keras (or pytorch)    # BootSnap's ML framework (TBD from 15.1)
gammatone                          # Gammatone filterbank (pip install gammatone)
librosa                            # Audio loading
pandas                             # CSV handling
matplotlib, seaborn                # Visualization
scipy                              # Statistical tests
```

## Phase 15 Gate

Before using results in the presentation:
- [ ] BootSnap repo cloned and model weights available (15.1)
- [ ] Gammatone preprocessing validated against BootSnap's own code (15.2)
- [ ] Bridge adapter runs end-to-end on single detection (15.3)
- [ ] Validation batch shows plausible results (15.4)
- [ ] Full dataset classified with master CSV (15.5)
- [ ] Wild vs. lab comparison statistics computed (15.5)
- [ ] Limitations documented (15.5)
- [ ] All Phase 15 tests pass
- [ ] py_compile passes on all new files

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Pretrained weights missing from repo | Email Reyhaneh Abbasi (reyhaneh.abbasi@oeaw.ac.at) requesting model files |
| Input format incompatible | Write custom Gammatone generator based on paper specs (filter order, center freqs, bandwidth) |
| Poor classification quality on our data | Fall back to broader groupings: 'no-jump' vs 'jumps' vs 'FP' (~95% F1 in paper) |
| Python/TensorFlow version conflicts | Create dedicated conda environment for BootSnap, isolated from main venv |
| Confidence scores systematically low | Report relative rankings instead of absolute thresholds; compare within-population |

## Key References

- Abbasi et al. (2022), PLOS Computational Biology — BootSnap: snapshot ensemble CNN for USV classification
- BootSnap GitHub: https://github.com/ReyhanehAbbasi/BootSnap
- Our detection pipeline: `src/usv_spectrogram/detection/` (Phase 1, DONE)
- ADR-001: 300 kHz sample rate, ADR-002: STFT parameters
