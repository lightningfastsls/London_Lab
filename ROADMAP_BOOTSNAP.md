# BootSnap USV Syllable Classification — Implementation Roadmap

> **Source plan:** `bootsnap_integration_plan.md`
> **Goal:** Classify detected USVs into syllable types (simple, complex, frequency-jump, etc.) using BootSnap's pretrained CNN ensemble classifier, enabling wild vs. lab mouse vocalization comparison for the May presentation.
> **Repo:** https://github.com/ReyhanehAbbasi/BootSnap (Apache-2.0, Python, Abbasi et al. 2022 PLOS Comp Bio)
> **Categories:** c, h, c2, c3, up, d, ui, s, f, u, us, composite, FP (false positive)

---

## How to Use This File

1. Work through modules **in order** within Phase 15 (dependencies are noted)
2. Each module has:
   - **What**: brief description of what to build
   - **`/implement` command**: copy-paste into Claude Code
   - **Test plan**: how to verify the module works
   - **Exit criteria**: what "done" looks like
3. After each module: commit, run review at the specified tier, fix issues, commit again
4. The phase gate must pass before declaring BootSnap integration complete

## Status Key

- **DONE** — Implemented and tested
- **READY** — Dependencies met, can start
- **BLOCKED** — Waiting on dependency or external input
- **FUTURE** — Not yet prioritized

---

## Phase 15: BootSnap USV Syllable Classification

### 15.1 Environment & Model Verification

**What:** Clone BootSnap repo, install dependencies into a separate environment, verify pretrained model weights load correctly, and create a verification script that confirms the classifier can accept input and produce predictions.
**Status:** READY
**Review Tier:** 1 (setup validation, no DSP logic)
**Depends on:** None (external repo)

/implement BootSnap Environment & Model Verification

Clone the BootSnap repository (https://github.com/ReyhanehAbbasi/BootSnap, Apache-2.0) and set up the classifier for use. The key directory is `applying_classifier_on_EV_data/` which contains the code for applying a pretrained model to new data. The `train_classifier/` directory is for training from scratch — skip it for now.

**Context:** BootSnap (Abbasi et al., 2022, PLOS Comp Bio) is a pretrained CNN ensemble classifier for mouse USV syllables, validated on both wild-derived and lab mice. It classifies into ~12 categories: 'c' (complex), 'h' (complex+harmonics), 'c2' (2 jumps), 'c3' (3+ jumps), 'up', 'd' (down), 'ui' (up-inverted), 's' (short), 'f' (flat), 'u' (unstructured), 'us' (ultra-short), 'composite', and 'FP' (false positive). The FP class is particularly useful as a denoising signal.

**Action items:**

1. Clone BootSnap repo into `external/BootSnap/` (gitignored — do NOT add to version control)
2. Read the README.md and `applying_classifier_on_EV_data/` entry point scripts
3. Identify pretrained model weights (`.h5`, `.pt`, `.pkl`, or similar) — they may be in the repo or may need separate download
4. Identify the Python framework (likely TensorFlow/Keras based on the paper's CNN architecture)
5. Document all required dependencies
6. Create a verification script

**Files to create:**

1. `scripts/verify_bootsnap_setup.py` (NEW) — Environment verification script

    ```python
    @dataclass
    class BootSnapSetupReport:
        repo_found: bool           # BootSnap repo exists at expected path
        weights_found: bool        # Pretrained model weights located
        deps_satisfied: bool       # All Python dependencies importable
        model_loads: bool          # Model weights load without error
        dummy_prediction: bool     # Model produces output on random input tensor
        framework: str             # "tensorflow", "pytorch", or "unknown"
        weight_files: list[str]    # Paths to found weight files
        missing_deps: list[str]    # Any missing Python packages
        input_shape: tuple | None  # Expected input tensor shape (H, W, C)
        num_classes: int | None    # Number of output classes
    ```

    Core logic:
    - `verify_repo(base_path) -> bool` — checks `external/BootSnap/` exists with expected structure
    - `find_weights(repo_path) -> list[Path]` — searches for model weight files recursively
    - `check_dependencies() -> tuple[bool, list[str]]` — tries importing each required package
    - `load_model(weight_path) -> tuple[bool, tuple | None, int | None]` — loads model, extracts input shape and num_classes
    - `run_dummy_prediction(model, input_shape) -> bool` — creates random tensor, runs forward pass, checks output shape
    - `generate_report() -> BootSnapSetupReport` — runs all checks, prints formatted report

2. `tests/test_bootsnap_setup.py` (NEW) — Setup verification tests (lightweight, mocked)

**Test plan:**
    ```
    1. verify_repo returns False when directory doesn't exist
    2. verify_repo returns True when directory has expected structure (mocked)
    3. find_weights locates .h5 / .pt / .pkl files in nested directories (using temp dir with dummy files)
    4. check_dependencies returns missing packages accurately (mock importlib)
    5. generate_report produces BootSnapSetupReport with all fields populated
    6. Script runs without error when BootSnap repo is absent (graceful failure, not crash)
    ```

**Exit criteria:**
- [ ] BootSnap repo cloned to `external/BootSnap/` and gitignored
- [ ] Verification script runs and prints clear pass/fail report
- [ ] Pretrained model weights located and documented
- [ ] Input tensor shape documented (height x width x channels)
- [ ] Framework identified (TensorFlow/Keras or PyTorch)
- [ ] All 6 tests pass
- [ ] py_compile passes on verify_bootsnap_setup.py

**Risk mitigation:**
- If pretrained weights are missing from the repo: email Reyhaneh Abbasi (reyhaneh.abbasi@oeaw.ac.at) requesting trained model files
- If Python version conflicts arise: create a dedicated conda environment for BootSnap and document activation steps

---

### 15.2 Gammatone Spectrogram Generator

**What:** Build a Gammatone spectrogram generator that matches BootSnap's preprocessing specifications. BootSnap expects Gammatone spectrograms (NOT standard STFT), with ~128 filters, center frequency midpoint at 68 kHz, and frequency range 20–120 kHz.
**Status:** BLOCKED
**Review Tier:** 3 (DSP-sensitive — frequency analysis, filter bank design)
**Depends on:** 15.1 (need to verify exact input specs from BootSnap code)

/implement Gammatone Spectrogram Generator

Build a Gammatone filter bank spectrogram generator that produces spectrograms matching BootSnap's expected input format. This is the critical DSP bridge between our 300 kHz WAV recordings and BootSnap's classifier input.

**Context:** BootSnap uses Gammatone spectrograms rather than standard STFT. Gammatone filters model the human auditory system's frequency decomposition — they're log-spaced in frequency (like mel scale) but with a different filter shape. For USV analysis, the filter bank is tuned to the 20–120 kHz range with center midpoint at 68 kHz. Our recordings are at 300 kHz sample rate (ADR-001), giving us a Nyquist of 150 kHz which comfortably covers the 20–120 kHz analysis range. Must specify `sr=300000` explicitly — never rely on library defaults.

**IMPORTANT:** Before implementing, read BootSnap's actual preprocessing code in `external/BootSnap/applying_classifier_on_EV_data/` to verify these specs. The paper says ~128 filters at 68 kHz midpoint, but the code is the ground truth. Document any differences.

**Files to create:**

1. `src/usv_spectrogram/classification/gammatone.py` (NEW) — Gammatone spectrogram generator

    ```python
    @dataclass(frozen=True)
    class GammatoneConfig:
        num_filters: int = 128        # Number of Gammatone filters
        freq_min_hz: float = 20000.0  # Low frequency bound (Hz)
        freq_max_hz: float = 120000.0 # High frequency bound (Hz)
        sample_rate: int = 300000     # MUST match recording sr (ADR-001)
        window_duration_ms: float = 10.0  # Analysis window duration
        hop_duration_ms: float = 2.5      # Hop between windows
        # Output image dimensions (verify from BootSnap code)
        output_height: int = 128      # Pixels (typically = num_filters)
        output_width: int = 128       # Pixels (may vary — check BootSnap)
        normalize: bool = True        # Apply per-spectrogram normalization

    @dataclass
    class GammatoneSpectrogram:
        image: np.ndarray             # Shape: (output_height, output_width, channels)
        config: GammatoneConfig
        time_range_ms: tuple[float, float]   # (start, end) in source audio
        freq_range_hz: tuple[float, float]   # (min, max) frequency coverage
        source_file: Path
    ```

    Core logic:
    - `GammatoneGenerator.__init__(config: GammatoneConfig)` — precomputes filter bank coefficients
    - `_compute_center_frequencies() -> np.ndarray` — ERB-spaced center frequencies from freq_min to freq_max
    - `_build_filterbank() -> np.ndarray` — Gammatone impulse responses for each filter
    - `generate(audio: np.ndarray, sr: int, time_range_ms: tuple) -> GammatoneSpectrogram` — applies filter bank to audio segment, produces spectrogram image
    - `_apply_filters(audio: np.ndarray) -> np.ndarray` — convolves audio with each Gammatone filter
    - `_extract_envelope(filtered: np.ndarray) -> np.ndarray` — Hilbert envelope extraction per channel
    - `_to_image(envelope: np.ndarray) -> np.ndarray` — resizes/normalizes to output dimensions

    Uses `scipy.signal` for filtering. May also use the `gammatone` Python package if BootSnap depends on it — check 15.1 dependency report.

2. `tests/test_gammatone.py` (NEW) — Gammatone generator tests

**Test plan:**
    ```
    1. Center frequencies are ERB-spaced between freq_min and freq_max — verify monotonically increasing
    2. Number of center frequencies equals num_filters (128)
    3. Output spectrogram shape matches (output_height, output_width, channels) from config
    4. Pure tone at 50 kHz produces peak energy in the correct filter band (within ±1 filter)
    5. Pure tone at 10 kHz (below freq_min) produces near-zero response
    6. Pure tone at 140 kHz (above freq_max) produces near-zero response
    7. Normalization produces values in [0, 1] range when normalize=True
    8. Silence input produces near-zero spectrogram (no NaN, no Inf)
    9. Very short segment (< 5ms) handled gracefully — padded or error raised, not crash
    10. Config with sr != 300000 raises ValueError (enforce ADR-001)
    ```

**Exit criteria:**
- [ ] Generated spectrograms visually match BootSnap's expected input (compare with repo examples if available)
- [ ] Output tensor shape matches what BootSnap's model expects (verified in 15.1)
- [ ] Pure tone test localizes to correct frequency band
- [ ] All 10 tests pass
- [ ] py_compile passes on gammatone.py
- [ ] DSP reviewer (`dsp-reviewer` agent) approves filter bank implementation

---

### 15.3 Classification Bridge

**What:** Adapter module that takes detection output (Candidate objects or CSV), extracts audio segments from source WAV files, generates Gammatone spectrograms, and runs them through the BootSnap pretrained model to produce syllable classifications with confidence scores.
**Status:** BLOCKED
**Review Tier:** 2 (standard module with clear logic, no novel DSP)
**Depends on:** 15.1 (model loaded), 15.2 (Gammatone generator)

/implement BootSnap Classification Bridge

Build the adapter that connects our detection pipeline output to BootSnap's classifier input and captures the results. This is the central orchestration module — it reads detected USVs, extracts audio, generates spectrograms, and runs inference.

**Context:** Our detection pipeline produces `Candidate` objects (see `src/usv_spectrogram/detection/candidate.py`) with `source_file` (Path to WAV), `start_ms`, `end_ms`, `peak_freq_hz`, and `peak_energy_db`. The bridge needs to: (1) extract the audio segment from the WAV file using these timestamps, (2) pass it through the Gammatone generator (Phase 15.2), (3) feed the spectrogram to BootSnap's pretrained model, (4) return the predicted syllable type and confidence. Follow the frozen dataclass pattern from `docs/architecture/patterns.md`.

**Files to create:**

1. `src/usv_spectrogram/classification/__init__.py` (NEW) — Package init

2. `src/usv_spectrogram/classification/config.py` (NEW) — Classification configuration

    ```python
    @dataclass(frozen=True)
    class ClassificationConfig:
        bootsnap_repo_path: Path           # Path to external/BootSnap/
        model_weights_path: Path | None = None  # Auto-detected if None
        gammatone_config: GammatoneConfig = field(default_factory=GammatoneConfig)
        confidence_threshold: float = 0.5  # Below this, mark as "low_confidence"
        batch_size: int = 32               # Inference batch size
        context_padding_ms: float = 10.0   # Extra audio around USV boundaries
        sample_rate: int = 300000          # Must match recording sr (ADR-001)
    ```

3. `src/usv_spectrogram/classification/bridge.py` (NEW) — Core classification bridge

    ```python
    @dataclass
    class ClassifiedUSV:
        candidate: Candidate             # Original detection
        syllable_type: str               # BootSnap category: c, h, c2, c3, up, d, ui, s, f, u, us, composite, FP
        confidence: float                # Model confidence (0-1)
        top_3_predictions: list[tuple[str, float]]  # [(class, confidence), ...] for analysis
        low_confidence: bool             # True if confidence < threshold
        spectrogram: GammatoneSpectrogram | None = None  # Optionally retain for visualization

    @dataclass
    class ClassificationReport:
        total_candidates: int
        classified: int
        failed: int                      # Candidates that couldn't be processed
        fp_count: int                    # Predicted as FP (false positive)
        low_confidence_count: int
        class_distribution: dict[str, int]  # syllable_type -> count
        mean_confidence: float
        processing_time_s: float
        failures: list[tuple[str, str]]  # [(candidate_id, error_message), ...]
    ```

    Core logic:
    - `BootSnapBridge.__init__(config: ClassificationConfig)` — loads model, initializes Gammatone generator
    - `_load_model() -> Any` — loads pretrained weights from BootSnap repo
    - `_extract_audio(candidate: Candidate) -> np.ndarray` — reads WAV segment with context padding
    - `classify_one(candidate: Candidate) -> ClassifiedUSV` — full pipeline for single USV
    - `classify_batch(candidates: list[Candidate]) -> list[ClassifiedUSV]` — batched inference for efficiency
    - `classify_from_csv(csv_path: Path, wav_dir: Path) -> list[ClassifiedUSV]` — reads detection CSV, resolves WAV paths
    - `_generate_report(results: list[ClassifiedUSV], failures, elapsed) -> ClassificationReport` — summary statistics

4. `scripts/classify_usvs.py` (NEW) — CLI entry point

    ```
    Usage: python scripts/classify_usvs.py --detections <CSV> --wav-dir <DIR> --output <CSV>
    Options:
      --detections    Path to detection CSV (Candidate format)
      --wav-dir       Directory containing source WAV files
      --output        Path for classified output CSV
      --confidence    Minimum confidence threshold (default: 0.5)
      --batch-size    Inference batch size (default: 32)
      --keep-fp       Include FP-classified USVs in output (default: exclude)
      --verbose       Print progress per file
    ```

5. `tests/test_classification_bridge.py` (NEW) — Bridge tests

**Test plan:**
    ```
    1. ClassifiedUSV has correct syllable_type from mocked model output
    2. Confidence below threshold sets low_confidence=True
    3. top_3_predictions sorted by descending confidence
    4. _extract_audio reads correct time range from WAV (verify sample indices match start_ms/end_ms)
    5. _extract_audio with context_padding_ms extends segment boundaries without exceeding file bounds
    6. classify_batch processes multiple candidates and returns same-length list
    7. Failed classification (corrupt WAV, missing file) adds to failures list, doesn't crash batch
    8. ClassificationReport.class_distribution sums to classified count
    9. ClassificationReport.fp_count matches count of syllable_type == "FP"
    10. classify_from_csv reads detection CSV and resolves WAV paths correctly
    11. CLI script runs with --help without error (py_compile + argparse check)
    12. Empty candidate list produces empty results, not crash
    ```

**Exit criteria:**
- [ ] Single USV classification produces valid ClassifiedUSV with syllable type and confidence
- [ ] Batch classification handles 100+ candidates without memory issues
- [ ] Failed candidates logged but don't halt the pipeline
- [ ] FP class correctly identified in output
- [ ] All 12 tests pass
- [ ] py_compile passes on all new files
- [ ] CLI help text displays correctly

---

### 15.4 Validation Batch Runner

**What:** Run BootSnap on a curated test batch of ~100 detected USVs from both wild and lab mice. Generate a validation report with classification distribution, confidence statistics, and visual inspection aids. This is the sanity-check gate before running on the full dataset.
**Status:** BLOCKED
**Review Tier:** 2 (validation script, analysis output)
**Depends on:** 15.3 (Classification Bridge working end-to-end)

/implement BootSnap Validation Batch Runner

Create a validation pipeline that processes a small curated set of ~100 USVs through the BootSnap classifier and generates a validation report. This is the critical quality gate before running on the full dataset.

**Context:** Before trusting BootSnap's output on the full dataset, we need to verify: (1) the classification distribution looks reasonable (not everything in one class), (2) the FP class catches actual noise detections, (3) confidence scores are discriminative (not all ~0.5), (4) visual spot-checks confirm labels make sense. This module uses the ClassificationBridge from Phase 15.3.

**Files to create:**

1. `scripts/validate_bootsnap.py` (NEW) — Validation script

    ```
    Usage: python scripts/validate_bootsnap.py --detections <CSV> --wav-dir <DIR> --output-dir <DIR>
    Options:
      --detections    Path to detection CSV
      --wav-dir       Directory containing source WAV files
      --output-dir    Directory for validation outputs
      --n-samples     Number of USVs to validate (default: 100)
      --stratify      Sample proportionally from each WAV file (default: random)
      --inspect-n     Number of classifications to render as spectrograms for visual check (default: 20)
    ```

    Core logic:
    - Sample ~100 candidates from detection CSV (stratified by source file or random)
    - Run through ClassificationBridge
    - Generate validation report (markdown + CSV):
      - Class distribution table (count and percentage per syllable type)
      - Confidence histogram (are scores discriminative?)
      - FP analysis: what do FP-classified USVs look like? (list with timestamps)
      - Low-confidence analysis: distribution of low-confidence predictions by class
      - Per-source-file breakdown (catches if one recording is systematically different)
    - Render ~20 classified spectrograms as PNG with predicted label overlay for visual inspection
    - Flag systematic issues: >50% single class, mean confidence <0.3, >30% FP

2. `src/usv_spectrogram/classification/validation.py` (NEW) — Validation report generator

    ```python
    @dataclass
    class ValidationReport:
        n_samples: int
        class_distribution: dict[str, int]
        confidence_stats: dict[str, float]    # mean, std, min, max, median
        fp_rate: float                        # proportion classified as FP
        low_confidence_rate: float
        per_file_distribution: dict[str, dict[str, int]]  # source_file -> class -> count
        warnings: list[str]                   # Systematic issues detected
        inspection_paths: list[Path]          # Paths to rendered spectrogram PNGs

    def generate_validation_report(
        results: list[ClassifiedUSV],
        output_dir: Path,
        inspect_n: int = 20,
    ) -> ValidationReport: ...

    def render_classified_spectrogram(
        classified: ClassifiedUSV,
        output_path: Path,
    ) -> None: ...
    ```

3. `tests/test_validation.py` (NEW) — Validation report tests

**Test plan:**
    ```
    1. generate_validation_report produces correct class counts from mock ClassifiedUSVs
    2. confidence_stats correctly computes mean, std, min, max, median
    3. fp_rate calculation: 10 FP out of 100 = 0.10
    4. Warning triggered when >50% of results are single class
    5. Warning triggered when mean confidence < 0.3
    6. Warning triggered when fp_rate > 0.30
    7. per_file_distribution groups by source file correctly
    8. Empty results list produces report with zero counts, not crash
    9. render_classified_spectrogram creates a PNG file at output_path (mock spectrogram data)
    ```

**Exit criteria:**
- [ ] Validation report generated with all statistics populated
- [ ] Systematic issue warnings fire correctly (test with adversarial distributions)
- [ ] ~20 visual inspection spectrograms rendered with predicted labels
- [ ] No single class dominates >50% of predictions (flag if it does)
- [ ] Confidence scores are discriminative (std > 0.1)
- [ ] All 9 tests pass
- [ ] py_compile passes on all new files
- [ ] Report reviewed by user — manual judgment on classification quality

---

### 15.5 Full Dataset Processing & Comparative Analysis

**What:** Batch-process ALL detected USVs from both wild and lab mouse recordings through the BootSnap classifier. Generate a master results CSV and comparative analysis showing syllable repertoire differences between wild and lab mice — the core deliverable for the May presentation.
**Status:** BLOCKED
**Review Tier:** 2 (batch processing + statistical analysis)
**Depends on:** 15.4 (Validation passed — user approved classification quality)

/implement BootSnap Full Dataset Processing & Comparative Analysis

Batch-classify all detected USVs and generate the wild-vs-lab comparison that is the project's core deliverable for the May presentation.

**Context:** After validation (Phase 15.4) confirms BootSnap produces reasonable classifications, this module runs on the full dataset. The key output is a comparison of syllable repertoires between wild field mice and laboratory mice. The plan notes several important caveats: BootSnap was trained on specific strains (may differ from ours), the pretrained model was not fine-tuned on our recording conditions, and confidence scores should be reported alongside classifications. These limitations must be documented in the analysis output.

**Files to create:**

1. `scripts/batch_classify.py` (NEW) — Full dataset batch processing script

    ```
    Usage: python scripts/batch_classify.py --config <YAML> --output-dir <DIR>
    Config YAML:
      detections:
        - path: "path/to/wild_detections.csv"
          group: "wild"
        - path: "path/to/lab_detections.csv"
          group: "lab"
      wav_dir: "path/to/wav/files"
      confidence_threshold: 0.5
      exclude_fp: true
      batch_size: 32
    ```

    Core logic:
    - Load all detection CSVs with group labels (wild/lab)
    - Run through ClassificationBridge in batches with progress bar (tqdm)
    - Write master CSV: candidate_id, source_file, start_ms, end_ms, peak_freq_hz, peak_energy_db, syllable_type, confidence, top_3, group (wild/lab)
    - Handle interruption gracefully (checkpoint after each batch, resume on restart)
    - Log processing summary: total processed, failures, time elapsed

2. `src/usv_spectrogram/classification/analysis.py` (NEW) — Comparative analysis

    ```python
    @dataclass
    class GroupAnalysis:
        group_name: str                       # "wild" or "lab"
        total_usvs: int
        syllable_counts: dict[str, int]       # syllable_type -> count
        syllable_proportions: dict[str, float] # syllable_type -> proportion
        complexity_ratio: float               # (complex types) / (simple types)
        mean_confidence: float
        unique_syllable_types: int            # Repertoire size

    @dataclass
    class ComparativeAnalysis:
        groups: dict[str, GroupAnalysis]       # group_name -> analysis
        repertoire_overlap: float             # Jaccard similarity of syllable types used
        complexity_comparison: dict[str, float] # group -> complexity_ratio
        significant_differences: list[str]    # Syllable types with notably different proportions
        limitations: list[str]                # Documented caveats

    def analyze_groups(
        results: list[ClassifiedUSV],
        group_labels: dict[str, str],  # candidate_id -> group
    ) -> ComparativeAnalysis: ...

    def generate_comparison_plots(
        analysis: ComparativeAnalysis,
        output_dir: Path,
    ) -> list[Path]: ...
    ```

    Plots to generate:
    - Grouped bar chart: syllable type distribution per group (wild vs. lab)
    - Stacked bar chart: proportion of each syllable type
    - Complexity comparison: simple vs. complex call ratios per group
    - Confidence distribution per group (violin or box plot)

3. `tests/test_analysis.py` (NEW) — Analysis tests

**Test plan:**
    ```
    1. GroupAnalysis.syllable_proportions sum to 1.0 (within float tolerance)
    2. complexity_ratio correctly categorizes: c, h, c2, c3 as complex; s, f, u, us as simple
    3. repertoire_overlap (Jaccard): identical sets = 1.0, disjoint = 0.0
    4. significant_differences flags types where proportion differs by >10% between groups
    5. Limitations list includes standard caveats (strain specificity, no fine-tuning, confidence caveat)
    6. Empty group produces GroupAnalysis with zero counts, not crash
    7. Single-group input produces ComparativeAnalysis without comparison errors
    8. generate_comparison_plots creates expected PNG files in output_dir (mocked matplotlib)
    9. Master CSV has all required columns and correct row count
    10. Batch processing resumes from checkpoint after simulated interruption
    ```

**Exit criteria:**
- [ ] Master CSV generated with all USVs classified + group labels
- [ ] Comparative analysis shows syllable repertoire differences (or confirms similarity)
- [ ] 4 comparison plots generated as PNG files
- [ ] Limitations section documents: strain specificity, no fine-tuning, confidence reporting
- [ ] All 10 tests pass
- [ ] py_compile passes on all new files
- [ ] Results ready for presentation integration

---

## Phase 15 Gate

Before declaring BootSnap integration complete:

- [ ] All 5 modules (15.1–15.5) at DONE status
- [ ] Total test count: ~47 new tests, all passing
- [ ] Master CSV exists with all detected USVs classified
- [ ] Wild vs. lab comparison analysis generated with plots
- [ ] FP class used to flag questionable detections from CNN pipeline
- [ ] Limitations documented for presentation
- [ ] Classification confidence reported alongside all predictions
- [ ] User has visually inspected ~20 classified spectrograms and confirmed quality

---

## Risk Mitigation Summary

| Risk | Mitigation | Phase |
|------|-----------|-------|
| Pretrained weights missing from repo | Email reyhaneh.abbasi@oeaw.ac.at requesting model files | 15.1 |
| Input format incompatible | Write custom Gammatone generator from paper specs (Sec 2.3) | 15.2 |
| Python version / framework conflicts | Dedicated conda env for BootSnap dependencies | 15.1 |
| Poor classification quality on our data | Fall back to 3-class grouping (no-jump / jumps / FP, ~95% F1 in paper) | 15.4 |
| Sample rate mismatch | Always specify sr=300000 (ADR-001); resample if BootSnap expects different sr | 15.2 |
| Memory issues on full dataset | Batch processing with configurable batch_size + checkpointing | 15.5 |

---

## Key Decisions to Document

After implementing Phase 15, create decision notes in `notes/` for:

1. **Gammatone vs. STFT** — Why BootSnap uses Gammatone filter banks instead of STFT (auditory model, log frequency spacing)
2. **Pretrained vs. fine-tuned** — Why we used the pretrained model directly (time constraints, May deadline, sufficient for preliminary comparison)
3. **FP class as denoising** — How BootSnap's false-positive class serves as a secondary validation of our CNN detector output
4. **Syllable complexity metric** — How we operationalize "complexity" for the wild vs. lab comparison
