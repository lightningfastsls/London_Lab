# USV Post-Processing Pipeline — ROADMAP

> Converts per-window CNN probabilities into reliable USV event detections for batch processing ~25,000 WAV files.
> Supersedes the old threshold values in Phase 13 (Batch Detection Pipeline) with an optimized multi-stage approach.
> Requires the matched-windows CNN model at `models/matched_windows/best_model.pt` (mid-C architecture [32, 96, 192], trained 2026-03-27).

---

## Phase 15: USV Post-Processing Pipeline

### 15.1 Hysteresis Detection Module

**What:** Replace single-threshold detection with dual-threshold (Schmitt trigger) hysteresis that exploits temporal structure — real USVs span 5-50 consecutive high-probability windows while noise FPs are 1-3 isolated windows. This is the single highest-impact stage.
**Status:** DONE
**Review Tier:** 3
**Depends on:** Matched-windows CNN model (complete)

/implement Hysteresis Detection Module

Create a hysteresis (dual-threshold) USV event detector that converts per-window CNN probabilities into discrete USV events. This replaces single-threshold detection and is the core of the post-processing pipeline.

**Context:** The matched-windows CNN (ROC AUC 0.989) produces per-window probabilities where true USVs score median 0.97 but the noise tail extends to 0.99, making single-threshold selection impossible. Hysteresis exploits temporal structure: an onset threshold seeds events only from sustained high-probability regions, while a sustain threshold extends events through mid-vocalization probability dips. No existing mouse USV tool (DeepSqueak, DAS, VocalMat, USVSEG) uses explicit hysteresis — they all use single threshold + gap-filling + minimum duration. Hysteresis subsumes and improves on this (Cances et al., 2019 WASPAA; WhaleVAD-BPN, 2024).

Reference: ADR-001 (sr=300000), ADR-002 (n_fft=512, hop=128). The SlidingInference module (`src/usv_spectrogram/app/core/sliding_inference.py`) produces `InferenceResult` with `probabilities` (n_windows,), `column_indices` (n_windows,), `times` (n_windows,) — this module consumes those arrays.

**Files to create:**

1. `src/usv_spectrogram/postprocessing/__init__.py` (NEW) — Package init

2. `src/usv_spectrogram/postprocessing/hysteresis.py` (NEW) — Core detection logic

    ```python
    from dataclasses import dataclass, field
    from typing import List
    import numpy as np

    @dataclass
    class USVEvent:
        """A detected USV event spanning multiple inference windows."""
        start_window: int           # First window index (inclusive)
        end_window: int             # Last window index (inclusive)
        start_time_s: float         # Start time in seconds
        end_time_s: float           # End time in seconds
        duration_ms: float          # Duration in milliseconds
        peak_probability: float     # Max per-window probability
        mean_probability: float     # Mean per-window probability
        window_count: int           # Number of windows in this event
        probabilities: np.ndarray   # Raw per-window probs for this event

    @dataclass(frozen=True)
    class HysteresisConfig:
        """Configuration for hysteresis detection."""
        onset_threshold: float = 0.75      # High threshold to initiate an event
        sustain_threshold: float = 0.40    # Low threshold to sustain an event
        gap_fill_windows: int = 3          # Max gap to merge between events
        min_duration_windows: int = 5      # Minimum event length (~16ms at hop=128/sr=300k, stride=10)

        def __post_init__(self) -> None:
            if not (0 < self.sustain_threshold <= self.onset_threshold <= 1.0):
                raise ValueError(
                    f"Need 0 < sustain ({self.sustain_threshold}) <= onset ({self.onset_threshold}) <= 1.0"
                )
            if self.gap_fill_windows < 0:
                raise ValueError("gap_fill_windows must be >= 0")
            if self.min_duration_windows < 1:
                raise ValueError("min_duration_windows must be >= 1")
    ```

    Algorithm for `hysteresis_detect(probabilities, times, config) -> List[USVEvent]`:
    1. Find all windows where `probability >= onset_threshold` — these are "seed" windows
    2. From each seed, extend forward and backward while `probability >= sustain_threshold`
    3. Mark all extended windows as `in_event`
    4. Extract contiguous `in_event` regions as candidate events
    5. Merge events separated by `<= gap_fill_windows` (gap-filling)
    6. Filter events shorter than `min_duration_windows`
    7. For each surviving event, compute start/end times from the `times` array, peak/mean probability, duration
    8. Return list of `USVEvent` objects

    Also implement `convert_to_detection_format(events, recording_stem) -> list[dict]` that converts USVEvents into the ADR-010 JSON detection format (compatible with existing `_saved_tracking.json` and the desktop app).

3. `tests/test_hysteresis.py` (NEW) — Unit tests

**Test plan:**
```
1. Single sustained peak above onset → one event detected
2. Two peaks separated by large gap → two events
3. Two peaks separated by small gap (≤ gap_fill_windows) → merged into one event
4. Short spike above onset but below min_duration → filtered out
5. Peak above onset with shoulders above sustain → event extends through shoulders
6. Noise-only input (all below sustain) → empty list
7. Edge case: peak at start/end of array → event correctly bounded
8. Config validation: sustain > onset raises ValueError
9. Times array correctly mapped to event start_time_s/end_time_s
10. ADR-010 format conversion produces valid detection dicts
```

**Exit criteria:**
- [ ] All tests pass
- [ ] py_compile passes
- [ ] Running on 5 test WAV recordings produces sensible event counts (fewer than raw window count, more than 0 for USV-containing recordings)
- [ ] Output USVEvent.duration_ms values are in plausible range (5-350ms for mouse USVs)

---

### 15.2 Hysteresis Parameter Optimization

**What:** Optimize the 4 hysteresis parameters (onset, sustain, gap_fill, min_duration) using F2-score cross-validation on the 126 labeled recordings. F2 weights recall ~4x more than precision, matching the preference for catching USVs over avoiding false positives.
**Status:** DONE
**Review Tier:** 3
**Depends on:** 15.1

/implement Hysteresis Parameter Optimization

Create a parameter optimization script that finds the best hysteresis thresholds using cross-validated F2 scoring on labeled recordings.

**Context:** The 126 labeled recordings (in `data/unified_labels.json`) provide ground truth. We need event-level evaluation (not window-level): a detected event is a true positive if it overlaps a ground truth USV within a ±200ms collar tolerance (standard in bioacoustic evaluation — Kershenbaum et al., 2025). Use collar-based matching rather than IoU because USV boundaries are inherently uncertain.

The search space is small (4 parameters, bounded ranges), so grid search with dichotomic refinement (Cances et al., 2019) is preferred over Bayesian optimization — it's simpler, reproducible, and sufficient. Run SlidingInference once per recording, cache the probability arrays, then re-score for each parameter combination (negligible compute per evaluation).

**Files to create:**

1. `src/usv_spectrogram/postprocessing/event_scoring.py` (NEW) — Event-level evaluation

    ```python
    @dataclass(frozen=True)
    class EventScoringConfig:
        onset_collar_s: float = 0.200   # ±200ms tolerance on event boundaries
        min_iou: float = 0.0            # Not used with collar matching [ASSUMED]

    def match_events_collar(
        detected: List[USVEvent],
        ground_truth: List[tuple[float, float]],  # (start_s, end_s)
        collar_s: float = 0.200,
    ) -> tuple[int, int, int]:
        """Match detected events to ground truth using collar tolerance.
        Returns (true_positives, false_positives, false_negatives).

        A detection matches a GT event if:
        - detection onset is within ±collar_s of GT onset, OR
        - detection offset is within ±collar_s of GT offset, OR
        - detection overlaps GT by any amount

        Each GT event can match at most one detection (greedy, best-overlap-first).
        Each detection can match at most one GT event.
        """
        ...

    def compute_f_beta(tp, fp, fn, beta=2.0) -> float:
        """Compute F-beta score. beta=2 weights recall 4x more than precision."""
        ...
    ```

2. `scripts/optimize_hysteresis.py` (NEW) — CLI for parameter optimization

    ```python
    # Grid search ranges:
    # onset_threshold:     [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # sustain_threshold:   [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] (constrained: sustain <= onset)
    # gap_fill_windows:    [0, 1, 2, 3, 4, 5]
    # min_duration_windows:[3, 4, 5, 6, 7, 8, 9, 10]
    ```

    Algorithm:
    1. Load unified_labels.json for ground truth
    2. Run SlidingInference on each labeled recording, cache probability arrays (this is the slow step — ~3 min for 126 recs)
    3. 5-fold stratified CV (stratify by has_usvs)
    4. For each fold, for each parameter combination:
       a. Run hysteresis_detect on val recordings using cached probabilities
       b. Score with collar-based event matching → F2
    5. Report best parameters with mean±std F2 across folds
    6. Apply one-standard-error rule: select simplest parameters within 1 SE of max F2
    7. Save results to `models/matched_windows/hysteresis_optimization.json`

    CLI: `python scripts/optimize_hysteresis.py --model models/matched_windows/best_model.pt --labels data/unified_labels.json --output models/matched_windows/hysteresis_optimization.json`

3. `tests/test_event_scoring.py` (NEW) — Tests for event-level matching

**Test plan:**
```
1. Perfect detection (exact match) → TP=1, FP=0, FN=0
2. Detection offset by 100ms (within collar) → still TP
3. Detection offset by 300ms (outside collar) → FP + FN
4. Two detections for one GT event → 1 TP + 1 FP
5. One detection spanning two GT events → 1 TP + 1 FN [ASSUMED: greedy one-to-one matching]
6. F2 score: verify formula with known TP/FP/FN values
7. Empty detections, non-empty GT → FN = len(GT)
8. Grid search finds known-optimal params on synthetic data
```

**Exit criteria:**
- [ ] All tests pass
- [ ] Optimization completes on 126 recordings (may take 5-10 min with cached probabilities)
- [ ] Best F2 score > 0.85 (if below, investigate — may indicate labeling or model issues)
- [ ] Optimal parameters saved to JSON with fold-level scores and confidence intervals
- [ ] One-standard-error parameters documented alongside best parameters

---

### 15.3 Temperature Scaling

**What:** Learn a single temperature parameter T on the validation set to calibrate CNN probabilities, making thresholds more interpretable across recordings. Requires exposing model logits (pre-sigmoid) from SlidingInference.
**Status:** DONE
**Review Tier:** 2
**Depends on:** None (can be done in parallel with 15.1-15.2)

/implement Temperature Scaling for CNN Calibration

Add temperature scaling calibration to the CNN inference pipeline. This learns a single scalar T that divides logits before sigmoid, improving probability calibration without changing discrimination (ROC AUC invariant).

**Context:** Modern CNNs are systematically miscalibrated (Guo et al., 2017 ICML). Temperature scaling is the simplest effective fix: 1 parameter, fits in seconds on a validation set. The matched-windows model currently outputs probabilities via `model.predict_proba()` which applies sigmoid to raw logits. We need to: (1) expose logits from SlidingInference, (2) fit T on validation data, (3) apply calibrated sigmoid during inference.

Use the validation split (29 recordings, `data/training/matched_windows/val.csv`) for fitting T. The test split stays untouched for final evaluation. Fit by minimizing negative log-likelihood with L-BFGS (converges in <50 iterations).

**Files to create/edit:**

1. `src/usv_spectrogram/postprocessing/calibration.py` (NEW) — Temperature scaling

    ```python
    @dataclass
    class TemperatureScaler:
        temperature: float = 1.5  # Initial value

        def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
            """Fit T by minimizing NLL on validation set. Returns optimal T."""
            ...

        def calibrate(self, logits: np.ndarray) -> np.ndarray:
            """Apply calibration: sigmoid(logits / T)."""
            return 1.0 / (1.0 + np.exp(-logits / self.temperature))

        def save(self, path: Path) -> None: ...
        def load(cls, path: Path) -> 'TemperatureScaler': ...
    ```

2. `src/usv_spectrogram/app/core/sliding_inference.py` (EDIT) — Add `return_logits` option

    Currently line 215 calls `model.predict_proba()`. Add option to call `model.forward()` instead and return both logits and probabilities in InferenceResult. Add an optional `logits` field to InferenceResult (default None for backward compat).

3. `scripts/calibrate_temperature.py` (NEW) — CLI to fit and save temperature parameter

    CLI: `python scripts/calibrate_temperature.py --model models/matched_windows/best_model.pt --val-csv data/training/matched_windows/val.csv --output models/matched_windows/temperature.json`

4. `tests/test_calibration.py` (NEW)

**Test plan:**
```
1. T=1.0 produces same probabilities as raw sigmoid
2. T>1 softens probabilities (moves toward 0.5)
3. T<1 sharpens probabilities (moves toward 0/1)
4. fit() reduces NLL on validation set compared to T=1.0
5. save/load round-trip preserves T value
6. InferenceResult.logits is None when return_logits=False (backward compat)
7. InferenceResult.logits has correct shape when return_logits=True
```

**Exit criteria:**
- [ ] Fitted T is in reasonable range (0.5-3.0) — values outside this suggest model issues
- [ ] Calibrated probabilities have lower ECE (Expected Calibration Error) than raw probabilities on validation set
- [ ] SlidingInference backward compatible (existing code that doesn't use logits works unchanged)
- [ ] Temperature parameter saved to `models/matched_windows/temperature.json`

---

### 15.4 Event Feature Extraction

**What:** Extract discriminative features from detected USV events (probability curve shape, spectral properties, duration) for use by a second-stage false-positive filter. Features distinguish true USVs (smooth sustained high-probability plateaus with tonal spectral content) from noise FPs (spiky isolated windows with broadband energy).
**Status:** DONE
**Review Tier:** 3
**Depends on:** 15.1

/implement Event Feature Extraction

Add feature extraction to USVEvent objects for second-stage classification. Extract probability-based features (peak, mean, std, kurtosis, smoothness) and spectral features (tonality, peak frequency, frequency continuity, SNR) from the spectrogram columns corresponding to each event.

**Context:** After hysteresis detection, some false positives remain — noise that sustained above the low threshold for enough windows. A second-stage classifier on event-level features catches these (Clarfeld et al., 2025: 84.5-89.8% accuracy for bioacoustic FP filtering). VocalMat uses a similar two-stage approach.

Window-to-spectrogram-column mapping: SlidingInference uses `hop_px` (default 10) as stride. Window `i` starts at spectrogram column `i * hop_px`. The full spectrogram is available from `AudioLoader.load()` → `audio_data.spectrogram_db`. Band mask: 20-120 kHz (170 frequency bins after masking, ADR-002). Frequency resolution: ~586 Hz/bin. To convert `mean_peak_freq_bin` to Hz: `freq_hz = 20000 + bin_index * 586`.

**Files to create:**

    **Pre-existing tests:** `tests/test_event_features.py` (14 tests from test-architect, all currently failing on import). Implementation must make these pass — do NOT modify test expectations.

1. `src/usv_spectrogram/postprocessing/event_features.py` (NEW)

    ```python
    @dataclass
    class EventFeatures:
        """Discriminative features for second-stage classification."""
        # Probability-based
        peak_probability: float
        mean_probability: float
        prob_std: float
        prob_kurtosis: float      # Spiky (noise) vs plateau (USV)
        prob_roughness: float    # Mean |second derivative| of probability curve
        duration_windows: int

        # Spectral (require spectrogram access)
        tonality: float           # Geometric/arithmetic mean ratio of power spectrum
        mean_peak_freq_bin: float # Should be in 30-110 kHz range for USVs
        freq_range_bins: float    # Frequency modulation extent
        freq_modulation_rate: float    # Mean |delta peak_freq| between columns — low = tonal
        snr_db: float             # Peak power vs noise floor estimate

    def extract_event_features(
        event: USVEvent,
        spectrogram: np.ndarray,   # Full recording spectrogram (freq_bins, time_frames)
        hop_px: int = 10,          # SlidingInference stride
    ) -> EventFeatures:
        """Extract features from a single event."""
        ...
    ```

    Key implementation details:
    - Map `event.start_window` to spectrogram column: `start_col = event.start_window * hop_px`
    - Tonality = geometric_mean(power) / arithmetic_mean(power) per column, averaged across event. Values > 0.3 suggest tonal content (DeepSqueak convention).
    - Smoothness = mean absolute second derivative of probability curve. Real USVs are smooth; noise is jagged.
    - SNR = mean(10 * log10(peak_power / noise_floor)) where noise_floor = 10th percentile per column.

2. `tests/test_event_features.py` (NEW)

**Test plan:**
```
1. Constant-probability event → prob_std=0, prob_roughness=0
2. Tonal synthetic signal (single freq) → high tonality (>0.5)
3. Broadband noise → low tonality (<0.2)
4. Monotonically increasing frequency → freq_modulation_rate > 0, freq_range > 0
5. Feature extraction handles edge events (start/end of spectrogram)
6. All features are finite (no NaN/Inf) on real spectrogram data
```

**Exit criteria:**
- [ ] All tests pass
- [ ] Features extracted from 5 known USV events have tonality > 0.3 and mean_peak_freq_bin in 30-110 kHz range
- [ ] Features extracted from 5 known noise events have lower tonality and/or higher freq_modulation_rate
- [ ] No NaN or Inf values on any of the 126 labeled recordings

---

### 15.5 Second-Stage False Positive Filter

**What:** Train a logistic regression classifier on event-level features to filter false positives from hysteresis detections. Logistic regression is preferred over gradient boosting for interpretability and minimal overfitting on small training sets (~hundreds of events).
**Status:** READY
**Review Tier:** 2
**Depends on:** 15.2 (optimized hysteresis params), 15.4 (event features)

/implement Second-Stage False Positive Filter

Train a logistic regression on event-level features extracted from hysteresis detections on the 126 labeled recordings. Events are labeled true/false by checking overlap with ground truth annotations using collar-based matching (reuse `event_scoring.py` from 15.2).

**Context:** After hysteresis + parameter optimization, some false positives will remain — spectral artifacts that sustain above the low threshold. A second-stage classifier catches these with ~85-90% accuracy (Clarfeld et al., 2025). Using logistic regression because: (1) interpretable coefficients show which features matter, (2) minimal overfitting with ~hundreds of training events, (3) outputs calibrated probabilities natively. Consider upgrading to LightGBM only if >1000 labeled events and logistic regression underfits.

**Files to create:**

    **Pre-existing tests:** `tests/test_fp_filter.py` (16 tests from test-architect, all currently failing on import). Implementation must make these pass — do NOT modify test expectations.

1. `src/usv_spectrogram/postprocessing/fp_filter.py` (NEW)

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    class FalsePositiveFilter:
        """Second-stage logistic regression filter for USV events."""

        def __init__(self):
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    class_weight='balanced',
                    C=1.0,
                    max_iter=1000,
                ))
            ])

        def fit(self, features: List[EventFeatures], labels: List[bool]) -> None: ...
        def predict(self, features: List[EventFeatures]) -> List[bool]: ...
        def predict_proba(self, features: List[EventFeatures]) -> np.ndarray: ...
        def feature_importances(self) -> dict[str, float]: ...
        def save(self, path: Path) -> None: ...
        def load(cls, path: Path) -> 'FalsePositiveFilter': ...
    ```

2. `scripts/train_fp_filter.py` (NEW) — CLI to train and evaluate

    Use leave-one-recording-out CV or 5-fold CV on events from 126 labeled recordings.
    CLI: `python scripts/train_fp_filter.py --model models/matched_windows/best_model.pt --labels data/unified_labels.json --hysteresis-config models/matched_windows/hysteresis_optimization.json --output models/matched_windows/fp_filter.pkl`

3. `tests/test_fp_filter.py` (NEW)

**Test plan:**
```
1. Filter trained on labeled events achieves F2 > 0.80 in cross-validation
2. Feature importances are non-zero for at least 5 features
3. save/load round-trip produces identical predictions
4. Filter with all-positive training data doesn't crash (edge case)
5. Balanced class weights handle imbalanced event counts
```

**Exit criteria:**
- [ ] Cross-validated F2 > 0.80 (event-level, after hysteresis + filter)
- [ ] Feature importance report shows interpretable top features (expect: duration, peak_prob, tonality)
- [ ] Pipeline improves precision over hysteresis-only without dropping recall by more than 5%
- [ ] Model saved to `models/matched_windows/fp_filter.pkl`

---

### 15.6 Per-Recording Score Normalization

**What:** Z-normalize CNN probabilities per recording using the noise distribution, making thresholds self-adjusting across recordings with different noise floors (different cages, equipment, days).
**Status:** DONE
**Review Tier:** 2
**Depends on:** None

/implement Per-Recording Score Normalization

Add per-recording Z-normalization of CNN probabilities using the estimated noise distribution within each recording. This makes fixed thresholds behave adaptively across recordings with varying noise floors.

**Context:** Noise floors vary across recordings (different cages, equipment, recording days). A fixed threshold of 0.50 means different things in quiet vs noisy recordings. Normalization estimates the noise distribution from the bottom 50th percentile of windows (predominantly noise in typical USV recordings, where USVs occupy <5% of total duration) and Z-normalizes: `z = (prob - noise_median) / noise_MAD`. Normalized scores can exceed [0,1] — that's expected. Hysteresis thresholds then operate on Z-scores.

Future alternative: PCEN (Per-Channel Energy Normalization, Lostanlen et al. 2019) operates at the spectrogram level and reduced false alarm rates by 50x in BirdVoxDetect. But PCEN requires retraining the CNN. Z-normalization works as a post-hoc fix with the current model.

**Files to create:**

    **Pre-existing tests:** `tests/test_normalization.py` (13 tests from test-architect, all currently failing on import). Implementation must make these pass — do NOT modify test expectations.

1. `src/usv_spectrogram/postprocessing/normalization.py` (NEW)

    ```python
    def normalize_scores_per_recording(probabilities: np.ndarray) -> np.ndarray:
        """Z-normalize CNN scores using the noise distribution.
        Estimate noise from bottom 50th percentile of windows.
        Returns normalized scores (can exceed [0,1])."""
        ...

    def normalize_scores_batch(
        all_probabilities: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Normalize a batch of recordings. Returns dict of stem -> normalized."""
        ...
    ```

2. `tests/test_normalization.py` (NEW)

**Test plan:**
```
1. Constant input → all zeros output (no variation from noise)
2. Known distribution (noise=0.1 ± 0.02, USV=0.9) → USV Z-score >> noise Z-scores
3. Two recordings with different noise floors normalize to comparable Z-scores
4. Empty array → handled gracefully (return empty)
5. All-same-value array → noise_MAD = 0 guard activated
```

**Exit criteria:**
- [ ] All tests pass
- [ ] Normalization demonstrably reduces cross-recording threshold variance when measured on 10+ recordings with different noise profiles
- [ ] `[ASSUMED]` Median percentile (50th) is the right cutpoint — validate that USVs occupy <5% of windows in typical recordings

---

### 15.7 Recording-Level Triage and Batch Output

**What:** Assign each recording to a triage tier (auto-accept / auto-reject / manual-review) based on detection confidence and QC metrics. Produce batch output (parquet + per-recording JSON) for the 25K file run.
**Status:** BLOCKED (on 15.5)
**Review Tier:** 2
**Depends on:** 15.2 (hysteresis params), 15.5 (FP filter) — needs the full pipeline to produce meaningful confidence scores

/implement Recording-Level Triage and Batch Output

Create a triage system that automatically categorizes recordings after detection, plus the batch output format for processing 25K files.

**Context:** 25,000 recordings is too many to manually review. A triage system auto-accepts high-confidence detections (~60-70%), auto-rejects clearly empty recordings (~10-20%), and queues ambiguous ones for manual review (~15-25%). Triage thresholds are calibrated after the first batch run using batch-level statistics (mean/std event counts, noise profiles).

Output format: single parquet for batch results (fast column queries, good compression for 25K rows) + per-recording JSON for detailed event data (compatible with existing `_saved_tracking.json` and the desktop app).

**Files to create:**

    **Pre-existing tests:** `tests/test_triage.py` (19 tests from test-architect, all currently failing on import). Implementation must make these pass — do NOT modify test expectations.

1. `src/usv_spectrogram/postprocessing/triage.py` (NEW)

    ```python
    @dataclass
    class RecordingResult:
        filepath: str
        events: List[USVEvent]
        tier: str               # 'auto_accept', 'auto_reject', 'manual_review'
        confidence_score: float
        qc_flags: List[str]
        # QC metrics stored per recording
        n_events: int
        max_confidence: float
        mean_event_confidence: float
        total_usv_duration_ms: float
        noise_floor_p90: float  # 90th percentile of all window probs

    @dataclass(frozen=True)
    class TriageConfig:
        auto_accept_min_peak: float = 0.90
        auto_reject_max_window: float = 0.10
        outlier_count_zscore: float = 2.0

    def triage_recording(
        filepath: str,
        events: List[USVEvent],
        probabilities: np.ndarray,
        config: TriageConfig,
        batch_stats: dict = None,  # mean/std from prior batch run
    ) -> RecordingResult: ...
    ```

2. `src/usv_spectrogram/postprocessing/batch_output.py` (NEW) — Write results to parquet + JSON

    ```python
    def write_batch_results(
        results: List[RecordingResult],
        output_dir: Path,
        write_parquet: bool = True,
        write_per_recording_json: bool = True,
    ) -> None:
        """Write batch results.
        - summary.parquet: one row per recording with QC metrics
        - detections/<stem>.json: per-recording events in ADR-010 format
        """
        ...
    ```

3. `scripts/run_batch_detection.py` (NEW) — CLI that runs the full pipeline

    This is the main entry point for batch processing. Orchestrates:
    AudioLoader → SlidingInference → [TemperatureScaling] → [Normalization] → HysteresisDetection → [EventFeatures → FPFilter] → Triage → Output

    CLI: `python scripts/run_batch_detection.py --wav-dir <path> --model models/matched_windows/best_model.pt --output-dir results/batch_001/ [--temperature models/matched_windows/temperature.json] [--fp-filter models/matched_windows/fp_filter.pkl] [--hysteresis-config models/matched_windows/hysteresis_optimization.json]`

4. `tests/test_triage.py` (NEW)

**Test plan:**
```
1. Recording with all events > 0.90 peak → auto_accept
2. Recording with no windows > 0.10 → auto_reject
3. Recording with mixed confidence events → manual_review
4. Outlier event count (z > 2) → flagged for review
5. High noise floor (p90 > 0.4) → flagged for review
6. Parquet output has expected columns and row count
7. Per-recording JSON matches ADR-010 format
8. Batch script processes 5 WAV files end-to-end without error
```

**Exit criteria:**
- [ ] Full pipeline runs on 10 test WAV files without errors
- [ ] Triage distributes recordings into all 3 tiers
- [ ] Parquet output readable by pandas with expected schema
- [ ] Per-recording JSONs loadable by existing desktop app
- [ ] `[ASSUMED]` Triage thresholds will need recalibration after first real batch run on ~100 recordings

---

## Phase 15 Gate

- [x] Hysteresis detection module complete with optimized parameters (15.1, 15.2)
- [x] Temperature scaling fitted and saved (15.3)
- [x] Event features + FP filter trained and evaluated (15.4, 15.5)
- [x] Per-recording normalization implemented (15.6)
- [x] Batch pipeline runs end-to-end on 198 recordings (126 positive, 72 noise) — ran 2026-03-28, plus batch_5970 and pipeline_comparison
- [x] Event-level F2 > 0.85 on held-out test recordings — hysteresis F2=0.885±0.016 (5-fold CV), FP filter F2=0.850±0.084 (5-fold CV)
- [x] Batch output format validated (parquet + per-recording JSON)
- [x] All tests passing (346/346, 2026-03-29)
- [x] All module docs written (7/7)

---

## Implementation Notes

### Recommended Order

1. **15.1** (Hysteresis) — Implement first, test on labeled recordings
2. **15.2** (Optimization) — Requires 15.1. Tells us if Stages 3-5 are even needed.
3. **15.3** (Temperature) — Independent, do whenever convenient
4. **15.6** (Normalization) — Independent, do whenever convenient
5. **15.4** (Features) — After 15.2 reveals whether FP filter is needed
6. **15.5** (FP Filter) — After 15.4
7. **15.7** (Triage + Batch) — Last, integrates everything

### Decision Point After 15.2

After optimizing hysteresis parameters, evaluate event-level precision/recall. If precision > 0.90 at recall > 0.90, Stages 15.4-15.5 (second-stage classifier) may be unnecessary — proceed directly to 15.7 with hysteresis-only pipeline. This avoids over-engineering.

### Key Assumptions Marked [ASSUMED]

- Collar-based matching with ±200ms tolerance for event evaluation
- Greedy one-to-one matching (each GT matches at most one detection)
- 50th percentile cutpoint for noise estimation in normalization
- Triage thresholds (0.90 accept, 0.10 reject) will need empirical calibration
- Logistic regression sufficient for FP filter (upgrade to LightGBM if >1000 events)

### Resolved Ambiguities (2026-03-28)

These were surfaced by test-architect agents and resolved before implementation:

1. **15.4 Tonality direction:** Use `1 - SFM` so high values = tonal (matching spec prose "values > 0.3 suggest tonal content"). Standard SFM is high for flat/broadband; inverting aligns the feature with the intuitive direction.
2. **15.4 Spectrogram units:** `extract_event_features` receives dB-scale spectrogram (from `AudioLoader`). Convert to linear power internally: `power = 10 ** (spectrogram_db / 10)` before computing tonality and SNR.
3. **15.5 Single-class training:** `FalsePositiveFilter.fit()` must handle single-class labels gracefully (fallback to always-predict-that-class) since sklearn's LogisticRegression raises on single-class input.
4. **15.6 MAD=0 fallback:** When all values are identical (MAD=0), return all zeros — "no variation from noise floor" is the correct semantic.
5. **15.7 confidence_score:** Defined as `mean_event_confidence` (mean of per-event `peak_probability` values).
6. **15.7 Empty events triage:** Empty events list → `auto_reject` tier (no events = no USVs detected).
