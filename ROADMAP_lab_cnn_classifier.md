# Lab-Cleaned USV Syllable Classifier — ROADMAP

> Train a 12-class syllable-type CNN on **lab-recorded** USVs using VocalMat (Apache 2.0, 12,954 labeled spectrograms; Grimsley 2011 taxonomy), our existing 4-layer cleaning stack, and DANN domain-adversarial training to enforce cage invariance.
> Source plan: `PLAN_lab_cnn_classifier.md` (in this worktree).
> Worktree: `worktree-lab-cnn-classifier-plan`.
> Production CNN (`models/hard_neg_retrain/best_model.pt`) and detection pipeline are **unchanged** — this is a downstream classifier.

---

## Phase 18: Lab USV Syllable Classifier (VocalMat-anchored, DANN)

**Goal:** Ship a 12-class syllable-type CNN that learns syllable structure rather than recording-environment ("cage") artifacts. Falsifiable cleaning gate (18.1) blocks all downstream work; honest failure preferred over silent success.

**Cross-phase constraints (apply to every module):**

| # | Constraint | Source |
|---|---|---|
| C1 | Resample 300→250 kHz via `scipy.signal.resample_poly(up=5, down=6)` only. Do NOT modify `corpus.py`. The canonical 300 kHz invariant remains intact for the rest of the project — this pipeline runs at 250 kHz internally. | /kcheck `[[project-lab-cnn-classifier-scope]]`; CLAUDE.md ADR-001 |
| C2 | Global MAD normalization: compute on whole spectrogram once, then crop window patches. Per-window MAD is a known trap (silently kills high-confidence USVs by clipping into noise). Reference: `src/usv_spectrogram/app/core/sliding_inference.py:389–424`. | /kcheck `[[feedback-cnn-inference-global-mad]]` |
| C3 | All 4 cleaning layers already exist — wire and validate, do NOT rebuild: soft-notch (`app/core/notch.py`), Boll baseline subtraction (`app/core/denoise.py`), global MAD (`app/core/sliding_inference.py`), per-recording Z-score (`postprocessing/normalization.py`, dormant). | /kcheck `[[project-cleaning-pipeline-inventory]]` |
| C4 | Soft-notch tonal library exists only for `lab_131204` (cage mask [50.4, 51.0] kHz). For 5970 and VocalMat: rely on baseline + MAD + Z-score only (soft-notch is a no-op). New cages would require `scripts/calibrate_lab_tonal_lines.py`. | /kcheck `[[project-lab-data-pipeline]]`; PLAN §"Existing Infrastructure" |
| C5 | All new code lives in `src/usv_spectrogram/classifier/` (new package). Do NOT modify production detection files (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `postprocessing/`). | CLAUDE.md "USV Red Flags"; PLAN §"What's NOT in this plan" |
| C6 | "Cage" = physical recording chamber (acoustic confound). "Rig" = compute infrastructure. Use the right term — confusion has produced incorrect causal claims before. | /kcheck `[[feedback-cage-not-rig-terminology]]` |

**Decisions baked in from user approval (2026-05-21):**

| # | Decision | Affected modules |
|---|---|---|
| D1 | Patch size 0.22s primary; defer 0.08s variant unless 18.5 (wild transfer) fails | 18.2, 18.5 |
| D2 | Wild labeling sequential after 18.4 (not parallel with 18.1–18.3) — bootstrap from v2 confidence scores | 18.5 |
| D3 | Perch 2.0 sidequest in 18.3: YES (1 day; parallel linear probe to ResNet-18) | 18.3 |
| D4 | DANN domain head granularity: 2-cage (lab_131204 vs vocalmat) for v1 | 18.4 |
| D5 | Minority class strategy: keep all 12 classes + class-weighted CE + focal loss + oversampling. Revisit (collapse/drop) only if v1 confusion matrix shows per-class precision < 0.20. | 18.3 |

---

### 18.1 Cleaning Validation Gate (BLOCKING)

**What:** Build a falsifiable diagnostic that proves our 4-layer cleaning stack suppresses cage-confound enough to make a syllable classifier honest. Re-run the VAE diagnostic from the `vae-pytorch-pivot` worktree on **cleaned** spectrograms across (a) VocalMat lab data, (b) lab 131204, (c) wild 5970, and verify four numeric pass criteria.
**Status:** READY
**Review Tier:** 3 (DSP + statistical methodology)
**Depends on:** None

/implement Cleaning Validation Gate

Build a falsifiable cleaning-validation diagnostic. The output decides go/no-go for the entire downstream classifier pipeline (Modules 18.2–18.5). Failure on any of four numeric criteria means STOP and iterate on cleaning before proceeding.

**Context:** The VAE comparison memo (`docs/handoffs/2026-05-18_vae_comparison_memo.md`) established that *raw* spectrograms produce near-disjoint cohort separation driven by cage acoustics (mean_power_db, tonality), not biology. Both VAE architectures (DeepSqueak's and ours) found the same signature — the issue is in the data, not the model. Phase 1.0 of the plan exists specifically to verify that enabling our 4-layer cleaning stack drops this signal below thresholds that would otherwise be conflated with biological signal. This is not a refactor or an engineering task — it is empirical hypothesis testing with concrete pass thresholds.

Reference: ADR-001 (sr=300000 for source; this module uses 250 kHz for VocalMat-aligned pipeline). ADR-002 (n_fft=512, hop=128 at 300 kHz). PLAN §"Phase 1.0 — Cleaning Validation Gate". `[[project-vae-comparison-complete]]` for benchmark definitions. Cross-phase constraints C1–C6 above all apply here.

**Files to create:**

1. `src/usv_spectrogram/classifier/__init__.py` (NEW) — Package init with version string

2. `src/usv_spectrogram/classifier/cleaning_pipeline.py` (NEW) — Orchestrates the 4-layer cleaning stack with each layer toggleable for ablation

    ```python
    from dataclasses import dataclass
    from pathlib import Path
    import numpy as np

    @dataclass(frozen=True)
    class CleaningConfig:
        """Configuration for the 4-layer cleaning stack.
        Each layer is independently toggleable so the diagnostic can ablate.
        """
        apply_soft_notch: bool = True              # Layer 1 — requires tonal library for cohort
        apply_baseline_subtraction: bool = True    # Layer 2 — Boll 1979 median_envelope mode
        apply_global_mad: bool = True              # Layer 3 — whole-spectrogram MAD (see C2)
        apply_per_recording_zscore: bool = True    # Layer 4 — per-recording Z-score normalization
        baseline_mode: str = "median_envelope"     # "percentile" | "median_envelope"
        tonal_library_path: Path | None = None     # None = skip soft-notch even if apply_soft_notch=True
        sample_rate_hz: int = 250_000              # VocalMat-aligned (NOT corpus.SAMPLE_RATE_HZ; see C1)

        def __post_init__(self) -> None:
            if self.baseline_mode not in {"percentile", "median_envelope"}:
                raise ValueError(f"baseline_mode must be 'percentile' or 'median_envelope', got {self.baseline_mode!r}")
            if self.sample_rate_hz not in {250_000, 300_000}:
                raise ValueError(f"sample_rate_hz must be 250000 or 300000, got {self.sample_rate_hz}")
            if self.apply_soft_notch and self.tonal_library_path is None:
                # Not an error — soft-notch will silently no-op. Document the choice in caller.
                pass

    def clean_spectrogram(spec: np.ndarray, cfg: CleaningConfig, recording_id: str) -> np.ndarray:
        """Apply the 4-layer cleaning stack to a single spectrogram (n_freq_bins, n_time_frames).
        Returns a cleaned spectrogram of the same shape, in dB.

        Layer order matters: notch → baseline → MAD → Z-score. Reordering is a behavior change.
        """
        ...
    ```

    The function imports the existing implementations from `app/core/notch`, `app/core/denoise`, and `postprocessing/normalization`. Global MAD must follow the whole-spectrogram-then-crop pattern from `sliding_inference.py:389–424` (C2). Per-recording Z-score must be wired correctly — the existing implementation is dormant; this module activates it for the first time outside production.

3. `src/usv_spectrogram/classifier/diagnostics.py` (NEW) — Three statistical diagnostics that constitute the gate

    ```python
    from dataclasses import dataclass
    import numpy as np

    @dataclass(frozen=True)
    class DiagnosticResult:
        """Result of one diagnostic test with pass/fail against threshold."""
        name: str                  # e.g., "notch_injection_migration"
        value: float               # measured value
        threshold: float           # pass threshold (interpretation: see docstring)
        threshold_direction: str   # "less_than" (most) | "greater_than"
        passed: bool
        details: dict              # method-specific metrics for the report

    def notch_injection_test(
        spectrograms_by_cohort: dict[str, np.ndarray],   # cohort_id -> (n_specs, n_freq, n_time)
        notch_band_khz: tuple[float, float] = (50.4, 51.0),  # lab_131204 cage mask (C4)
        notch_depth_db: float = 20.0,
    ) -> DiagnosticResult:
        """Inject a synthetic cage tone into the OTHER cohort's spectrograms and measure
        whether a small encoder confuses injected-noise samples with the original cage cohort.
        Pass: migration rate < 30% (raw baseline: 91.7% on our VAE, 58.5% on DeepSqueak's).
        """
        ...

    def per_band_cohens_d(
        spectrograms_by_cohort: dict[str, np.ndarray],
        band_edges_khz: list[tuple[float, float]] = None,  # default: 10 kHz bands from 20 to 120
    ) -> DiagnosticResult:
        """Compute mean spectral power per 10 kHz sub-band per cohort, then max Cohen's d
        between any two cohorts across all sub-bands. Pass: max d < 0.3 (raw: 0.4–2.0).
        """
        ...

    def knn_same_cohort_rate(
        embeddings_by_cohort: dict[str, np.ndarray],   # cohort_id -> (n_samples, embed_dim)
        k: int = 5,
    ) -> DiagnosticResult:
        """For each sample, compute fraction of k-NN that share its cohort label.
        Mean across samples. Pass: < 0.85 (raw: 0.98–1.0).
        """
        ...

    def raw_pixel_pca_d(
        spectrograms_by_cohort: dict[str, np.ndarray],
        n_components: int = 1,
    ) -> DiagnosticResult:
        """Run PCA on flattened spectrograms, compute Cohen's d on PC1 scores between cohorts.
        Pass: |d| < 1.5 (raw: +5.83 on our VAE).
        """
        ...

    def train_diagnostic_vae(
        spectrograms: np.ndarray,
        latent_dim: int = 32,
        n_epochs: int = 6,
        device: str = "cuda",
    ) -> np.ndarray:
        """Train a small VAE (4–8 epochs is enough for diagnostic per PLAN), return latent
        embeddings for all input spectrograms. Used to feed knn_same_cohort_rate on encoder
        embeddings rather than raw pixels (more sensitive test).
        """
        ...
    ```

    Cohen's d formula: `(mean_A - mean_B) / pooled_std` where pooled_std = sqrt((var_A + var_B) / 2). Threshold direction "less_than" means lower value = better invariance.

4. `scripts/cnn_cleaning_validation.py` (NEW) — Diagnostic runner with ablation matrix

    ```python
    # CLI structure
    # python scripts/cnn_cleaning_validation.py \
    #     --vocalmat-sample data/vocalmat_sample/ \
    #     --lab-131204-sample <wav-dir> \
    #     --wild-5970-sample <wav-dir> \
    #     --sample-size 200 \
    #     --output-dir results/cleaning_validation/ \
    #     --report docs/handoffs/cleaning-validation-report.md

    def run_ablation(cohort_specs_by_layer_config, diagnostics):
        """Run every diagnostic on every layer combination.
        Layer combinations:
          - raw (no cleaning)
          - +soft-notch only
          - +baseline only
          - +mad only
          - +zscore only
          - all 4 layers
        Total: ~6 layer configs × 4 diagnostics × 3 cohort pairings = 72 cells.
        """
        ...
    ```

    Follow Script CLI Pattern (patterns.md §4): `parents[1]` bootstrap, separate `parse_args()`, return exit codes. Sample size 200 spectrograms per cohort is enough for diagnostic-grade Cohen's d / K-NN; full pull happens in 18.2.

5. `docs/handoffs/cleaning-validation-report.md` (NEW) — Generated output, not hand-written. Includes:
    - All 4 numeric criteria with pass/fail
    - Ablation table (which layers move the needle most)
    - Cohort-specific notes (soft-notch is a no-op for 5970 and VocalMat per C4)
    - Visual diagnostics: 6×3 grid of mean-spectrum-by-cohort PNGs (one column per cohort, one row per cleaning config)
    - **Go/no-go decision section** — REQUIRED. Either "All 4 criteria pass → Module 18.2 unlocked" or "Criteria X, Y failed → STOP, iterate."

**Test plan:**
    ```
    1. CleaningConfig __post_init__ rejects invalid baseline_mode and sample_rate_hz
    2. CleaningConfig with apply_soft_notch=True and tonal_library_path=None does not raise
       (soft-notch is allowed to no-op for cohorts without tonal libraries)
    3. clean_spectrogram preserves shape: input (n_freq, n_time) → output (n_freq, n_time)
    4. clean_spectrogram with all layers disabled returns input unchanged within float32 epsilon
    5. clean_spectrogram applies layers in correct order — verified by mocking each layer
       and asserting call sequence: notch → baseline → MAD → zscore
    6. notch_injection_test on synthetic spectra with NO injected cage tone yields migration
       rate ≤ 5% (sanity check — baseline should not falsely flag clean data)
    7. per_band_cohens_d on two identical spectrogram populations yields d ≈ 0 within ±0.05
    8. knn_same_cohort_rate on two well-separated cohort populations yields rate ≈ 1.0;
       on overlapping populations yields rate ≈ 0.5
    9. raw_pixel_pca_d on identical populations yields d ≈ 0; on shifted populations yields
       |d| > 1.0 (positive control)
    10. train_diagnostic_vae returns embedding array of shape (n_input, latent_dim=32)
        and trains for exactly n_epochs without NaN losses
    11. CLI script with --sample-size 0 produces empty report with clear error message
    12. End-to-end smoke: tiny synthetic 3-cohort dataset runs full ablation in <60s on CPU
        and produces a valid Markdown report with all 4 criteria rows present
    ```

**Exit criteria:**
- [ ] All tests pass (`pytest tests/classifier/test_cleaning_pipeline.py tests/classifier/test_diagnostics.py -v`)
- [ ] `py_compile` passes on all 3 new Python files
- [ ] Script runs end-to-end on real data: `python scripts/cnn_cleaning_validation.py --vocalmat-sample <real path> --lab-131204-sample <real path> --wild-5970-sample <real path>`
- [ ] `docs/handoffs/cleaning-validation-report.md` exists and includes all 4 criteria + ablation table + go/no-go decision
- [ ] If ANY of 4 criteria fail → STOP. Do not mark module complete. Surface failure to user with proposed iteration plan.
- [ ] Module doc written to `docs/modules/cnn-cleaning-validation.md`
- [ ] No production-detection files modified (`git diff --stat` shows only `src/usv_spectrogram/classifier/**` and `scripts/cnn_cleaning_validation.py` and `docs/` paths)

---

### 18.2a Sample Download for Phase 1.0 Gate

**What:** Download a small representative VocalMat sample (~200 spectrograms per class) sufficient for Module 18.1's real-data gate diagnostic. Resolves the chicken-and-egg between Module 18.1 (needs real data for its go/no-go verdict) and Module 18.2 (which would normally download the data but is blocked by 18.1 passing).
**Status:** READY (after 18.1 code complete)
**Review Tier:** 2 (download script + manifest only; no novel logic)
**Depends on:** 18.1 (code complete; gate verdict still pending)

/implement Sample Download for Phase 1.0 Gate

Build a minimal OSF download utility that pulls a ~200-spectrogram-per-class sample of VocalMat data plus a representative WAV per lab/wild cohort, sufficient to feed Module 18.1's `scripts/cnn_cleaning_validation.py` on REAL data and produce the binding go/no-go verdict.

**Context:** PLAN §"Phase 1.1 — Data Preparation" originally specified the full 12 GB VocalMat download. The split into 18.2a + 18.2b was decided 2026-05-21 to break the architectural deadlock: Module 18.1 deliverable requires real-data verdict, but 18.2 (full download) was blocked by 18.1 passing. 18.2a unblocks 18.1's verdict by providing just enough real data for the diagnostic; 18.2b (full pull) runs only if the gate passes.

**Files to create:**

1. `scripts/cnn_download_vocalmat_sample.py` (NEW) — Small-scale OSF download CLI
    - Pulls ~200 spectrograms balanced across 12 Grimsley classes from `https://osf.io/bk2uj/`
    - Optional `--full` flag bridges into 18.2b's full pull (only after gate passes)
    - Writes to `data/vocalmat_sample/{class}/*.png` with manifest CSV
2. `data/vocalmat_sample/.gitignore` (NEW) — `*` to prevent committing downloaded data

**Test plan:**
    ```
    1. Script with --dry-run prints download plan without fetching
    2. Manifest CSV has expected columns and balanced class counts
    3. Re-run is idempotent (skips already-downloaded files)
    ```

**Exit criteria:**
- [ ] Script downloads ~200 × 12 = ~2,400 spectrograms (~500 MB) into `data/vocalmat_sample/`
- [ ] Module 18.1 CLI accepts `--vocalmat-sample data/vocalmat_sample/` and produces a non-synthetic report
- [ ] Module 18.1 go/no-go verdict captured in `docs/handoffs/cleaning-validation-report.md`
- [ ] If the gate passes: 18.2b unlocks; if it fails: STOP, iterate cleaning

---

### 18.2b Full Data Preparation

**What:** Download the full VocalMat dataset (12,954 spectrograms, ~12 GB), resample our lab+wild WAVs to 250 kHz, apply the validated cleaning stack uniformly, generate 227×227 RGB spectrogram patches (VocalMat-compatible), and build stratified recording-level-grouped 80/10/10 splits.
**Status:** BLOCKED (waiting on 18.2a + 18.1 gate to pass)
**Review Tier:** 2 (pipeline plumbing; class-balance is the only complex piece)
**Depends on:** 18.2a (sample download) AND 18.1 gate passed (verdict in cleaning-validation-report.md)

/implement Full Data Preparation

Build the training-data preparation pipeline. Downloads VocalMat (12,954 labeled spectrograms across 12 classes), resamples our own 300 kHz lab+wild data to 250 kHz, applies the cleaning stack validated in 18.1, generates VocalMat-format spectrogram patches, and splits with strict no-leakage guarantees.

**Context:** PLAN §"Phase 1.1 — Data Preparation". VocalMat dataset characteristics verified 2026-05-20: 12,954 images, single-rater training labels (expect 5–15% noise in minority classes), test set is dual-rater consensus (gold standard, held-out). Source: 5 mouse strains × 3 neonatal ages × both sexes — already heterogeneous, good for cage invariance. **Class imbalance is severe**: top class (Step-up, 1,814) vs bottom (Multi-steps, 74) = 24.5×. With 80/10/10 split, Multi-steps gets ~59 training examples — ResNet-18's reasonable lower bound. D5 commits to keep-all-12-classes + class-weighted CE + focal loss + oversampling.

Reference: PLAN §"VocalMat Dataset Characteristics", §"Phase 1.1". `[[project-lab-cnn-classifier-scope]]`. Cross-phase constraints C1, C2, C3, C4 apply.

**Files to create:**

1. `src/usv_spectrogram/classifier/resample.py` (NEW) — 300→250 kHz resampling

    ```python
    import numpy as np
    from scipy.signal import resample_poly

    SOURCE_SAMPLE_RATE_HZ = 300_000   # our LMT hardware (corpus.SAMPLE_RATE_HZ)
    TARGET_SAMPLE_RATE_HZ = 250_000   # VocalMat-aligned
    RESAMPLE_UP = 5
    RESAMPLE_DOWN = 6                 # 300_000 * 5 / 6 = 250_000

    def resample_to_vocalmat(samples: np.ndarray) -> np.ndarray:
        """Resample mono audio from 300 kHz → 250 kHz using polyphase filtering.
        Anti-aliasing handled by resample_poly's built-in Kaiser window FIR.
        """
        if samples.ndim != 1:
            raise ValueError(f"samples must be mono (1D), got shape {samples.shape}")
        return resample_poly(samples, up=RESAMPLE_UP, down=RESAMPLE_DOWN).astype(np.float32)
    ```

2. `src/usv_spectrogram/classifier/dataset.py` (NEW) — Dataset manifest + class balance utilities

    ```python
    from dataclasses import dataclass, field
    from pathlib import Path
    import pandas as pd

    GRIMSLEY_12_CLASSES = (
        "Noise", "Step up", "Down-FM", "Short", "Chevron", "Up-FM",
        "Flat", "Two steps", "Step down", "Complex", "Reverse Chevron", "Multi-steps",
    )

    @dataclass(frozen=True)
    class DatasetSplit:
        train_csv: Path
        val_csv: Path
        test_csv: Path
        class_weights: dict[str, float]   # for class-weighted CE
        oversample_targets: dict[str, int]  # minority class target counts post-oversample

    def build_stratified_split(
        manifest: pd.DataFrame,        # columns: path, class, source_recording, duration_ms
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 1729,
    ) -> DatasetSplit:
        """Stratified 80/10/10 split with RECORDING-LEVEL grouping.
        Hard constraint: no call from the same source_recording appears in both train and val/test.
        This prevents leakage from recording-environment correlations.

        Class weights: inverse-frequency, normalized so mean weight = 1.0.
        Oversample targets: minority class counts brought up to median class count via random
        resampling WITH replacement on training set only.
        """
        ...
    ```

3. `scripts/cnn_prepare_training_data.py` (NEW) — End-to-end data prep CLI

    ```python
    # CLI structure
    # python scripts/cnn_prepare_training_data.py \
    #     --vocalmat-source <osf-download-dir> \
    #     --lab-wav-dirs USV_lab_131204/ \
    #     --wild-wav-dirs '5970 USV/' USV_3452_sample_reviewed/ \
    #     --output-dir data/lab_cnn_training/ \
    #     --patch-duration-s 0.22 \
    #     --workers 4

    # Steps:
    # 1. Verify VocalMat checksums (optional --skip-checksum-verify flag)
    # 2. Build VocalMat manifest CSV with columns: path, class, source_recording, duration_ms
    # 3. For each lab/wild WAV: load → resample → apply cleaning stack → STFT → patch into
    #    0.22s 227×227 RGB patches using VocalMat's STFT params (Hamming 256, hop 128,
    #    NFFT 1024 at 250 kHz). Apply MAD globally per WAV (C2).
    # 4. Visual sanity check: emit 50 random patches per cohort to data/lab_cnn_training/
    #    sanity_patches/ for human inspection (the PLAN risk: "--subtract-baseline interacts
    #    badly with VocalMat's spectrogram conventions").
    # 5. Build stratified 80/10/10 split with recording-level grouping.
    # 6. Write {train,val,test}/manifest.csv + corresponding PNG patches into split dirs.
    ```

    PLAN risk explicitly called out: "validate output PNGs visually in Phase 1.1 ablation" — step 4 is the human-in-the-loop checkpoint.

4. `data/vocalmat/.gitignore` (NEW) — Single line `*` to prevent committing the 12 GB dataset

5. `docs/modules/cnn-data-preparation.md` (NEW) — Module doc explaining VocalMat STFT params, why 250 kHz vs 300 kHz (C1), recording-level grouping rationale, class-balance strategy (D5)

**Test plan:**
    ```
    1. resample_to_vocalmat preserves duration_s within 1 sample tolerance:
       len(out) ≈ len(in) * 5 / 6
    2. resample_to_vocalmat raises ValueError on stereo input
    3. resample_to_vocalmat returns float32 dtype
    4. resample_to_vocalmat on a 60 kHz pure tone at 300 kHz produces a 60 kHz pure tone
       at 250 kHz (no aliasing — 60 kHz < new Nyquist 125 kHz, well within band)
    5. resample_to_vocalmat on a 140 kHz pure tone at 300 kHz produces output with energy
       at 110 kHz or attenuated by >40 dB (anti-aliasing positive control — 140 > new Nyquist)
    6. build_stratified_split enforces recording-level grouping: every source_recording in
       val/test is absent from train (hard zero-leakage assertion)
    7. build_stratified_split preserves per-class proportions within ±2% across splits
    8. build_stratified_split class_weights sum to (n_classes * 1.0) within 1e-6 (normalized)
    9. build_stratified_split oversample_targets bring all minority classes up to median
    10. End-to-end smoke: small synthetic VocalMat-like dataset (12 classes × 5 examples each)
        runs full prep pipeline in <30s and produces valid splits
    ```

**Exit criteria:**
- [ ] All tests pass
- [ ] `py_compile` passes
- [ ] Script runs on real VocalMat download + at least one lab + one wild WAV
- [ ] `data/lab_cnn_training/sanity_patches/` populated with 50 patches per cohort — human-verified by user before 18.3 starts (CHECKPOINT: pause for user review)
- [ ] Manifest CSVs exist in train/val/test with correct columns
- [ ] No production-detection files modified

---

### 18.3 Baseline ResNet-18 Training

**What:** Train a 12-class syllable classifier using `timm`'s ResNet-18 (ImageNet-pretrained), AdamW, cosine LR, ~50 epochs, with SpecAugment + audio augmentation, class-weighted CE + focal loss + oversampling (D5). Run Perch 2.0 linear probe in parallel (D3) for embedding-quality comparison.
**Status:** BLOCKED (waiting on 18.2)
**Review Tier:** 3 (ML training + class imbalance + augmentation correctness)
**Depends on:** 18.2

/implement Baseline ResNet-18 Training

Train ResNet-18 v1 baseline and produce per-class metrics + held-out 845 verdict evaluation. Also run Perch 2.0 embeddings + linear probe as parallel comparator (1-day sidequest, D3 approved).

**Context:** PLAN §"Phase 1.2 — Baseline ResNet-18 Training". ResNet-18 is right capacity for ~13k examples; abundant transfer-learning support via `timm`. EfficientNet-B0 is the documented fallback if overfitting appears. The 845 hand-curated lab 131204 verdicts (`classified_detections_lab_131204_clean.csv`) are held out from training and used as an independent acceptance test — these were dual-rater-quality (better than VocalMat's single-rater training labels). Perch 2.0 (arXiv:2512.03219, 2025) is a bioacoustic-pretrained audio embedding model; recent evidence shows bioacoustic-pretrained embeddings often beat ImageNet-pretrained CNNs cross-domain.

Reference: PLAN §"Phase 1.2", §"Validation criteria". Cross-phase constraints C5, C6 apply. D3 (Perch sidequest), D5 (minority class strategy) baked in.

**Files to create:**

1. `src/usv_spectrogram/classifier/model.py` (NEW) — Model factory + classifier head

    ```python
    import timm
    import torch
    import torch.nn as nn

    NUM_CLASSES = 12   # Grimsley 2011 taxonomy

    def build_resnet18_classifier(
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> nn.Module:
        """timm ResNet-18 ImageNet-pretrained backbone with 12-class head.
        Returns model with .forward(x) -> logits of shape (B, num_classes).
        """
        return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
    ```

2. `src/usv_spectrogram/classifier/augmentation.py` (NEW) — SpecAugment + audio aug

    ```python
    from dataclasses import dataclass
    import numpy as np
    import torch

    @dataclass(frozen=True)
    class AugmentationConfig:
        time_mask_max_width_frames: int = 20    # SpecAugment T_max
        time_mask_n: int = 2                    # SpecAugment m_T
        freq_mask_max_width_bins: int = 16      # SpecAugment F_max
        freq_mask_n: int = 2                    # SpecAugment m_F
        pitch_shift_max_pct: float = 0.10       # ±10%
        time_stretch_max_pct: float = 0.20      # ±20%
        random_crop_max_pct: float = 0.05       # ±5%
        cage_noise_inject_prob: float = 0.25    # probability of injecting noise from 845 verdict negatives
        cage_noise_paths: tuple[str, ...] = ()  # paths to verdict-negative patches

    def specaugment(spec: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
        """SpecAugment time+freq masking. Park et al. 2019 (arXiv:1904.08779)."""
        ...

    def inject_cage_noise(spec: np.ndarray, cfg: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
        """Sample a verdict-negative patch and additively blend into spec.
        Targets the cage-confound directly — novel-to-this-codebase per PLAN."""
        ...
    ```

3. `src/usv_spectrogram/classifier/losses.py` (NEW) — Class-weighted CE + focal loss

    ```python
    import torch
    import torch.nn.functional as F

    def focal_loss(
        logits: torch.Tensor,        # (B, C)
        targets: torch.Tensor,       # (B,) long
        class_weights: torch.Tensor, # (C,)
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Lin et al. 2017 focal loss with per-class weighting (D5 strategy).
        Reduces relative loss for well-classified examples; up-weights minority classes."""
        ...
    ```

4. `src/usv_spectrogram/classifier/training.py` (NEW) — Training loop

    ```python
    from dataclasses import dataclass
    from pathlib import Path

    @dataclass(frozen=True)
    class TrainingConfig:
        epochs: int = 50
        batch_size: int = 64
        learning_rate: float = 1e-3
        weight_decay: float = 1e-4
        warmup_epochs: int = 3
        cosine_min_lr: float = 1e-5
        early_stop_patience: int = 8
        confusion_matrix_every_epochs: int = 5
        focal_gamma: float = 2.0
        device: str = "cuda"

    def train_classifier(
        train_loader, val_loader, test_loader,    # PyTorch DataLoaders
        cfg: TrainingConfig,
        output_dir: Path,
        held_out_845_csv: Path,                    # lab 131204 dual-rater verdicts
    ) -> dict:
        """Full training loop with:
        - AdamW + cosine LR + warmup
        - SpecAugment + cage-noise injection (augmentation.py)
        - Class-weighted focal loss (losses.py)
        - Per-class precision/recall + confusion matrix every 5 epochs
        - Best checkpoint by macro F1 on val
        - Final eval on test split + held-out 845 USV/noise + 845 syllable-type entropy

        Returns metrics dict with all PLAN §"Validation criteria" measurements.
        """
        ...
    ```

5. `scripts/train_lab_classifier.py` (NEW) — Training CLI

    ```python
    # python scripts/train_lab_classifier.py \
    #     --train-csv data/lab_cnn_training/train/manifest.csv \
    #     --val-csv   data/lab_cnn_training/val/manifest.csv \
    #     --test-csv  data/lab_cnn_training/test/manifest.csv \
    #     --held-out-845 classified_detections_lab_131204_clean.csv \
    #     --output-dir results/lab_classifier_v1/ \
    #     --epochs 50 --batch-size 64
    ```

6. `scripts/train_perch2_probe.py` (NEW, D3 sidequest) — Perch 2.0 linear probe

    ```python
    # Embed every training/val/test patch with Perch 2.0 (frozen),
    # train a 12-class linear classifier on those embeddings.
    # Compare macro F1 vs ResNet-18 baseline.
    # Output: results/perch2_probe/eval_report.md
    ```

7. `docs/modules/lab-classifier-v1.md` (NEW) — Architecture, hyperparams, results

**Test plan:**
    ```
    1. build_resnet18_classifier returns model with output shape (B, 12) for input (B, 3, 227, 227)
    2. AugmentationConfig __post_init__ rejects negative masking widths
    3. specaugment preserves spectrogram shape
    4. specaugment with all mask widths = 0 returns input unchanged
    5. inject_cage_noise preserves shape; with cage_noise_inject_prob=0 returns input unchanged
    6. focal_loss reduces to weighted CE when gamma=0
    7. focal_loss positive control: minority class with high weight → larger gradient than majority
    8. TrainingConfig __post_init__ rejects epochs <= 0, batch_size <= 0, lr <= 0
    9. End-to-end smoke: 12-class synthetic dataset (10 samples per class) trains 2 epochs in
       <90s on CPU without NaN losses, produces a checkpoint
    10. Held-out 845 evaluation function returns dict with keys: usv_noise_acc, syllable_entropy_mean
    11. Perch 2.0 probe: embedding dim is fixed across all inputs (no NaNs)
    12. Perch 2.0 probe: linear classifier reaches macro F1 > 0.30 on smoke dataset (sanity)
    ```

**Exit criteria:**
- [ ] All tests pass
- [ ] `py_compile` passes
- [ ] Training runs to completion on real data, produces `models/lab_classifier_v1/best.pt`
- [ ] `results/lab_classifier_v1/eval_report.md` exists with:
    - Macro F1 on VocalMat val split > 0.65 (PLAN pass)
    - Per-class precision: all ≥ 0.40 (PLAN pass; if any < 0.20, revisit D5)
    - Held-out 845 USV/noise accuracy > 0.80
    - Held-out 845 syllable-type entropy mean ≤ log(6)
- [ ] `results/perch2_probe/eval_report.md` exists with Perch vs ResNet-18 macro F1 comparison
- [ ] Confusion matrix PNG at `results/lab_classifier_v1/confusion_matrix.png`
- [ ] Module doc at `docs/modules/lab-classifier-v1.md`
- [ ] If per-class precision < 0.20 on Multi-steps or Reverse-Chevron: STOP and revisit D5 (collapse step-family or drop) before 18.4

---

### 18.4 DANN Cage-Confound-Aware Training

**What:** Add a gradient-reversal domain discriminator head to the v1 architecture (2-cage granularity per D4: `lab_131204` vs `vocalmat`), train with λ schedule 0 → 1 per Ganin et al. 2015, then verify cage-invariance via linear probe on encoder features + VAE falsifiable test on encoder embeddings.
**Status:** BLOCKED (waiting on 18.3)
**Review Tier:** 3 (adversarial training + encoder collapse risk)
**Depends on:** 18.3

/implement DANN Cage-Confound-Aware Training

Extend the v1 training pipeline with a Domain-Adversarial Neural Network (DANN) head. The encoder is shared between syllable classification (12-way) and cage discrimination (2-way: lab_131204 vs vocalmat). A gradient-reversal layer between encoder and domain head causes the encoder to learn to be *bad* at cage discrimination while staying *good* at syllable classification — yielding a cage-invariant representation.

**Context:** PLAN §"Phase 1.3 — Cage-Confound-Aware Training". Ganin & Lempitsky 2015 (arXiv:1409.7495). The DANN head is ~50 lines on top of v1. The risk is encoder collapse: if λ is too aggressive, the encoder can find a trivial cage-invariant representation that also destroys syllable signal. PLAN sets validation criterion "syllable F1 vs v1 baseline no worse than −0.05" exactly to catch this. D4 commits to 2-cage granularity (minimum aggressiveness) — per-recording (50–100 cages) is documented as a more aggressive option if needed.

Reference: PLAN §"Phase 1.3". Ganin & Lempitsky 2015 ICML, "Unsupervised Domain Adaptation by Backpropagation". Cross-phase constraints C2, C5, C6 apply.

**Files to create:**

1. `src/usv_spectrogram/classifier/dann.py` (NEW) — Gradient-reversal layer + domain head + λ schedule

    ```python
    from dataclasses import dataclass
    import torch
    import torch.nn as nn

    class GradientReversal(torch.autograd.Function):
        """Forward: identity. Backward: multiplies gradient by -λ.
        Standard Ganin 2015 implementation."""
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambda_, None

    def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
        return GradientReversal.apply(x, lambda_)

    class DomainHead(nn.Module):
        """2-way cage discriminator (D4: lab_131204 vs vocalmat)."""
        def __init__(self, feature_dim: int, num_domains: int = 2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_domains),
            )
        def forward(self, features, lambda_):
            return self.net(grad_reverse(features, lambda_))

    @dataclass(frozen=True)
    class LambdaSchedule:
        """Ganin 2015 schedule: λ(p) = 2 / (1 + exp(-10p)) − 1, p = epoch/total_epochs."""
        total_epochs: int
        gamma: float = 10.0
        def lambda_at(self, epoch: int) -> float:
            p = epoch / max(1, self.total_epochs)
            return float(2.0 / (1.0 + (-self.gamma * p).__class__(0).__class__(0)))  # see real impl
        # (real implementation uses math.exp; placeholder above shows the formula source)

    class ResNet18DANN(nn.Module):
        """Shared ResNet-18 encoder + class head + domain head."""
        def __init__(self, num_classes: int = 12, num_domains: int = 2):
            ...
        def forward(self, x, lambda_: float = 0.0):
            features = self.encoder(x)
            class_logits = self.class_head(features)
            domain_logits = self.domain_head(features, lambda_)
            return class_logits, domain_logits, features
    ```

2. `src/usv_spectrogram/classifier/cage_probe.py` (NEW) — Linear cage probe on frozen encoder features

    ```python
    def linear_cage_probe(
        encoder: nn.Module,
        train_loader,
        val_loader,
        num_cages: int = 2,
        device: str = "cuda",
    ) -> float:
        """Freeze encoder, train a linear classifier on features to predict cage.
        Return val accuracy. Pass threshold: < 0.65 (vs random ~0.50 for 2 cages).
        Lower = more cage-invariant encoder."""
        ...
    ```

3. `scripts/train_lab_classifier_v2.py` (NEW) — DANN training CLI (extends v1 script)

    ```python
    # python scripts/train_lab_classifier_v2.py \
    #     --train-csv data/lab_cnn_training/train/manifest.csv \
    #     --val-csv   data/lab_cnn_training/val/manifest.csv \
    #     --test-csv  data/lab_cnn_training/test/manifest.csv \
    #     --v1-checkpoint models/lab_classifier_v1/best.pt   # warm start, optional
    #     --output-dir results/lab_classifier_v2/ \
    #     --domain-granularity 2cage          # D4 default; "per_recording" override
    #     --epochs 50 --batch-size 64
    ```

4. `scripts/run_vae_diagnostic_on_encoder.py` (NEW) — Re-runs 18.1 VAE diagnostic but on **encoder features** instead of raw spectrograms

    ```python
    # Loads v2 encoder, extracts features for all val patches across cohorts,
    # runs the same notch_injection/per_band_d/knn/PC1 tests from 18.1 on features.
    # Output: results/lab_classifier_v2/cage_invariance_probe.md
    ```

5. `docs/modules/lab-classifier-v2-dann.md` (NEW) — Module doc explaining DANN intuition, λ schedule choice, encoder collapse risk and detection

**Test plan:**
    ```
    1. GradientReversal forward returns input unchanged
    2. GradientReversal backward returns gradient × −λ (test with λ=1.0, λ=0.5, λ=0)
    3. DomainHead with 2-way output produces logits of shape (B, 2)
    4. LambdaSchedule.lambda_at(0) ≈ 0.0; lambda_at(total_epochs) ≈ 1.0; monotonically increasing
    5. ResNet18DANN.forward returns 3-tuple (class_logits, domain_logits, features) with correct shapes
    6. ResNet18DANN with lambda_=0.0 produces identical class_logits across calls (deterministic)
    7. linear_cage_probe on a frozen encoder that was trained WITHOUT DANN on a 2-cage
       dataset where cohorts differ in mean intensity yields accuracy > 0.90 (positive control)
    8. linear_cage_probe on a randomly-initialized encoder yields accuracy ≈ 0.50 ± 0.05 (negative control)
    9. Training smoke: 2 cages × 6 classes × 8 samples per class trains 2 epochs without NaN
    10. Training smoke: v2 macro F1 within ±0.10 of v1 baseline on smoke dataset (collapse detection)
    ```

**Exit criteria:**
- [ ] All tests pass
- [ ] `py_compile` passes
- [ ] v2 training runs to completion, produces `models/lab_classifier_v2/best.pt`
- [ ] Linear cage probe accuracy < 0.65 on v2 encoder features (PLAN pass)
- [ ] Syllable macro F1 no worse than v1 baseline − 0.05 (PLAN pass — collapse detection)
- [ ] VAE falsifiable test re-run on v2 encoder features: all 4 criteria pass (stronger evidence than 18.1's raw-spectrogram pass)
- [ ] `results/lab_classifier_v2/cage_invariance_probe.md` written with all metrics
- [ ] `results/lab_classifier_v2/comparison_v1_vs_v2.md` written (side-by-side metric table)
- [ ] Module doc at `docs/modules/lab-classifier-v2-dann.md`
- [ ] If F1 drops by > 0.05 OR cage probe stays > 0.65: STOP. Encoder collapsed or λ schedule wrong — surface to user with proposed λ schedule alternatives.

---

### 18.5 Wild Transfer Evaluation (USER-ENABLED)

**What:** User produces ~200 labeled wild USV calls from 5970. Run inference with v1 (baseline) and v2 (DANN) classifiers, report per-class precision/recall + confusion matrix + confidence distributions, decide ship-as-production-classifier vs needs-Phase-2.
**Status:** BLOCKED (waiting on 18.4 + user labeling per D2)
**Review Tier:** 2 (eval + reporting)
**Depends on:** 18.4, user-produced labels at `data/wild_labels/5970_human_verified.csv`

/implement Wild Transfer Evaluation

Evaluate v1 and v2 classifiers on user-curated wild data to test lab → wild generalization. This is the decision point for the entire plan: does v2 generalize meaningfully better than v1 → ship; or no → Phase 2 needed.

**Context:** PLAN §"Phase 1.4 — Wild Transfer Evaluation". Domain-gap reality: VocalMat training already spans 5 strains × 3 ages, so strain monoculture is NOT the cause if wild transfer fails. Real candidates are (1) cage acoustics (mitigated by DANN in 18.4 + cleaning stack), (2) call-duration distribution (wild 30–50 ms vs lab 30–200 ms — same mechanism that killed κ=0.13 VocalMat transfer; `[[project-vocalmat-transfer-test]]`). D1 deferred the 0.08s short-patch variant — re-evaluate it here if v2 fails.

User labeling protocol: PLAN §"Phase 1.4 Pre-requisite" + `[[feedback-labeling-queue-folder]]` — PNGs copied to flat folder with stable-ID prefixes (`typ01_*.png`). Single-rater is acceptable for OOD eval. ~200 calls at ~30s each ≈ 100 minutes labeling time. 200 calls / 12 classes ≈ 17/class average — descriptive statistics only, not inferential.

Reference: PLAN §"Phase 1.4". Cross-phase constraints C5, C6 apply.

**Files to create:**

1. `scripts/build_wild_labeling_queue.py` (NEW) — Pre-labeling tool

    ```python
    # python scripts/build_wild_labeling_queue.py \
    #     --detections classified_detections_5970.csv \
    #     --output-dir labeling_queue_wild_5970/ \
    #     --n-calls 200 \
    #     --balance-by-v2-prediction   # use v2 confidence to bootstrap stratification (D2)

    # Copy detection PNGs to flat folder with stable-ID filenames typXX_origID.png.
    # Emit empty CSV template at data/wild_labels/5970_human_verified.csv for user to fill.
    ```

2. `scripts/eval_wild_transfer.py` (NEW) — Eval CLI

    ```python
    # python scripts/eval_wild_transfer.py \
    #     --labels data/wild_labels/5970_human_verified.csv \
    #     --v1-checkpoint models/lab_classifier_v1/best.pt \
    #     --v2-checkpoint models/lab_classifier_v2/best.pt \
    #     --wav-dirs '5970 USV/' \
    #     --output-dir results/wild_transfer_eval/

    # For each labeled call:
    # 1. Extract patch from WAV using same cleaning + resampling as training (18.2)
    # 2. Run v1 inference → top-1 prediction + confidence
    # 3. Run v2 inference → top-1 prediction + confidence
    # 4. Tally per-class precision/recall vs human label
    # 5. Build confusion matrices (v1 and v2 separately)
    # 6. Compare confidence distributions on agreed vs disagreed predictions
    # 7. Decision: v2 macro F1 vs v1 macro F1 — Δ > +0.05 → ship; else Phase 2 memo
    ```

3. `data/wild_labels/5970_human_verified.csv` (NEW, USER-PRODUCED) — Schema:
    ```
    wav_stem, det_start_s, det_end_s, syllable_type_grimsley
    ```
    The training pipeline produces this template; user fills it.

4. `results/wild_transfer_eval/decision_memo.md` (NEW, GENERATED) — Ship-or-Phase-2 decision

5. `docs/modules/wild-transfer-evaluation.md` (NEW) — Methodology + result interpretation guide

**Test plan:**
    ```
    1. build_wild_labeling_queue produces N PNG files in output dir with typXX_ prefix
    2. build_wild_labeling_queue produces empty CSV template with correct headers
    3. eval_wild_transfer raises clear error if --labels file missing or has wrong schema
    4. eval_wild_transfer raises clear error if either checkpoint missing
    5. eval_wild_transfer on 12 synthetic labeled calls produces:
       - Confusion matrices with shape (12, 12) for both v1 and v2
       - Per-class precision/recall dict with all 12 Grimsley classes as keys
       - Confidence distribution arrays
       - Decision memo with explicit "ship" or "Phase 2" recommendation
    6. eval_wild_transfer uses the SAME cleaning + resampling pipeline as training
       (verified by importing from src/usv_spectrogram/classifier/{cleaning_pipeline,resample}.py
       and asserting no duplicate implementation)
    7. Decision logic positive control: synthetic case where v2 F1 = v1 F1 + 0.10 →
       memo recommends "ship"
    8. Decision logic negative control: synthetic case where v2 F1 = v1 F1 − 0.02 →
       memo recommends "Phase 2"
    ```

**Exit criteria:**
- [ ] All tests pass
- [ ] `py_compile` passes
- [ ] User has labeled ≥ 100 wild calls (200 ideal; PLAN allows descriptive-only at 200)
- [ ] `results/wild_transfer_eval/v1_eval.md` and `v2_eval.md` written with confusion matrices
- [ ] `results/wild_transfer_eval/decision_memo.md` written with explicit ship-or-Phase-2 recommendation + rationale
- [ ] If v2 fails: re-evaluate D1 (short-patch 0.08s variant) and surface to user with cost estimate
- [ ] Module doc at `docs/modules/wild-transfer-evaluation.md`

---

## Phase 18 Gate

- [ ] 18.1 gate passed (4 cleaning-validation criteria) — recorded in `cleaning-validation-report.md`
- [ ] 18.2 data prep produces train/val/test splits + sanity patches user-approved
- [ ] 18.3 v1 baseline meets PLAN validation criteria (macro F1 > 0.65, per-class ≥ 0.40, held-out 845 > 0.80)
- [ ] 18.3 Perch 2.0 sidequest evaluated; comparison written
- [ ] 18.4 v2 DANN meets cage-invariance threshold (linear probe < 0.65) AND no syllable F1 regression > 0.05
- [ ] 18.4 VAE falsifiable test re-run on v2 encoder features passes all 4 criteria
- [ ] 18.5 wild transfer decision memo written; v2 either shipped to production *or* Phase 2 scoped
- [ ] All 5 module docs in `docs/modules/`
- [ ] No production-detection files modified across all 5 modules (`corpus.py`, `sliding_inference.py`, batch detection script remain untouched)
- [ ] Plan-author's "Success Definition" met: 12-class lab classifier with macro F1 > 0.65 OR honest documented failure

---

## Open Items Carried Forward From Plan

These are not blocking but should be tracked:

- Lab tonal library only covers 131204 — limits Phase 2 (multi-cage lab training). Mitigation: `scripts/calibrate_lab_tonal_lines.py` exists but unused for new lab data.
- VocalMat training is neonatal (P5–P15); our wild + lab are adult. If 18.5 fails, this is the second most likely cause after cage acoustics. Mitigation in PLAN §"Risk Register": "filter VocalMat data to older ages (P15) if available in metadata, or accept the gap and document."
- VocalMat training set is single-rater; expect 5–15% label noise in minority classes. Don't treat as gold-standard. PLAN risk register mitigation: "consider active-learning relabeling in Phase 2 if minority-class precision is too poor."
