# USV Labeling SIS Benchmark — ROADMAP

> Multi-method benchmark to determine which labeling scheme produces the highest Syntax Information Score (SIS / MI at lag 1) on our 7,518-call 5970 dataset. Tests five hypotheses about what makes USV labels sequentially informative: (1) the labels we already have, (2) rule-based pitch-jump detection (Hertz 2020 iMSA), (3) FM+AM ridge vectorization + clustering (Oren 2024), (4) autoencoder features + PCA + clustering (Stoumpou 2022 AMVOC), (5) direct SIS optimization via SIM (Hertz 2020).
>
> Paper triangulation handoff: `docs/handoffs/three-paper-deep-reads-2026-04-15.md`.
> Reference metric: our current MI at lag 1 on 7-type Scattoni labels = **0.093 bits**.
> Hertz 2020 published values: iVoICE 0.10 bits, iMUPET 0.13 bits, iMSA 0.22 bits, SIM 0.23 bits (depth 1, 346K syllables).

**Scope note:** All modules operate on the 5970 dataset (animal `usv_lmt_034`, 7,518 classified USVs in `classified_detections_full.csv`). Extending to 3452/9252 is Phase B work and out of scope here.

**Signal processing constants** (per ADR-001 + ADR-002): `sr=300_000`, `n_fft=512`, `hop_length=128`.

---

## Phase 17: USV Labeling SIS Benchmark

### 17.1 SIS Baselines on Existing Labels

**What:** Compute MI at lag 1 for the three existing labelings (Scattoni-7, DeepSqueak-27, HDBSCAN-3) and report results in a single table. Serves as the decision gate for the rest of the phase.
**Status:** READY
**Review Tier:** 2
**Depends on:** None

/implement SIS Baselines on Existing Labels

Build a driver script + reusable function that computes MI at lag 1 (SIS-equivalent at depth 1) for every labeling already present in our results directory, so we know the ceiling we need Phase 17's new methods to beat.

**Context:**
- Hertz et al. 2020 defined SIS as `H(X_n) - H(X_n | X_{n-1}..X_{n-D})` which at depth D=1 equals `I(X_n; X_{n-1})` — mutual information between consecutive syllables.
- Our current known value: MI at lag 1 = 0.093 bits on 7-type Scattoni labels (from Phase A2 `results/sequential_structure/`).
- Hertz compared iVoICE (0.10), iMUPET (0.13), iMSA (0.22) bits on 346K syllables. Our 7,518-call dataset is 43× smaller, so we're constrained to depth 1.
- This module is the "free baseline" — no new features, no new clustering, just MI on labels we already have. If any existing labeling is already ≥0.15 bits, feature engineering may not be needed.

**Input data:**
- `classified_detections_full.csv` — has `label` (DeepSqueak 27-cluster k-means), `syllable_type` (Scattoni 7-type), plus per-call `file`, `begin_time_s` columns.
- `results/recluster_umap_hdbscan/reclassified_detections.csv` — has `hdbscan_label` (3-cluster manifold).
- Existing `usv_language/analysis/information_theory.py::mutual_information_at_lag(sequence, K, lag)` — reuse this; do not reimplement.

**Files to create:**

1. `src/usv_spectrogram/classification/sis_baselines.py` (NEW) — Reusable SIS computation

    ```python
    from dataclasses import dataclass
    import numpy as np
    from usv_language.analysis.information_theory import mutual_information_at_lag

    @dataclass(frozen=True)
    class SISResult:
        """SIS result for one labeling."""
        name: str                    # e.g. "scattoni-7"
        n_calls: int                 # total calls in the sequence
        n_labels: int                # K, alphabet size
        mi_at_lag_1: float           # bits, depth-1 SIS
        marginal_entropy: float      # bits, H(X_n)
        conditional_entropy: float   # bits, H(X_n | X_{n-1})
        entropy_reduction_pct: float # (MI / marginal_entropy) * 100

    def compute_sis_depth_1(
        labels: np.ndarray,
        name: str,
        sort_by_time: np.ndarray | None = None,
    ) -> SISResult:
        """Compute depth-1 SIS for a sequence of integer labels.

        If sort_by_time is given, labels are reordered by that key first.
        Returns SISResult with marginal/conditional entropies and MI.
        """
    ```

   Implementation notes:
   - Cast labels to contiguous integers [0..K-1] via `pd.factorize` (handles string labels like 'Flat', 'Down', etc.)
   - Sort by `(file, begin_time_s)` before MI computation — sequential order matters
   - K = number of unique labels
   - Use `mutual_information_at_lag(sequence, K, lag=1)` for the MI value
   - Compute marginal entropy H(X) with log2 — match the existing module's convention

2. `scripts/run_sis_baselines.py` (NEW) — Driver

    Follow Pattern 4 (Script CLI) from `docs/architecture/patterns.md`.
    Args: `--classified-csv PATH`, `--umap-csv PATH`, `--output-dir DIR`.
    Default to `classified_detections_full.csv` and `results/recluster_umap_hdbscan/reclassified_detections.csv`.

    Pipeline:
    1. Load both CSVs, join on the call ID column (likely `id` or `det_index`)
    2. For each labeling column — `syllable_type` (Scattoni-7), `label` (DeepSqueak-27), `hdbscan_label` (HDBSCAN) — call `compute_sis_depth_1`
    3. Write `results/sis_baselines/baselines.csv` with one row per labeling
    4. Write `results/sis_baselines/baselines.png` — bar chart of MI at lag 1 with horizontal reference lines at Hertz's 0.10 / 0.13 / 0.22 bit values
    5. Print a summary table to stdout (name, K, MI, % entropy reduction)

3. `tests/test_sis_baselines.py` (NEW)

**Test plan:**
```
1. compute_sis_depth_1 on a perfectly periodic sequence [A,B,A,B,...] → MI = 1.0 bit (binary, perfect predictability)
2. compute_sis_depth_1 on an i.i.d. random sequence → MI ≈ 0 bits
3. SISResult.entropy_reduction_pct is in [0, 100]
4. sort_by_time reorders labels correctly before MI computation (test with shuffled input)
5. String labels (e.g. ['Flat','Down','Flat']) are handled via pd.factorize
6. Script end-to-end on synthetic CSV produces baselines.csv with 3 rows
7. Empty sequence returns MI = 0 without crash
8. Single-label sequence (all same label) returns MI = 0, marginal_entropy = 0
```

**Exit criteria:**
- [ ] `results/sis_baselines/baselines.csv` exists with MI values for Scattoni-7, DeepSqueak-27, HDBSCAN-3
- [ ] Scattoni-7 MI ≈ 0.093 bits (matches prior Phase A2 result — reproducibility check)
- [ ] `baselines.png` shows bar chart with Hertz reference lines
- [ ] All tests pass
- [ ] py_compile passes

**Decision gate after this module:** If all three baselines are below 0.05 bits, the sequential structure of our data may be intrinsically weak and feature engineering may not help. Discuss before proceeding to 17.2+.

---

### 17.2 Spectrogram Pre-Filtering Module

**What:** DSP module that cleans a single USV spectrogram before ridge extraction or autoencoder encoding — amplitude thresholding (noise floor), median filter, frequency band masking. Shared infrastructure for 17.3, 17.5, 17.6.
**Status:** READY
**Review Tier:** 3
**Depends on:** None

/implement Spectrogram Pre-Filtering Module

Build a stateless DSP module that takes a magnitude spectrogram and returns a cleaned version with noise masked out, suitable for ridge extraction and autoencoder input.

**Context:**
- Our wild-mouse recordings have substantial cage noise, scratching transients, and occasional amplitude-modulated noise bands.
- Simple argmax ridge extraction on unfiltered spectrograms will latch onto: silent column noise (random frequencies), broadband transients (outlier frequencies), low-SNR onset/offset columns.
- AMVOC autoencoder training on unfiltered spectrograms will waste capacity reconstructing noise.
- Three defenses: (1) amplitude threshold by local noise floor, (2) 3×3 median filter to remove isolated pixel noise, (3) frequency band mask outside 25-120 kHz (removes equipment hum, ventilation, cage clatter).

**Files to create:**

1. `src/usv_spectrogram/features/spectrogram_filter.py` (NEW)

    ```python
    from dataclasses import dataclass
    import numpy as np
    from scipy.ndimage import median_filter

    @dataclass(frozen=True)
    class FilterConfig:
        """Pre-filtering parameters for USV spectrograms."""
        sample_rate: int = 300_000
        noise_floor_multiplier: float = 3.0   # mask below 3× local median
        noise_floor_window_cols: int = 20     # columns for local-median window
        median_filter_size: int = 3           # 3×3 kernel
        freq_min_hz: float = 25_000.0
        freq_max_hz: float = 120_000.0

        def __post_init__(self) -> None:
            if self.noise_floor_multiplier <= 1.0:
                raise ValueError("noise_floor_multiplier must be > 1")
            if self.freq_min_hz >= self.freq_max_hz:
                raise ValueError("freq_min_hz must be < freq_max_hz")
            if self.median_filter_size % 2 != 1:
                raise ValueError("median_filter_size must be odd")


    def prefilter_spectrogram(
        magnitude: np.ndarray,          # shape (n_freq_bins, n_time_cols), linear magnitude
        freqs_hz: np.ndarray,           # shape (n_freq_bins,), frequency of each bin
        cfg: FilterConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (cleaned_magnitude, mask).

        Steps:
        1. Apply 3×3 median filter to magnitude (removes isolated noise pixels)
        2. Compute local noise floor per column: rolling median over cfg.noise_floor_window_cols
        3. mask = magnitude > cfg.noise_floor_multiplier × local_noise_floor
        4. mask &= (freqs_hz >= freq_min_hz) & (freqs_hz <= freq_max_hz) broadcast per row
        5. cleaned = magnitude * mask (zero out masked pixels)

        Returns cleaned magnitude array (same shape) and boolean mask.
        """
    ```

   Implementation notes:
   - Use `scipy.ndimage.median_filter` with `size=cfg.median_filter_size`
   - Local noise floor: for each time column, take median over a window of `noise_floor_window_cols` centered on that column. Handle edges with `mode='reflect'`.
   - Frequency mask applied via broadcasting: `freq_mask = (freqs_hz >= cfg.freq_min_hz) & (freqs_hz <= cfg.freq_max_hz)`, then `mask &= freq_mask[:, None]`
   - Return `cleaned = magnitude * mask.astype(magnitude.dtype)` — preserves dtype

2. `tests/test_spectrogram_filter.py` (NEW)

**Test plan:**
```
1. Pure-tone spectrogram (synthetic): peak is preserved after filtering
2. Spectrogram with one high-amplitude pixel outlier: median filter removes it
3. Silent column + signal column: silent columns get mostly masked out, signal column passes
4. Low-frequency content (<25 kHz): fully masked to zero
5. Frequency mask shape broadcasts correctly on (129, 1000) test input
6. FilterConfig validation: freq_min >= freq_max raises; even median_filter_size raises
7. Input shape (n_freq_bins, n_time_cols) is preserved
8. Edge case: very short signal (n_time_cols < noise_floor_window_cols) — no crash
9. All-zero input returns all-zero output without NaN
```

**Exit criteria:**
- [ ] Filter reduces broadband noise on a synthetic noisy tone by >10 dB SNR improvement
- [ ] Frequency bins outside [25, 120] kHz are zero after filtering
- [ ] All tests pass
- [ ] py_compile passes

---

### 17.3 DP-Based Ridge Tracker

**What:** Extract the dominant frequency trajectory (pitch contour) from a cleaned spectrogram using Viterbi-like dynamic programming with a transition penalty, preventing harmonic jumps. Produces FM (frequency) and AM (amplitude) trajectories with NaN for silent columns.
**Status:** READY
**Review Tier:** 3
**Depends on:** 17.2

/implement DP-Based Ridge Tracker

Implement a dynamic-programming ridge tracker that finds the MAP sequence of frequency indices through a spectrogram, penalizing large frequency jumps. Reference: MATLAB `tfridge` algorithm sketch, Oren 2024 methods.

**Context:**
- Naive argmax per column produces a ridge that jumps between harmonics, noise spikes, and the fundamental — broken for ~30% of harmonic-containing mouse USVs.
- Viterbi-style tracking: for column t and frequency bin f, optimal path reward = magnitude[f,t] − λ × |f − f_prev|. Solved via forward DP in O(F²T) or O(F·T·W) with window W.
- For silent columns (low cleaned magnitude), emit NaN — no valid ridge exists.
- Output shape: `(n_steps,)` FM trajectory + `(n_steps,)` AM trajectory, NaN where silent.

**Files to create:**

1. `src/usv_spectrogram/features/ridge_tracker.py` (NEW)

    ```python
    from dataclasses import dataclass
    import numpy as np

    @dataclass(frozen=True)
    class RidgeConfig:
        """DP ridge tracker parameters."""
        transition_penalty: float = 0.1       # λ: cost per bin of frequency jump
        max_jump_bins: int = 10               # search window W around previous frequency
        silence_threshold: float = 1e-6       # magnitude below this → NaN ridge

        def __post_init__(self) -> None:
            if self.transition_penalty < 0:
                raise ValueError("transition_penalty must be >= 0")
            if self.max_jump_bins < 1:
                raise ValueError("max_jump_bins must be >= 1")


    def track_ridge(
        magnitude: np.ndarray,    # cleaned, shape (n_freq_bins, n_time_cols)
        freqs_hz: np.ndarray,     # shape (n_freq_bins,)
        cfg: RidgeConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (fm_hz, am) both shape (n_time_cols,).

        Algorithm:
          1. Mask silent columns: `is_silent[t] = magnitude[:,t].max() < silence_threshold`
          2. On non-silent columns, run Viterbi forward pass:
             - reward[f, t] = magnitude[f, t]
             - transition_cost[f, g] = transition_penalty * |f - g|
             - path[f, t] = reward[f, t] + max_g(path[g, t-1] - transition_cost[f, g])
               (only over g in [f - max_jump_bins, f + max_jump_bins])
          3. Back-trace from argmax at last non-silent column
          4. FM(t) = freqs_hz[ridge_idx[t]] on non-silent columns, NaN elsewhere
          5. AM(t) = magnitude[ridge_idx[t], t] on non-silent columns, NaN elsewhere

        Returns (fm_hz, am) with NaN where silent.
        """
    ```

   Implementation notes:
   - Use windowed DP (O(F·T·W)) not full O(F²T) — W=10 is enough since mouse USVs have smooth pitch trajectories between jumps
   - Vectorize the inner transition step with numpy slicing, not nested loops
   - For back-trace, store argmax indices in a parallel array
   - Handle the all-silent spectrogram case: return arrays of all-NaN
   - Seed the first non-silent column with argmax reward (no prior path)

2. `tests/test_ridge_tracker.py` (NEW)

**Test plan:**
```
1. Pure tone at 60 kHz for 100 cols: FM returns ~60 kHz on every column
2. Linearly-sweeping tone 50→80 kHz: FM is monotonic, AM ~constant
3. Spectrogram with harmonic at 2×fundamental: tracker stays on fundamental (transition penalty prevents jump)
4. Silent column in middle of signal: that column's FM/AM = NaN, surrounding cols intact
5. All-silent spectrogram: returns all-NaN arrays
6. Discontinuous pitch jump > max_jump_bins: tracker follows, documents behavior in test docstring
7. RidgeConfig validation: transition_penalty < 0 raises
8. Output shapes match input n_time_cols
9. Regression: on a synthetic USV-like signal, reconstructed FM is within 1 kHz RMSE of ground truth
```

**Exit criteria:**
- [ ] On synthetic FM-sweep input, reconstructed FM has RMSE < 2 kHz
- [ ] Harmonic-suppression test passes (tracker on fundamental, not 2× harmonic)
- [ ] All tests pass
- [ ] py_compile passes

---

### 17.4 iMSA Pitch-Jump Classifier (Hertz 2020)

**What:** Implement Hertz's iMSA rule-based labeling — detect pitch jumps from the ridge, classify each syllable by pitch-jump presence + trajectory shape. Produces discrete labels per call.
**Status:** READY
**Review Tier:** 3
**Depends on:** 17.3

/implement iMSA Pitch-Jump Classifier

Implement a rule-based syllable classifier from Hertz et al. 2020 that detects step discontinuities in the pitch contour and classifies syllables into a small number of types (e.g., Up, Down, Flat, Complex) based on jump presence and slope. This is the algorithm that achieved the highest published SIS (0.22 bits, depth 1) on mouse USVs.

**Context:**
- iMSA (independent Mouse Syllable Analyzer) uses the ridge (pitch contour) from 17.3 to detect *pitch jumps* — abrupt frequency changes above a threshold (Hertz 2020 used ~10 kHz).
- Classification rules (operationalized from Hertz's description):
  - `Flat`: slope magnitude < 5 kHz per 10 ms, no jumps
  - `Up`: monotonic positive slope > 5 kHz/10ms, no jumps
  - `Down`: monotonic negative slope > 5 kHz/10ms, no jumps
  - `U-shape`: sign of slope changes twice (down then up), no jumps
  - `Inverted-U`: sign of slope changes twice (up then down), no jumps
  - `Complex`: any call with ≥1 pitch jump
- Published rule count is 6-8 classes depending on the variant. Aim for 6: `Flat, Up, Down, U, InvertedU, Complex`.
- The highest SIS contributor in Hertz was *self-repetition* — same label following itself. iMSA's skewed label distribution (most calls are Flat) is precisely why it scored high: consecutive Flats are predictable.

**Files to create:**

1. `src/usv_spectrogram/features/imsa_classifier.py` (NEW)

    ```python
    from dataclasses import dataclass
    from enum import Enum
    import numpy as np

    class IMSALabel(Enum):
        FLAT = "Flat"
        UP = "Up"
        DOWN = "Down"
        U_SHAPE = "U"
        INVERTED_U = "InvertedU"
        COMPLEX = "Complex"

    @dataclass(frozen=True)
    class IMSAConfig:
        """iMSA pitch-jump classifier parameters."""
        pitch_jump_threshold_hz: float = 10_000.0    # min inter-column delta to count as jump
        flat_slope_threshold_hz_per_s: float = 500_000.0  # |slope| below this = Flat
        min_valid_cols: int = 3                      # need ≥3 non-NaN cols to classify
        smooth_before_slope: bool = True             # mean filter FM before slope/jump detection

        def __post_init__(self) -> None:
            if self.pitch_jump_threshold_hz <= 0:
                raise ValueError("pitch_jump_threshold_hz must be > 0")


    def classify_imsa(
        fm_hz: np.ndarray,           # shape (n_steps,), NaN-allowed ridge from 17.3
        am: np.ndarray,              # shape (n_steps,), amplitude trajectory
        hop_s: float,                # time between FM samples (seconds)
        cfg: IMSAConfig,
    ) -> IMSALabel:
        """Classify a single USV call from its FM trajectory.

        Algorithm:
          1. Drop NaN values from fm_hz (silent cols excluded)
          2. If len(fm_valid) < cfg.min_valid_cols: return Flat (degenerate)
          3. Optionally smooth fm_valid with mean filter (window=5) per Oren spec
          4. Compute deltas: d = diff(fm_valid)
          5. Jumps: |d| > pitch_jump_threshold_hz at any index → Complex
          6. Else compute overall slope = (fm_valid[-1] - fm_valid[0]) / duration
             - |slope| < flat_slope_threshold: return Flat
             - All deltas same sign positive: Up
             - All deltas same sign negative: Down
             - One sign change in slope (through deltas): U or InvertedU based on direction
             - Else: Complex (multiple sign changes without a jump)

        Returns IMSALabel.
        """
    ```

   Implementation notes:
   - Use `np.diff` on the smoothed FM trajectory
   - For U/InvertedU: count `np.sign` changes in `diff` (ignoring zero-slope segments)
   - Consider hysteresis on zero-crossing to avoid spurious sign changes from noise

2. `scripts/run_imsa_labeling.py` (NEW) — Driver

    Follow Pattern 4 (Script CLI).
    Args: `--classified-csv`, `--wav-search-dirs` (list), `--output-dir`.

    Pipeline:
    1. Load call list from `classified_detections_full.csv`
    2. For each call: load WAV segment, compute STFT, apply `prefilter_spectrogram` (17.2), call `track_ridge` (17.3), call `classify_imsa`
    3. Save per-call iMSA label alongside existing labels: `results/imsa/imsa_labels.csv` with columns [call_id, file, begin_time_s, imsa_label]
    4. Compute SIS on iMSA labels via `compute_sis_depth_1` (17.1) and append to `results/sis_baselines/baselines.csv` (or write parallel `imsa_sis.csv`)
    5. Print label distribution and SIS value

3. `tests/test_imsa_classifier.py` (NEW)

**Test plan:**
```
1. Pure flat tone (constant FM): classify_imsa → Flat
2. Monotonic rising FM: Up
3. Monotonic falling FM: Down
4. V-shaped FM (down then up): U_SHAPE
5. Inverted-V FM (up then down): INVERTED_U
6. FM with one >10 kHz jump: Complex
7. All-NaN FM: Flat (degenerate handling)
8. 2-column FM (too short): Flat (below min_valid_cols)
9. Realistic noisy FM with small oscillations (<5 kHz/10ms): Flat (robust to noise)
10. IMSAConfig validation: pitch_jump_threshold_hz <= 0 raises
11. End-to-end on synthetic WAV with 3 Flat + 3 Up + 2 Down syllables: label distribution matches
```

**Exit criteria:**
- [ ] iMSA labels produced for all 7,518 calls
- [ ] `results/imsa/imsa_labels.csv` exists
- [ ] iMSA SIS value computed and recorded (comparison target: Hertz's 0.22 bits)
- [ ] Label distribution is skewed (Flat should dominate, matching Hertz)
- [ ] All tests pass
- [ ] py_compile passes

---

### 17.5 Oren 80D Vectorization

**What:** Produce a per-call feature vector by concatenating time-resampled FM (40D) + AM (40D) trajectories from the ridge, with configurable step count. No clustering — this module only produces feature vectors.
**Status:** READY
**Review Tier:** 2
**Depends on:** 17.3

/implement Oren 80D Vectorization

Adapt Oren et al. 2024's 80D call vectorization to mouse USVs: extract ridge (17.3), resample to fixed time-step count, concatenate FM + AM trajectories, append duration as an explicit scalar feature.

**Context:**
- Oren 2024 used 40 time steps × 2 trajectories = 80 dims for 1-2 s marmoset calls.
- Mouse USVs are 10-100 ms — at our 0.427 ms hop, native STFT columns range 23-234 per call. Resampling to 40 involves upsampling short calls and downsampling long ones.
- Oren normalized AM and FM per-caller to [0,1]. For mouse USVs, absolute FM matters (a 50 kHz call vs a 90 kHz call are different types). Keep **both** raw and normalized versions as output; downstream modules pick.
- Duration is the one piece of information that gets discarded by time-resampling. Append it as an explicit 81st scalar feature so clustering can use it.
- Sweep `n_steps ∈ {20, 30, 40, 60}` as a hyperparameter in 17.7 — the module accepts it as a config value.

**Files to create:**

1. `src/usv_spectrogram/features/omer_vectorize.py` (NEW)

    ```python
    from dataclasses import dataclass
    import numpy as np
    from scipy.interpolate import interp1d

    @dataclass(frozen=True)
    class OmerVectorConfig:
        """Oren-style 80D vectorization parameters."""
        n_steps: int = 40
        am_smooth_window: int = 6       # median filter size (Oren)
        fm_smooth_window: int = 5       # mean filter size (Oren)
        nan_fill_strategy: str = "interpolate"  # 'interpolate' | 'zero'


    def vectorize_call(
        fm_hz: np.ndarray,      # shape (n_native_cols,) with NaN for silent
        am: np.ndarray,          # shape (n_native_cols,)
        duration_s: float,
        cfg: OmerVectorConfig,
    ) -> np.ndarray:
        """Return 1D feature vector of shape (2 * n_steps + 1,).

        Steps:
          1. Handle NaN per cfg.nan_fill_strategy (interpolate across gaps or zero-fill)
          2. Smooth AM (median, window=am_smooth_window), FM (mean, window=fm_smooth_window)
          3. Resample both to cfg.n_steps via 1D linear interpolation
          4. Concatenate: [am_resampled (n_steps), fm_resampled (n_steps), duration_s (1)]

        Returns shape (2 * n_steps + 1,).
        """


    def normalize_vectors(
        vectors: np.ndarray,   # shape (n_calls, 2*n_steps + 1)
        n_steps: int,
        mode: str = "raw",      # 'raw' | 'minmax' | 'zscore'
    ) -> np.ndarray:
        """Normalize per-feature across the dataset.

        mode='raw': return vectors unchanged
        mode='minmax': rescale each of the 2*n_steps+1 feature dimensions to [0,1]
        mode='zscore': StandardScaler per feature dim
        """
    ```

2. `scripts/run_omer_vectorize.py` (NEW) — Driver

    Follow Pattern 4 (Script CLI).
    Args: `--classified-csv`, `--wav-search-dirs`, `--n-steps` (default 40), `--output-dir`.

    Pipeline:
    1. For each call: load WAV, STFT, prefilter (17.2), ridge (17.3), vectorize_call
    2. Stack all vectors into `np.ndarray` shape (7518, 2*n_steps + 1)
    3. Write `results/omer_vectorize/vectors_raw_{n_steps}steps.npy` + companion `call_ids.csv`

3. `tests/test_omer_vectorize.py` (NEW)

**Test plan:**
```
1. vectorize_call on synthetic FM sweep: output shape is (2 * n_steps + 1,)
2. Duration is the last element of the vector
3. n_steps=40 → shape (81,); n_steps=20 → shape (41,); n_steps=60 → shape (121,)
4. NaN-interpolate strategy: NaN in middle of ridge gets interpolated, no NaN in output
5. NaN-zero strategy: NaN bins become 0 in output
6. All-NaN ridge: returns zero vector (no crash)
7. FM smoothing window 5 preserves trajectory shape within small tolerance
8. normalize_vectors minmax: each feature dim has min 0, max 1 across the dataset
9. normalize_vectors zscore: each feature dim has mean ~0, std ~1
10. OmerVectorConfig with unknown nan_fill_strategy raises
```

**Exit criteria:**
- [ ] Vectors produced for all 7,518 calls at n_steps=40
- [ ] `results/omer_vectorize/vectors_raw_40steps.npy` exists with shape (7518, 81)
- [ ] Companion `call_ids.csv` has 7,518 rows matching vector row order
- [ ] All tests pass
- [ ] py_compile passes

---

### 17.6 AMVOC Autoencoder Features

**What:** Train AMVOC-style convolutional autoencoder from scratch on our 7,518 filtered USV spectrograms, extract the 1,280D bottleneck, apply variance thresholding + PCA (explaining 95% variance) to get a cluster-ready feature matrix.
**Status:** READY
**Review Tier:** 3
**Depends on:** 17.2

/implement AMVOC Autoencoder Features

Train a convolutional autoencoder on our domain-matched USV spectrograms (wild mice, not AMVOC's lab strains), extract bottleneck features, and reduce via variance threshold + PCA. Purpose: capture the *axes of variation the model needs to preserve* for reconstruction — a learned analog to handcrafted ridge features.

**Context:**
- AMVOC architecture (Stoumpou 2022): 3 conv layers + 3 maxpool → bottleneck 8×8×20 = 1,280D → 3 upsample layers → reconstruction. Input shape 1×64×160 (64 time × 160 freq).
- AMVOC trains for only 2 epochs deliberately — they argue longer training hurts clustering utility.
- Training on our data (not transferring from AMVOC's pretrained) because our wild mice differ from their C57BL/6J / B6D2F1/J strains.
- Post-bottleneck pipeline (per AMVOC): (1) keep only high-variance dims, (2) StandardScaler, (3) PCA keeping components that explain 95% variance.
- Output: cluster-ready feature matrix with the number of PCA components determined by the 95% variance criterion (typically ~20-50 dims).
- User-provided rationale: "autoencoders are one of the best forms of analysis — the bottleneck + PCA reveals concepts."

**Files to create:**

1. `src/usv_spectrogram/features/amvoc_autoencoder.py` (NEW)

    ```python
    from dataclasses import dataclass
    import torch
    import torch.nn as nn

    @dataclass(frozen=True)
    class AMVOCConfig:
        """AMVOC autoencoder training parameters."""
        input_time: int = 64            # input time frames
        input_freq: int = 160           # input frequency bins
        bottleneck_filters: int = 20    # AMVOC's choice (gives 8×8×20 = 1280)
        epochs: int = 2                 # deliberate under-training per AMVOC
        batch_size: int = 32
        lr: float = 1e-3
        device: str = "cuda"            # or 'cpu'


    class AMVOCAutoencoder(nn.Module):
        """3-layer conv encoder + 3-layer conv decoder.

        Encoder: conv(64,3x3) → pool → conv(32,3x3) → pool → conv(20,3x3) → pool
                 → bottleneck 8×8×20 = 1280D
        Decoder: mirrored with ConvTranspose2d (or upsample + conv)
        """

        def __init__(self, cfg: AMVOCConfig) -> None: ...

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Return bottleneck activations shape (batch, 1280)."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return reconstruction shape (batch, 1, 64, 160)."""


    def train_amvoc(
        spectrograms: torch.Tensor,     # shape (N, 1, 64, 160)
        cfg: AMVOCConfig,
    ) -> AMVOCAutoencoder:
        """Train on pre-filtered, standardized spectrograms. Returns trained model."""


    def extract_bottleneck(
        model: AMVOCAutoencoder,
        spectrograms: torch.Tensor,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return bottleneck activations shape (N, 1280)."""


    def reduce_features(
        bottleneck: np.ndarray,         # shape (N, 1280)
        variance_threshold: float = 1e-4,
        pca_variance: float = 0.95,
    ) -> tuple[np.ndarray, dict]:
        """Apply variance threshold → StandardScaler → PCA.

        Returns:
          reduced: shape (N, k) where k is # components explaining pca_variance
          info: dict with kept_dims_after_variance, pca_n_components, explained_variance_ratio_
        """
    ```

2. `scripts/run_amvoc_features.py` (NEW) — Driver

    Follow Pattern 4 (Script CLI).
    Args: `--classified-csv`, `--wav-search-dirs`, `--output-dir`, `--device`, `--epochs`.

    Pipeline:
    1. Load each call's WAV segment, compute STFT, apply `prefilter_spectrogram` (17.2)
    2. Resize/pad each spectrogram to 64×160 (time-center + frequency-crop or interpolate)
    3. Normalize to [0,1] per-spectrogram
    4. Stack into tensor (N, 1, 64, 160)
    5. Call `train_amvoc` → trained model
    6. Call `extract_bottleneck` → (N, 1280) array
    7. Call `reduce_features` → (N, k) reduced array + info dict
    8. Save: `results/amvoc/bottleneck.npy`, `results/amvoc/reduced_features.npy`, `results/amvoc/info.json`, `results/amvoc/model.pt`

3. `tests/test_amvoc_autoencoder.py` (NEW)

**Test plan:**
```
1. AMVOCAutoencoder forward pass: input (2,1,64,160) → output (2,1,64,160)
2. encode() returns shape (batch, 1280)
3. train_amvoc on 10 synthetic spectrograms for 1 epoch runs without crashing
4. extract_bottleneck on 5 inputs returns shape (5, 1280)
5. reduce_features on synthetic 1280D noise: output dims < 1280 (variance threshold drops some)
6. reduce_features PCA 95% variance → info dict has correct explained_variance_ratio_ summing to >= 0.95
7. reduce_features handles all-zero bottleneck without crash
8. AMVOCConfig validation: epochs < 1 raises
9. Training loss decreases over epochs on synthetic data (sanity check: model is learning)
10. CPU-only path works (device='cpu')
```

**Exit criteria:**
- [ ] Trained model saved to `results/amvoc/model.pt`
- [ ] Bottleneck extracted for all 7,518 calls, shape (7518, 1280)
- [ ] Reduced features saved, shape (7518, k) with k typically 20-50
- [ ] `info.json` records kept dims + PCA variance explained
- [ ] All tests pass
- [ ] py_compile passes

---

### 17.7 K-means Clustering Sweep

**What:** K-means clustering on both Oren (17.5) and AMVOC (17.6) feature sets across k ∈ [5, 7, 10, 15, 20, 27]. Produces per-call cluster labels at every (feature_set, k) combination.
**Status:** READY
**Review Tier:** 2
**Depends on:** 17.5 + 17.6

/implement K-means Clustering Sweep

Run k-means on the two feature matrices from 17.5 and 17.6 across a k-sweep, producing labels for every (feature_source, k) combination. Feeds 17.8 (SIM) and 17.9 (benchmark).

**Context:**
- K values [5, 7, 10, 15, 20, 27] span: Scattoni-matched (7), mid-range exploration (10, 15, 20), DeepSqueak-matched (27), and a lower-granularity anchor (5).
- Two feature sources: Oren (n_steps=40 default, 81D) and AMVOC (PCA-reduced from 1280D).
- K-means initialization: k-means++, n_init=10 for stability. Random state fixed for reproducibility.
- Store silhouette scores for each (source, k) combo for interpretation — not the primary metric, but useful signal.

**Files to create:**

1. `src/usv_spectrogram/classification/cluster_sweep.py` (NEW)

    ```python
    from dataclasses import dataclass
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    @dataclass(frozen=True)
    class ClusterSweepConfig:
        k_values: tuple[int, ...] = (5, 7, 10, 15, 20, 27)
        random_state: int = 42
        n_init: int = 10
        silhouette_sample_size: int = 2000   # subsample for speed on 7518 calls


    def run_sweep(
        features: np.ndarray,    # shape (N, D)
        cfg: ClusterSweepConfig,
    ) -> dict[int, dict]:
        """Return {k: {'labels': np.ndarray, 'inertia': float, 'silhouette': float, 'centers': np.ndarray}}."""
    ```

2. `scripts/run_cluster_sweep.py` (NEW) — Driver

    Follow Pattern 4 (Script CLI).
    Args: `--oren-vectors`, `--amvoc-vectors`, `--output-dir`.

    Pipeline:
    1. Load both feature matrices
    2. Run `run_sweep` on each
    3. Save `results/cluster_sweep/oren_labels_k{K}.csv` and `amvoc_labels_k{K}.csv` for each k
    4. Save combined summary `cluster_sweep_summary.csv` (columns: source, k, inertia, silhouette)

3. `tests/test_cluster_sweep.py` (NEW)

**Test plan:**
```
1. run_sweep on synthetic blob data (3 well-separated clusters): k=3 has highest silhouette
2. Output dict has one entry per k in cfg.k_values
3. Each entry has labels of length N, centers of shape (k, D)
4. Inertia is monotonically non-increasing as k increases (within random_state tolerance)
5. Silhouette computed via subsample for speed, matches full silhouette within 0.05
6. Empty feature matrix raises ValueError
7. N < max(k_values) gracefully skipped (warn, don't crash)
8. Reproducibility: same random_state → same labels
```

**Exit criteria:**
- [ ] 12 label CSVs generated (2 sources × 6 k values)
- [ ] `cluster_sweep_summary.csv` has 12 rows
- [ ] All tests pass
- [ ] py_compile passes

---

### 17.8 SIM Optimization (Hertz 2020)

**What:** Iteratively reassign cluster labels to maximize SIS at depth 1, starting from each labeling produced so far. Reports best final SIS per starting point.
**Status:** READY
**Review Tier:** 2
**Depends on:** 17.1

/implement SIM Optimization

Implement Syntax Information Maximization: given an initial clustering, propose point reassignments; accept moves that increase SIS at depth 1. Iterate until no improvement.

**Context:**
- Hertz 2020 SIM: starting from an initial labeling (iMUPET or iMSA), iteratively perturb labels to maximize SIS. Published result: SIM on iMUPET start matched or exceeded iMSA's standalone score.
- Our depth-1 simplification: `SIS = MI(X_n; X_{n-1})` — can be computed from the K×K transition count matrix in O(K²) after each move.
- Naive algorithm: for each call i, for each candidate label c ≠ current[i], propose move, accept if SIS increases. One pass = N × K evaluations.
- Optimization: maintain transition count matrix; a single label change at position i modifies at most 4 counts (the transitions into and out of position i, for both old and new label). So each proposed move is O(1).
- Multiple starting points: run SIM independently from each labeling produced by 17.1 (Scattoni-7, DeepSqueak-27, HDBSCAN-3), 17.4 (iMSA), 17.7 (Oren-kmeans, AMVOC-kmeans) — all "source labelings" get optimized, results compared.
- User's explicit decision: run option (a) — SIM on every starting labeling, not just the best one.

**Files to create:**

1. `src/usv_spectrogram/classification/sim_optimizer.py` (NEW)

    ```python
    from dataclasses import dataclass
    import numpy as np

    @dataclass(frozen=True)
    class SIMConfig:
        max_iterations: int = 50             # passes through the dataset
        min_sis_improvement: float = 1e-4    # stop if one pass improves by less
        random_order: bool = True             # shuffle call visit order per pass
        random_state: int = 42


    @dataclass(frozen=True)
    class SIMResult:
        initial_labels: np.ndarray           # starting labeling
        optimized_labels: np.ndarray
        initial_sis: float                   # bits
        final_sis: float                     # bits
        iterations_used: int
        sis_history: list[float]             # SIS per iteration (for convergence plot)


    def optimize_sis(
        initial_labels: np.ndarray,    # shape (N,), integer labels
        cfg: SIMConfig,
    ) -> SIMResult:
        """Iteratively reassign labels to maximize MI(X_n; X_{n-1}).

        Algorithm:
          1. Build K×K transition count matrix from initial_labels
          2. Compute initial SIS from counts
          3. For iteration in range(max_iterations):
             a. For each call index i (optionally in random order):
                current = labels[i]
                best_label = current
                best_gain = 0.0
                for candidate in unique_labels:
                    if candidate == current: continue
                    # Compute SIS delta from swapping labels[i] from current → candidate
                    # (affects at most 2 outgoing + 2 incoming transitions)
                    delta = compute_delta(counts, labels, i, current, candidate)
                    if delta > best_gain:
                        best_gain = delta
                        best_label = candidate
                if best_label != current:
                    apply swap (update counts, update labels[i])
             b. If total improvement this pass < min_sis_improvement: stop

        Returns SIMResult with history.
        """
    ```

    Implementation notes:
    - `compute_delta` is the critical hot loop — O(1) after precomputing row/col sums of the transition matrix.
    - Keep a running K×K count matrix; update in place on accepted moves.
    - At each proposal, compute SIS from current counts (entropy of joint, marginals, difference).
    - Use `np.log2` consistently.

2. `scripts/run_sim_optimization.py` (NEW) — Driver

    Args: `--labelings-dir` (points to dir with all candidate starting labelings), `--output-dir`.
    Pipeline:
    1. Discover all labeling CSVs in inputs (17.1 baselines + 17.4 iMSA + 17.7 cluster_sweep outputs)
    2. For each starting labeling: call `optimize_sis`
    3. Save per-start results: `results/sim/{source}_optimized.csv` + `results/sim/{source}_history.png`
    4. Aggregate summary: `results/sim/sim_summary.csv` (columns: source, initial_sis, final_sis, improvement, iterations)

3. `tests/test_sim_optimizer.py` (NEW)

**Test plan:**
```
1. optimize_sis on i.i.d. random labels: final_sis > initial_sis (some improvement from noise)
2. optimize_sis on a highly structured sequence ABABAB...: already optimal, final_sis ≈ initial_sis
3. compute_delta matches naive full-recomputation for 10 random test cases (correctness of incremental update)
4. Running optimize_sis twice with same random_state produces identical output
5. max_iterations=0: returns initial labels unchanged, iterations_used=0
6. K=2 labels: handles binary case
7. K > N: handles degenerate case without crash
8. Empty initial_labels returns empty SIMResult without crash
9. SIS history is monotonically non-decreasing across iterations
```

**Exit criteria:**
- [ ] SIM run on ≥5 starting labelings (Scattoni-7, DeepSqueak-27, HDBSCAN-3, iMSA, Oren-kmeans-best, AMVOC-kmeans-best)
- [ ] `sim_summary.csv` has per-start initial + final SIS
- [ ] Final SIS values recorded for 17.9 comparison
- [ ] All tests pass
- [ ] py_compile passes

---

### 17.9 SIS Benchmark + Comparison Report

**What:** Aggregate all labelings produced across 17.1, 17.4, 17.7, 17.8 into a single comparison table and report. Produce the decision artifact: which method wins, by how much, and which hypothesis does that support?
**Status:** BLOCKED (on 17.1, 17.4, 17.7, optionally 17.8)
**Review Tier:** 2
**Depends on:** 17.1, 17.4, 17.7, (17.8)

/implement SIS Benchmark + Comparison Report

Produce the final comparison table and accompanying plots that show which labeling scheme achieves the highest SIS at depth 1 on our 5970 dataset.

**Context:**
- Every prior module in Phase 17 has recorded labels + SIS values in `results/`. This module unifies them.
- Output is the artifact the user will look at to decide which method is worth advancing to Phase B (3452 / 9252 datasets).
- Comparison axes: method family (existing vs iMSA vs Oren-kmeans vs AMVOC-kmeans vs SIM-variants) × K (label count) × normalization.
- Secondary analyses: label distribution (is the winner skewed like iMSA, uniform like HDBSCAN, or in between?), confusion matrices between top methods (do they agree on which calls are "hard"?).

**Files to create:**

1. `scripts/run_sis_benchmark.py` (NEW) — Driver

    Follow Pattern 4 (Script CLI).
    Args: `--results-root` (defaults to `results/`), `--output-dir` (defaults to `results/sis_benchmark/`).

    Pipeline:
    1. Load all labeling CSVs from: `sis_baselines/`, `imsa/`, `cluster_sweep/`, `sim/`
    2. For any that haven't been SIS-computed yet, compute via `compute_sis_depth_1` (17.1)
    3. Aggregate into master table: `results/sis_benchmark/benchmark.csv` with columns [name, family, K, mi_at_lag_1, marginal_entropy, conditional_entropy, entropy_reduction_pct]
    4. Produce plots:
       - `benchmark_bar.png`: bar chart of MI at lag 1, sorted descending, with horizontal reference lines at Hertz's 0.10/0.13/0.22/0.23 values
       - `benchmark_by_k.png`: MI vs K for cluster_sweep methods (lines per feature source)
       - `label_distribution_grid.png`: histograms of label frequencies for the top 5 methods
       - `confusion_top3.png`: confusion/agreement matrix between the top 3 labelings (Cramér's V or Adjusted Rand Index)
    5. Produce `results/sis_benchmark/report.md` with:
       - Winner + runner-up, MI values, interpretation in one paragraph
       - What hypothesis each supports (rule-based / ridge features / autoencoder / direct SIS optimization)
       - Caveats: scale mismatch vs Hertz's 346K dataset, wild vs lab mouse strain differences, depth-1 limitation

2. `tests/test_sis_benchmark.py` (NEW)

**Test plan:**
```
1. Aggregator assembles a master table from synthetic per-method CSVs with correct row count
2. Benchmark bar chart includes Hertz reference lines at 0.10, 0.13, 0.22, 0.23
3. benchmark_by_k.png produced only for methods with multiple K values
4. Confusion matrix only computed for methods with matching call_ids
5. Report.md includes the winner name and MI value
6. Script runs end-to-end on synthetic results_root with 2 methods and produces all outputs
7. Missing per-method result → warn and skip, do not crash
```

**Exit criteria:**
- [ ] `results/sis_benchmark/benchmark.csv` has rows for all methods
- [ ] `benchmark_bar.png`, `benchmark_by_k.png`, `label_distribution_grid.png`, `confusion_top3.png` all exist
- [ ] `report.md` identifies the winning method with rationale
- [ ] All tests pass
- [ ] py_compile passes

---

## Phase 17 Gate

**Purpose:** Decide which labeling method to promote to Phase B (cross-dyad comparison on 3452 / 9252).

- [ ] All 9 modules complete
- [ ] `results/sis_benchmark/benchmark.csv` exists
- [ ] `results/sis_benchmark/report.md` identifies winner
- [ ] Winner's MI is recorded in `ops/goals.md` as the new reference SIS value
- [ ] Decision note written to `notes/` explaining why the winner won and what that implies for the USV-as-language hypothesis
- [ ] Losing methods archived with brief "why not" rationale

**Decision questions the gate answers:**
1. Which method wins: rule-based (iMSA), ridge features (Oren), learned features (AMVOC), or direct optimization (SIM)?
2. Is the winning MI value meaningfully above our 0.093 baseline, or is sequential structure weak regardless of labeling?
3. Does the winner's label distribution look like Hertz's (skewed, Flat-dominated) or something different?
4. If SIM variants dominate, the finding is "labels matter more than features" — pursue SIM on Phase B.
5. If AMVOC dominates, the finding is "learned representations beat handcrafted ones" — pursue autoencoder elsewhere (e.g., VAE for bout-level embeddings).
6. If iMSA dominates, the finding is "pitch-jump structure carries the sequential signal" — simple rules suffice.
7. If Oren dominates, the finding is "continuous FM+AM shape captures the informative axes" — handcrafted ridge features win.

---

## Directory Plan

New directories this phase creates:
```
src/usv_spectrogram/features/           (NEW — houses 17.2, 17.3, 17.5, 17.6)
results/sis_baselines/                  (17.1 outputs)
results/imsa/                           (17.4 outputs)
results/omer_vectorize/                 (17.5 outputs)
results/amvoc/                          (17.6 outputs)
results/cluster_sweep/                  (17.7 outputs)
results/sim/                            (17.8 outputs)
results/sis_benchmark/                  (17.9 outputs)
```

## References

- `docs/handoffs/three-paper-deep-reads-2026-04-15.md` — paper ingestion summary
- `docs/architecture/patterns.md` — Pattern 1 (frozen dataclass), Pattern 4 (script CLI), Pattern 7 (STFT core), Pattern 8 (import bootstrap)
- `notes/Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable.md`
- `notes/iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy.md`
- `notes/Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space.md`
- `notes/AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering.md`
- `notes/AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression.md`
- `notes/ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory.md`
- `notes/raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs.md`

ADR references: ADR-001 (sample rate = 300000), ADR-002 (n_fft = 512, hop_length = 128).
