# USV Analysis Pipeline — ROADMAP

> Extracts biological meaning from classified USV data. Two dyads of wild mice
> (same species): 5970 (usv_lmt_034, 7,518 calls) and 3452 (usv_lmt_035, ~349 events).
> Builds on detection pipeline + two classification schemes (traditional taxonomy + UMAP/HDBSCAN).
> The original DeepSqueak k-means 27-cluster solution is **deprecated** — it imposed structure the
> data doesn't support (UMAP/HDBSCAN showed one continuous manifold). Use traditional taxonomy for
> literature comparability and UMAP embedding for structure analysis.
> Prose plan: `docs/analysis-roadmap.md`.

**Important:** Both datasets record USVs from a *dyad* (pair of interacting wild mice).
Individual calls cannot be attributed to a specific mouse without acoustic localization.
Cross-dataset comparison is cross-dyad, not cross-individual.

---

## Phase 16: USV Repertoire Analysis

### 16.1 Temporal Dynamics (A1)

**What:** Analyze call rate, type composition, and bout structure over time in the 5970 dataset.
**Status:** DONE
**Review Tier:** 2
**Depends on:** None

**Script:** `scripts/analyze_temporal_dynamics.py`
**Results:** `results/temporal_dynamics/`
- `call_rate_hourly.png` — call counts binned by hour
- `type_composition_hourly.png` — stacked syllable type proportions over time
- `bout_structure.png` — bout length distributions, first-call analysis
- `ici_distribution.png` — inter-call interval histogram
- `call_raster.png` — spike-raster-style temporal plot
- `temporal_summary.csv` — key metrics (1,579 bouts, median ICI 0.19s, 32.4h span)

---

### 16.2 Sequential Structure (A2)

**What:** Analyze transition probabilities, entropy rates, idioms, and sequential memory in USV sequences.
**Status:** DONE
**Review Tier:** 3
**Depends on:** None

**Script:** `scripts/analyze_sequential_structure.py`
**Results:** `results/sequential_structure/`
- `transition_matrix.png` — P(type_B | type_A) heatmap
- `entropy_convergence.png` — entropy rate across n-gram orders (1–5)
- `mutual_information_lag.png` — MI at lag 1–10
- `zipf_distribution.png` — rank-frequency plot
- `idiom_report.csv` — 1,843 significant recurring n-grams
- `sequential_structure_summary.csv` — marginal entropy 2.54 bits, conditional 2.45, 3.7% reduction

**Key findings:** Weak but significant sequential structure. Entropy reduction ~3.7%. Memory depth ~10 calls. Self-transitions elevated (mean 0.26 vs chance 0.14). Complex→Complex is the strongest idiom.

---

### 16.3 Acoustic Feature Deep-Dive (A3)

**What:** Characterize the continuous acoustic space — correlations, principal components, and UMAP overlays — to understand the shape of the USV manifold beyond categorical labels.
**Status:** READY
**Review Tier:** 2
**Depends on:** None

/implement Acoustic Feature Deep-Dive

Create an analysis script that explores the 10-dimensional acoustic feature space of 5970 USV calls, bridging the discrete traditional taxonomy and the continuous UMAP manifold.

**Context:** UMAP+HDBSCAN showed USVs form one continuous manifold (96.6% in one cluster), while traditional taxonomy assigns 7 discrete types. This module answers: *which acoustic dimensions create the perceived type boundaries?* Are the traditional types cutting the space at density ridges, or at arbitrary thresholds?

**Input data:**
- `results/traditional_taxonomy/classified_traditional.csv` — 7,864 rows with 10 acoustic features + `syllable_type` + `classification_confidence`
- `results/recluster_umap_hdbscan/reclassified_detections.csv` — same rows with `umap_x`, `umap_y`, `hdbscan_label`
- Acoustic features: `call_length_s`, `principal_freq_hz`, `low_freq_hz`, `high_freq_hz`, `bandwidth_hz`, `freq_std_dev_hz`, `slope`, `sinuosity`, `mean_power_db`, `tonality`

**Files to create:**

1. `scripts/analyze_acoustic_features.py` (NEW) — Main analysis script

    ```python
    # Constants
    ACOUSTIC_FEATURES = [
        "call_length_s", "principal_freq_hz", "low_freq_hz", "high_freq_hz",
        "bandwidth_hz", "freq_std_dev_hz", "slope", "sinuosity",
        "mean_power_db", "tonality",
    ]
    SYLLABLE_TYPES = ["Flat", "Down", "Chevron", "Short", "Complex", "Frequency_Jump", "Up"]
    ```

    Follow the Script CLI Pattern (Pattern 4 in `docs/architecture/patterns.md`).
    Arguments: `--input-taxonomy CSV`, `--input-umap CSV`, `--output-dir DIR`.

    **Analyses to implement:**

    a. **Correlation matrix** — Pearson + Spearman on the 10 features. Output: `feature_correlation.png` (heatmap with both triangles: Pearson upper, Spearman lower).

    b. **PCA** — StandardScaler → PCA on 10 features. Output:
       - `pca_variance.png` — scree plot (explained variance per component, cumulative line)
       - `pca_loadings.png` — heatmap of component loadings (top 5 PCs × 10 features)
       - `pca_scatter.png` — PC1 vs PC2 colored by syllable_type (7 colors + legend)

    c. **UMAP feature overlays** — For each of the 10 acoustic features, create a scatter of (umap_x, umap_y) colored by that feature (continuous colormap, e.g. viridis). Output: `umap_overlay_<feature>.png` (10 files). Also create a 2×5 panel figure: `umap_overlay_panel.png`.

    d. **Within-type variability** — For each feature × type combination: violin plot showing distribution. Output: `within_type_variability.png` (10-panel figure, one per feature, each with 7 violins).

    e. **Boundary analysis** — Filter to `classification_confidence == "low"`. Plot these on the PCA and UMAP spaces, highlighting them against the full dataset. Output: `boundary_cases.png`. Also compute: which type transitions are most common among low-confidence calls (i.e., "if not type A, what would it be?").

    f. **Power & tonality deep-dive** (prose plan Section 7) — Dedicated analysis of these two features:
       - Power by type: box plot of `mean_power_db` grouped by `syllable_type`. Are some types louder?
       - Tonality distribution: histogram + UMAP colored by tonality. Which calls are most tonal vs noisy?
       - Tonality vs detection confidence: scatter `tonality` vs `det_prob_max`. Are tonal calls easier to detect?
       - Power over time: `mean_power_db` binned by file timestamp (hourly). Does intensity drift across sessions?
       Output: `power_tonality_analysis.png` (4-panel figure).

    g. **Frequency drift analysis** (Open Q4) — Plot `principal_freq_hz` over absolute time (parsed from filenames). Bin by hour and session. Linear regression to detect systematic drift. Output: `frequency_drift.png`. This checks for equipment drift, temperature effects, or fatigue.

    h. **Flat call temporal distribution** (Open Q7) — Is the dominant type (Flat, 32%) uniformly distributed in time, or does it cluster? Compare Flat call rate vs non-Flat call rate across time bins. Output: included in `power_tonality_analysis.png` or separate `flat_call_temporal.png`.

    i. **Summary CSV** — Per-type: mean, std, median for each feature. Plus: PCA variance explained (first 5 PCs), top 3 loadings per PC. Frequency drift slope + p-value. Power/tonality by-type statistics.

    Output directory: `results/acoustic_features/`

**Test plan:**
```
1. Correlation matrix is symmetric and has correct shape (10×10)
2. PCA components sum to 1.0 (explained variance)
3. PCA loadings have correct shape (n_components × 10)
4. UMAP overlay function handles NaN features gracefully (skip row, warn)
5. Violin plot handles types with < 5 calls without crashing
6. Summary CSV has 7 rows (one per type) and expected columns
7. Low-confidence filter produces non-empty subset
8. Power/tonality panel produces 4 subplots
9. Frequency drift regression returns slope and p-value
10. Flat call temporal analysis handles bins with zero Flat calls
11. Script runs end-to-end on synthetic data and produces all expected output files
```

**Exit criteria:**
- [ ] All analysis outputs generated (correlation, PCA×3, UMAP panel, violins, boundary, power/tonality, freq drift, summary)
- [ ] Summary CSV written with per-type statistics + drift + power/tonality stats
- [ ] Script runs end-to-end: `python scripts/analyze_acoustic_features.py --input-taxonomy results/traditional_taxonomy/classified_traditional.csv --input-umap results/recluster_umap_hdbscan/reclassified_detections.csv --output-dir results/acoustic_features/`
- [ ] py_compile passes
- [ ] All tests pass

---

### 16.4 Detection Confidence Analysis (A4)

**What:** Assess CNN detection reliability across call types and identify systematic quality patterns.
**Status:** READY
**Review Tier:** 2
**Depends on:** None

/implement Detection Confidence Analysis

Create an analysis script examining how CNN detection confidence relates to call types, match quality, and acoustic features in the 5970 dataset.

**Context:** The CNN model (`hard_neg_retrain/best_model.pt`) scores each detection window with a probability. Some call types may be systematically harder to detect (e.g., Short calls at ~4ms near the detection floor). Understanding detection bias is critical before cross-dyad comparison — if one dyad produces more Short calls, apparent repertoire differences could be artifacts of detection sensitivity.

**Input data:**
- `results/traditional_taxonomy/classified_traditional.csv` — columns include `det_prob_max`, `det_prob_mean`, `det_user_action`, `accepted`, `syllable_type`, `classification_confidence`
- `results/recluster_umap_hdbscan/reclassified_detections.csv` — includes `hdbscan_label` (37 noise points to investigate)

**Files to create:**

1. `scripts/analyze_detection_confidence.py` (NEW) — Analysis script

    Arguments: `--input-taxonomy CSV`, `--input-umap CSV`, `--output-dir DIR`.

    **Analyses to implement:**

    a. **Confidence by type** — Box/violin plot of `det_prob_max` grouped by `syllable_type`. Output: `confidence_by_type.png`. Compute: Kruskal-Wallis test for overall difference, pairwise Mann-Whitney with Bonferroni correction.

    b. **Match quality cross-tab** — If `match_quality` column exists [ASSUMED: may be absent in some CSVs], cross-tabulate with `syllable_type`. Output: `match_quality_by_type.png` (stacked bar). If column absent, skip with warning.

    c. **HDBSCAN noise inspection** — Filter `hdbscan_label == -1` (37 noise points). Characterize: feature distributions vs main cluster, `det_prob_max` distribution, syllable type breakdown. Output: `noise_point_analysis.png` (multi-panel: feature box plots + type pie chart).

    d. **Confidence vs acoustic features** — Scatter plots of `det_prob_max` vs each of: `call_length_s`, `mean_power_db`, `tonality`, `sinuosity`. Color by type. Output: `confidence_vs_features.png` (2×2 panel).

    e. **Calibration check** — Among reviewed calls (`det_user_action` is not null/NaN): compare accepted vs rejected by confidence bin. Output: `confidence_calibration.png`. Compute: precision at different confidence thresholds.

    f. **Summary CSV** — Per-type: mean/median `det_prob_max`, n_reviewed, accept_rate (where reviewed), n_noise_points (hdbscan -1).

    Output directory: `results/detection_confidence/`

**Test plan:**
```
1. Confidence-by-type plot handles types with zero reviewed calls
2. Match quality analysis gracefully handles missing column
3. Noise point filter produces correct count (37 per current data)
4. Kruskal-Wallis test returns valid p-value
5. Calibration analysis handles case where no calls have user_action
6. Summary CSV has expected shape and no NaN in count columns
7. Script runs end-to-end on synthetic data
```

**Exit criteria:**
- [ ] All applicable outputs generated (5-6 figures + summary CSV)
- [ ] Statistical tests computed with proper multiple-comparison correction
- [ ] Graceful handling of missing columns (match_quality, det_user_action)
- [ ] py_compile passes
- [ ] All tests pass

---

## Phase 16 Gate (A)

- [ ] 16.3: Acoustic feature analysis complete — correlations, PCA, UMAP overlays, power/tonality, frequency drift
- [ ] 16.4: Detection confidence analysis complete, systematic biases documented
- [ ] Key finding documented: which acoustic dimensions best separate traditional types
- [ ] Key finding documented: whether detection confidence varies systematically by type
- [ ] Key finding documented: whether frequency or power drift over time (equipment vs biology)
- [ ] All tests passing

---

### 16.5 3452 Feature Extraction & Classification (B1)

**What:** Extract acoustic features from the 3452 dyad's detected USVs (currently only CNN metadata) and apply both classification schemes for cross-dyad comparison.
**Status:** READY
**Review Tier:** 3
**Depends on:** None (but informed by 16.3/16.4 findings)

/implement 3452 Feature Extraction & Classification

Replicate the same DeepSqueak MATLAB pipeline used for 5970 to extract acoustic features from 3452 detections, then apply traditional taxonomy and UMAP projection.

**Context:** The 5970 acoustic features came from DeepSqueak's MATLAB `CalculateStats` function, NOT custom Python extraction. The full pipeline was:

```
CNN detection → Raven export → DeepSqueak .mat creation → MATLAB CalculateStats → Excel → Python merge
```

The 3452 dataset must follow the **same pipeline** for feature comparability. Writing a Python approximation of CalculateStats would introduce systematic differences that confound cross-dyad comparison.

**3452 batch detection state:**
- `results/batch_3452_sample/` — 5,409 WAVs, 93 with events, **349 total events**
- `results/batch_3452_reviewed/` — 855 WAVs (curated subset), 17 with events, **56 events**
- Detection JSON format: list of `{start_time_s, end_time_s, duration_s, start_col, end_col, max_probability, mean_probability}`

**Important scale note:** 3452 has ~349 events vs 5970's 7,518. This 20:1 ratio is itself a finding (one dyad vocalizes far less, or in a less detectable range). The cross-dyad comparison (16.6) must account for this.

**WAV file location:** `USV_3452_sample/` contains the WAV files. File paths in summary.parquet use relative paths (e.g., `USV_3452_sample/USV_2/usv_lmt_035/...`).

**Files to create/modify:**

#### Step 1: Raven Export (Python — reuse existing tool)

1. `scripts/export_3452_raven_tables.py` (NEW) — Thin wrapper around existing Raven export

    Use `scripts/export_raven_tables.py` with `--batch-format` as reference. The existing `src/usv_spectrogram/classification/raven_export.py` module handles the conversion. This script:
    a. Reads detection JSONs from `results/batch_3452_sample/detections/` (only the 93 files with events)
    b. Converts each to a Raven selection table (`.Table.1.selections.txt`)
    c. Outputs to `results/batch_3452_classified/raven_tables/`

    The Raven format: tab-delimited TSV with columns `Selection, View, Channel, Begin Time (s), End Time (s), Low Freq (Hz), High Freq (Hz)`. Use 25000–110000 Hz as default freq band (matching detection config).

#### Step 2: DeepSqueak Processing (MATLAB — manual step)

2. **MATLAB scripts** — Adapt existing scripts for 3452:

    a. `scripts/create_deepsqueak_mats_3452.m` (NEW) — Adapted from `scripts/create_deepsqueak_mats.m`
       - Input: Raven tables from Step 1 + WAV files in `USV_3452_sample/`
       - Output: `.mat` files in DeepSqueak Detections folder
       - Key change: WAV search path points to `USV_3452_sample/` instead of `5970/`

    b. `scripts/deepsqueak_classify_3452.m` (NEW) — Adapted from `scripts/deepsqueak_batch_classify.m`
       - Input: .mat files from (a)
       - Output: Same .mat files with cluster labels
       - Use same settings as 5970: k-means, shape=3, freq=2, duration=1

    c. `scripts/deepsqueak_export_stats_3452.m` (NEW) — Adapted from `scripts/deepsqueak_export_stats.m`
       - Input: Classified .mat files
       - Output: `deepsqueak_output_3452/classified_Stats.xlsx`
       - Same `CalculateStats` thresholds: ENTROPY_THRESHOLD=0.215, AMPLITUDE_THRESHOLD=0.825

    **These MATLAB steps are manual** — they require a MATLAB environment with DeepSqueak v3.1 installed. Document the exact commands in a README within `results/batch_3452_classified/`.

#### Step 3: Import & Classify (Python)

3. `scripts/import_and_classify_3452.py` (NEW) — Merge + classify

    a. **DeepSqueak import** — Reuse `src/usv_spectrogram/classification/deepsqueak_import.py` to merge DeepSqueak Excel with batch detection JSONs. Same 75ms tolerance, greedy 1:1 matching. Output: `classified_detections_3452.csv`.

    b. **Traditional taxonomy** — Import `classify_call()` from `scripts/classify_traditional_taxonomy.py` (or inline the same logic). Apply to merged CSV. Output: `classified_3452_traditional.csv` (adds `syllable_type`, `classification_confidence`).

    c. **UMAP projection** — Load the fitted UMAP reducer from 5970 [ASSUMED: saved as pickle in `results/recluster_umap_hdbscan/` — if not, refit jointly on both datasets]. StandardScaler the 10 features using the **5970 scaler** (not refit on 3452) to maintain comparability. Project into 5970 UMAP space. Output: `classified_3452_umap.csv` (adds `umap_x`, `umap_y`). Also generate: `umap_3452_overlay.png`.

    d. **Type distribution comparison** — Bar chart comparing 5970 vs 3452 type proportions. Output: `type_comparison_5970_vs_3452.png`.

    Output directory: `results/batch_3452_classified/`

    **Note on frequency units:** DeepSqueak `CalculateStats` outputs frequency values in **kHz** (not Hz), despite column names containing `_hz`. The 5970 data has this same quirk — values like 18–91 in `principal_freq_hz` are actually kHz. Ensure consistent units when applying taxonomy thresholds.

**Test plan:**
```
1. Raven export produces valid TSV with correct column headers
2. Raven table count matches number of detection JSONs with events (93)
3. Each Raven table has correct number of selections matching JSON event count
4. DeepSqueak import merge matches expected tolerance (75ms)
5. classify_call() produces same output as original script for identical feature values
6. UMAP projection handles the case where fitted model isn't available (clear error + fallback to joint fit)
7. Import script handles missing WAV files gracefully (skips, reports count)
8. Output CSV has expected columns and no duplicate rows
9. Frequency unit consistency check: values in same range as 5970 data
```

**Exit criteria:**
- [ ] Raven tables exported for all 3452 events
- [ ] MATLAB pipeline documented with exact commands in README
- [ ] DeepSqueak Excel imported and merged with CNN detections
- [ ] Traditional taxonomy applied, type distribution reported
- [ ] UMAP projection works (or clear documentation of why not + alternative approach)
- [ ] Feature value ranges consistent with 5970 (same units, same order of magnitude)
- [ ] py_compile passes on all new Python files
- [ ] All tests pass

---

### 16.6 Cross-Dyad Comparison (B2)

**What:** Statistical comparison of vocal repertoires between the 5970 and 3452 dyads — type distributions, acoustic space overlap, and sequential structure differences.
**Status:** BLOCKED (on 16.5)
**Review Tier:** 3
**Depends on:** 16.5

/implement Cross-Dyad Comparison

Comprehensive statistical comparison of USV repertoires between two dyads of wild mice, using the classified data from both 5970 and 3452 datasets.

**Context:** Both datasets are dyads of the same species of wild mice. USVs cannot be attributed to individual mice without localization — the comparison is at the dyad level. Key challenge: sample size imbalance (~7,500 vs ~350 calls). All statistical tests must be robust to unequal n. The tools in `src/usv_spectrogram/classification/repertoire_stats.py` were designed for this: PERMANOVA, chi-squared, JSD, Shannon entropy.

**Input data:**
- 5970: `results/traditional_taxonomy/classified_traditional.csv` + `results/recluster_umap_hdbscan/reclassified_detections.csv`
- 3452: `results/batch_3452_classified/classified_3452_traditional.csv` + `results/batch_3452_classified/classified_3452_umap.csv`

**Files to create:**

1. `scripts/analyze_cross_dyad.py` (NEW) — Cross-dyad comparison script

    Arguments: `--dyad1-taxonomy CSV`, `--dyad1-umap CSV`, `--dyad2-taxonomy CSV`, `--dyad2-umap CSV`, `--output-dir DIR`.

    **Analyses to implement:**

    a. **Repertoire comparison** — Side-by-side type proportion bar charts (normalized). Chi-squared test (with Monte Carlo p-value for small expected counts). Jensen-Shannon Divergence. Shannon entropy comparison. Output: `repertoire_comparison.png`, stats in summary CSV.

    b. **Acoustic space overlap** — Overlay 3452 on 5970's UMAP space. Compute: convex hull overlap area, nearest-neighbor distance distributions (3452→5970 vs within-5970 bootstrap). Output: `umap_overlap.png`, `nn_distance_distribution.png`.

    c. **Per-feature comparison** — For each of 10 acoustic features: violin plots side by side (5970 vs 3452), Mann-Whitney U test with effect size (rank-biserial correlation). Output: `feature_comparison.png` (10-panel).

    d. **Transition matrix comparison** — Compute 7×7 transition matrices for both dyads. Visualization: side-by-side heatmaps + difference matrix. If 3452 has enough multi-call bouts, compute: conditional entropy comparison, transition divergence (per-row KL divergence). Output: `transition_comparison.png`.

    e. **Call rate normalization** — The 20:1 event count difference needs context. Compute: calls per recording hour (normalizing for recording duration), calls per WAV file, proportion of files with any USV. Output: included in summary CSV.

    f. **Summary report** — CSV with all test statistics, effect sizes, and a text summary of key differences.

    Output directory: `results/cross_dyad_comparison/`

    Use `repertoire_stats.py` functions where available: `shannon_entropy()`, `type_proportions()`, `jsd()`, `chi_squared_type_distribution()`, `transition_matrix()`. Import from `src/usv_spectrogram/classification/repertoire_stats.py`.

**Test plan:**
```
1. Chi-squared handles zero-count types in one dyad (expected for rare types in 3452)
2. JSD returns value in [0, 1] for any pair of distributions
3. UMAP overlap computation handles non-overlapping convex hulls (overlap = 0)
4. Transition comparison handles dyad with too few transitions (< 10) — skip with warning
5. Call rate normalization correctly parses timestamps from filenames
6. Mann-Whitney handles highly unequal n (7500 vs 350) without error
7. Summary CSV has all expected statistics
8. Script runs on synthetic data with known distributional differences
```

**Exit criteria:**
- [ ] All 5 comparison analyses produced
- [ ] Statistical tests use appropriate corrections for multiple comparisons
- [ ] Sample size caveat prominently noted in summary output
- [ ] All `repertoire_stats.py` functions used where applicable
- [ ] py_compile passes
- [ ] All tests pass

---

## Phase 16 Gate (B)

- [ ] 16.5: 3452 features extracted, both classification schemes applied
- [ ] 16.6: Cross-dyad comparison complete
- [ ] Key finding documented: how do the two dyads differ?
- [ ] Key finding documented: is the 20:1 event count ratio a real biological difference or a detection artifact?
- [ ] All tests passing

---

## Future Phases (Not Yet Specced)

### Phase C — LMT Behavioral Correlation

**What:** Link USV call types to social behavior events from Live Mouse Tracker.
**Status:** BLOCKED (needs `.sqlite` behavioral annotation database)
**Depends on:** External data (LMT output)

Modules planned:
- C1. Peri-event time histograms (PETH) for USV rate around behavioral events
- C2. Type-specific PETH — do specific call types correlate with specific behaviors?
- C3. USV-behavior temporal coupling (cross-correlation at varying lags)

**Existing tools:** `src/usv_spectrogram/lmt/` has `db_loader.py`, `synchronizer.py`, `event_triggered.py` — implemented but unused in production.

### Phase D — Publication Synthesis

**What:** Generate publication-ready figures and a comprehensive repertoire report.
**Status:** BLOCKED (on Phases A + B completion)
**Depends on:** 16.3, 16.4, 16.5, 16.6

Figures planned:
- Repertoire pie/bar chart (traditional types)
- UMAP with type overlay (continuum structure)
- Transition matrix heatmap
- Temporal raster (spike-raster style)
- Cross-dyad comparison panel
- Example spectrogram panel (one per type — galleries already exist)

---

## Open Questions

1. **Is the 20:1 call count ratio real?** One dyad producing far fewer USVs could reflect social dynamics (less interaction), individual differences (quieter animals), or recording/detection artifacts (different noise floor, equipment placement). Phase 16.4 and 16.6 should help disambiguate.

2. **Can calls be attributed to individuals?** Not with current single-microphone setup. If LMT data (Phase C) shows behavioral events from specific mice, temporal correlation could provide *probabilistic* attribution.

3. **UMAP projection vs joint fit:** Projecting 3452 into 5970's UMAP space assumes the feature distributions are comparable. If 3452 occupies a different region, a joint UMAP (fit on both datasets) may be more appropriate. Decision point in 16.5.

4. **Traditional taxonomy transferability:** The rule thresholds were calibrated on 5970 data distributions. They may not generalize to 3452 if that dyad produces acoustically different calls. Monitor classification confidence distribution in 16.5.

5. **Sinuosity as arousal-dependent complexity:** Complex calls (sinuosity > 3.5) are 8.7% of the repertoire. Does complexity increase in specific temporal or behavioral contexts? 16.3's feature overlays + temporal analysis can address this partially; full answer requires Phase C (LMT behavioral data).

6. **Are Flat calls a default state?** At 32% of the repertoire, Flat dominates. Is it uniformly distributed in time, or does it fill gaps between more structured calls? 16.3 includes a dedicated Flat temporal analysis to address this.

7. **Is the continuous manifold truly continuous?** UMAP showed one big blob, but finer-grained density analysis (HDBSCAN with smaller min_cluster_size) could reveal density ridges or subtypes within. Not directly addressed in current modules — could become a follow-up if 16.3 PCA reveals multi-modal structure.
