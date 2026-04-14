# Handoff: A3 — Acoustic Feature Deep-Dive

**Date:** 2026-04-14
**Commit:** `8cbe136d` on `main`
**Status:** Complete

---

## What This Is

A3 is the third analysis phase in the USV vocalization study. It examines the 10 acoustic features extracted by DeepSqueak for 7,864 classified USV calls from cage 5970 (animal usv_lmt_034). The central question: **does the 7-type traditional syllable taxonomy (Short, Flat, Up, Down, Chevron, Complex, Frequency_Jump) cut the acoustic space at natural boundaries, or does it impose arbitrary categories on a continuum?**

This matters because Goffinet et al. (2021) showed that mouse USVs form a continuous acoustic manifold. If our classification thresholds split density peaks instead of sitting at density valleys, any downstream analysis (cross-animal comparisons, behavioral correlations) that relies on categorical types could be misleading.

---

## Why We Did Each Analysis (and What It Tells You)

### 1. Feature Correlation Matrix (`plot_correlation_matrix`)

**Why:** Before analyzing 10 features, we need to know if some are redundant. If two features are highly correlated (r > 0.7), they're measuring roughly the same thing, and including both inflates the apparent dimensionality.

**Method:** Pearson correlation on the raw (unstandardized) features. We use a seaborn `clustermap` which adds hierarchical clustering to both rows and columns — this automatically groups correlated features together rather than displaying them in arbitrary order.

**What we found:**
- Mean power and tonality are nearly identical (r = 0.94) — louder calls are tonally cleaner, likely because higher SNR produces cleaner spectral peaks
- Bandwidth and freq SD are strongly correlated (r = 0.85) — both measure "how spread out the call is"
- Low freq and bandwidth are negatively correlated (r = -0.73) — calls starting at lower frequencies tend to span wider ranges

**Implication:** The effective feature space is ~7-8 dimensions, not 10. This means PCA and UMAP are working in a slightly lower-dimensional space than the raw feature count suggests.

### 2. PCA — Principal Component Analysis (`run_pca`)

**Why:** PCA finds the orthogonal axes along which the data varies most. It answers: "what are the main directions of acoustic variation?" If PC1 alone explains 80% of variance, the space is essentially 1D. If variance is spread evenly, the space is truly high-dimensional.

**Method:** Standardize features first (zero mean, unit variance) so that features with large numeric ranges (slope: -25000 to +20000) don't dominate features with small ranges (tonality: 0.16 to 0.67). Then fit PCA on all 10 components.

**Three outputs:**
- **Scree plot** — bar chart of variance explained per component, with cumulative line. Look for an "elbow" where adding more components stops helping.
- **Biplot** — scatter plot of PC1 vs PC2, with each point colored by syllable type AND arrows showing how each original feature maps onto these axes. Arrow direction = which PC a feature loads on. Arrow length = how important the feature is.
- **Loadings CSV** — the full numeric table of how each feature contributes to each PC.

**Key finding:** PC1 (33.4%) and PC2 (27.0%) together explain 60.4% — moderate, not dominant. The loadings are distributed across many features (no single feature dominates either axis). Critically, **slope barely appears in PC1 or PC2** — the Up/Down/Flat distinction operates along an axis orthogonal to the main variance directions. This means the traditional taxonomy partly captures variation that PCA under-weights.

**Important caveat:** Naming PCA components (e.g., "PC1 = spectral complexity") is interpretive, not mathematical. With loadings of 0.49, 0.47, and -0.46 on the top three features, plus 0.20-0.28 on six more, there is no clean label. The biplot arrows are the honest visualization.

### 3. UMAP Embedding (`compute_umap`, `plot_umap_by_type`, `plot_umap_by_features`)

**Why:** PCA is linear — it can only find flat axes. UMAP (Uniform Manifold Approximation and Projection) is nonlinear and preserves local neighborhoods, so it can reveal clusters and manifold structure that PCA misses. If USVs form discrete groups, UMAP will show islands. If they form a continuum, UMAP will show a smooth connected shape.

**Method:** UMAP with `n_neighbors=15` (how many nearby points define "local structure"), `min_dist=0.1` (how tightly points can pack), `random_state=42` (reproducibility). Applied to the 10 standardized features, NOT to CNN embeddings — we specifically want the acoustic feature manifold, not the learned representation.

**Why standardize before UMAP:** Without standardization, bandwidth (range 0-164 kHz) would dominate the distance calculation while tonality (range 0.16-0.67) would be invisible. Standardization gives each feature equal weight in defining distances.

**Two plots:**
- **By type** — each point colored by its assigned syllable type. If types are natural, you'd see color-coherent islands. What we see: overlap everywhere except Short (which clusters on the right edge due to its extreme duration values).
- **By feature** — the same embedding, colored by the actual value of each feature. This reveals what drives the spatial structure. Duration creates a smooth left-right gradient. Sinuosity and bandwidth create a lower-left lobe. Slope creates an upper-vs-lower gradient. Mean power and tonality show no clear spatial pattern.

**Why save coordinates:** UMAP is expensive to compute (~20s on 7,864 points). Saving `umap_coordinates.csv` means future analyses can reuse the embedding without recomputing.

### 4. Within-Type Violin Plots (`plot_within_type_violins`)

**Why:** This is the most direct test of the taxonomy. For each feature, we show how the values are distributed *within* each syllable type. If a type is a natural category, its violin should be tight (low spread). If it's an arbitrary slice, the violin will be wide and overlap heavily with adjacent types.

**Method:** Seaborn violin plots with `inner="quartile"` (showing median and IQR inside each violin). The critical addition: **red dashed reference lines at the exact classification thresholds** used in `classify_traditional_taxonomy.py`:
- Duration: 0.015s (Short threshold)
- Sinuosity: 1.8 (Chevron) and 3.5 (Complex)
- Bandwidth: 25 kHz (Chevron) and 55 kHz (Freq Jump)
- Slope: ±200 (Up/Down)

**How to read:** If a threshold line sits at a density valley between two distinct peaks, the boundary is natural. If it cuts through the middle of a single peak, the boundary is arbitrary.

**What we found:** The slope thresholds (±200) are clean cuts — Up and Down distributions are well-separated from Flat. The sinuosity 1.8 threshold (Chevron boundary) cuts through a continuous distribution, not at a valley. This is the most problematic boundary.

**CV (coefficient of variation):** We compute CV = std/mean for each feature within each type. High mean CV across features = "loose" type (catch-all category). Short has the highest CV (1.48) because it's defined only by duration — spectrally, Short calls are all over the place.

### 5. Boundary Case Analysis (`analyze_boundary_cases`)

**Why:** The classification script assigns a confidence level (high/medium/low) based on how close a call's feature values are to the threshold boundaries (within 20% of threshold = low confidence). Examining where these low-confidence calls sit reveals which boundaries are ambiguous.

**Method:**
- Filter calls where `classification_confidence == "low"` (1,130 calls, 14.4%)
- Plot them on the UMAP (highlighted in color against gray background) to see where they cluster spatially
- Overlay histograms of 4 key features (duration, slope, sinuosity, bandwidth) comparing low-confidence vs high/medium confidence distributions

**What we found:** Chevron (301) and Frequency_Jump (295) account for 53% of all low-confidence calls despite being only 24% of the dataset. The Chevron boundary is the leakiest. Up has almost no low-confidence calls (16) — the slope threshold is clean.

### 6. Auto-Generated Summary (`write_summary`)

**Why:** Programmatically extract key findings so the analysis script is self-documenting. The summary captures: number of strong correlations, PCA variance explained, loosest/tightest types, boundary case counts. This means you can re-run the script on different data (e.g., cage 3452) and get an updated summary without reading plots.

---

## Architecture Decisions

### Single script vs. module

This is a standalone `scripts/analyze_acoustic_features.py`, not a module in `src/`. Rationale: this is exploratory analysis that produces plots and CSVs. It doesn't define reusable APIs that other code calls. The existing pattern (`classify_traditional_taxonomy.py`, `analyze_temporal_dynamics.py`, `analyze_sequential_structure.py`) uses scripts for analysis and `src/` for library code.

### 10 features, not all 33 columns

The CSV has 33 columns. Only 10 are acoustic features — the rest are detection metadata (CNN confidence, match distance, file paths, etc.). Including detection metadata in PCA/UMAP would conflate "what does the call sound like?" with "how was the call detected?" — a methodological error.

### Standardize before PCA/UMAP, not before correlation

Pearson correlation is already scale-invariant (it normalizes by standard deviation). Standardizing before computing correlation would give identical results. But PCA and UMAP use Euclidean distance, which IS scale-dependent — without standardization, bandwidth (range 0-164) would dominate over tonality (range 0.16-0.67).

### UMAP on acoustic features, not CNN embeddings

The project has a `clustering/visualizer.py` that computes UMAP on 128-dimensional CNN embeddings. A3 specifically asks about the *acoustic* feature space — the 10 interpretable features from DeepSqueak. CNN embeddings are a learned representation that may or may not correspond to the same manifold structure.

### UMAP fallback to t-SNE

If `umap-learn` is not installed, the script falls back to `sklearn.manifold.TSNE`. This is a weaker alternative (t-SNE doesn't preserve global structure as well and can create artificial clusters), but it's always available via scikit-learn. In practice, UMAP 0.5.11 is installed in this project's venv.

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/analyze_acoustic_features.py` | Main analysis script (~500 lines) |
| `tests/test_acoustic_feature_analysis.py` | 16 synthetic-data tests |
| `docs/human/a3-acoustic-feature-deep-dive.md` | Human-readable report (markdown) |
| `docs/human/a3-acoustic-feature-deep-dive.pdf` | Report with all figures embedded (3 MB) |
| `docs/human/generate_a3_pdf.py` | PDF generator (rerun after editing markdown) |
| `results/acoustic_feature_analysis/` | All output files (plots, CSVs, summary) |

---

## Input Data

- **Source:** `results/traditional_taxonomy/classified_traditional.csv`
- **Rows:** 7,921 total, 7,864 with complete features, 57 unclassified (NaN features)
- **Produced by:** `scripts/classify_traditional_taxonomy.py` (Phase B traditional taxonomy classification)
- **Feature origin:** DeepSqueak acoustic feature extraction via the Python-MATLAB bridge

---

## How to Re-Run

```bash
# Full analysis (produces all plots + CSVs + summary)
.venv/bin/python scripts/analyze_acoustic_features.py

# With custom input/output
.venv/bin/python scripts/analyze_acoustic_features.py \
    --input results/traditional_taxonomy_3452/classified_traditional.csv \
    --output-dir results/acoustic_feature_analysis_3452/

# Regenerate PDF after editing the markdown report
.venv/bin/python docs/human/generate_a3_pdf.py

# Run tests
.venv/bin/python -m pytest tests/test_acoustic_feature_analysis.py -v
```

---

## Key Findings (for quick reference)

1. **Continuum confirmed.** UMAP shows smooth gradients, not discrete clusters. Consistent with Goffinet et al. (2021) and our own HDBSCAN result (Phase B).

2. **Three strong feature correlations.** Mean power ↔ tonality (r=0.94), bandwidth ↔ freq SD (r=0.85), low freq ↔ bandwidth (r=-0.73). Effective dimensionality is ~7-8.

3. **PCA captures 60.4% in 2 axes.** PC1 = spectral spread (bandwidth, freq SD, low freq). PC2 = power/clarity (mean power, tonality, sinuosity). Slope is absent from both — the Up/Down/Flat distinction is orthogonal to the main variance.

4. **Short, Up, and Down are clean categories.** Duration and slope thresholds sit at natural boundaries.

5. **Chevron and Frequency_Jump boundaries are arbitrary.** Sinuosity 1.8 and bandwidth 55 kHz thresholds cut through density peaks. These two types produce 53% of all low-confidence classifications.

6. **Short is internally the most variable type** (CV = 1.48). Defined only by duration, it groups spectrally diverse calls together.

---

## What Comes Next

- **A4 (Detection quality audit):** CNN confidence by type, HDBSCAN noise point inspection, ultra-short cluster investigation
- **B1 (Cross-animal comparison):** Re-run this same script on cage 3452's classified data to compare acoustic feature structure between animals
- **Possible follow-up:** Re-evaluate whether Chevron should be merged with Flat/Complex given the boundary ambiguity
