# A3: Acoustic Feature Deep-Dive

**Researcher:** Shachar | **PI:** Prof. London | **Date:** April 14, 2026

**Subject:** Do the 7 traditional syllable types correspond to natural categories in acoustic space?

---

## 1. Overview

We have 7,864 classified USV calls from cage 5970 (usv_lmt_034), each labeled with one of 7 traditional syllable types (Short, Flat, Up, Down, Chevron, Complex, Frequency_Jump). Prior UMAP+HDBSCAN analysis (Phase B) suggested calls form a continuum rather than discrete clusters. This analysis asks the sharper question: **where exactly do the traditional category boundaries fall relative to the natural structure of acoustic space?**

We analyze the 10 acoustic features extracted by DeepSqueak: duration, principal frequency, low/high frequency, bandwidth, frequency standard deviation, slope, sinuosity, mean power, and tonality.

## 2. Feature Correlations

The 10 features are not independent. Hierarchical clustering of the correlation matrix reveals three groups:

**Strongly correlated pairs (|r| > 0.7):**

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| Mean power (dB) | Tonality | r = 0.94 |
| Bandwidth (kHz) | Freq SD (kHz) | r = 0.85 |
| Low freq (kHz) | Bandwidth (kHz) | r = -0.73 |

Mean power and tonality are almost the same measurement — louder calls have cleaner tonal structure, presumably because they have higher signal-to-noise ratio. Bandwidth and frequency standard deviation both capture "how spread out the call is in frequency." Low frequency correlates negatively with bandwidth because calls starting at lower frequencies tend to span a wider range.

**Implication:** The effective dimensionality of the feature space is closer to 7–8 than 10, because 2–3 features are near-redundant.

### Figure: Correlation Matrix

## 3. PCA — Principal Axes of Variation

PCA on the 10 standardized features reveals that the first two components explain **60.4% of total variance** (PC1: 33.4%, PC2: 27.0%). This is moderate — enough to capture the dominant structure, but 40% of variance lives in PC3–PC10, meaning the feature space is not low-dimensional.

**Important note on interpreting PCA:** Principal components are weighted sums of all features, not single features. Naming them is interpretive, not mathematical. The loadings below show that multiple features contribute substantially to each axis — there is no clean "this PC = this feature" mapping.

### PC1 Loadings (33.4% variance)

| Feature | Loading |
|---------|---------|
| Bandwidth (kHz) | +0.49 |
| Freq SD (kHz) | +0.47 |
| Low freq (kHz) | -0.46 |
| Call length (s) | +0.28 |
| Principal freq (kHz) | -0.26 |
| Sinuosity | +0.25 |
| High freq (kHz) | +0.20 |
| Mean power (dB) | +0.20 |
| Tonality | +0.20 |
| Slope | -0.03 |

The top three loadings (bandwidth, freq SD, low freq) all relate to *how spectrally spread out the call is*. The remaining seven features each contribute 0.20–0.28. Slope is essentially absent from PC1.

### PC2 Loadings (27.0% variance)

| Feature | Loading |
|---------|---------|
| Mean power (dB) | +0.51 |
| Tonality | +0.46 |
| Sinuosity | -0.37 |
| High freq (kHz) | -0.34 |
| Call length (s) | +0.32 |
| Principal freq (kHz) | -0.27 |
| Bandwidth (kHz) | -0.22 |
| Freq SD (kHz) | -0.20 |
| Slope | +0.04 |
| Low freq (kHz) | -0.02 |

PC2 mixes power/tonality (positive) with sinuosity and high frequency (negative). Again, slope and low freq are absent.

**Key observation:** Slope — the feature that separates Up, Down, and Flat types — barely appears in either PC1 or PC2. This means the Up/Down/Flat distinction captures a direction of variation that is *orthogonal* to the dominant axes. The traditional taxonomy partly cuts the space in ways PCA doesn't prioritize.

### Figure: PCA Scree Plot

### Figure: PCA Biplot

## 4. UMAP Embedding — Acoustic Manifold Structure

UMAP (Uniform Manifold Approximation and Projection) projects the 10-dimensional feature space into 2D while preserving local neighborhood structure. Unlike PCA, UMAP is nonlinear and can reveal clusters that linear methods miss.

### Colored by syllable type

The UMAP embedding shows:

- **Short calls** (red) form a distinct cluster at the right edge — duration alone separates them
- **Complex** and **Chevron** (orange/brown) overlap heavily in the lower-left region
- **Flat**, **Down**, and **Up** (blue/purple/green) are intermixed in the center
- **Frequency_Jump** (pink) scatters throughout but concentrates in the lower-left

Types do **not** separate into discrete islands. The acoustic space is a continuum with local gradients, not clean boundaries. This confirms the HDBSCAN finding from Phase B.

### Figure: UMAP by Syllable Type

### Colored by individual features

Coloring the same UMAP by each feature individually reveals what drives the spatial structure:

- **Duration** shows a smooth gradient from short (right) to long (left) — not a binary split
- **Slope** shows positive (green, upper-right) vs negative (purple, lower-right) calls, but with no sharp boundary
- **Sinuosity** concentrates high values (yellow) in the lower-left lobe — this is where Complex and Chevron live
- **Bandwidth** mirrors sinuosity — wide-bandwidth calls are in the same lower-left region
- **Mean power** and **tonality** vary smoothly across the entire manifold — no spatial structure

### Figure: UMAP by Features

## 5. Within-Type Variability

Violin plots show the distribution of each feature within each syllable type, with red dashed lines marking the classification thresholds.

### Key findings

**Tight categories (low within-type variability):**
- **Down** (mean CV = 0.31) and **Up** (CV = 0.35) — the slope threshold (±200) separates genuinely distinct populations. Most calls are well above/below the threshold.
- **Frequency_Jump** (CV = 0.41) — the 55 kHz bandwidth threshold sits at a relative gap in the distribution.

**Loose categories (high within-type variability):**
- **Short** (mean CV = 1.48) — defined only by duration < 15ms, so any spectral shape is allowed. Short calls are internally extremely diverse in frequency, slope, and sinuosity.
- **Complex** (CV = 0.77) and **Chevron** (CV = 0.65) — the sinuosity thresholds (3.5 and 1.8) cut through continuous distributions rather than sitting at valleys. The boundary between Chevron and Flat is especially arbitrary.

### Figure: Within-Type Violin Plots

## 6. Boundary Cases — Where the Taxonomy Breaks Down

14.4% of calls (1,130 / 7,864) were classified with low confidence, meaning their feature values fall near a classification threshold.

**Leakiest boundaries:**

| Type | Low-confidence count | % of all low-conf |
|------|---------------------|-------------------|
| Chevron | 301 | 26.6% |
| Frequency_Jump | 295 | 26.1% |
| Flat | 168 | 14.9% |
| Down | 127 | 11.2% |
| Short | 116 | 10.3% |
| Complex | 107 | 9.5% |
| Up | 16 | 1.4% |

Chevron and Frequency_Jump together account for **53% of all low-confidence calls** despite being only 24% of the dataset. The Chevron boundary (sinuosity > 1.8 AND bandwidth > 25 kHz) is the most problematic — it cuts through a density peak in the joint sinuosity-bandwidth distribution.

By contrast, Up has almost no low-confidence calls — the slope > 200 threshold is clean.

### Figure: Boundary Cases

## 7. Summary and Implications

### What the taxonomy gets right
- **Short** — duration is a genuine binary feature; these calls are physically distinct
- **Up / Down** — slope threshold separates real subpopulations; clean boundaries
- The taxonomy provides a useful *language* for discussing call properties

### What the taxonomy gets wrong
- **Chevron vs Flat vs Complex** — these boundaries are arbitrary cuts through a continuum. The sinuosity 1.8 threshold doesn't correspond to a natural break point
- **Frequency_Jump** — the 55 kHz bandwidth threshold is somewhat arbitrary; many borderline cases
- **Short is internally incoherent** — it groups calls with wildly different spectral properties (CV = 1.48) just because they're brief

### The big picture

The 7-type taxonomy is a useful simplification but not a natural carving of acoustic space. The data supports a model where:

1. **Duration** creates one real boundary (short vs. everything else)
2. **Slope** creates another (rising vs. falling vs. flat)
3. **Sinuosity/bandwidth** form a continuum, not categories — Complex, Chevron, and Frequency_Jump are arbitrary slices of a continuous modulation spectrum

For downstream analyses (cross-animal comparison, behavioral correlation), the traditional types are a reasonable first pass, but results should be verified against continuous acoustic features — especially for Chevron and Frequency_Jump, where category membership is ambiguous for ~25% of calls.

## 8. Output Files

All outputs saved to `results/acoustic_feature_analysis/`:

| File | Description |
|------|-------------|
| `correlation_matrix.png` | Hierarchical clustermap of 10 features |
| `pca_scree.png` | Variance explained per principal component |
| `pca_biplot.png` | PC1 vs PC2 with loading arrows, colored by type |
| `pca_loadings.csv` | Full 10×10 loadings table |
| `umap_by_type.png` | UMAP colored by 7 syllable types |
| `umap_by_feature.png` | UMAP colored by 6 key features (2×3 grid) |
| `umap_coordinates.csv` | Reusable UMAP coordinates (7,864 rows) |
| `within_type_violins.png` | Feature distributions per type with threshold lines |
| `boundary_cases.png` | Low-confidence calls on UMAP + feature histograms |
| `analysis_summary.md` | Auto-generated key findings |
