# A3: Acoustic Feature Deep-Dive — Summary

**Dataset:** 401 classified USV calls (7 traditional types)
**Features:** 10 acoustic features, standardized for PCA/UMAP
**Embedding:** UMAP (n_neighbors=15, min_dist=0.1)

## 1. Feature Correlations

**Strong correlations (|r| > 0.7):** 3 pairs
- Bandwidth (kHz) <-> Freq SD (kHz): r = 0.923
- Mean power (dB) <-> Tonality: r = 0.917
- High freq (kHz) <-> Bandwidth (kHz): r = 0.732

## 2. PCA Results

**Variance explained by PC1 + PC2:** 61.5%
- PC1 (39.2%): dominated by bandwidth_hz (Bandwidth (kHz))
- PC2 (22.3%): dominated by call_length_s (Duration (s))

**Top PC1 loadings:**
- Bandwidth (kHz): 0.473
- Freq SD (kHz): 0.460
- Sinuosity: 0.409

**Top PC2 loadings:**
- Duration (s): 0.443
- Tonality: 0.437
- Low freq (kHz): -0.429

## 3. UMAP Embedding

See `umap_by_type.png` and `umap_by_feature.png` for visual assessment of
whether the 7 traditional types map onto discrete clusters or a continuum.

## 4. Within-Type Variability

**Loosest types (highest mean coefficient of variation):**
- Chevron: mean CV = 35.736
- Short: mean CV = 6.886
- Flat: mean CV = 3.318

**Tightest types:**
- Frequency_Jump: mean CV = 1.394
- Down: mean CV = 0.375
- Up: mean CV = 0.289

## 5. Boundary Cases

**Low-confidence calls:** 40 / 401 (10.0%)

**Leakiest type boundaries (most low-confidence calls):**
- Short: 16 (40.0% of all low-confidence)
- Chevron: 12 (30.0% of all low-confidence)
- Flat: 6 (15.0% of all low-confidence)
- Complex: 2 (5.0% of all low-confidence)
- Frequency_Jump: 2 (5.0% of all low-confidence)
- Up: 1 (2.5% of all low-confidence)
- Down: 1 (2.5% of all low-confidence)

## Interpretation Guide

- If violin plots show threshold lines cutting through density peaks rather than
  valleys, the taxonomy boundary is arbitrary at that point.
- If UMAP shows smooth color gradients rather than discrete patches for types,
  the acoustic space is a continuum (consistent with Goffinet et al. 2021).
- High within-type CV suggests that type is a catch-all rather than a coherent category.
- Many low-confidence calls at a type boundary suggest the rule cascade is ambiguous there.
