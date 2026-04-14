# A3: Acoustic Feature Deep-Dive — Summary

**Dataset:** 7864 classified USV calls (7 traditional types)
**Features:** 10 acoustic features, standardized for PCA/UMAP
**Embedding:** UMAP (n_neighbors=15, min_dist=0.1)

## 1. Feature Correlations

**Strong correlations (|r| > 0.7):** 3 pairs
- Mean power (dB) <-> Tonality: r = 0.941
- Bandwidth (kHz) <-> Freq SD (kHz): r = 0.854
- Low freq (kHz) <-> Bandwidth (kHz): r = -0.728

## 2. PCA Results

**Variance explained by PC1 + PC2:** 60.4%
- PC1 (33.4%): dominated by bandwidth_hz (Bandwidth (kHz))
- PC2 (27.0%): dominated by mean_power_db (Mean power (dB))

**Top PC1 loadings:**
- Bandwidth (kHz): 0.487
- Freq SD (kHz): 0.471
- Low freq (kHz): -0.456

**Top PC2 loadings:**
- Mean power (dB): 0.514
- Tonality: 0.461
- Sinuosity: -0.373

## 3. UMAP Embedding

See `umap_by_type.png` and `umap_by_feature.png` for visual assessment of
whether the 7 traditional types map onto discrete clusters or a continuum.

## 4. Within-Type Variability

**Loosest types (highest mean coefficient of variation):**
- Short: mean CV = 1.481
- Complex: mean CV = 0.768
- Chevron: mean CV = 0.647

**Tightest types:**
- Frequency_Jump: mean CV = 0.412
- Up: mean CV = 0.349
- Down: mean CV = 0.311

## 5. Boundary Cases

**Low-confidence calls:** 1130 / 7864 (14.4%)

**Leakiest type boundaries (most low-confidence calls):**
- Chevron: 301 (26.6% of all low-confidence)
- Frequency_Jump: 295 (26.1% of all low-confidence)
- Flat: 168 (14.9% of all low-confidence)
- Down: 127 (11.2% of all low-confidence)
- Short: 116 (10.3% of all low-confidence)
- Complex: 107 (9.5% of all low-confidence)
- Up: 16 (1.4% of all low-confidence)

## Interpretation Guide

- If violin plots show threshold lines cutting through density peaks rather than
  valleys, the taxonomy boundary is arbitrary at that point.
- If UMAP shows smooth color gradients rather than discrete patches for types,
  the acoustic space is a continuum (consistent with Goffinet et al. 2021).
- High within-type CV suggests that type is a catch-all rather than a coherent category.
- Many low-confidence calls at a type boundary suggest the rule cascade is ambiguous there.
