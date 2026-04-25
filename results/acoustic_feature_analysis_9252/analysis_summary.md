# A3: Acoustic Feature Deep-Dive — Summary

**Dataset:** 597 classified USV calls (7 traditional types)
**Features:** 10 acoustic features, standardized for PCA/UMAP
**Embedding:** UMAP (n_neighbors=15, min_dist=0.1)

## 1. Feature Correlations

**Strong correlations (|r| > 0.7):** 3 pairs
- Mean power (dB) <-> Tonality: r = 0.928
- Bandwidth (kHz) <-> Freq SD (kHz): r = 0.899
- Principal freq (kHz) <-> Low freq (kHz): r = 0.705

## 2. PCA Results

**Variance explained by PC1 + PC2:** 60.8%
- PC1 (37.6%): dominated by freq_std_dev_hz (Freq SD (kHz))
- PC2 (23.2%): dominated by principal_freq_hz (Principal freq (kHz))

**Top PC1 loadings:**
- Freq SD (kHz): 0.469
- Bandwidth (kHz): 0.467
- Mean power (dB): -0.379

**Top PC2 loadings:**
- Principal freq (kHz): 0.572
- Low freq (kHz): 0.562
- High freq (kHz): 0.346

## 3. UMAP Embedding

See `umap_by_type.png` and `umap_by_feature.png` for visual assessment of
whether the 7 traditional types map onto discrete clusters or a continuum.

## 4. Within-Type Variability

**Loosest types (highest mean coefficient of variation):**
- Complex: mean CV = 2.795
- Short: mean CV = 2.037
- Flat: mean CV = 1.361

**Tightest types:**
- Frequency_Jump: mean CV = 0.504
- Down: mean CV = 0.368
- Up: mean CV = 0.319

## 5. Boundary Cases

**Low-confidence calls:** 50 / 597 (8.4%)

**Leakiest type boundaries (most low-confidence calls):**
- Short: 19 (38.0% of all low-confidence)
- Complex: 13 (26.0% of all low-confidence)
- Chevron: 10 (20.0% of all low-confidence)
- Flat: 4 (8.0% of all low-confidence)
- Frequency_Jump: 2 (4.0% of all low-confidence)
- Up: 1 (2.0% of all low-confidence)
- Down: 1 (2.0% of all low-confidence)

## Interpretation Guide

- If violin plots show threshold lines cutting through density peaks rather than
  valleys, the taxonomy boundary is arbitrary at that point.
- If UMAP shows smooth color gradients rather than discrete patches for types,
  the acoustic space is a continuum (consistent with Goffinet et al. 2021).
- High within-type CV suggests that type is a catch-all rather than a coherent category.
- Many low-confidence calls at a type boundary suggest the rule cascade is ambiguous there.
