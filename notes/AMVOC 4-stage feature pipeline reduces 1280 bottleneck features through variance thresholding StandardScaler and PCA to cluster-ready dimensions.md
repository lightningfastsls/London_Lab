---
description: "variance threshold at 1.2x mean removes ~75 percent of features before scaling and PCA — critically clustering operates in PCA space not t-SNE space, a common methodological error the pipeline explicitly avoids"
type: method
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification-methodology]]"
---

# AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions

After extracting 1,280-dimensional bottleneck vectors from the autoencoder, AMVOC applies a 4-stage post-processing pipeline before clustering:

**Stage 1 — Variance Thresholding:** Remove features with variance below `v_t = 1.2 × mean(variance_per_feature)` using `sklearn.feature_selection.VarianceThreshold`. This eliminates ~75% of features (1,280 → ~320), discarding bottleneck dimensions that are near-constant across USVs and therefore uninformative for discrimination.

**Stage 2 — Standard Scaling:** Z-score normalization per feature via `sklearn.preprocessing.StandardScaler`. Ensures all remaining features contribute equally to distance calculations rather than being dominated by high-variance dimensions.

**Stage 3 — PCA:** Retain the smallest number of components preserving 95% of variance via `sklearn.decomposition.PCA`. The number of components varies across datasets. The code iteratively increases component count if 95% is not reached with the initial guess.

**Stage 4 — t-SNE for visualization only.** This is the critical methodological point: clustering operates in the PCA-reduced space, NOT in the t-SNE 2D projection. t-SNE distorts distances and densities, making it unsuitable for distance-based clustering algorithms. Using t-SNE coordinates for clustering is a common error in the field.

The dimensionality cascade is dramatic:
```
Input:           10,240  (spectrogram pixels)
Bottleneck:       1,280  (8× compression by autoencoder)
Var threshold:     ~320  (~4× reduction)
PCA (95%):     variable  (further reduction)
t-SNE:              2    (visualization only)
```

This pipeline is sensible but predates the field's convergence on [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]]. The modern equivalent would replace t-SNE with UMAP and replace user-specified k-means with HDBSCAN. The variance thresholding → StandardScaler → PCA chain remains valid regardless of downstream method.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]] — produces the 1,280 features this pipeline processes
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] — the modern replacement for stages 3-4
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] — our own application of modern pipeline to similar data
- [[separating representation learning from discretization enables richer feature discovery]] — AMVOC's 4-stage pipeline embodies this principle: the autoencoder learns representations first, then variance thresholding + PCA reduce dimensionally before clustering, keeping representation and categorization as distinct phases
- [[CASE benchmark systematically compared 48 unsupervised clustering methods for animal vocalizations]] — CASE provides the systematic comparison AMVOC's 4-stage pipeline lacks: stages 3-4 could be swapped for any of 48 methods Schneider 2022 tested, making the PCA+t-SNE+k-means path one point in a much larger design space

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
