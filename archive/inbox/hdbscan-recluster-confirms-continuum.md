---
title: HDBSCAN re-clustering confirms mouse USV continuum hypothesis
source: Own analysis (2026-04-03)
type: empirical-finding
tags: [clustering, HDBSCAN, UMAP, mouse-USV, continuum]
---

## Finding

UMAP + HDBSCAN re-clustering of 7,864 mouse USV calls (10 acoustic features from DeepSqueak, StandardScaler normalized) found only **3 natural clusters**, replacing 27 k-means clusters:

- **Cluster 2 (96.6%):** The main continuous manifold containing all standard frequency-modulated USV types
- **Cluster 0 (1.2%):** Ultra-short calls (~4ms duration), minimal frequency sweep
- **Cluster 1 (1.7%):** Outlier group, 77/131 from old k-means Cluster_27
- **Noise (0.5%):** 37 ambiguous points

## Significance

This empirically confirms Goffinet et al. 2021's finding that GMM model selection supports k<=2 for mouse USVs. The 27 k-means clusters imposed structure the data doesn't naturally support — the contingency matrix shows nearly every old cluster maps almost entirely to the single main HDBSCAN cluster.

## Parameters

UMAP: n_neighbors=15, min_dist=0.1, 2D (viz) + 8D (HDBSCAN input). HDBSCAN: min_cluster_size=50, min_samples=10.

## Limitation

Analysis used raw DeepSqueak acoustic features, not learned embeddings. Field standard (Best et al. 2023) uses pretrained encoder embeddings before UMAP. Results are feature-dependent.

## Links

- Supports: Goffinet 2021 GMM finding (k<=2)
- Supports: USV continuum hypothesis
- Contradicts: DeepSqueak 27-cluster classification
- Output: `results/recluster_umap_hdbscan/reclassified_detections.csv`
- Script: `scripts/recluster_umap_hdbscan.py`
