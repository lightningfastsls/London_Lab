---
description: "Field convergence on embed-reduce-cluster paradigm — extract learned embeddings, UMAP to 8 dimensions, HDBSCAN for density-based clustering with automatic cluster count"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations

The bioacoustics field converged between 2023 and 2025 on a three-stage unsupervised pipeline: (1) extract embeddings from a pretrained model, (2) reduce dimensionality with UMAP, (3) cluster with HDBSCAN. This was formalized by Best et al. (2023), who demonstrated the approach across six species including Bengalese finches, humpback whales, and bottlenose dolphins.

HDBSCAN became the default clustering algorithm for several reasons that matter for bioacoustic data: it automatically determines cluster count (no need to pre-specify k), handles variable cluster shapes and densities, explicitly models noise points (critical for noisy field recordings), and scales with O(n log n) complexity. These properties make it far more suitable than k-means, which requires pre-specifying cluster count and assumes spherical clusters.

Notably, AMVOC (Stoumpou et al. 2022) predates this convergence and does NOT use UMAP or HDBSCAN — it uses t-SNE for visualization only and clusters in PCA space with user-specified k (K-Means, GMM, or Agglomerative at k=6). Despite this older methodology, AMVOC achieved strong results (37% over handcrafted baselines in blinded evaluation), which raises the question of whether embedding quality matters more than downstream clustering method — see [[AMVOC t-SNE plus user-specified k versus field-standard UMAP plus HDBSCAN for bioacoustic clustering]].

The pipeline achieved NMI scores of 0.5-0.88 across eight datasets, with 46-97% of clusters meeting a 90% purity threshold. For mouse USVs specifically, this pipeline represents the modern replacement for DeepSqueak's built-in k-means clustering. We applied it to our own 5970 dataset (7,864 calls): UMAP n_neighbors=15, min_dist=0.1, 8D reduction; HDBSCAN min_cluster_size=50, min_samples=10. The result — [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] — confirmed the pipeline's ability to reject imposed structure that k-means enforces. However, the choice of embedding model matters enormously — since [[supervised bioacoustic foundation models vastly outperform self-supervised for species-level clustering]], the upstream model selection is the critical decision.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- this note covers the same paper but the pipeline convergence finding is broader
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- HDBSCAN's soft clustering handles this continuum better than k-means
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] -- AMVOC's pre-UMAP/HDBSCAN pipeline used PCA+t-SNE+k-means instead; the variance thresholding and StandardScaler stages remain valid preprocessing regardless of whether downstream uses UMAP+HDBSCAN or the older approach
- [[AMVOC t-SNE plus user-specified k versus field-standard UMAP plus HDBSCAN for bioacoustic clustering]] -- the tension between AMVOC's pre-convergence methodology and the current field standard; AMVOC's success suggests embedding quality may be the dominant factor

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
