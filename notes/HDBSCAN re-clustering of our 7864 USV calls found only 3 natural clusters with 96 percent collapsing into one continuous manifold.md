---
description: "Independent replication of Goffinet's k<=2 finding using HDBSCAN instead of GMM on our own 5970 dataset — cross-method convergence strengthens the continuum claim"
type: finding
confidence: proven
created: 2026-04-06
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold

UMAP + HDBSCAN re-clustering of 7,864 mouse USV calls from our 5970 dataset (lmt_034) found only **3 natural clusters**, replacing 27 k-means clusters from DeepSqueak's built-in classification:

- **Cluster 2 (96.6%):** The main continuous manifold — all standard frequency-modulated USV types collapse here
- **Cluster 0 (1.2%):** Ultra-short calls (~4ms), minimal frequency sweep — a genuine acoustic subtype
- **Cluster 1 (1.7%):** Outlier group, 77/131 from old k-means Cluster_27 — a true outlier subpopulation
- **Noise (0.5%):** 37 ambiguous points assigned to no cluster

This independently corroborates [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] using a completely different method (density-based HDBSCAN vs. model-selection GMM) on a completely different dataset. Cross-method convergence is strong evidence that the continuum is a property of the data, not an artifact of any particular analysis.

The contingency matrix between old k-means clusters and new HDBSCAN assignments is telling: nearly every one of the 27 k-means clusters maps almost entirely to the single main HDBSCAN cluster. This means [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] imposed structure the data doesn't naturally support — k-means *must* find k clusters whether they exist or not, because it optimizes within-cluster variance without modeling noise or variable density.

However, this analysis used 10 raw DeepSqueak acoustic features (duration, principal frequency, bandwidth, etc.), not learned embeddings. Since [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] established encoder embeddings as the field standard, our results are feature-dependent. It remains an open question whether [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]].

Parameters: UMAP n_neighbors=15, min_dist=0.1, 2D (visualization) + 8D (HDBSCAN input). HDBSCAN min_cluster_size=50, min_samples=10.

---

Source: [[hdbscan-recluster-confirms-continuum]] (own analysis, 2026-04-03)

Relevant Notes:
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] — corroborates with different method and dataset
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] — the theoretical claim our data supports
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] — the imposed structure we falsified
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] — the pipeline we applied
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] — our result strengthens this concern
- [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]] — the limitation that qualifies this finding
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] — our data independently confirms the paradigm shift away from Holy & Guo discrete types
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] — AMVOC clustered in PCA-reduced autoencoder-embedding space and still found structure at k=6 with blinded human validation; our raw-feature HDBSCAN found k=3; the difference could reflect feature quality (learned 1280D embeddings vs raw 10D metrics) rather than genuine cluster count
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] — validates the unsupervised branch: UMAP+HDBSCAN reveals the continuum that k-means obscures
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] — if 96% of calls form one manifold, distributional comparison within that manifold captures variation that categorical analysis discards

Topics:
- [[unsupervised-usv-discovery]]
- [[classification]]
