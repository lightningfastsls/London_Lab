---
description: "Schneider et al 2022 tested community detection, affinity propagation, HDBSCAN, and fuzzy clustering with DTW and cross-correlation features — open benchmark still cited in 2025"
type: baseline
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# CASE benchmark systematically compared 48 unsupervised clustering methods for animal vocalizations

CASE (Cluster and Analyze Sound Events, Schneider et al. 2022) provides the most systematic comparison of unsupervised vocal classification methods available. It tested 48 clustering methods — including community detection, affinity propagation, HDBSCAN, and fuzzy clustering — paired with classifiers including k-nearest neighbor, dynamic time warping, and cross-correlation. The benchmark uses a windowed, multi-feature extraction approach and provides an open evaluation framework.

While published in 2022, CASE remains relevant and continues to be cited in 2025 clustering evaluations. Its value is methodological: rather than comparing tools (DeepSqueak vs AMVOC vs DAS), it compares the underlying algorithms in a controlled setting. For our pipeline, CASE could serve as a reference when evaluating which clustering algorithm to apply to VQ-VAE codebook assignments or to continuous embedding spaces.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] -- HDBSCAN emerged as the winner from benchmarks like CASE
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- clustering methods must handle continuous variation
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] -- our empirical application of HDBSCAN (one of CASE's 48 methods) on mouse USVs confirmed density-based clustering finds far fewer clusters than k-means

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
