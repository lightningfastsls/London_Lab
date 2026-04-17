---
description: "AMVOC achieved strong results (37 percent over baseline) with PCA plus t-SNE plus k-means-6 before the UMAP plus HDBSCAN paradigm solidified — raises the question of whether embedding quality matters more than downstream clustering method"
type: open-question
created: 2026-04-15
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification-methodology]]"
---

# AMVOC t-SNE plus user-specified k versus field-standard UMAP plus HDBSCAN for bioacoustic clustering

## Quick Test

Can AMVOC's PCA → t-SNE → user-specified-k approach produce results competitive with the modern UMAP → HDBSCAN pipeline? Evidence suggests yes — AMVOC's deep features achieved 37% higher global annotation scores than handcrafted baselines using K-Means/GMM at k=6, and the blinded evaluation scores were strong. But this doesn't tell us whether UMAP+HDBSCAN would have done even better on the same features.

## When Each Pole Wins

**AMVOC-style (PCA + user-specified k)** wins when: the number of categories is approximately known from prior work (e.g., traditional USV taxonomy has ~7-11 types), the goal is consistent assignment to predefined types across studies, or when interpretability of fixed cluster centroids matters.

**UMAP + HDBSCAN** wins when: the true number of clusters is unknown, cluster shapes are non-spherical, noise points should be explicitly identified, or the data may form a continuum rather than discrete groups. Since [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]], the continuum case is exactly our situation.

## Dissolution Attempts

The tension partially dissolves if the primary discriminant is embedding quality, not clustering method. If autoencoder embeddings are rich enough, even k-means at k=6 produces meaningful clusters. If embeddings are poor, even HDBSCAN cannot find real structure. AMVOC's success may owe more to the autoencoder than to the clustering approach.

However, the tension reasserts for continuum data: k-means forces k clusters by definition, while HDBSCAN can return k=1 (or k=3, as we found). The choice of clustering method determines whether you can *discover* that the data is a continuum rather than having discrete structure imposed.

## Practical Applications

For our pipeline: use UMAP+HDBSCAN to discover natural structure, then optionally impose k-means/GMM for cross-study comparability. The methods answer different questions — "what structure exists?" versus "given k categories, how do assignments compare?"

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] — the modern paradigm
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] — AMVOC's approach
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] — evidence favoring HDBSCAN for discovery
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] — the deeper reason the clustering method matters: k-means imposes k clusters by definition, preventing discovery that the data is a continuum
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] — our VQ-VAE codebook is a third approach: data-driven discretization at K=64 that over-discretizes the continuum deliberately to capture sub-structure that density-based clustering merges
- [[CASE benchmark systematically compared 48 unsupervised clustering methods for animal vocalizations]] — the systematic framework for answering this tension empirically: Schneider 2022 tested HDBSCAN, affinity propagation, and others on the same features, providing the comparison AMVOC's paper did not
- [[DeepSqueak v3.1 added VAE-based contour-invariant clustering as upgrade over k-means for continuous USV variation]] — a fourth pole beyond AMVOC-style and UMAP+HDBSCAN: DeepSqueak's response was to replace the clustering method (k-means to VAE) while keeping the workflow structure, suggesting the embedding-quality-vs-clustering-method tension is resolved in practice by upgrading both

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
