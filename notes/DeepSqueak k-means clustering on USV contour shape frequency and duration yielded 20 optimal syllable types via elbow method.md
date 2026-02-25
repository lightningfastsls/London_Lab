---
description: "DeepSqueak's unsupervised k-means operates on z-scored contour shape, frequency, and duration features with elbow method yielding k=20 for mouse USVs"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[representation-learning]]"
---

# DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method

DeepSqueak's primary classification method is unsupervised k-means clustering operating on three feature types: **contour shape** (1st derivative at 10 segments), **frequency** (contour reduced to 10 segments), and **duration** -- all z-score normalized with user-adjustable weighting. The elbow method on within-cluster error automatically determines the optimal cluster count; in the original paper this yielded **20 optimal syllable types** for mouse USVs.

Version 3.1 added VAE-based contour-invariant clustering as a significant upgrade for capturing continuous variation, and t-SNE visualization is built in for inspecting cluster distributions. The k-means approach can also be followed by supervised CNN training: users produce clean clusters via unsupervised methods, then use those labeled exemplars to train a supervised network for faster future classification.

The 20-cluster result from k-means sits between VocalMat's 11 predefined types and [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] at 100-140 types. This range of cluster counts across tools illustrates that the "right" number of USV categories depends heavily on the feature representation and clustering method -- a reality that [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] later formalized as a continuum rather than naturally discrete categories.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Coffey et al. (2019), *Neuropsychopharmacology*

Relevant Notes:
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- unsupervised predecessor discovering far more types with different features
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- challenges the assumption that any fixed k is correct
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our VQ-VAE K=64 contextualizes within the 11-140 range
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- supervised approach with k=11 predefined
- [[DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection]] -- the detection side of the same tool
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- method for evaluating whether k=20 captures meaningful structure
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- per-animal proportions of these 20 syllable types feed the Bray-Curtis dissimilarity matrix
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- these syllable type proportions become the P and Q distributions for JSD computation

Topics:
- [[classification]]
- [[representation-learning]]
