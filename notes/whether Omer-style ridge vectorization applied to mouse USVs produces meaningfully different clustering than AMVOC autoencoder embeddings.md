---
description: "Open question: AMVOC mode 3 (90D frequency contour) and Omer ridge (40D FM + 40D AM) are architecturally similar — do they produce the same clusters on mouse USV data?"
type: open-question
confidence: speculative
conditions:
  - requires implementing Omer vectorization in Python and running on classified_detections_full.csv data
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# whether Omer-style ridge vectorization applied to mouse USVs produces meaningfully different clustering than AMVOC autoencoder embeddings

AMVOC feature mode 3 produces a 90-dimensional resampled frequency contour per USV (Stoumpou et al. 2022). The Omer lab ridge vectorization produces 80 dimensions: 40 FM (frequency contour) + 40 AM (amplitude along ridge). These are architecturally similar but differ in:

1. **AM component:** Omer includes amplitude trajectory; AMVOC mode 3 does not
2. **Smoothing:** Omer applies median smoothing (window=6) to AM and mean smoothing (window=5) to FM; AMVOC smoothing differs
3. **Normalization scope:** Omer normalizes per-caller; AMVOC normalizes differently
4. **Dimensionality:** 80D vs 90D (different target time steps)

The empirical question: when both representations are embedded via UMAP and clustered via HDBSCAN on our 7,518 classified USV calls, do they produce the same structure? Given that [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]], the null hypothesis is that both representations collapse to a similar continuous manifold — but the AM component in Omer vectorization might reveal substructure that the FM-only representation misses.

This is testable with existing data (classified_detections_full.csv) once a Python implementation of Omer vectorization is built, using the [[Oren 2024 Zenodo repository provides complete MATLAB implementation of spectrogram ridge vectorization for adaptation|Zenodo source code]] as reference.

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)

Relevant Notes:
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- the AMVOC comparison point
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] -- current clustering result to compare against
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the continuum finding that both methods must respect
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the Omer method
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] -- the downstream clustering method both representations would feed, holding the pipeline constant so the comparison isolates the embedding
- [[AMVOC SVM-smoothed frequency contour resampled to 90 dimensions is architecturally similar to peak-frequency vectorization]] -- the specific AMVOC feature mode 3 that is most directly comparable to Omer ridge vectorization, differing mainly in the AM component
- [[per-caller normalization of AM and FM features to 0-1 prevents individual acoustic idiosyncrasies from dominating classification]] -- confound to control for: Omer normalizes per-caller, AMVOC does not; any meaningful clustering difference could be attributed to normalization scope rather than FM-vs-AM+FM

Topics:
- [[unsupervised-usv-discovery]]
- [[classification]]
