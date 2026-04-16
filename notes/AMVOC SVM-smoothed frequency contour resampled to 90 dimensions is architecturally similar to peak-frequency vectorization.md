---
description: "feature mode 3 extracts peak energy per frame then smooths via RBF-kernel SVM regression before resampling to fixed-length vector — adds SVM denoising compared to raw peak-frequency-per-column"
type: method
confidence: likely
created: 2026-04-15
meta_state: current
topics:
  - "[[classification-methodology]]"
---

# AMVOC SVM-smoothed frequency contour resampled to 90 dimensions is architecturally similar to peak-frequency vectorization

AMVOC's feature_mode 3 extracts a frequency contour from each USV spectrogram using a pipeline that closely mirrors peak-frequency vectorization approaches:

1. **Peak energy detection:** For each time frame, identify the frequency bin with maximum energy across 30–110 kHz
2. **Thresholding:** Retain only frames where peak energy exceeds the 20th percentile of maximum values (removes frames with no clear USV signal)
3. **SVM regression smoothing:** Fit an RBF-kernel SVM (C=1e3, gamma=0.1) mapping time → frequency, producing a smooth contour that tracks the dominant frequency trajectory
4. **Resampling:** Resample the smoothed contour to a fixed 90-dimensional vector

This is conceptually identical to Didi's peak-frequency-per-column approach, with two key differences: AMVOC adds SVM smoothing (step 3), which denoises the contour before vectorization, and AMVOC derives summary statistics from the contour (feature_mode 2: 4 features) rather than using the raw contour as the feature vector (feature_mode 3: 90 features).

The comparison matters because AMVOC's deep features beat this contour-based approach by 37% in human evaluation — since [[AMVOC deep autoencoder features scored 37 percent higher than 4-feature handcrafted baselines in blinded human evaluation]], the magnitude of improvement over vectorization-like features is substantial. This suggests that if our pipeline currently uses a peak-frequency vectorization approach, investing in learned representations (autoencoder, VQ-VAE, or pretrained embeddings) would likely yield significant clustering improvements.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[AMVOC deep autoencoder features scored 37 percent higher than 4-feature handcrafted baselines in blinded human evaluation]] — the quantitative gap between this approach and deep features
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] — another contour-based approach
- [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]] — the broader question this comparison informs
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] — the closest analog: Omer's 40D FM + 40D AM is a superset of AMVOC mode 3's 90D FM-only contour; the AM component is the key difference
- [[per-caller normalization of AM and FM features to 0-1 prevents individual acoustic idiosyncrasies from dominating classification]] — Oren normalizes contour features per-caller before classification; AMVOC mode 3 does not, making normalization scope a confound when comparing the two approaches

Topics:
- [[classification-methodology]]
