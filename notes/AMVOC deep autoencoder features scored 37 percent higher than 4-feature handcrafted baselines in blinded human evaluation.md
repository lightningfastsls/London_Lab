---
description: "blinded 4-annotator evaluation (2 experts + 2 non-experts) across K-Means GMM and Agglomerative at k=6 — all significant at p less than 0.01 for global scores, though point-level approval was not significant at p=0.18"
type: finding
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification-methodology]]"
---

# AMVOC deep autoencoder features scored 37 percent higher than 4-feature handcrafted baselines in blinded human evaluation

Stoumpou et al. (2022) compared their autoencoder's 1,280-dimensional bottleneck features against a 4-feature handcrafted baseline for USV clustering quality. The evaluation used blinded human annotation — 4 annotators (2 domain experts, 2 non-experts) who did not know which method produced which clustering. Results across three clustering algorithms (K-Means, GMM, Agglomerative, all at k=6):

- **Global annotation scores:** 37% higher for deep features on average (p < 0.01 for all three configs)
- **Cluster-level scores:** 30% higher for deep features
- **Point-level approval rate:** Higher for deep features but NOT significant (p = 0.18, ~100 points per config)

The handcrafted baseline used only 4 features: (1) duration, (2) normalized time position of minimum frequency, (3) normalized time position of maximum frequency, and (4) normalized bandwidth (freq_start - freq_end) / mean_frequency. This is a deliberately simple baseline — not the best handcrafted features one could design. The 12-feature extended set (feature_mode 1) includes min/max/mean frequency, frequency change rates, and delta statistics, but was not the default comparison.

The non-significance at point level (p = 0.18) is worth noting: while deep features produce better-structured clusters overall, individual USV assignments may not be dramatically better. This aligns with the continuum hypothesis — since [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]], the "correct" assignment of a USV near a category boundary is inherently ambiguous.

For our pipeline, this confirms that [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]] — but the magnitude of improvement depends heavily on how many handcrafted features you start with. Our DeepSqueak features (10 dimensions) are richer than AMVOC's 4-feature baseline, so the improvement from learned representations may be smaller for us.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] — consistent finding that learned representations outperform handcrafted features
- [[raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs]] — directly relevant open question for our pipeline
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] — the pipeline that produced the deep features evaluated here
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] — why point-level agreement may be inherently limited
- [[DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality]] — DeepSqueak's 16 handcrafted features are richer than AMVOC's 4-feature baseline, so the 37% improvement over a 16-feature set would likely be smaller; contextualizes the gap magnitude
- [[a generic cross-species autoencoder performs nearly as well as species-specific models suggesting shared vocalization structure]] — the 37% advantage of deep features over handcrafted generalizes across taxa, suggesting the learned-representation benefit is not mouse-specific and a pretrained multi-species AE could yield similar gains without mouse-specific training

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
