---
description: "Our HDBSCAN analysis used raw DeepSqueak features, not learned embeddings — encoder representations might reveal sub-structure within the main cluster that raw features miss"
type: open-question
created: 2026-04-06
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification-methodology]]"
---

# raw acoustic features versus learned embeddings may yield different clustering structure for mouse USVs

Our HDBSCAN re-clustering used 10 raw DeepSqueak acoustic features (duration, principal frequency, low/high frequency, mean frequency, delta frequency, sinuosity, and contour statistics) — hand-crafted measurements of the spectrogram contour. The field standard established by [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] instead uses pretrained encoder embeddings before UMAP+HDBSCAN.

The question is whether learned representations would reveal meaningful sub-structure within the 96.6% main cluster that [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] found. Raw features capture explicit acoustic measurements but miss higher-order patterns — spectrotemporal dynamics, harmonic structure, noise texture — that learned embeddings encode implicitly. If those hidden dimensions carry biological signal, the "one big cluster" result could partially reflect feature limitations rather than pure continuum structure.

This doesn't undermine the core finding — [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] used 32-dim VAE latent space (learned embeddings) and still found k<=2, so the continuum likely holds regardless. But there may be *more* structure than raw features reveal, even if it falls short of 20-27 discrete categories.

AMVOC provides the most direct quantitative evidence for this gap: since [[AMVOC deep autoencoder features scored 37 percent higher than 4-feature handcrafted baselines in blinded human evaluation]], learned autoencoder representations substantially outperform even the simplest handcrafted features (duration, normalized min/max frequency positions, normalized bandwidth). However, their baseline was only 4 features — our DeepSqueak features are 10-dimensional and richer, so the magnitude of improvement from learned representations may be smaller for us. The specific comparison AMVOC omits is the one we most need: how does their 90-dimensional SVM-smoothed contour (feature_mode 3, since [[AMVOC SVM-smoothed frequency contour resampled to 90 dimensions is architecturally similar to peak-frequency vectorization]]) compare to autoencoder features? That gap remains open.

Resolution paths: (1) apply a pretrained bioacoustic encoder (BEATs, Perch 2.0, or HuBERT fine-tuned) to our spectrograms and re-run UMAP+HDBSCAN, (2) train our own autoencoder on the 5970 dataset and compare clustering outcomes, (3) compare cluster stability metrics across feature spaces.

---

Source: [[hdbscan-recluster-confirms-continuum]] (own analysis, 2026-04-03)

Relevant Notes:
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] — the field standard our analysis deviates from
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] — the finding this limitation qualifies
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] — learned embeddings also found k<=2, suggesting continuum holds regardless
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] — the pipeline whose input features matter
- [[BEATs self-distilled discrete tokenizer achieves the highest BEANS benchmark score among bioacoustic encoders]] — candidate encoder for resolution path 1
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- if learned embeddings reveal more structure, the "one big continuum" result from raw features may understate meaningful variation
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- the choice of feature space directly affects the quality of latent-space distributional comparisons
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the unsupervised branch's results depend critically on whether raw or learned features are used
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- a third feature category between DeepSqueak raw stats (10D) and AMVOC learned embeddings (1280D); mid-complexity handcrafted option that could triangulate whether the "one big cluster" result reflects feature limitations or genuine continuum

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
