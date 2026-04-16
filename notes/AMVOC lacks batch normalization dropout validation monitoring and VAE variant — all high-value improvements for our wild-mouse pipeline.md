---
description: "actionable gap analysis from 2022 paper — each missing element maps to a concrete design decision for our VQ-VAE autoencoder, with wild-mouse variability making several gaps more consequential"
type: decision
confidence: likely
created: 2026-04-15
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
---

# AMVOC lacks batch normalization dropout validation monitoring and VAE variant — all high-value improvements for our wild-mouse pipeline

AMVOC's convolutional autoencoder (Stoumpou et al. 2022) is effective but architecturally minimal. Identifying what it lacks — and why each gap matters more for our wild-mouse data — creates an actionable design checklist for our VQ-VAE:

**1. No batch normalization.** Modern autoencoders use batch norm between conv layers to stabilize training and enable faster convergence. For our pipeline, batch norm would allow training for more than 2 epochs without instability, since [[AMVOC trains for only 2 epochs deliberately because the undercomplete bottleneck acts as implicit regularizer]].

**2. No dropout.** Combined with the undercomplete bottleneck, the network relies solely on compression for regularization. Dropout would provide additional regularization for more variable data, especially important for our wild-mouse recordings where within-species variability is higher than lab strains.

**3. No validation set monitoring.** AMVOC uses a fixed 2-epoch training with 80/20 split but no early stopping based on validation loss. This works for their 22K lab-strain syllables but may undertrain or overtrain on different distributions.

**4. No VAE variant.** The standard autoencoder's latent space has no smoothness constraints — points may be arranged discontinuously, with gaps between clusters that produce meaningless interpolations. A VAE with KL regularization would yield a smoother latent space, and since [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]], adding VQ would provide the discrete codebook we need for information-theoretic analysis.

**5. No UMAP.** AMVOC uses t-SNE for visualization only and clusters in PCA space. UMAP better preserves global structure and is the modern standard for the embed-reduce-cluster paradigm.

**6. Shallow architecture (3 conv layers).** With wild-mouse USVs potentially exhibiting more complex spectrotemporal patterns than lab strains, deeper architectures (4-5 conv layers) may capture hierarchical features that 3 layers cannot.

The authors themselves acknowledge several of these gaps — they mention VAEs in their introduction and note the potential for improved architectures. Each item above maps to a concrete parameter or architectural choice in our pipeline design.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] — our VQ-VAE would address gap #4
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] — why gaps matter more for wild-mouse data
- [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]] — addresses gap #5
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] — our CNN already implements gap #1 (BatchNorm) and uses Dropout(0.5) — concrete evidence that the improvements AMVOC lacks are already standard practice in our codebase
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] — directly addresses gap #4 (VAE variant): our VQ-VAE pipeline includes four collapse-prevention mechanisms that AMVOC's standard autoencoder doesn't need but a VAE variant would require
- [[DeepSqueak v3.1 added VAE-based contour-invariant clustering as upgrade over k-means for continuous USV variation]] — gap #4 (no VAE) has already been partially closed in the ecosystem: DeepSqueak's response to continuous USV variation was to add a VAE clustering mode, validating that the VAE extension of AMVOC's autoencoder is the correct direction
- [[perceptual loss outperforms pixel-level MSE for autoencoder spectrogram representation learning]] — a seventh gap beyond the six listed: AMVOC uses BCE on raw pixels rather than perceptual loss, which Best et al 2023 showed produces better-clustering embeddings across species

Topics:
- [[unsupervised-usv-discovery]]
