---
description: "Convolutional autoencoders with perceptual loss achieved NMI 0.5-0.88 across 8 datasets and 6 species; 46-97% cluster purity; established UMAP+HDBSCAN pipeline"
type: finding
confidence: proven
conditions:
  - "8 datasets"
  - "6 species: Bengalese finch, Cassin vireo, California thrasher, black-headed grosbeak, humpback whale, bottlenose dolphin"
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species

Best, Paris, Glotin, and Marxer (PLOS ONE 2023, "Deep Audio Embeddings for Vocalisation Clustering") trained convolutional autoencoders (NOT variational, NOT VQ) with perceptual loss for bioacoustic repertoire discovery. Key findings across 8 datasets spanning 6 species:

1. **NMI scores 0.5-0.88** across datasets (updated range from full study), with **46-97% of clusters meeting a 90% purity threshold**, demonstrating learned representations consistently outperform handcrafted features for clustering
2. **Generic (cross-species) autoencoders matched species-specific ones** -- a single model trained across species performed comparably to specialized models, suggesting shared acoustic structure (see [[a generic cross-species autoencoder performs nearly as well as species-specific models suggesting shared vocalization structure]])
3. **UMAP to 8 dimensions before HDBSCAN** -- the specific pipeline used 256-dim bottleneck embeddings reduced via UMAP, establishing what became the dominant paradigm (see [[UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations]])
4. Published as a **Python package** for the bioacoustics community

This work is architecturally distinct from our VQ-VAE approach: it uses continuous embeddings without any discrete codebook, and the autoencoder is convolutional (not variational). The cross-species generalization finding is relevant to our pipeline -- if generic embeddings work, our VQ-VAE codebook might similarly generalize beyond mouse USVs. However, the lack of discretization means this approach cannot produce the symbolic sequences needed for [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] and other information-theoretic analyses.

The perceptual loss used here (matching intermediate representations rather than raw pixel/spectrogram values) is worth noting since our pipeline debates [[MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction]].

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Best et al. (2023), PLOS ONE. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283396

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- continuous VAE approach for mice
- [[Garrobe Fonollosa 2024 showed VAE plus temporal convolutional network achieved AUC over 0.9 for sperm whale click classification]] -- another continuous VAE approach for bioacoustics, extending to cetaceans
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- Best 2023 uses continuous AE (not VQ-VAE), appearing in the gap analysis table
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- Best 2023 uses continuous AE, not VQ-VAE
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- another unsupervised clustering approach

Topics:
- [[unsupervised-usv-discovery]]
- [[classification]]
