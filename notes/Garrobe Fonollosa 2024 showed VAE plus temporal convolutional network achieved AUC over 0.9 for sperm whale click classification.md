---
description: "VAE for unsupervised feature extraction from cetacean recordings outperformed handcrafted features on weakly labeled data"
type: finding
confidence: proven
conditions:
  - "sperm whale click recordings"
  - "4-minute recording segments"
  - "weakly labeled dataset"
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# Garrobe Fonollosa 2024 showed VAE plus temporal convolutional network achieved AUC over 0.9 for sperm whale click classification

Garrobe Fonollosa et al. (arXiv October 2024, "Temporal Feature Learning in Weakly Labelled Bioacoustic Cetacean Datasets") combined a VAE with a Temporal Convolutional Network (TCN) for sperm whale click classification from 4-minute recordings. The VAE was used for unsupervised feature extraction, and the TCN leveraged temporal patterns across the learned features.

Key results:
- AUC > 0.9 on sperm whale click detection/classification
- Outperformed handcrafted acoustic features
- Worked with weakly labeled data (only recording-level labels, not click-level)

This extends the VAE-for-bioacoustics literature beyond mouse USVs ([[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]]) to cetaceans. Like Goffinet's AVA tool, the VAE here produces continuous representations without discretization. The combination with a temporal model (TCN) parallels our [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]], though TCN uses dilated convolutions rather than self-attention.

The weak labeling setup is relevant -- our pipeline also deals with recording-level rather than fine-grained labels in early stages, and learned representations could help bridge this gap.

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Garrobe Fonollosa et al. (2024), arXiv. https://arxiv.org/abs/2410.17006

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- VAE applied to USVs
- [[Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space]] -- quantitative baseline for VAE information retention in another species (mice vs whales)
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] -- another learned embedding approach across multiple species
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- Garrobe Fonollosa is a continuous VAE approach, appearing in the gap analysis
- [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] -- our temporal modeling approach

Topics:
- [[unsupervised-usv-discovery]]
- [[classification]]
