---
description: "AVA's VAE latent space retained most traditional feature information, establishing a baseline for VQ-VAE information retention"
type: baseline
confidence: proven
conditions:
  - "31440 mouse USV syllables"
  - "VAE architecture"
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space

Goffinet et al. (eLife 2021) trained their AVA (Animal Vocalization Analysis) VAE on 31,440 mouse USV syllables and measured how much traditional feature information was retained in the learned latent space. The result: 64-95% of traditional feature information was captured, establishing that unsupervised representation learning can preserve most of the information that domain experts manually engineered.

This quantitative result provides an important baseline for our VQ-VAE pipeline. Our discrete codebook ([[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]]) must retain at minimum a comparable fraction of information. The 64-95% range sets expectations: some information loss is inherent in compression (whether continuous VAE or discrete VQ-VAE), but the majority should be preserved.

The result also strengthens the argument from [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]]: the VAE's continuous latent space captures real acoustic structure, not noise. Discretizing this space via VQ-VAE is scientifically motivated because it imposes structure on a space already known to carry meaningful information.

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Goffinet et al. (2021), eLife -- AVA tool

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the parent finding
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our codebook must retain comparable information
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] -- quantifying the information loss from discretization
- [[Garrobe Fonollosa 2024 showed VAE plus temporal convolutional network achieved AUC over 0.9 for sperm whale click classification]] -- VAE information retention demonstrated in another species (cetaceans)
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- the 64-95% retention quantifies how well learned features capture what the traditional taxonomy describes

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
