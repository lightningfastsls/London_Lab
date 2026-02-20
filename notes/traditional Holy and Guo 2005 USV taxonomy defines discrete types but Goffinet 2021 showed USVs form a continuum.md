---
description: "The paradigm shift from predefined discrete USV categories (Holy & Guo 2005) to continuous representations (Goffinet 2021) shaped our VQ-VAE approach"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# Traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum

The traditional USV classification framework from Holy & Guo (2005) defines discrete call types based on acoustic features — categories like simple sweeps, complex calls, frequency jumps, etc. This taxonomy became standard in the field and is used by tools like [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]]. However, [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] — challenging the fundamental assumption that USVs fall into discrete types. Our approach navigates this paradigm shift by using VQ-VAE to discover data-driven discrete codes that may not align with traditional categories. Rather than forcing USVs into Holy & Guo categories, we let the codebook find its own segmentation of the continuum.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- Holy & Guo (2005); Goffinet et al. (2021)

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the challenge to discrete taxonomy
- [[Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space]] -- quantifies how much of the traditional feature information the learned representation retains
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- tool based on the traditional taxonomy
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our data-driven alternative
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- unsupervised clustering already finds 100-140 types, far exceeding Holy & Guo's ~10-15, foreshadowing the continuum finding
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- the next step beyond both taxonomic and unsupervised approaches

Topics:
- [[classification]]
- [[representation-learning]]
