---
description: "MUPET's unsupervised approach discovers 100-140 data-driven types — a philosophical predecessor to our VQ-VAE but using simpler methods"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[representation-learning]]"
---

# MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types

MUPET takes an unsupervised approach to USV classification: it uses a gammatone filterbank for feature extraction followed by k-means clustering, typically discovering 100-140 data-driven USV types. This is philosophically closer to our VQ-VAE approach than VocalMat's supervised classification — both MUPET and our pipeline let the data determine categories rather than imposing predefined taxonomies. However, MUPET's method is simpler: k-means on fixed features versus our learnable VQ-VAE codebook on transformer-learned representations. The 100-140 cluster count from MUPET provides an empirical precedent for the order of magnitude of USV variety, contextualizing our [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] at K=64.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)

Relevant Notes:
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our K=64 vs MUPET's 100-140 types
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- supervised alternative
- [[separating representation learning from discretization enables richer feature discovery]] -- our VQ-VAE learns representations MUPET's k-means cannot
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- MUPET's 100-140 types already challenge the traditional ~10-15 categories
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- MUPET's high cluster count foreshadows Goffinet's continuum finding
- [[STSG spectrogram token skip-gram achieved only 0.559 AUC versus 0.810 for transfer learning on bioacoustic classification]] -- another K-means-based approach that dramatically underperformed, suggesting fixed clustering of spectrograms has inherent limitations

Topics:
- [[classification]]
