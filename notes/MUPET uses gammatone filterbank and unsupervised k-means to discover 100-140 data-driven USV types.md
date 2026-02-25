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

MUPET's choice of gammatone filterbank was independently validated by BootSnap: [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]]. Two different tools from different research groups converged on gammatone as the preferred spectral representation for USV analysis, suggesting the auditory-model-inspired frequency spacing genuinely captures USV features better than uniform STFT bins. This convergence strengthens the case for considering gammatone spectrograms in our classification pipeline alongside our current STFT approach.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23) -- BootSnap evidence for gammatone > STFT

Relevant Notes:
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our K=64 vs MUPET's 100-140 types
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- supervised alternative
- [[separating representation learning from discretization enables richer feature discovery]] -- our VQ-VAE learns representations MUPET's k-means cannot
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- MUPET's 100-140 types already challenge the traditional ~10-15 categories
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- MUPET's high cluster count foreshadows Goffinet's continuum finding
- [[STSG spectrogram token skip-gram achieved only 0.559 AUC versus 0.810 for transfer learning on bioacoustic classification]] -- another K-means-based approach that dramatically underperformed, suggesting fixed clustering of spectrograms has inherent limitations
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- independent validation of MUPET's gammatone choice
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- the tool providing the gammatone validation

Topics:
- [[classification]]
