---
description: "Spectrograms contain identity information beyond call type — DNNs can distinguish male from female emitters at 77-84 percent accuracy"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# Ivanenko et al 2020 showed DNNs achieve 77-84 percent accuracy classifying emitter sex from spectrograms

Ivanenko et al. (2020) demonstrated that deep neural networks can classify the sex of the USV emitter from spectrogram data alone, achieving 77-84% accuracy. This reveals that spectrograms contain identity information beyond call type — subtle acoustic features encode information about the vocalizer itself. This finding is relevant to our research because it suggests that if population-level metadata is available (since [[whether population-level metadata is available for context-dependent VQ-VAE analysis]]), the VQ-VAE codebook might capture emitter-related features alongside call-type features. It also raises the question of whether wild and lab mice can be distinguished at the individual spectrogram level, which connects to [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]].

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- Ivanenko et al. (2020)

Relevant Notes:
- [[whether population-level metadata is available for context-dependent VQ-VAE analysis]] -- metadata needed to leverage identity information
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- can wild/lab distinction be detected in spectrograms?
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- our CNN architecture processes the same spectrogram features that Ivanenko showed encode identity
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- VQ-VAE codebook entries might capture emitter-level features at different abstraction levels

Topics:
- [[classification-methodology]]
