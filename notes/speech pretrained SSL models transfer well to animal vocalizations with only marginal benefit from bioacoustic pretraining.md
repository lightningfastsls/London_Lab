---
description: "Sarkar 2025 found HuBERT pretrained on human speech matched animal-pretrained models for bioacoustic tasks"
type: finding
confidence: proven
conditions:
  - "HuBERT and similar SSL architectures"
  - "marmoset and dog vocalizations"
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
  - "[[classification]]"
---

# Speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining

Sarkar and Magimai-Doss (ICASSP 2025, "Comparing SSL Models for Bioacoustics") systematically compared self-supervised learning models pretrained on human speech against models pretrained specifically on animal vocalizations. The key finding: speech-pretrained models (HuBERT and others) matched animal-pretrained ones, with bioacoustic pretraining providing only marginal improvements.

This has several implications for our pipeline:

1. **Data efficiency**: We do not need a large bioacoustic corpus for pretraining. Off-the-shelf speech SSL models already capture relevant acoustic structure shared between human speech and animal vocalizations -- frequency modulation patterns, harmonic structure, temporal dynamics.

2. **Bootstrap strategy**: If our autoregressive transformer struggles with training data volume, fine-tuning a speech-pretrained HuBERT + VQ-VAE could bootstrap discrete representations without starting from scratch. [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]] demonstrated this with general audio pretraining.

3. **Shared acoustic structure**: The fact that speech representations transfer well supports the hypothesis that mammals share fundamental vocalization production mechanisms, lending credibility to approaches that originated in speech processing like [[Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128]].

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Sarkar & Magimai-Doss (2025), ICASSP. https://arxiv.org/abs/2501.05987

Relevant Notes:
- [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]] -- corroborates transfer learning from non-bioacoustic domains
- [[Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128]] -- speech-domain architecture that transfers to our work
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- our architecture that could benefit from pretrained speech features

Topics:
- [[bioacoustic-ssl]]
- [[classification]]
