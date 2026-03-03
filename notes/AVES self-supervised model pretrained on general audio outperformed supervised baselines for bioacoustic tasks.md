---
description: "HuBERT-based AVES pretrained on FSD50K/AudioSet/VGGSound beat supervised models for bioacoustic classification and detection"
type: finding
confidence: proven
conditions:
  - "pretrained on FSD50K, AudioSet, VGGSound"
  - "HuBERT architecture"
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
  - "[[classification]]"
---

# AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks

Hagiwara (ICASSP 2023) developed AVES (Animal Vocalization Encoder based on Self-Supervision), a HuBERT architecture pretrained on three general audio datasets (FSD50K, AudioSet, VGGSound). AVES outperformed supervised baselines on bioacoustic classification and detection tasks despite never seeing animal vocalizations during pretraining.

This is significant for our VQ-VAE pipeline in two ways:

1. **Potential backbone**: If we encounter training data volume limitations for our autoregressive transformer, AVES or similar SSL models could serve as a pretrained encoder whose representations we then discretize with VQ-VAE. This is an alternative to [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- instead of training our own transformer from scratch, we could leverage AVES features.

2. **Representation quality benchmark**: AVES provides continuous representations that set a performance ceiling. Our VQ-VAE discretization must retain enough information that the discrete codes remain useful, since [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]].

Notably, AVES uses continuous representations with no discrete codebook. Combining AVES with VQ-VAE remains untested, and [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] suggests the specific pretraining domain may matter less than the architecture.

**Tension (March 2026):** Muenster et al. (2025, arXiv 2504.06710) tested 15 bioacoustic models and found that [[supervised bioacoustic foundation models vastly outperform self-supervised for species-level clustering]]. The top 6 models were all supervised. This may not contradict AVES's original claim (which was about classification/detection tasks), but it suggests the advantage of self-supervised models is task-dependent — SSL may work well for classification but poorly for unsupervised clustering, where taxonomic structure from supervised training provides stronger inductive bias.

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Hagiwara (2023), ICASSP. https://arxiv.org/abs/2210.14493

Relevant Notes:
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- our current approach; AVES could be an alternative backbone
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- Sarkar 2025 finding on SSL transfer
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- AVES uses continuous representations, gap remains
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- practical method to adapt AVES for USV-specific tasks between frozen features and full retraining

Topics:
- [[bioacoustic-ssl]]
- [[classification]]
