---
description: "Mentzer et al ICLR 2024 showed FSQ matches VQ-VAE performance with simpler implementation and no collapse risk"
type: finding
confidence: proven
conditions:
  - "speech codecs at 400-700 bps"
  - "not tested on bioacoustics"
meta_state: current
topics:
  - "[[representation-learning]]"
---

# FSQ eliminates codebook collapse by construction achieving 100 percent utilization through fixed scalar quantization

Mentzer et al. (ICLR 2024) introduced Finite Scalar Quantization (FSQ), which eliminates the codebook learning problem entirely. Instead of learning codebook entries via nearest-neighbor lookup (VQ-VAE) or Gumbel-softmax relaxation (GVQ), FSQ rounds each scalar dimension of the representation to a fixed set of discrete levels. The discrete vocabulary emerges from combinations of quantized dimensions.

Key results:
- **100% codebook utilization by construction** -- every combination of scalar levels is reachable, making collapse impossible
- **Competitive with VQ-VAE** on standard benchmarks with substantially simpler implementation
- Applied to **speech codecs at 400-700 bps**, demonstrating viability for audio domains
- **NOT applied to bioacoustics** yet

This strengthens the case for [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] by providing concrete evidence rather than just theoretical appeal. Given that [[Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments]] and our own pipeline requires multiple mechanisms for [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]], FSQ's collapse-free design is increasingly attractive.

The tradeoff remains: FSQ vocabulary size is the product of per-dimension levels (e.g., 5 levels x 4 dimensions = 625 codes) rather than a directly set K parameter, which may affect interpretability compared to [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]].

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Mentzer et al. (2024), ICLR. https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf

Relevant Notes:
- [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] -- the open question this evidence informs
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- the complexity FSQ eliminates
- [[Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments]] -- another quantization approach that failed
- [[discrete audio token taxonomy from 2025 survey covers quantization methods beyond simple VQ]] -- FSQ is one entry in a comprehensive taxonomy that catalogs 8+ quantization methods
- [[whether flow matching could replace VQ-VAE for unsupervised USV representation learning]] -- FSQ could serve as a post-hoc quantizer for flow matching's continuous trajectories, combining flow matching's training stability with discrete interpretability

Topics:
- [[representation-learning]]
