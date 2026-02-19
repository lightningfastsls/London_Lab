---
description: "2025 survey catalogs K-means, RVQ, SVQ, GVQ, FSQ, MSRVQ, CSRVQ, PQ for audio but omits bioacoustics entirely"
type: finding
confidence: proven
conditions:
  - "speech, music, general audio domains"
  - "no bioacoustics coverage"
meta_state: current
topics:
  - "[[representation-learning]]"
---

# Discrete audio token taxonomy from 2025 survey covers quantization methods beyond simple VQ

"Discrete Audio Tokens: More Than a Survey!" (2025) provides a comprehensive taxonomy of discrete audio tokenization methods: K-means, Residual VQ (RVQ), Stacked VQ (SVQ), Gumbel VQ (GVQ), Finite Scalar Quantization (FSQ), Multi-Scale RVQ (MSRVQ), Conditional Sequential RVQ (CSRVQ), and Product Quantization (PQ). The survey covers speech, music, and general audio domains but contains no bioacoustics coverage, reinforcing [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]].

Key insights from the taxonomy relevant to our pipeline:

1. **Codebook collapse is identified as a critical challenge** across methods, validating that [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] addresses a recognized problem
2. **RVQ (Residual VQ)** uses multiple codebooks in sequence, each encoding the residual from the previous -- this could address [[single codebook with V=50 was insufficient for complex vocalization structure in discrete token experiments]] by expanding effective vocabulary exponentially
3. **Product Quantization (PQ)** factorizes the codebook into independent sub-spaces, offering another path to larger vocabularies
4. **Gumbel VQ** is documented as collapse-prone, consistent with [[Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments]]

The absence of bioacoustics from this comprehensive survey (despite covering speech, music, environmental sounds, and audio events) underscores the novelty of our work.

---

Source:
- [[learn-vqvae-bioacoustics-state-of-art-2026-02]] (inbox)
- "Discrete Audio Tokens: More Than a Survey!" (2025). https://arxiv.org/abs/2506.10274

Relevant Notes:
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- our current collapse prevention
- [[FSQ eliminates codebook collapse by construction achieving 100 percent utilization through fixed scalar quantization]] -- one method from the taxonomy
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our K=64 choice in context of taxonomy

Topics:
- [[representation-learning]]
