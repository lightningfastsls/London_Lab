---
description: "Sarkar 2025 found V=50 single codebook insufficient for marmoset calls, suggesting K=64 may need RVQ or larger K"
type: finding
confidence: likely
conditions:
  - "marmoset vocalizations"
  - "post-hoc VQ on HuBERT"
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[classification]]"
---

# Single codebook with V=50 was insufficient for complex vocalization structure in discrete token experiments

Sarkar and Magimai-Doss (2025) identified a key limitation in their bioacoustic discrete token experiments: a single codebook with V=50 entries was insufficient to capture the complexity of marmoset vocalization structure. This is a notable finding because our planned VQ-VAE uses [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- a similar order of magnitude (K=64 vs V=50).

However, the comparison is not direct. Sarkar's V=50 was applied as post-hoc quantization on frozen HuBERT features, where the codebook must compress all information into a single discrete token per frame. Our architecture uses end-to-end training where the encoder can learn to organize the latent space specifically for quantization, potentially making each codebook entry more informative.

Still, this finding motivates considering:
1. **Residual Vector Quantization (RVQ)** -- multiple codebooks in sequence, each encoding the residual error from the previous, effectively expanding vocabulary exponentially
2. **Larger K** -- testing K=128 or K=256 alongside K=64
3. **Product Quantization (PQ)** -- factoring the codebook into independent sub-spaces

The comprehensive [[discrete audio token taxonomy from 2025 survey covers quantization methods beyond simple VQ]] documents the full landscape of alternatives.

---

Source:
- [[learn-vqvae-bioacoustics-state-of-art-2026-02]] (inbox)
- Sarkar & Magimai-Doss (2025), NeurIPS Workshop

Relevant Notes:
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our current choice at similar scale
- [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] -- the experiment that revealed this limitation
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- codebook adequacy may vary by layer

Topics:
- [[representation-learning]]
