---
description: "Sarkar 2025 showed post-hoc VQ on HuBERT achieves 35% vs 49% UAR, proving discretization after feature learning loses information"
type: finding
confidence: proven
conditions:
  - "marmoset vocalizations"
  - "V=50 codebook"
  - "frozen HuBERT features"
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[classification]]"
---

# Post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training

Empirical evidence from [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] demonstrates that applying vector quantization as a post-processing step on frozen continuous representations loses substantial discriminative information. On the Bosshard marmoset dataset, post-hoc VQ tokens achieved 35% UAR compared to 49% for linear probing of the same continuous HuBERT embeddings -- a 14 percentage point gap.

This result has direct implications for our VQ-VAE pipeline. It empirically validates the architectural choice of [[separating representation learning from discretization enables richer feature discovery]] while also suggesting that the discretization step itself must be jointly trained rather than applied post-hoc. In our architecture, [[transformer-first then VQ-VAE avoids forcing premature discretization]] by first learning rich continuous representations, but the VQ-VAE phase still trains encoder-codebook-decoder end-to-end on those representations rather than just quantizing frozen features.

The performance gap also raises the question of whether our K=64 codebook ([[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]]) combined with end-to-end training can close the gap that post-hoc VQ could not.

---

Source:
- [[learn-vqvae-bioacoustics-state-of-art-2026-02]] (inbox)
- Sarkar & Magimai-Doss (2025), NeurIPS Workshop

Relevant Notes:
- [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] -- the primary evidence
- [[separating representation learning from discretization enables richer feature discovery]] -- our architecture's principle
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- our two-phase approach that avoids the post-hoc trap

Topics:
- [[representation-learning]]
