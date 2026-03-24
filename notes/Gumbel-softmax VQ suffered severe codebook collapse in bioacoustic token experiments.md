---
description: "Sarkar 2025 found Gumbel-softmax VQ collapsed when applied to animal vocalizations, validating standard VQ-VAE choice"
type: finding
confidence: proven
conditions:
  - "marmoset vocalizations"
  - "V=50 codebook"
  - "frozen HuBERT features"
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
---

# Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments

In the first published attempt to apply discrete audio tokens to bioacoustics, [[Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations]] and tested both standard post-hoc VQ and Gumbel-softmax VQ (GVQ). The GVQ variant suffered severe codebook collapse -- most codebook entries went unused while a small subset dominated all assignments.

This is a negative result with practical implications. Gumbel-softmax reparameterization, which uses a temperature-controlled continuous relaxation to make discrete selection differentiable, apparently fails to maintain codebook diversity in the bioacoustic setting. The collapse mechanism likely relates to the temperature schedule and the distributional properties of vocalization embeddings.

This result provides external validation for our pipeline's choice of standard VQ-VAE with explicit [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]]. It also strengthens the case for considering [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]], since FSQ achieves 100% codebook utilization by construction, avoiding both VQ-VAE collapse and GVQ collapse pathways entirely.

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Sarkar & Magimai-Doss (2025), NeurIPS Workshop. https://arxiv.org/abs/2511.10190

Relevant Notes:
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- our countermeasures
- [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] -- alternative that avoids collapse by design
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our codebook at similar scale (K=64 vs V=50)

Topics:
- [[unsupervised-usv-discovery]]
