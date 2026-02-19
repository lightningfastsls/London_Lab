---
description: "VQ-VAE codebook stability requires all four mechanisms simultaneously -- removing any one risks degenerate codebooks"
type: method
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization

VQ-VAE training is prone to codebook collapse -- a failure mode where most codebook entries go unused and all inputs map to a few codes. Preventing collapse requires four mechanisms working simultaneously: (1) EMA (Exponential Moving Average) updates with decay=0.99 for smooth codebook evolution; (2) dead code reset with threshold=2.0 to replace unused entries; (3) k-means initialization on encoder outputs for meaningful starting positions; (4) L2 normalization on encoder outputs for stable distance computation. Each mechanism addresses a different failure mode, and removing any single one risks collapse. The commitment weight (beta=0.25) balances encoder commitment to codebook entries against codebook adaptation to encoder outputs. If these mechanisms prove insufficient, [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] remains an open alternative.

### ROADMAP Context

ROADMAP provides implementation specifics for each mechanism: k-means initialization uses sklearn KMeans with n_init=10 applied to ~5000 encoder outputs sampled at the start of training. L2 normalization is applied via F.normalize(z, dim=-1). The dead code threshold is 2.0 (entries with EMA count below this are reset). An optional fifth mechanism — entropy regularization — encourages uniform codebook utilization by penalizing low-entropy code distributions. If ALL four (or five) defenses fail and collapse persists, [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] (Mentzer et al., ICLR 2024) achieves 100% utilization by design and is the designated fallback.

---

Source:
- DECISIONS.md (ADR-007) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the architecture using this VQ-VAE
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- the codebook being stabilized
- [[whether FSQ provides more stable discretization than VQ-VAE for USV codebook learning]] -- fallback approach
- [[separating representation learning from discretization enables richer feature discovery]] -- collapse prevention is specifically needed because discretization is a separate phase operating on already-learned representations

Topics:
- [[classification]]
