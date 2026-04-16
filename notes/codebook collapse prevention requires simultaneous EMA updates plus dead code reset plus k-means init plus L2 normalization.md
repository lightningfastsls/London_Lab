---
description: "VQ-VAE codebook stability requires all four mechanisms simultaneously -- removing any one risks degenerate codebooks"
type: method
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[representation-learning]]"
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
- [[Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments]] -- external validation: GVQ collapsed in Sarkar 2025, confirming collapse is a real risk requiring our multi-mechanism prevention
- [[FSQ eliminates codebook collapse by construction achieving 100 percent utilization through fixed scalar quantization]] -- the alternative that eliminates collapse by design (Mentzer ICLR 2024)
- [[discrete audio token taxonomy from 2025 survey covers quantization methods beyond simple VQ]] -- 2025 survey identifies codebook collapse as a critical challenge across quantization methods
- [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] -- parallel stability mechanism: LoRA prevents catastrophic weight drift through low-rank constraint, while these four mechanisms prevent codebook drift through EMA/reset/init/normalization; both address the same fundamental challenge of stable adaptation without degeneration
- [[AMVOC semi-supervised retraining combines reconstruction KL divergence and pairwise constraint losses with uncertainty-based annotation priority]] -- AMVOC addresses the stable-update challenge differently: weighted multi-loss (BCE 0.5, KL 0.2, pairwise 0.001) balances competing objectives during iterative refinement, a complementary strategy to the four-mechanism defense here

Topics:
- [[representation-learning]]
