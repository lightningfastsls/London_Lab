---
description: "First published work applying discrete audio tokens to bioacoustics, using post-hoc VQ on HuBERT not end-to-end VQ-VAE"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# Sarkar and Magimai-Doss 2025 applied post-hoc VQ to frozen HuBERT embeddings for marmoset and dog vocalizations

Sarkar and Magimai-Doss (NeurIPS 2025 Workshop, "Towards Leveraging Sequential Structure in Animal Vocalizations") represent the closest published work to applying VQ-VAE in bioacoustics, but their approach is fundamentally different from end-to-end VQ-VAE training. They applied post-hoc vector quantization and Gumbel-softmax VQ to frozen HuBERT embeddings with vocabulary size V=50, testing on marmoset datasets (3 datasets, up to 72K samples) and dogs (8K samples).

Key results: VQ tokens could discriminate call types and individual callers, but substantially underperformed linear probing baselines. On the Bosshard marmoset dataset, VQ tokens achieved 35% UAR versus 49% for linear probing -- a significant gap showing that discretization via post-hoc quantization loses substantial information compared to continuous representations.

This work narrows but does not close [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]], since post-hoc VQ on frozen features is architecturally distinct from end-to-end VQ-VAE where encoder, codebook, and decoder are jointly optimized. The performance gap also motivates end-to-end approaches: [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]].

Separately, [[GmSLM is a London-Omer collaboration applying self-supervised speech models to marmoset vocalizations|GmSLM]] (Sternberg et al. 2025, EMNLP Findings), co-authored by Mickey London and David Omer, also applies self-supervised speech models to marmoset vocalizations. This is closer to Sarkar's approach but with a direct collaboration link to our lab, and establishes that the Omer lab is actively exploring SSL approaches alongside the ridge vectorization technique from Oren 2024.

---

Source:
- learn-vqvae-bioacoustics-state-of-art-2026-02 (archived to archive/inbox/)
- Sarkar & Magimai-Doss, "Towards Leveraging Sequential Structure in Animal Vocalizations", NeurIPS 2025 Workshop. https://arxiv.org/abs/2511.10190
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, April 2026) — GmSLM collaboration context

Relevant Notes:
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- the gap this work narrows but does not close
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- updated gap analysis table explicitly positions Sarkar as closest approach
- [[Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128]] -- end-to-end approach our project follows
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our K=64 is comparable to their V=50
- [[Gumbel-softmax VQ suffered severe codebook collapse in bioacoustic token experiments]] -- their negative result with GVQ
- [[GmSLM is a London-Omer collaboration applying self-supervised speech models to marmoset vocalizations]] -- London-Omer SSL approach with direct lab collaboration link

Topics:
- [[unsupervised-usv-discovery]]
- [[classification]]
