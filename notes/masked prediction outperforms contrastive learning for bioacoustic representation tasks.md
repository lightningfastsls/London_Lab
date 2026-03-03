---
description: "BEATs and HuBERT consistently beat wav2vec2 on BEANS and multi-species benchmarks — predicting discrete pseudo-labels is more effective than contrastive future prediction"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# Masked prediction outperforms contrastive learning for bioacoustic representation tasks

BEATs (masked prediction with self-distilled tokenizer) scores 97.98 AUROC on BEANS versus wav2vec2 (contrastive), which consistently underperforms HuBERT and AVES across marmoset, marine mammal, and dog datasets. The key architectural difference is that contrastive loss requires careful negative sampling, while masked prediction avoids this entirely by predicting discrete pseudo-labels derived from k-means clustering.

In the direct comparison by Sarkar and Magimai-Doss (2025), HuBERT achieved 64.35% UAR on marmoset vocalizations and 94.18% on marine mammal calls, while wav2vec2 scored 62.40% and 94.25% respectively. The gap is consistent but not enormous, which suggests the advantage comes from the training paradigm's stability rather than a fundamental representational superiority. Masked prediction simply has fewer failure modes because it does not need to construct informative negatives.

CPC (van den Oord 2018) was foundational but has been superseded by two waves — masked prediction (2020-2022) then masked autoencoding (2022-2024). The "contrastive" component increasingly appears as a regularizer in hybrid approaches rather than the primary training objective. This trajectory matters for our pipeline because it validates the masked prediction family of approaches, which our VQ-VAE transformer architecture belongs to, over contrastive alternatives that would require additional engineering effort around negative sampling strategies.

---

Source:
- cpc-vs-mae-bioacoustic-representation-learning-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] — confirms masked prediction family dominates
- [[AVES self-supervised model pretrained on general audio outperformed supervised baselines for bioacoustic tasks]] — AVES uses HuBERT (masked prediction)

Topics:
- [[bioacoustic-ssl]]
