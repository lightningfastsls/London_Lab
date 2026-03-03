---
description: "BEATs iteratively learns its own audio tokenizer through self-distillation, scoring 97.98 AUROC on BEANS — ahead of EAT, BirdMAE, AudioMAE, and AVES"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# BEATs self-distilled discrete tokenizer achieves the highest BEANS benchmark score among bioacoustic encoders

BEATs (Microsoft) uses a unique training approach: it iteratively learns both the audio encoder and a discrete tokenizer through self-distillation. Unlike HuBERT, which uses external k-means clustering to generate pseudo-labels, BEATs learns its own tokenizer that captures acoustically meaningful units. This architectural choice proves consequential on the BEANS benchmark (Schwinger et al., Aug 2025), where BEATs scored 97.98 AUROC with attentive probing, ahead of EAT (97.51), BirdMAE (97.27), AudioMAE (97.19), and AVES (97.16).

However, the probing method matters significantly. With linear probing (simpler, less expressive), BEATs dropped to 94.10 while AudioMAE fell further to 84.47 — demonstrating that transformer models need attentive probing to unlock their full representational power. This suggests that the representations learned by BEATs are rich but distributed across attention heads, and therefore a simple linear layer cannot adequately project them into task-relevant spaces.

NatureLM-audio uses BEATs as its audio encoder, validating that BEATs representations encode bioacoustically relevant features that generalize across downstream tasks. The self-distilled tokenizer approach is particularly relevant to our VQ-VAE pipeline because it demonstrates that learned discrete tokens can outperform externally-derived pseudo-labels, supporting the principle that end-to-end training of the discretization mechanism yields better representations than post-hoc quantization.

---

Source:
- cpc-vs-mae-bioacoustic-representation-learning-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[masked prediction outperforms contrastive learning for bioacoustic representation tasks]] — BEATs belongs to masked prediction family
- [[post-hoc vector quantization substantially underperforms continuous representations motivating end-to-end VQ-VAE training]] — BEATs' learned tokenizer contrasts with post-hoc approaches

Topics:
- [[bioacoustic-ssl]]
