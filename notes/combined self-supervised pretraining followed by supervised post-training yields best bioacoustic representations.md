---
description: "OpenBEATs pretrained on 20K hours of mixed audio outperformed BEATs and even 1.2B-parameter Dasheng — data diversity in both training stages matters more than architecture"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# Combined self-supervised pretraining followed by supervised post-training yields best bioacoustic representations

Miron et al. (Aug 2025) showed that the best bioacoustic encoders combine SSL pretraining with supervised post-training on mixed bioacoustic and general audio data. OpenBEATs, trained on 20K hours spanning music, environmental, and bioacoustic audio, outperformed both BEATs and Dasheng (1.2B parameters) on cross-domain evaluation. The key finding is that data diversity matters more than model architecture choice — a smaller model trained on diverse data beats a larger model trained on narrower data.

This validates hybrid training approaches over pure SSL or pure supervised methods, but it complicates the narrative around domain-specific pretraining. While Bird-MAE showed that domain-specific MAE dramatically outperforms generic MAE, the combined training results suggest that mixing domain-specific and general audio during both pretraining stages yields even better representations. The mechanism is likely that general audio provides broad acoustic priors while bioacoustic data provides domain-relevant fine structure, and therefore the combination captures both.

The practical implication for our VQ-VAE pipeline is that a two-stage training approach — first self-supervised on a mix of USV and general audio, then supervised fine-tuning on labeled USV syllable types — would likely outperform either stage alone. However, this requires labeled USV data that we are still building through the detection and labeling pipeline.

---

Source:
- cpc-vs-mae-bioacoustic-representation-learning-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[masked prediction outperforms contrastive learning for bioacoustic representation tasks]] — training paradigm choice
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] — complicates the story (marginal gains from domain-specific SSL alone, but combined training works)

Topics:
- [[bioacoustic-ssl]]
