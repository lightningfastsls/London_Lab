---
description: "2025 study showed HuBERT pretrained on English speech transfers equally well to primate calls and bird songs — acoustic features like spectral envelopes and temporal modulations are fundamental"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# Phylogenetic proximity to humans does not influence transfer learning effectiveness from speech models to animal vocalizations

A 2025 cross-species transfer study (arXiv 2509.04166) found that HuBERT pretrained on English speech transfers equally well to primate calls and bird songs. Phylogenetic distance from humans — which intuition would suggest matters for speech model transfer — has no measurable effect on transfer learning effectiveness.

This means the acoustic features learned by speech models (spectral envelopes, temporal modulations, harmonic structure) are truly fundamental across vocal production systems. They are not human-specific linguistic features but rather general properties of structured sound that all vocal organisms produce. Since [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]], this finding explains WHY: the features are universal, so the training domain barely matters.

For mouse USV analysis, this validates using speech-pretrained models as feature extractors without concern about the evolutionary distance between humans and mice. The relevant acoustic primitives (frequency sweeps, amplitude contours, harmonic stacking) are shared.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[speech pretrained SSL models transfer well to animal vocalizations with only marginal benefit from bioacoustic pretraining]] -- this finding provides the mechanistic explanation
- [[a generic cross-species autoencoder performs nearly as well as species-specific models suggesting shared vocalization structure]] -- same universal structure principle
- [[domain-specific MAE pretraining dramatically outperforms generic Audio-MAE for bioacoustic tasks]] — apparent tension: phylogenetic distance doesn't matter but domain-specific pretraining helps; resolution is that acoustic features transfer universally but pretraining data distribution still matters for fine-grained tasks
- [[NatureLM-audio combines BEATs encoder with Llama 3.1-8B for zero-shot bioacoustic species identification]] — cross-domain training consistent with universal acoustic features; language model adds semantic structure

Topics:
- [[bioacoustic-ssl]]
