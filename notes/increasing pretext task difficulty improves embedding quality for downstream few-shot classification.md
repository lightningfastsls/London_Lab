---
description: "Perch 2.0 found that harder classification problems during training produce better-separable embedding spaces — suggesting curriculum design matters as much as architecture choice"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# increasing pretext task difficulty improves embedding quality for downstream few-shot classification

Perch 2.0 (Google DeepMind, 2025) reported a counterintuitive finding: "increasing the difficulty of the classification problem increases overall quality of the embedding model." Harder pretext tasks during supervised pretraining — meaning more fine-grained distinctions, more species, more confusable classes — produce embedding spaces where novel classes are better separated.

This has direct implications for USV representation learning. If training our transformer on an easy binary task (USV vs noise) produces embeddings that are poorly structured for downstream syllable classification, then we should consider making the pretraining task harder. This could mean distinguishing closely related syllable subtypes, or even using contrastive objectives that push apart similar-but-different calls.

The finding echoes curriculum learning research showing that difficult examples drive representation quality more than easy ones. It also provides a potential explanation for why [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] — the training task is too easy, producing representations that don't capture the fine-grained structure needed for classification. A harder pretraining objective on more diverse data would force the model to learn more discriminative features.

Combined with the finding that [[combined self-supervised pretraining followed by supervised post-training yields best bioacoustic representations]], this suggests a two-phase approach: first pretrain with a challenging SSL objective on diverse USV data, then fine-tune with a demanding supervised classification task on labeled syllable types.

---

Source:
- few-shot-learning-animal-sound-classification-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[combined self-supervised pretraining followed by supervised post-training yields best bioacoustic representations]] — training strategy matters
- [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] — our current training task may be too easy

Topics:
- [[classification-methodology]]
- [[bioacoustic-ssl]]
