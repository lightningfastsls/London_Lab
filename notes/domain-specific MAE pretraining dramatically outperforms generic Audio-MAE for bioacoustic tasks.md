---
description: "Bird-MAE improved MAP from 44.69 to 55.28 on BirdSet HSN — a 10.6 point gain — and even beat supervised Perch (41.12 MAP) on some datasets"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
---

# Domain-specific MAE pretraining dramatically outperforms generic Audio-MAE for bioacoustic tasks

Generic Audio-MAE pretrained on AudioSet performs worse than simple supervised spectrogram features on bioacoustic tasks, which is a striking result because it means the largest and most expensive foundation models can actually hurt downstream performance when applied out of domain. The solution is domain-specific pretraining: Bird-MAE pretrained on bird audio improved MAP from 44.69 to 55.28 on the BirdSet HSN dataset, a 10.6 point gain that represents a qualitative shift in usefulness.

This matters for our USV pipeline because it demonstrates that off-the-shelf foundation models cannot simply be applied to ultrasonic vocalization data. Domain-specific pretraining on recordings that share the acoustic characteristics of USVs would likely be needed to achieve meaningful performance gains. The improvement from domain-specific MAE was large enough to outperform even fully supervised models — Bird-MAE achieved 55.26 MAP on the POW dataset versus Perch's 41.12 MAP — suggesting that self-supervised pretraining on in-domain data can discover features that supervised training on labeled data misses.

The implication is therefore twofold: first, we should not expect pretrained audio models to transfer well to 300 kHz USV recordings without adaptation, and second, if we were to apply MAE-style pretraining, we would need a corpus of USV recordings rather than relying on general audio datasets like AudioSet.

---

Source:
- cpc-vs-mae-bioacoustic-representation-learning-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] — domain-specific features matter
- [[TweetyBERT self-supervised masked spectrogram prediction addresses temporal resolution limitations of speech SSL models for animal vocalizations]] — another domain-specific SSL approach, operating at 2.7ms resolution for birdsong
- [[CPC has been superseded by masked prediction and masked autoencoding for audio representation learning]] — MAE is the winning paradigm; domain-specific MAE amplifies the advantage further
- [[phylogenetic proximity to humans does not influence transfer learning effectiveness from speech models to animal vocalizations]] — generic speech models fail across species, reinforcing the case for domain-specific pretraining

Topics:
- [[bioacoustic-ssl]]
