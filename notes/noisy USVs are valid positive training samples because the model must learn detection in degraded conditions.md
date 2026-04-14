---
description: "USVs embedded in background noise are labeled positive — the CNN must generalize to both clean and noisy conditions"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# Noisy USVs are valid positive training samples because the model must learn detection in degraded conditions

A USV does not stop being a USV because there is background noise around it. When labeling training data, USVs embedded in noisy regions are still labeled positive. The ideal positive is a bright coherent frequency trace against a dark background, but noisy positives are equally valid and important for training robustness. If the model only trains on clean examples, it will fail on noisy recordings — which are common in real-world conditions. This labeling decision directly affects training data composition: the positive class includes a spectrum from pristine to heavily degraded signals.

This complements the negative sampling strategy where [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] — together they define the full labeling policy: positives include noise-embedded USVs, negatives must be unambiguously non-USV.

The need for noise-robust training is grounded in the recording environment: since [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]], noisy recordings are the norm rather than the exception.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- defines the negative side of the same labeling policy
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- the quality criterion for the other class
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- the failure mode this labeling policy is designed to address
- [[PCEN normalization is more robust than log-mel spectrograms for few-shot bioacoustic scenarios]] -- PCEN's adaptive gain control could help the model better distinguish USVs from noise in these degraded samples by normalizing per-channel energy before classification
- [[BirdVoxDetect PCEN reduced false alarm rates 50x near-field and 5x far-field]] -- empirical evidence that adaptive normalization dramatically reduces false alarms from variable-energy signals, directly relevant because noisy USVs exhibit the same distance-dependent energy variation that drove the 50x improvement

Topics:
- [[classification]]
- [[experimental-methods]]
