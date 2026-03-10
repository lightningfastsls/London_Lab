---
description: "Negatives should be pure noise, electrical artifacts, silence, or transient cage sounds — ambiguous cases degrade training quality"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# Good negative training samples must be unambiguously not USV to prevent label noise

For the negative class, the quality criterion is unambiguity: good negatives are patches that are clearly not USVs. The researcher identifies four categories of unambiguous negatives: pure noise, electrical artifacts, silence, and transient cage sounds. This quality principle complements the diversity principle in [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] — the three sources (random chunks, inter-USV gaps, low-energy regions) ensure coverage of the negative distribution, while unambiguity ensures each individual sample is correctly labeled. Ambiguous negatives (patches where a faint USV might be present) would introduce label noise that degrades training. This is the mirror of [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] — noisy positives are valid, but noisy negatives (where the label itself is uncertain) are harmful.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- diversity of negatives (what); this note covers quality of negatives (how)
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- the mirror principle for the positive class
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- the general pattern this quality criterion applies to
- [[electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs]] -- one unambiguous negative category: constant-frequency electrical artifacts
- [[transient cage noises produce broadband vertical smears rejected by the minimum duration filter]] -- another unambiguous negative category: broadband transient cage impacts
- [[negative prototype construction is critical for few-shot detection without explicit negative annotations]] -- in few-shot settings, the negative class must be constructed implicitly since only positive examples are provided, making our explicit negative curation even more valuable
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- BootSnap uses an explicit noise class with high-quality negatives as a second-pass filter

Topics:
- [[classification]]
- [[experimental-methods]]
