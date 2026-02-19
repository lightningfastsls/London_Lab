---
description: "Selection bias in training data -- 0.997 mean probability on random audio chunks revealed the model had never seen non-candidate audio"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio

When the CNN was trained exclusively on candidates from the energy detector, it achieved 0.997 mean probability on random audio chunks -- classifying essentially everything as USV. The root cause was selection bias: the energy detector pre-filters audio, so the model only ever saw audio that had already been flagged as potentially interesting. It never learned what "normal" audio looks like and therefore had no basis for rejecting it. This was the most consequential preprocessing insight — not a spectrogram parameter issue but a pipeline design issue where upstream filtering silently shaped downstream training data. This diagnostic finding led to the development of [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]]. The insight generalizes: since [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]], any detection pipeline with a pre-filtering stage will face this selection bias issue.

---

Source:
- DECISIONS.md (ADR-008) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- the solution
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- the general pattern
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the pipeline that created the bias

Topics:
- [[classification]]
