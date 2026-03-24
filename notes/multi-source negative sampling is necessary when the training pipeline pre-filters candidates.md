---
description: "General pattern -- when candidate generation pre-filters, the model never sees the full negative distribution unless explicitly provided"
type: pattern
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# multi-source negative sampling is necessary when the training pipeline pre-filters candidates

When a detection pipeline uses a candidate generation stage (like an energy detector) to pre-filter data before classification, the classifier only sees pre-filtered examples during training. It never learns what "normal" (non-candidate) data looks like, since [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]]. The solution is multi-source negative sampling: deliberately generating negative examples from multiple sources that collectively span the full distribution of non-target data. In this project, [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] using random chunks, inter-USV gaps, and low-energy regions. This pattern generalizes to any detection pipeline with a pre-filtering stage -- the pre-filter creates a selection bias that must be explicitly counteracted in the training data.

---

Source:
- DECISIONS.md (ADR-008) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] -- the problem
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- the solution
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the pipeline creating the bias

Topics:
- [[classification]]
- [[experimental-methods]]
