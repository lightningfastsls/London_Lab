---
description: "Negative samples from random chunks, inter-USV gaps, and low-energy regions each teach a different aspect of 'not USV'"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# Three-source negative sampling teaches the CNN the full spectrum of non-USV audio

The CNN was originally trained only on energy-detector candidates, since [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]]. The fix is generating negative training samples from three distinct sources: (1) random chunks -- random time slices from recordings with no energy filtering, teaching the model about normal background noise; (2) inter-USV gaps -- audio between known USV detections, teaching about near-USV silence; (3) low-energy regions -- deliberately quiet segments, preventing false triggers on quiet artifacts. Concrete examples of unambiguous negatives include pure noise, electrical artifacts, silence, and transient cage sounds — since [[good negative training samples must be unambiguously not USV to prevent label noise]]. Together, these three sources teach the model the full spectrum of "not USV" audio. This pattern generalizes: since [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]].

### ROADMAP Context

ROADMAP Phase 9 specifies the fraction allocation across the three negative sources: 50% random chunks, 30% inter-USV gaps, 20% low-energy regions. The overall negatives-to-positives ratio is controlled by a configurable neg_ratio parameter (default 1.0, meaning one negative per positive). The DatasetAssembler class orchestrates negative generation as part of the unified assembly pipeline, so all three source types are produced in a single coordinated pass rather than separately. See [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] for how this dataset feeds the broader iterative improvement loop.

---

Source:
- DECISIONS.md (ADR-008) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] -- the diagnostic finding
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- general pattern
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- the model being trained
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- quality criterion (unambiguity) that complements this diversity criterion (three sources)

Topics:
- [[classification]]
- [[experimental-methods]]
