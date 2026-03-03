---
description: "In 5-shot bioacoustic detection only positive examples are annotated -- novel negative selection strategies for constructing representative negative prototypes improved F-measure to 0.703"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[classification]]"
---

# negative prototype construction is critical for few-shot detection without explicit negative annotations

A fundamental challenge in few-shot bioacoustic detection is that only positive examples (5 annotated instances of the target sound) are provided -- there are no explicitly annotated negative examples. This means the negative prototype (representing "not the target sound") must be constructed carefully from the unlabeled portions of the recording. Novel adaptive learning with negative selection strategies (DCASE 2024) achieved F-measure 0.703, a 12.84% improvement, by constructing representative negative prototypes from background audio.

The problem is subtle but consequential. If the negative prototype is built from random background segments, it may not represent the acoustic diversity of non-target sounds -- missing quiet passages, environmental noise, or other species' calls. Conversely, if negative segments happen to include unannotated instances of the target sound (which is likely in long recordings), the negative prototype becomes contaminated and the decision boundary blurs. The DCASE 2024 solution addressed this by adaptively selecting negative segments that are maximally informative -- distant from the positive prototype in embedding space but representative of the recording's acoustic variety.

This problem maps directly to USV classification: our energy detector produces many candidates, and the "not USV" class has no curated examples. How we construct the negative prototype determines whether the classifier learns to reject noise versus merely memorizing the few positive examples. The parallel with our existing discovery is striking -- we previously found that [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]], which is essentially the same negative-class construction problem manifesting in a different framework. Therefore, negative prototype strategies from the few-shot literature could directly inform our CNN training pipeline.

---

Source:
- few-shot-learning-animal-sound-classification-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] -- the same negative-class problem
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- permissive first stage creates imbalanced negative class

Topics:
- [[classification-methodology]]
