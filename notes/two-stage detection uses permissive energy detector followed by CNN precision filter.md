---
description: "Architectural pattern where energy detector maximizes recall and CNN classifier maximizes precision in sequence"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# Two-stage detection uses permissive energy detector followed by CNN precision filter

The USV detection pipeline uses a two-stage architecture. The first stage is a permissive energy detector that scans the spectrogram for regions with elevated energy in the USV frequency band (25-110 kHz). This stage is tuned for high recall -- it catches most USVs but also generates many false positives. The second stage is a CNN classifier that examines each candidate and filters for precision. This separation of concerns means each stage can be optimized independently: the energy detector for sensitivity, the CNN for specificity. The pattern generalizes beyond USV detection -- since [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]]. Beyond detection, this pipeline's output is the sole data source for the downstream representation learning system: detected USVs are assembled into bouts that feed the transformer, whose hidden states are then discretized by VQ-VAE (see [[transformer-first then VQ-VAE avoids forcing premature discretization]]). Any systematic bias in detection — missed call types, noise leakage — propagates directly into the learned codebook.

---

Source:
- DECISIONS.md (ADR-003) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- the specific threshold choice
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- the CNN architecture
- [[recall versus precision tradeoff in two-stage USV detection]] -- the designed tradeoff
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- competitive positioning vs the most widely used alternative
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- detection output feeds the representation learning pipeline; detection bias propagates into VQ-VAE codebook quality
- [[DCASE class-dependent post-processing parameters improved F1 from 37 to 44 percent]] -- empirical evidence that post-processing optimization alone (without model changes) yields 7pp F1 gain, justifying our investment in hysteresis parameter tuning as the post-processing layer

Topics:
- [[detection]]
- [[classification]]
