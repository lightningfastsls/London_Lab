---
description: "3.0x multiplier on top of raw class ratio (~11.8) yields ~35.4 effective pos_weight, biasing strongly toward recall"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# 3x class weight boost compensates for USV class imbalance in CNN training

USV candidates are heavily imbalanced -- noise/non-USV samples substantially outnumber true USVs. Without class weighting, the model learns to always predict "not USV." The base pos_weight is computed from class ratios (~11.8 raw), with an additional 3.0x multiplier applied on top, giving a final effective pos_weight of ~35.4. This extreme recall bias is acceptable because [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the pipeline can tolerate some false positives from the CNN since users can adjust the classification threshold post-hoc. However, since [[class weight boosting biases toward recall at the cost of precision]], the tradeoff should be monitored as the dataset scales.

---

Source:
- DECISIONS.md (ADR-005) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- pipeline context making recall bias acceptable
- [[class weight boosting biases toward recall at the cost of precision]] -- the explicit tradeoff
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- the model being trained

Topics:
- [[classification]]
- [[experimental-methods]]
