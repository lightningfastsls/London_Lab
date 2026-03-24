---
description: "Effective pos_weight of ~35.4 creates extreme recall bias, acceptable only in the context of the two-stage pipeline"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# class weight boosting biases toward recall at the cost of precision

The 3.0x multiplier on the raw class ratio (~11.8) produces an effective pos_weight of ~35.4 -- an extreme recall bias. The model will predict USV even with relatively weak evidence, accepting many false positives to minimize missed USVs. This is acceptable specifically because [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- users can adjust the classification threshold post-hoc to shift the precision/recall balance. However, this means the raw CNN outputs cannot be interpreted as calibrated probabilities. The 3.0x boost was an empirical choice that improved detection coverage without unacceptable precision loss at the chosen operating thresholds. As the dataset scales, the class imbalance ratio may change, potentially requiring recalibration of the boost factor.

---

Source:
- DECISIONS.md (ADR-005) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- the specific parameter choice
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- why this extreme bias is tolerable
- [[recall versus precision tradeoff in two-stage USV detection]] -- the broader tradeoff context
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- both detection stages independently bias toward recall, compounding the effect

Topics:
- [[classification]]
- [[experimental-methods]]
