---
description: "When two classes have ~74 and ~136 training examples in a 12-class taxonomy, the choices are keep them and oversample (faithful but risky), merge them into a parent (loses granularity), or drop them entirely (incomplete taxonomy) — no option is free"
type: tension
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[training-methodology]]"
---

# Minority class strategy trades taxonomy faithfulness against trainability

The Grimsley 2011 mouse USV taxonomy has 12 syllable types because that's how the Grimsley group decided to carve the acoustic space — the 12 categories reflect biological distinctions the researchers found meaningful. The VocalMat training distribution honors this taxonomy, but unevenly: Multi-steps gets 74 examples and Reverse-Chevron gets 136, while top classes have 1,500–2,000. For a downstream classifier, this is a forced choice with no clean answer.

**Pole A — Keep all 12 classes with class-weighted CE + focal loss + oversampling.** Most faithful to the taxonomy; preserves the biological distinctions that justified Grimsley's framework in the first place. Risk: ~59 Multi-steps examples in train split may be insufficient signal for ResNet-18 to learn even with all three corrective techniques. Per-class precision may fall below 0.20 — the model effectively never predicts the minority class, producing "everything-becomes-Step-up" failure mode. The 12-class label space is intact in name but a 10-class model in practice.

**Pole B — Collapse step-family.** Merge Multi-steps + Two-steps + Step-up + Step-down into one "Multi-step family" class → 9-class model. Loses biological granularity Grimsley designed for. Risk: downstream consumers expecting Grimsley-12 outputs see an incompatible 9-class output and have to map; the merged class may also be too acoustically heterogeneous to be a clean target. The 9-class model produces no minority-class failures but throws away signal the taxonomy was designed to preserve.

**Pole C — Drop Multi-steps + Reverse-Chevron entirely.** 10-class model that ignores rare-but-real syllable types. Risk: downstream usage on real recordings hits Multi-steps occasionally; the model produces incorrect predictions for them or assigns them to the nearest class. The taxonomy is incomplete by design, which is honest, but reduces usefulness.

### When Each Pole Wins

| Situation | Pick |
|---|---|
| Downstream user explicitly needs full Grimsley | A |
| Per-class precision on rare classes < 0.20 in pole A | B (if step-family) or C (otherwise) |
| Acoustic homogeneity of merged class is high | B |
| Researcher cares about precision on rare types | A or C, never B |

### Dissolution Attempts

The plan adopts pole A as the v1 baseline and commits to revisit if pole A fails empirically. This is the most defensible sequencing — A is the strongest commitment to the taxonomy, and the trigger for falling back to B or C is concrete (per-class precision < 0.20). The decision isn't avoided, it's *deferred* until evidence supports one pole over another.

The tension does not dissolve under inspection. With 74 examples, ResNet-18 may simply lack the capacity to learn Multi-steps as a separate class regardless of loss tricks. If pole A fails, the irreducible trade is taxonomy faithfulness vs honest per-class performance, and the user has to commit. The plan's choice is to make that commitment on evidence rather than upfront.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[class imbalance 24x breaks naive cross-entropy and requires class-weighted plus focal plus oversampling]] — pole A's main defensive mechanism
- [[VocalMat test set quality dual-rater consensus exceeds training set quality single-rater]] — context: minority classes are also where label noise concentrates

Topics:
- [[classification-methodology]]
- [[training-methodology]]
