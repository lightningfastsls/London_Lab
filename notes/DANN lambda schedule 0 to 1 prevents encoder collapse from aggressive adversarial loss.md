---
description: "If the gradient-reversal coefficient is full-strength from epoch 1, the encoder finds trivial cage-invariant features that also kill class signal — annealing from 0 to 1 lets the class head establish useful features before the adversary applies pressure"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[model-adaptation]]"
  - "[[training-methodology]]"
---

# DANN lambda schedule 0 to 1 prevents encoder collapse from aggressive adversarial loss

DANN's gradient-reversal layer multiplies the domain-head gradient by −λ before it reaches the shared encoder. If λ is held constant at 1.0 from epoch 1, the encoder receives full-strength negative pressure to confuse the domain head before it has had any opportunity to learn class-discriminative features. The encoder can satisfy this pressure trivially — by collapsing to a representation that contains very little information at all, neither class nor domain. The downstream class head then has nothing to work with, and macro F1 craters compared to the no-DANN baseline.

Ganin's 2015 fix is to schedule λ from 0 to 1 over training, typically with the sigmoid schedule λ(p) = 2 / (1 + exp(−10p)) − 1 where p = current_epoch / total_epochs. Early in training, the encoder receives almost no adversarial pressure and learns standard discriminative features just as the class head asks. As training proceeds, λ ramps up smoothly and the encoder is progressively pushed toward invariance — but only after class-discriminative structure has been established that the encoder is unwilling to throw away cheaply. The net effect is that the encoder finds a representation that retains class signal while *also* being cage-invariant, rather than one that lacks both.

The Phase 1.3 plan formalizes the detection criterion for collapse: syllable macro F1 vs the v1 baseline should not drop by more than 0.05. A larger drop is evidence the encoder collapsed and λ must be re-scheduled — typical fixes are reducing the maximum λ (e.g., to 0.5), starting the ramp later (e.g., p shifted by 0.2), or using a more conservative schedule. The bound is empirical, not theoretical, but it provides a falsifiable check: if the criterion fails, the methodology is wrong and the fix is mechanical, not philosophical.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[DANN gradient-reversal enforces invariance without per-batch domain matching]] — the architecture this schedule supports
- [[falsifiable cleaning gates with numeric thresholds beat vibes-based judgment]] — same pattern: collapse criterion is a numeric falsifiable threshold

Topics:
- [[model-adaptation]]
- [[training-methodology]]
