---
description: "Naive cross-entropy with majority-class dominant data collapses minority classes to near-random; a triple stack of inverse-frequency weighting, focal loss, and oversampling targets three different failure modes simultaneously"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[training-methodology]]"
  - "[[classification-methodology]]"
---

# Class imbalance 24x breaks naive cross-entropy and requires class-weighted plus focal plus oversampling

VocalMat's 12-class training distribution spans Step-up (n=1,814) at the top and Multi-steps (n=74) at the bottom — a 24.5× imbalance ratio. With a stratified 80/10/10 split, Multi-steps gets 59 training examples, near the lower bound for what ResNet-18 can learn at all. Naive cross-entropy on this distribution produces a degenerate optimum: the model learns to never predict Multi-steps because the loss reduction from getting majority classes right dwarfs the gain from getting any minority class right. Every minority class collapses toward the prior, and per-class precision drops below 0.20 — the failure mode the Phase 1.2 validation criteria explicitly check for.

Three corrective techniques attack different failure modes:

**Class-weighted CE** rescales the per-sample loss by inverse class frequency. The minority class loss is amplified so that the gradient contribution per minority example matches what an equal-sized class would produce. This fixes the optimization-target imbalance but doesn't help if the minority class has no useful structure to learn from.

**Focal loss** (Lin et al. 2017) multiplies CE by (1 − p_t)^γ where p_t is the predicted probability of the true class. Well-classified examples get exponentially reduced loss; hard examples — typically minority — dominate the gradient. This fixes the *gradient* imbalance without requiring class weights, and prevents the model from "coasting" on already-correct majority predictions.

**Oversampling** repeats minority-class examples in the training set with replacement, so each epoch the model sees minority classes more often. This fixes the *exposure* imbalance — without oversampling, the model can spend dozens of epochs barely encountering Multi-steps examples even if the loss is correctly weighted.

The three together address three independent problems (target weighting, gradient distribution, exposure frequency). Removing any one usually leaves a failure mode. The Phase 1.2 plan adopts all three for the v1 baseline (decision D5) and explicitly says revisit collapsing or dropping minority classes only if per-class precision still falls below 0.20 — confirming empirically what the triple-stack is supposed to prevent.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[good negative training samples must be unambiguously not USV to prevent label noise]] — adjacent training data quality concern
- [[VocalMat test set quality dual-rater consensus exceeds training set quality single-rater]] — context: minority-class label noise also matters here

Topics:
- [[training-methodology]]
- [[classification-methodology]]
