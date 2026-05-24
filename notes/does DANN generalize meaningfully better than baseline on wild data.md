---
description: "Open empirical question — the cage-invariance argument for DANN is theoretically sound, but it has not been measured against the baseline on actual lab→wild transfer; the answer determines whether v2 ships or Phase 2 is needed"
type: claim
confidence: open
conditions:
  - "Awaiting Module 18.5 wild transfer evaluation; user labeling of ~200 wild calls is sequential after v2"
meta_state: current
topics:
  - "[[model-adaptation]]"
  - "[[wild-lab-vocal-comparison]]"
  - "[[classification-methodology]]"
---

# Does DANN generalize meaningfully better than baseline on wild data

The DANN domain-adversarial training architecture is justified theoretically as a cheap way to enforce cage invariance in the classifier encoder, with a linear cage probe accuracy < 65% as its in-distribution proof. But the in-distribution proof does not directly answer the out-of-distribution question: does v2 (DANN-trained) actually generalize better than v1 (baseline ImageNet ResNet-18) on wild 5970 data the model has never seen?

The plan's decision logic at Phase 1.4: if Δ(macro F1) = v2 F1 − v1 F1 > +0.05 on the user-labeled 200-call wild set, ship v2 as the production lab+wild classifier. If Δ ≤ +0.05 (no meaningful improvement) or Δ < 0 (regression), document and defer to Phase 2. The boundary at +0.05 is the smallest improvement that's larger than the noise floor at this sample size — anything smaller is consistent with chance.

Three possible outcomes:
- **v2 wins clearly (Δ > +0.10)**: DANN delivered on its theoretical promise. Ship v2. Cage acoustics were indeed the main obstacle to lab→wild transfer.
- **v2 wins marginally (Δ in 0.05–0.10)**: DANN helped but not transformatively. Ship v2 with caveat. The cage-acoustics hypothesis is partially confirmed but not the whole story.
- **v2 loses or ties (Δ ≤ +0.05)**: Either DANN didn't enforce enough invariance (linear probe should have caught this — re-examine probe results), or cage acoustics aren't the main domain gap. Patch-duration mismatch ([[patch-duration mismatch was the mechanistic cause of VocalMat-on-wild kappa 0.13 failure]]) becomes the prime suspect; revisit open question 1 (0.08s short-patch variant).

The question is open because Module 18.5 has not been executed and the user has not produced the 200 wild labels yet (sequential after v2 per D2). Until both happen, the cage-invariance theoretical argument remains untested in the regime that matters most — actual wild data the model has never seen.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[DANN gradient-reversal enforces invariance without per-batch domain matching]] — the architecture under test
- [[DANN lambda schedule 0 to 1 prevents encoder collapse from aggressive adversarial loss]] — training discipline that determines whether the architecture works
- [[patch-duration mismatch was the mechanistic cause of VocalMat-on-wild kappa 0.13 failure]] — the alternative hypothesis if DANN doesn't deliver
- [[200 call OOD evaluation gives descriptive not inferential statistics at 12 classes]] — caveat on how to read the answer

Topics:
- [[model-adaptation]]
- [[wild-lab-vocal-comparison]]
- [[classification-methodology]]
