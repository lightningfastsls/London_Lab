---
description: "Dividing a small labeled set across many classes leaves single-digit samples per class — chi-squared and confidence intervals stop being meaningful and reports must be framed as descriptive observations only"
type: claim
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification-methodology]]"
---

# 200 call OOD evaluation gives descriptive not inferential statistics at 12 classes

Phase 1.4 of the lab CNN classifier plan budgets ~200 user-labeled wild calls from 5970 for out-of-distribution evaluation of the lab-trained classifier. At 12 Grimsley classes, that averages 17 calls per class — but the realized distribution will be heavier on common classes and may give 1–2 samples to rare ones (Multi-steps, Reverse-Chevron). The statistical implication is binding: any per-class precision/recall computed at this scale has confidence intervals roughly ±20 percentage points by binomial bounds. A reported per-class precision of 0.60 could be anywhere from 0.40 to 0.80 with that sample size, and class-level claims need to be framed as "the model produces N predictions on this class and M of them matched the label" rather than "the model has X precision on this class."

This is not a flaw of the experiment — it's an honest acknowledgement of what 200 samples can support. The OOD evaluation answers one question well: does the lab classifier produce *any* reasonable predictions on wild data at all? If even the common classes show macro F1 in the 0.4+ range, the model has generalized in some meaningful sense and is worth shipping for further use. If macro F1 is 0.1, the model has not generalized and no amount of additional labeling will save it. The threshold for "does it work" is robust to sample size; the threshold for "exactly how well does it work per class" is not.

The methodological discipline this requires: in the Phase 1.4 report, write "of 14 Step-up predictions, 11 matched the human label" rather than "Step-up precision = 0.79." Reserve aggregate metrics (macro F1, mean confidence) for between-model comparisons (v1 vs v2) where directional change matters more than absolute value. Apply no inferential statistics (chi-squared, McNemar) without an explicit power calculation showing the test is sensitive at this sample size. This kind of honesty in small-sample reporting is what separates an OOD probe (which 200 calls supports) from an OOD evaluation (which it does not).

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[normalization statistics must be computed on training set only to prevent data leakage]] — adjacent evaluation hygiene principle
- [[recording groups 5970 3452 2379 are all wild mouse dyads not different strains]] — wild-data sampling context

Topics:
- [[experimental-methods]]
- [[classification-methodology]]
