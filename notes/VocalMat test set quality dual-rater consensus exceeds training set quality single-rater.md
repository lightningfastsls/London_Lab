---
description: "Fonseca 2021 used different labeling protocols for training (one experienced experimenter) vs test (two investigators plus consensus arbitration) — the test set is therefore a stricter evaluation than the train split suggests"
type: claim
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-tools]]"
  - "[[classification-methodology]]"
---

# VocalMat test set quality dual-rater consensus exceeds training set quality single-rater

The Fonseca et al. 2021 VocalMat paper (eLife 10:e59161) describes its labeling protocol explicitly in the Methods section. The **training set** of 12,954 spectrograms was labeled by a single "experienced experimenter" — no second rater, no inter-rater agreement reported, no quantified label noise rate. The **test set** of 4,441 USVs (across 7 recordings) was independently labeled by ≥2 investigators with consensus arbitration by a third investigator on disagreements — the gold-standard multi-rater protocol that produces near-perfect label quality on the agreed subset. These two datasets are different in kind, not just in size, and the difference matters for any downstream use.

For consumers of the dataset, the practical implication is asymmetric: predictions evaluated on the test set are scored against high-confidence ground truth, but predictions evaluated on a held-out portion of the training set are scored against noisy labels. A model with 92% accuracy on test data and 85% accuracy on train-split data has not necessarily learned worse — the 85% may include 5–15% labels that are simply wrong, especially in minority classes (Reverse-Chevron n=136, Multi-steps n=74 in training) where one rater's edge-case judgments carry disproportionate weight.

This is a clean example of why labeling protocols belong in dataset documentation — not just class counts and provenance, but the inter-rater discipline that produced each subset. A paper that reports only "12,954 labeled images" without distinguishing single-rater training from dual-rater test would leave downstream users assuming uniform quality and reaching wrong conclusions about model performance differentials. VocalMat documents both protocols explicitly; many published bioacoustics datasets do not.

The takeaway for our lab CNN classifier: any evaluation that compares model predictions to held-out VocalMat training labels should expect (and tolerate) 5–15% disagreement floor in minority classes that is label noise rather than model error. Comparisons to the dual-rater test set, or to our 845 dual-rater lab 131204 verdicts, are the harder and more honest evaluations.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] — the broader dataset claim this refines
- [[held-out dual-rater verdicts serve as independent acceptance test for single-rater-trained classifiers]] — methodology this enables

Topics:
- [[classification-tools]]
- [[classification-methodology]]
