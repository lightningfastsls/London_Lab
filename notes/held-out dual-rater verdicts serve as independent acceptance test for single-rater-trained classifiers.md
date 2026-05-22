---
description: "When training data is single-rater (5-15 percent label noise expected) but a separate higher-quality dual-rater set exists, that set becomes the acceptance test even though it never saw training — the quality differential is the leverage"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[experimental-methods]]"
---

# Held-out dual-rater verdicts serve as independent acceptance test for single-rater-trained classifiers

VocalMat's 12,954 training spectrograms were labeled by a single experienced experimenter. No inter-rater agreement was reported, no quantified label noise rate is published, and minority-class noise estimates of 5–15% are the standard expectation in the bioacoustics literature for single-rater protocols on rare event classes. Our 845 hand-curated lab 131204 verdicts (`classified_detections_lab_131204_clean.csv`) were produced under a different protocol — they were dual-rater quality with consensus arbitration on disagreements. The two datasets are not equivalent: the 845 are *better labeled* than the 12,954.

The methodological move is to hold out the 845 from training entirely and treat them as an acceptance test. The classifier is trained on the 12,954 single-rater set; evaluated normally on a held-out portion of the same set; and *separately* evaluated on the 845. Because the 845 are higher-quality labels, disagreements between model predictions and the 845 verdicts are more likely to be real model errors and less likely to be label noise. This is the inverse of the more common situation where train labels are gold and held-out labels are inferred — here, train labels are noisy and held-out labels are gold.

Two evaluation metrics matter on this set. **USV/noise accuracy** asks whether the model can at least separate vocalizations from non-vocalizations (the easiest task). **Syllable-type entropy** asks whether predictions are peaky — if mean entropy ≤ log(6), the model is producing committed predictions rather than spreading probability across all 12 classes. The PLAN sets both as binding pass criteria: > 0.80 USV/noise accuracy AND mean entropy ≤ log(6). A model that passes the standard validation split but fails the 845 is a model that has overfit to the single-rater label noise — high training-set performance hides genuine pathology.

The principle generalizes: when label quality varies across your evaluation data, use the highest-quality subset for the acceptance test, even if it's small. Quality differential is the leverage.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[normalization statistics must be computed on training set only to prevent data leakage]] — different leakage concern but same general principle of evaluation hygiene
- [[good negative training samples must be unambiguously not USV to prevent label noise]] — related: label noise is the failure mode this guards against

Topics:
- [[classification-methodology]]
- [[experimental-methods]]
