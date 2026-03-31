---
description: "Monotonic transformation of probabilities preserves ranking — discrimination unchanged, but calibrated scores let thresholds generalize across recordings and model versions"
type: finding
confidence: proven
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[training-methodology]]"
---

# ROC AUC is invariant to temperature scaling but threshold interpretability improves

ROC AUC measures a model's ability to discriminate between positive and negative examples — specifically, the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative example. Because temperature scaling is a monotonic transformation (dividing all logits by the same constant T), it preserves the rank ordering of all predictions. If example A had a higher raw probability than example B before calibration, it still does after. Therefore, ROC AUC, which depends only on ranking, is mathematically guaranteed to remain unchanged.

This invariance means temperature scaling cannot improve a poorly discriminating model — if the CNN cannot distinguish USVs from noise, calibration will not help. However, what calibration does change is the interpretability and portability of threshold values. An onset threshold of 0.7 on a calibrated model means "the model estimates 70% probability of USV presence," which has a clear semantic interpretation. The same threshold on an uncalibrated model means something model-specific that changes whenever the model is retrained.

This distinction becomes practically important in two scenarios. First, when applying thresholds across different recordings with varying noise characteristics: calibrated probabilities should generalize better because they are anchored to actual event frequencies rather than arbitrary model-internal scales. Second, when updating the CNN model: if both the old and new models are calibrated, threshold values should transfer without requiring a full re-optimization of hysteresis parameters.

Therefore, temperature scaling serves a different purpose than improving model quality — it improves the reliability and transferability of the entire downstream post-processing pipeline that depends on probability thresholds. This is why calibration belongs in the pipeline even though it has zero effect on the metric most commonly used to evaluate classifiers.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[temperature scaling is the simplest effective calibration — one scalar divides logits before sigmoid]] -- the specific calibration method whose interaction with AUC this note explains
- [[modern CNNs are systematically miscalibrated — confidence does not match accuracy]] -- the underlying problem that makes threshold interpretability unreliable without calibration

Topics:
- [[detection]]
- [[training-methodology]]
