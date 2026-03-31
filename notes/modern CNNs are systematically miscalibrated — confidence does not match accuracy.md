---
description: "Guo et al 2017 ICML showed post-hoc calibration is essential — raw sigmoid outputs from deep networks overstate certainty, affecting downstream threshold selection"
type: finding
confidence: proven
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[training-methodology]]"
---

# Modern CNNs are systematically miscalibrated — confidence does not match accuracy

Guo et al. (2017, ICML) demonstrated that modern deep neural networks, despite achieving higher accuracy than their predecessors, are significantly less calibrated. A well-calibrated model produces probability estimates that match empirical accuracy — if it predicts 0.8 probability for a class, that class should appear roughly 80% of the time. Modern CNNs systematically violate this: they tend toward overconfidence, producing probability distributions that are sharper than warranted by their actual accuracy.

This miscalibration arises partly from increased model capacity (more parameters than needed to fit the training data), batch normalization, and training beyond the point of zero training loss. These techniques improve discrimination (the model ranks examples correctly) but distort the probability scale. The practical consequence is that a threshold of 0.5 on a miscalibrated model does not mean "more likely USV than noise" — it means something harder to interpret.

For our pipeline, this matters directly because hysteresis detection operates on CNN probability streams. If the onset threshold is set to 0.7 assuming calibrated probabilities, but the model's 0.7 actually corresponds to 0.55 true probability, then the threshold is effectively too permissive. Conversely, if the model is overconfident in the other direction (rarely the case with modern CNNs), thresholds would be too strict. Either way, threshold values lose their intuitive meaning and become model-specific magic numbers that must be re-tuned whenever the model changes.

Post-hoc calibration — applying a learned transformation to model outputs after training — restores the probability-accuracy correspondence without retraining. Temperature scaling is the simplest and most effective method for this purpose, requiring only a single parameter fitted on held-out validation data.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[temperature scaling is the simplest effective calibration — one scalar divides logits before sigmoid]] -- the specific calibration method we apply to address this problem
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- the baseline performance context where calibration becomes relevant
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- miscalibration exacerbates this by making noisy-region confidences appear higher than warranted

Topics:
- [[detection]]
- [[training-methodology]]
