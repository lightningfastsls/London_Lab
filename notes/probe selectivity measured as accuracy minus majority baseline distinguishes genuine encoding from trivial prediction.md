---
description: "Raw probe accuracy misleads on imbalanced targets; selectivity (accuracy minus majority class baseline) reveals whether hidden states genuinely encode the probed property"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# probe selectivity measured as accuracy minus majority baseline distinguishes genuine encoding from trivial prediction

A probing classifier that achieves 90% accuracy on a classification task sounds impressive, but if the majority class comprises 85% of all examples, a trivial classifier that always predicts the majority class would achieve 85% accuracy with zero understanding of the hidden states. The probe's 5% improvement over this baseline — its selectivity — is the true measure of whether the transformer's hidden states genuinely encode the probed property, and 5% selectivity is negligible in most practical contexts.

This distinction matters critically for USV probing experiments because several target properties are likely to be imbalanced. The is_voiced classification may be heavily skewed toward voiced (most USV frames are tonal), frequency_direction may be dominated by "flat" segments, and bout_position categories (early/middle/late) may have unequal representation depending on how bouts are segmented. Without the selectivity correction, a probe could appear to "encode" a property when it is merely exploiting class imbalance, which means the resulting layer-property heatmap would be misleading.

Selectivity = accuracy - majority_baseline provides a simple, interpretable correction. A selectivity of 0% means the probe learned nothing beyond the prior; 10% means modest encoding; 30%+ means strong, genuine encoding. This metric is well-established in the NLP probing literature as "accuracy gain" or "improvement over majority." For multi-class problems, the majority baseline is the frequency of the most common class; for binary problems, it is max(p, 1-p) where p is the positive class frequency.

For regression targets (peak_frequency, spectral_centroid, energy), this correction is not needed because R-squared already handles it: a constant predictor (predicting the mean) achieves R-squared = 0 by definition. Therefore, any positive R-squared directly indicates that the hidden states carry information about the target property beyond the unconditional mean. This asymmetry between classification and regression metrics means the layer-property heatmap should use selectivity for categorical columns and R-squared for continuous columns, which makes the cells directly comparable in their interpretation as "information content above trivial baseline."

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- selectivity is the primary correction applied to probe accuracy results in that analysis

Topics:
- [[representation-learning]]
- [[experimental-methods]]
