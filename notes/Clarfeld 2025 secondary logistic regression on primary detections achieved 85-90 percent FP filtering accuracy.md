---
description: "Ecological Informatics study validates the two-stage pattern — extract features from initial detections, train lightweight classifier to reject false positives, applicable across taxa"
type: finding
confidence: proven
conditions: []
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
---

# Clarfeld 2025 secondary logistic regression on primary detections achieved 85-90 percent FP filtering accuracy

Clarfeld et al. (2025, Ecological Informatics) demonstrated that a two-stage detection approach — where a primary detector generates candidates and a secondary logistic regression classifier filters false positives based on features extracted from those candidates — achieves 84.5-89.8% accuracy across multiple bioacoustic taxa. This finding is significant because it validates the two-stage detection pattern as a general principle rather than a taxon-specific trick. The secondary classifier does not need to be complex; logistic regression suffices because the feature space extracted from primary detections is already informative enough to separate true positives from false positives.

The approach works because primary detectors are typically tuned for high recall (catch everything), which inevitably admits many false positives. Rather than trying to build a single detector that simultaneously achieves high recall and high precision, the two-stage design separates these objectives: the first stage maximizes recall while the second stage recovers precision. This decomposition is easier to optimize because each stage has a simpler objective function.

This directly validates our Phase 15.5 false positive filter design, which uses a LogisticRegression trained on event-level features (duration, bandwidth, mean probability, probability variance, and spectral features) extracted from hysteresis-detected candidates. Our cross-validated F2 score of 0.850 is consistent with the Clarfeld accuracy range, providing independent confirmation that our implementation is performing as expected. The consistency across their multi-taxa dataset and our mouse USV dataset suggests that the two-stage pattern is robust to the specific characteristics of the vocalization being detected, because the underlying statistical separation between true vocalizations and noise artifacts generalizes across signal types.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- our pipeline adds hysteresis as the primary stage before FP filtering
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- related class imbalance handling in the primary CNN stage
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- the specific FP failure mode that Clarfeld's two-stage pattern addresses; our FP filter targets these structural noise mimics
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- Clarfeld's work is the strongest empirical validation of this general pattern across multiple bioacoustic taxa
- [[unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives]] -- unsupervised clustering offers a label-free alternative to Clarfeld's supervised secondary classifier, achieving comparable precision gains
- [[BootSnap includes an explicit false-positive class alongside 11 USV syllable categories]] -- unified classification with noise class versus Clarfeld's modular two-stage approach represent different integration points for FP rejection
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- Clarfeld's secondary classifier inherits the same pre-filtering bias; its training set must include representative false positives from the primary detector

Topics:
- [[detection]]
