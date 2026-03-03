---
description: Empirical baseline: CNN classifier achieves F1 91.7% at threshold 0.05 on ~840 labeled candidates with recording-level splits, validating the two-stage pipeline design.
type: baseline
confidence: proven
topics:
  - "[[classification]]"
  - "[[detection]]"
---

# CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach

The trained CNN classifier achieves 89.7% precision and 93.8% recall at optimal threshold 0.05, yielding F1 91.7%. These numbers come from approximately 840 labeled candidates evaluated with recording-level splits, meaning no candidate from a held-out recording appeared in training. This baseline validates that [[two-stage detection uses permissive energy detector followed by CNN precision filter]] operates as designed: the energy detector provides high recall (catching most real USVs), and the CNN classifier adds precision (filtering out noise detections).

The optimal threshold of 0.05 — far below the conventional 0.5 — reflects the extreme class imbalance in the detection task. The energy detector produces many more noise candidates than true USVs, so the CNN must be calibrated to maintain recall even at very low confidence scores. [[3x class weight boost compensates for USV class imbalance in CNN training]] is the training-time mechanism that shifts the learned decision boundary in the appropriate direction, but threshold tuning on the validation set remains necessary to find the operating point that optimizes F1.

The 93.8% recall means approximately 6% of real USVs are missed by the CNN stage. Whether this is acceptable depends on the downstream use case: for behavioral ethology, missing 6% of calls in a session is tolerable if the detected calls are representative; for fine-grained temporal analysis of call sequences, this miss rate may introduce bias. The recall versus precision tradeoff at different thresholds is documented in [[recall versus precision tradeoff in two-stage USV detection]] and should be revisited as the labeled dataset grows beyond 840 candidates.

The CNN architecture producing these results is described in [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]], confirming that a compact model is sufficient at this data scale.

This baseline establishes the starting point for the [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]], where each subsequent milestone (5K, 10K, 20K, 30K labels) should improve upon these numbers. The precision and recall values also inform the [[split ratio inconsistency between DECISIONS.md 80-10-10 and ROADMAP Phase 9 70-15-15 needs resolution]], since the ~840-label test set that produced these metrics may be too small for reliable confidence intervals under 80/10/10 splits.

---

Source: [ROADMAP](../ROADMAP.md)
