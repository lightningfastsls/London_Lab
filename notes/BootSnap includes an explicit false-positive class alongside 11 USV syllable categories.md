---
description: "Abbasi et al 2022 PLOS Comp Bio — training the classifier to explicitly recognize noise as a category rather than just filtering low-confidence detections, a principled alternative to post-hoc FP filtering"
type: finding
confidence: proven
conditions: []
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[classification-tools]]"
---

# BootSnap includes an explicit false-positive class alongside 11 USV syllable categories

BootSnap (Abbasi et al., 2022, PLOS Computational Biology) takes a fundamentally different approach to false positive handling compared to two-stage filter pipelines. Instead of detecting candidates first and then filtering them with a separate classifier, BootSnap trains a single multi-class classifier that learns 12 categories: 11 USV syllable types plus one explicit false-positive class. The model therefore learns what noise looks like during training, rather than relying on post-hoc threshold heuristics or secondary classifiers to reject it.

This approach is conceptually cleaner because the noise class competes directly with USV classes in the softmax output. A candidate that resembles noise more than any syllable type gets classified as a false positive through the same learned decision boundaries that distinguish syllable types from each other. The unified training objective means the model allocates representational capacity to the noise boundary jointly with the syllable boundaries, which should in principle produce better-calibrated rejection decisions than a separate FP filter that never sees the syllable classification features.

However, the trade-off is that this design requires labeled noise examples in the training data — the model cannot learn to recognize false positives without seeing representative examples during training. This creates a bootstrapping problem: you need a preliminary detector to generate candidate false positives, then a human must label them, then you retrain. BootSnap addresses this with an iterative bootstrapping approach, but the labeling burden is nontrivial.

Our approach — a separate LogisticRegression FP filter trained on event-level features — is more modular but less integrated. The modularity means we can update the FP filter independently of the CNN, and we can use different feature representations for detection versus filtering. But the separation means the CNN and FP filter cannot share learned representations, potentially leaving precision on the table compared to a unified model.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[VocalMat two-stage morphological filtering plus CNN noise classification achieves over 98 percent detection rate]] -- another two-stage approach but with hand-engineered first stage
- [[Clarfeld 2025 secondary logistic regression on primary detections achieved 85-90 percent FP filtering accuracy]] -- the two-stage pattern our pipeline follows
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- our current approach to class imbalance, which would interact with an explicit noise class

Topics:
- [[detection]]
- [[classification-tools]]
