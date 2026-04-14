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

However, the trade-off is that this design requires labeled noise examples in the training data — the model cannot learn to recognize false positives without seeing representative examples during training. This creates a bootstrapping problem: you need a preliminary detector to generate candidate false positives, then a human must label them, then you retrain. BootSnap addresses this with an iterative bootstrapping approach (a form of [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]]), but the labeling burden is nontrivial. The quality of those noise examples matters: since [[good negative training samples must be unambiguously not USV to prevent label noise]], BootSnap's noise class training data must be curated to avoid ambiguous cases that would blur the decision boundary.

Our approach — a separate LogisticRegression FP filter trained on event-level features — is more modular but less integrated. The modularity means we can update the FP filter independently of the CNN, and we can use different feature representations for detection versus filtering. But the separation means the CNN and FP filter cannot share learned representations, potentially leaving precision on the table compared to a unified model. BootSnap's unified training also sidesteps the selection bias problem where [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] — because the noise class is part of the training objective, the model inherently learns what non-USV audio looks like.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[VocalMat two-stage morphological filtering plus CNN noise classification achieves over 98 percent detection rate]] -- another two-stage approach but with hand-engineered first stage
- [[Clarfeld 2025 secondary logistic regression on primary detections achieved 85-90 percent FP filtering accuracy]] -- the two-stage pattern our pipeline follows; BootSnap's unified approach is the architectural alternative
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- our current approach to class imbalance, which would interact with an explicit noise class
- [[multi-source negative sampling is necessary when the training pipeline pre-filters candidates]] -- BootSnap's bootstrapping problem (needing labeled noise examples) is a special case of this general pattern: the noise class requires representative false positives from a preliminary detector
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- the specific structural noise mimics that BootSnap's explicit noise class learns to reject through direct competition in the softmax
- [[unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives]] -- unsupervised clustering offers a third approach to FP rejection alongside BootSnap's unified noise class and Clarfeld's two-stage filter, requiring no labels at all
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- BootSnap's noise class operates on gammatone spectrograms, which may provide better separation between USV and noise than standard STFTs
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- companion note: how this pattern applies to our pipeline as a second-pass quality control layer
- [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]] -- the selection bias failure mode that BootSnap's unified noise class avoids by construction
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- the general two-stage pattern that BootSnap's unified approach is the architectural alternative to
- [[six USV detection architectural approaches span object detection to speech model transfer with distinct tradeoff profiles]] -- BootSnap falls in the "hybrid pipeline" category of this architectural taxonomy
- [[whether BootSnap code is publicly available or must be requested from Abbasi Zala Penn at Vienna]] -- practical availability question for adopting this approach
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- VocalMat also includes noise samples alongside its 11 predefined USV types, but treats noise as training data rather than an explicit competing class in the softmax
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- BootSnap's 12-class scheme could serve as the supervised branch, with its noise class providing built-in FP rejection

Topics:
- [[detection]]
- [[classification-tools]]
