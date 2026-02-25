---
description: "VocalMat's GitHub provides 10,871 USVs across 11 categories plus 2,083 noise samples -- the largest freely available labeled USV dataset for transfer learning"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# VocalMat provides 12954 labeled USV spectrograms freely available as training data

VocalMat provides **12,954 labeled spectrograms** freely available on GitHub: 10,871 USVs across 11 categories plus 2,083 noise samples. This is the largest publicly available labeled dataset for USV classification and the most accessible source of training data for transfer learning.

The practical value for our pipeline is significant: rather than labeling thousands of spectrograms from scratch, we can **fine-tune a pretrained backbone** (MobileNetV2 or ResNet-18) on VocalMat's labeled data, then adapt to our recordings. VocalMat itself achieved **~86% accuracy on 11 syllable categories** using fine-tuned AlexNet (details in [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]]).

BootSnap's labeled data from wild-derived and lab mice is also available through their PLOS Computational Biology supplementary materials. However, a critical limitation applies: since [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]], VocalMat's lab-mouse data alone will not produce accurate classification of our wild mouse recordings. We should supplement with our own labeled data. The [[including a noise-false-positive class in the USV classifier catches residual detection errors]] recommendation means the 2,083 noise samples in VocalMat's dataset are especially valuable for training a noise rejection class.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- VocalMat GitHub repository

Relevant Notes:
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- the tool and method that produced this dataset
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- limitation of using lab-mouse training data alone
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- the 2,083 noise samples enable this class
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] -- strategy for supplementing VocalMat data with our own labels
- [[good negative training samples must be unambiguously not USV to prevent label noise]] -- quality criterion for VocalMat's noise samples
- [[model size should scale with labeled dataset size to balance underfitting and overfitting]] -- 12,954 samples supports a medium-sized model

Topics:
- [[classification]]
- [[experimental-methods]]
