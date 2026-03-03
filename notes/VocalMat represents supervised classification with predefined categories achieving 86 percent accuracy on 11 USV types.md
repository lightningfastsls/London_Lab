---
description: "VocalMat uses AlexNet CNN for supervised classification into 11 predefined USV categories — the traditional approach our VQ-VAE aims to surpass"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types

VocalMat uses a **fine-tuned AlexNet CNN architecture** for supervised classification of USVs into 11 predefined categories, achieving approximately 86% accuracy. Critically, VocalMat provides **12,954 labeled spectrograms** freely on GitHub -- 10,871 USVs across 11 categories plus 2,083 noise samples (see [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]]). This makes it not just a classification tool but the largest publicly available labeled USV dataset for transfer learning.

This represents the traditional approach to USV classification: define categories a priori based on acoustic features (following taxonomies like [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]]), then train a classifier to assign each call. The AlexNet backbone is dated by 2026 standards -- more modern architectures (MobileNetV2, ResNet-18) would likely improve on the 86% accuracy. Our VQ-VAE approach deliberately sidesteps this predefined-category paradigm since [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] — letting the codebook discover its own vocabulary rather than classifying into human-defined types.

VocalMat has been **inactive since ~2021**, meaning it will not receive updates for new architectures or training methods. However, its labeled dataset retains lasting value as training data for any supervised approach.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23) -- AlexNet architecture detail, 12,954 labeled dataset, inactive since 2021

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the finding that challenges predefined categories
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our approach vs VocalMat's 11 types
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- another competitor tool
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- unsupervised alternative that discovers 100-140 types vs VocalMat's 11 predefined categories
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- the paradigm VocalMat implements vs the continuum reality
- [[VocalMat provides 12954 labeled USV spectrograms freely available as training data]] -- the 12,954-sample dataset available for transfer learning
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- VocalMat's lab-mouse data has limited generalization

Topics:
- [[classification-methodology]]
