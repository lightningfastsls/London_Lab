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

VocalMat uses an AlexNet CNN architecture for supervised classification of USVs into 11 predefined categories, achieving approximately 86% accuracy. This represents the traditional approach to USV classification: define categories a priori based on acoustic features (following taxonomies like [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]]), then train a classifier to assign each call. Our VQ-VAE approach deliberately sidesteps this predefined-category paradigm since [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] — letting the codebook discover its own vocabulary rather than classifying into human-defined types.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the finding that challenges predefined categories
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our approach vs VocalMat's 11 types
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- another competitor tool

Topics:
- [[classification]]
