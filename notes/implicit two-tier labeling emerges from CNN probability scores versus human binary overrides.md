---
description: "CNN detections carry probability scores while manually saved detections are flagged user-saved without probability — creating two confidence tiers"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[experimental-methods]]"
---

# Implicit two-tier labeling emerges from CNN probability scores versus human binary overrides

The detection system produces an implicit two-tier labeling structure. Automatic CNN detections save with their probability scores, providing a continuous confidence measure. Manually saved detections are marked as "user saved" but carry no probability — they represent cases where the researcher spotted something the CNN missed or wanted to override. This creates two distinct label types: machine-generated with graded confidence, and human-generated as binary assertions. The human override acts as a third recall mechanism beyond the [[two-stage detection uses permissive energy detector followed by CNN precision filter]] automated stages. For training, this means the positive class contains examples with different provenance and implicit quality — CNN-detected positives have been scored, while human-saved positives are high-confidence by definition (a human judged them worth saving).

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the automated pipeline that human override supplements
- [[recall versus precision tradeoff in two-stage USV detection]] -- human override acts as a recall safety net beyond the two automated stages
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] -- human overrides feed back into training data

Topics:
- [[detection]]
- [[experimental-methods]]
