---
description: "Quiet USVs near the noise floor and very short calls are hardest to detect — they were the original source of training bias in earlier CNN iterations"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# Low-amplitude and short-duration USVs are the primary source of false negatives and training bias

Quiet USVs (low amplitude, near the noise floor) and very short calls are the hardest to detect and were the original source of training bias in earlier CNN iterations. The energy detector created this selection bias by filtering out quiet USVs, which then biased CNN training — since [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]], the upstream filtering silently shaped the downstream training distribution. Early training sets under-represented faint/short signals, causing the CNN to learn a biased distribution skewed toward louder, longer, more obvious calls. This bias was a key driver behind the active learning approach where [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] — specifically, the "mine" step targets regions where the current model is uncertain, which disproportionately surfaces these faint/short signals. The [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] exists specifically to catch these low-amplitude signals before CNN filtering.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- deliberately permissive to catch faint signals
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] -- iterative process to correct the initial training bias
- [[recall versus precision tradeoff in two-stage USV detection]] -- false negatives from faint signals directly affect recall

Topics:
- [[detection]]
- [[classification]]
