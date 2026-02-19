---
description: "Dataset splits by recording file stem, not individual candidates -- prevents the model from memorizing recording-specific noise"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# Recording-level splits prevent data leakage in USV classification

USVs from the same recording are temporally correlated -- they share the same background noise, equipment characteristics, and recording conditions. If chunks from the same recording appear in both training and test sets, the model can "cheat" by memorizing recording-specific noise patterns rather than learning genuine USV features. All dataset splits are performed by recording file stem with seed=42 (80/10/10 train/val/test). This prevents data leakage but since [[recording-level splits reduce effective training set size but prevent data leakage]], the tradeoff is a smaller effective training set because chunks from the same recording cannot be distributed across splits. This requires enough distinct recordings for meaningful splits.

---

Source:
- DECISIONS.md (ADR-004) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[recording-level splits reduce effective training set size but prevent data leakage]] -- the explicit tradeoff
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- another data preparation decision
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- the model trained on these splits

Topics:
- [[experimental-methods]]
- [[classification]]
