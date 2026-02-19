---
description: Tension between model capacity and data availability across active learning milestones, with scaling boundaries that need empirical validation.
type: finding
confidence: likely
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# model size growth versus available labeled data at each training milestone

A persistent tension in the active learning pipeline is matching model capacity to available data. Too small a model underfits at high label counts, leaving performance on the table; too large a model overfits at low label counts, producing inflated validation metrics that collapse on held-out recordings. The scaling guide maps three size regimes: small CNN (~101K params) for fewer than 5K labels, medium CNN (~400K params) for 5–15K labels, and large CNN (~1.6M params) for 15K and above.

These boundaries are estimates derived from general deep learning scaling intuitions applied to the USV domain, not from empirical ablations on this specific dataset. The actual thresholds depend on the effective diversity of the training set — which is constrained by [[recording-level splits reduce effective training set size but prevent data leakage]]. Because splits are recording-level, 5K labels from 50 recordings is less diverse than 5K labels from 500 recordings, making the effective capacity requirement lower than a naive label count suggests.

The [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] defines five milestones (2K → 5K → 10K → 20K → 30K labels) that provide natural checkpoints to test scaling decisions. At each checkpoint, the recommendation is to retrain the current size model and the next size up, compare validation metrics, and only upgrade size if the larger model shows clear improvement without overfitting on the held-out recordings. This makes model sizing an empirical question answered during the labeling campaign rather than a prior commitment.

The connection to [[model size should scale with labeled dataset size to balance underfitting and overfitting]] captures the general principle, while [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] documents the current working architecture that anchors the small regime.

---

Source: [[ROADMAP.md]]

Relevant Notes:
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] -- defines the milestones where size decisions are evaluated
- [[recording-level splits reduce effective training set size but prevent data leakage]] -- constrains effective data diversity at each milestone
- [[model size should scale with labeled dataset size to balance underfitting and overfitting]] -- the general principle this note elaborates
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- the small regime architecture

Topics:
- [[classification]]
- [[experimental-methods]]
