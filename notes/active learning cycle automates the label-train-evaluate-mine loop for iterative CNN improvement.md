---
description: Automated pipeline cycles through data assembly, CNN training, threshold optimization, hard negative mining, and report generation to scale labels from 2K to 30K.
type: method
confidence: likely
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement

Scaling a labeled dataset from a few thousand examples to tens of thousands requires a systematic process that alternates between model training and targeted data collection. An active learning cycle makes this process tractable by automating every step except human labeling itself. The human annotator's time is the bottleneck; automation ensures that each hour of labeling produces maximum model improvement.

Each cycle executes five automated stages in sequence. First, training data is assembled by combining all currently labeled positives (with constrained jittering to multiply their count) and negatives sampled via [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]]. Second, the CNN is trained for a fixed number of epochs with [[3x class weight boost compensates for USV class imbalance in CNN training]]. Third, the model is evaluated on the held-out test set and a precision-recall curve is computed. Fourth, the detection threshold is optimized to balance precision and recall for the target application. Fifth, hard negatives — examples where the current model is confidently wrong — are mined from unannotated recordings and surfaced for human review.

Each cycle concludes by generating a markdown report containing precision, recall, F1, and a comparison to the previous cycle's metrics. This report closes the feedback loop: if a cycle does not improve metrics, the hard negatives mined in that cycle likely expose the model's current failure mode, and the human labeler can focus on those examples in the next annotation round.

The five milestones (2K, 5K, 10K, 20K, 30K labels) define checkpoints at which model capacity may be increased according to [[model size should scale with labeled dataset size to balance underfitting and overfitting]]. The starting point for this scaling is [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]], established on ~840 labels. The raw candidate pool from which hard negatives are mined comes from [[batch detection with skip-existing enables incremental processing of large WAV collections]], which must have completed at least a partial run before the active learning cycle can mine effectively. The confidence rating is "likely" because the specific cycle count needed to reach each milestone depends on annotation throughput, which varies across sessions.

An alternative to scaling up labels through active learning is using fewer labels more efficiently: [[DCASE few-shot bioacoustic detection improved from F1 40 percent to 70 percent across 2021-2024 challenge editions]] demonstrates that metric learning and embedding-based approaches can achieve reasonable performance with as few as 5 labeled examples. Similarly, [[foundation model embeddings enable few-shot classification via simple linear probes without end-to-end training]] suggests that frozen pretrained embeddings plus a linear probe could reduce the label count needed per milestone. However, since [[frequency shifting USVs into the audible range could enable classification with standard audio foundation models]] remains unvalidated, these few-shot alternatives are not yet applicable to our 300 kHz USV domain. The annotation workflow itself is a limiting factor, as [[mouse USV annotation tools focus on detection and segmentation rather than human review and labeling workflows]].

---

---

Relevant Notes:
- [[AMVOC semi-supervised retraining combines reconstruction KL divergence and pairwise constraint losses with uncertainty-based annotation priority]] -- AMVOC implements a parallel active learning loop for unsupervised clustering: prioritize most-uncertain USVs for human pairwise constraint annotation, then retrain autoencoder with constraints; our cycle mines hard negatives for the CNN detector while AMVOC mines uncertain cluster assignments for the autoencoder, but both follow the same principle of focusing human effort on model uncertainty

Source: [ROADMAP](../ROADMAP.md), Phase 2
