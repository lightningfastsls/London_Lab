---
description: "Whombat, OpenSoundscape, and DAS all support iterative annotate-train-predict-review cycles — reducing human effort by focusing annotation on uncertain predictions"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[classification]]"
---

# active learning annotation workflows are the frontier in bioacoustic tools

The emerging frontier in bioacoustic annotation is active learning — iterative cycles where models predict, humans correct the uncertain cases, and the model retrains on the expanded dataset. Whombat, OpenSoundscape, and DAS all support this workflow pattern. The efficiency gain comes from focusing human annotation effort on examples where the model is uncertain, rather than labeling the entire dataset.

This paradigm is directly applicable to our USV pipeline: after initial energy detection, a CNN classifier predicts labels, and human reviewers focus on low-confidence predictions, progressively building a high-quality training set. Our existing [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] describes the concept, but the broader ecosystem trend shows this is where all bioacoustic tools are heading.

DAS achieves the highest reported USV detection metrics (98% precision, 99% recall) through exactly this iterative human-model collaboration. OpenSoundscape provides active learning workflows with native Raven format support and CNN transfer learning. Whombat adds collaborative multi-annotator support to the loop, enabling multiple researchers to review predictions simultaneously.

The convergence of multiple independent tools toward this same workflow pattern validates the approach as more than a convenience — it represents a fundamental insight about how human expertise and ML capability complement each other in bioacoustic research. The human provides domain judgment on ambiguous cases; the model provides consistent throughput on clear cases.

AMVOC (Stoumpou et al. 2022) provides an earlier example of this pattern applied specifically to mouse USV classification: its semi-supervised retraining loop prioritizes the most uncertain USVs (lowest max-probability of cluster assignment) for human pairwise constraint annotation, then retrains the autoencoder with the constraints incorporated. Though not labeled "active learning," this implements the core loop. See [[AMVOC semi-supervised retraining combines reconstruction KL divergence and pairwise constraint losses with uncertainty-based annotation priority]].

---

Source:
- bioacoustic-annotation-tools-landscape-2025-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] — our pipeline's implementation of this concept
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] — active learning extends the two-stage concept with iterative human refinement

Topics:
- [[detection]]
- [[classification]]
