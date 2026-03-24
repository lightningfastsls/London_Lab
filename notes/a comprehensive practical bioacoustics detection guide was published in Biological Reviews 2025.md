---
description: "Covers best practices for training data, evaluation metrics, and cross-dataset generalization — a reference for standardizing bioacoustic detection methodology"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection-landscape]]"
---

# a comprehensive practical bioacoustics detection guide was published in Biological Reviews 2025

A comprehensive guide for biologists and computer scientists on automatic detection for bioacoustic research was published in Biological Reviews (2025), one of the highest-impact ecology journals. The guide covers best practices for training data collection, evaluation metrics, and cross-dataset generalization — three areas where the USV detection field lacks standardization.

The publication is significant because bioacoustic detection methods are typically evaluated with different metrics (precision/recall, F1, Dice, IoU), different overlap thresholds (IoU >= 0.5, 0.6, 0.7), and on different datasets, making cross-study comparison unreliable. A standardized methodology guide from a high-impact journal could drive convergence on evaluation practices.

For our pipeline, this guide could inform how we report detection performance and ensure our evaluation methodology aligns with emerging standards. The paper was published at doi.org/10.1111/brv.13155 but was inaccessible during initial research (403 error), so the specific recommendations remain to be reviewed.

The broader context is that [[no single bioacoustic tool covers the full detection-annotation-review-export pipeline]], and this methodological fragmentation extends to evaluation practices. Standardized evaluation would make it easier to compare our two-stage energy-detection-plus-CNN approach against alternatives like DAS, SqueakOut, and DeepSqueak on equal footing.

---

Source:
- usv-detection-methods-landscape-2024-2026-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[no single bioacoustic tool covers the full detection-annotation-review-export pipeline]] — methodological standardization is part of the broader fragmentation problem
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] — our approach that would benefit from standardized evaluation

Topics:
- [[detection]]
