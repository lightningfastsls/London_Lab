---
description: "Ecological Informatics 2025 pipeline achieved 93% correct annotation through unsupervised clustering of extracted features after initial detection — clustering as precision filter"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[detection-landscape]]"
  - "[[classification]]"
---

# Unsupervised clustering as post-detection filtering eliminates 88 percent false positives while retaining 95 percent true positives

An automated annotation pipeline (Ecological Informatics, 2025) achieved 93% correct annotation of target species acoustic notes through unsupervised clustering of extracted features. The key finding is that clustering served as a post-processing step after initial detection, eliminating 88% of false positives while retaining 95% of true positives.

This validates using unsupervised methods as a precision filter in a multi-stage detection pipeline — conceptually similar to our two-stage approach where energy detection maximizes recall and CNN classification maximizes precision. The difference is that unsupervised clustering achieves precision improvement without labeled training data. Since [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]], adding an unsupervised clustering stage after our CNN could further reduce false positives without additional labeling effort.

However, this approach works best when false positives cluster differently from true positives in feature space — which depends on the embedding quality and the nature of the noise.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- unsupervised clustering could be a third stage
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- supervised noise class vs unsupervised noise clustering

Topics:
- [[unsupervised-usv-discovery]]
- [[detection]]
- [[classification-methodology]]
