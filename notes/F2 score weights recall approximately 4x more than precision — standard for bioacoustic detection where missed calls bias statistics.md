---
description: "With beta=2 the formula penalizes false negatives heavily — appropriate when undercounting vocalizations distorts behavioral analyses more than overcounting does"
type: method
confidence: proven
conditions: []
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
---

# F2 score weights recall approximately 4x more than precision — standard for bioacoustic detection where missed calls bias statistics

The F-beta score generalizes the F1 score by introducing a parameter beta that controls the relative importance of recall versus precision: F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall). When beta=2, the formula weights recall approximately four times more heavily than precision, because the beta^2 term (4) multiplies precision in the denominator, making precision violations less costly to the overall score.

This weighting is standard in bioacoustic detection because the downstream consequences of false negatives and false positives are asymmetric. Missing real vocalizations (false negatives) directly biases vocalization rate estimates downward, distorts social interaction metrics that depend on call counts, and skews repertoire analyses that depend on observing the full range of syllable types. These biases are systematic and difficult to correct post-hoc because you cannot analyze what you did not detect. False positives, by contrast, can be identified and removed through manual review or downstream filtering — they add work but do not create irrecoverable information loss.

Therefore the F2 score aligns the optimization objective with the scientific use case: prioritize finding all vocalizations even at the cost of admitting some false positives that can be filtered later. Our hysteresis optimization uses micro-averaged F2 across recordings, which means we compute F2 on the pooled confusion matrix rather than averaging per-recording F2 scores. Micro-averaging is appropriate because it weights each detection equally regardless of which recording it came from, preventing recordings with few events from disproportionately influencing the metric. The Phase 15 gate criterion requires F2 > 0.85, establishing a minimum quality bar that ensures the detection pipeline catches the vast majority of vocalizations before any results are used for behavioral analysis.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- the detection method whose parameters are optimized using F2
- [[Clarfeld 2025 secondary logistic regression on primary detections achieved 85-90 percent FP filtering accuracy]] -- the FP filter whose performance is also evaluated with F2

Topics:
- [[detection]]
