---
description: "Fonseca et al 2021 eLife — the morphological pre-filter uses spectrogram curvature to reject non-USV shapes before CNN classification, an alternative to our probability-based feature approach"
type: finding
confidence: proven
conditions: []
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[classification-tools]]"
---

# VocalMat two-stage morphological filtering plus CNN noise classification achieves over 98 percent detection rate

VocalMat (Fonseca et al., 2021, eLife) implements a two-stage detection pipeline that first applies morphological filtering based on spectrogram contour curvature, then passes surviving candidates through a CNN that classifies them into USV syllable types or noise. The morphological pre-filter analyzes the shape of connected components in the thresholded spectrogram — genuine USVs tend to have smooth, elongated contours with characteristic curvature profiles, while noise artifacts tend to be irregular, fragmented, or geometrically implausible for vocalizations. This shape-based rejection removes obvious non-USV candidates before the more expensive CNN classification step.

The reported detection rate exceeding 98% is impressive, but the morphological approach carries an important assumption: that USVs have characteristic contour shapes that distinguish them from noise. This assumption holds well under clean recording conditions where USVs appear as clear frequency-modulated sweeps, but may break down when recordings have low SNR, overlapping calls, or unusual vocalization types that do not conform to expected morphological templates. The curvature-based filtering is therefore somewhat brittle to recording quality variation.

Our approach differs fundamentally because we use probability-based features extracted from the CNN itself rather than morphological features from the spectrogram. This is more model-agnostic — the features reflect what the CNN has learned to be informative rather than what human-designed shape descriptors capture. The trade-off is that our approach depends on the CNN being reasonably well-calibrated, whereas VocalMat's morphological stage is independent of any learned model. Both approaches validate the general principle that two-stage detection outperforms single-stage detection for bioacoustic applications, but they disagree on whether the first stage should be hand-engineered (VocalMat) or model-derived (our pipeline).

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[no existing mouse USV tool uses explicit hysteresis for event detection]] -- VocalMat uses single-threshold detection despite its two-stage architecture
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- another comparison of detection architectures
- [[Clarfeld 2025 secondary logistic regression on primary detections achieved 85-90 percent FP filtering accuracy]] -- independent validation of the two-stage pattern across taxa

Topics:
- [[detection]]
- [[classification-tools]]
