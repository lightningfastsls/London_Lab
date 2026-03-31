---
description: "Landscape survey of DeepSqueak, DAS, VocalMat, USVSEG, MUPET — all use single threshold + gap-fill + min-duration, making dual-threshold detection a genuine methodological gap"
type: baseline
confidence: proven
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[classification-tools]]"
---

# No existing mouse USV tool uses explicit hysteresis for event detection

A systematic survey of the five most widely used mouse USV detection tools reveals a uniform architectural pattern: every tool applies a single detection threshold, then cleans up the resulting binary mask with gap-filling and minimum-duration post-hoc filters. DeepSqueak (Coffey et al., 2019) runs Faster R-CNN or YOLO on spectrograms and applies tonality filtering but uses a single confidence threshold for final detection. DAS (Steinfath et al., 2021) trains a temporal convolutional network and applies threshold plus gap-fill plus minimum duration. VocalMat (Fonseca et al., 2021) uses spectral peak detection followed by curvature-based filtering, again with a single threshold. USVSEG applies bandpass filtering with a single amplitude threshold, and MUPET uses a gammatone filterbank with fixed energy thresholding.

None of these tools implement explicit Schmitt trigger / dual-threshold hysteresis, where a higher onset threshold must be crossed to begin an event and a lower sustain threshold keeps the event alive until the signal falls below it. This is notable because hysteresis is well established in electronic circuit design and industrial signal processing for exactly this class of problem — detecting events in noisy signals without chattering.

The one bioacoustic system that does use hysteresis is WhaleVAD-BPN (2024, arXiv:2510.21280), which applies it to whale call detection. However, whale calls occupy a fundamentally different signal regime (lower frequencies, longer durations, ocean noise characteristics), so WhaleVAD-BPN does not validate the approach for mouse USVs specifically. This positions our Phase 15 hysteresis module as genuinely novel in the mouse USV detection space — not a reinvention but a principled transfer from established signal processing practice into an application domain where it has not yet been applied.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries]] -- a different architectural limitation of the same tool ecosystem
- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] -- the strongest single-threshold baseline we are comparing against
- [[DeepSqueak uses monolithic Faster R-CNN detection whereas our two-stage pipeline allows independent tuning of recall and precision]] -- our architectural advantage that hysteresis extends further

Topics:
- [[detection]]
- [[classification-tools]]
