---
description: "Combines 64-element acoustic camera with 4 ultrasonic microphones — 3x better than prior systems, approaching physical limits of mouse snout diameter (~10 mm)"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[experimental-methods]]"
---

# HyVL hybrid beamforming achieves 3 to 5 mm USV localization precision with 91 percent source assignment

HyVL (Sterling et al., eLife 2023) integrates a 64-element acoustic camera (Cam64) with 4 high-quality ultrasonic microphones (USM4) for USV localization. Acoustic beamforming on the Cam64 delivers median absolute error ~4-5 mm, while the USM4 arm uses the SLIM algorithm (MAE ~11-14 mm). Hybrid fusion achieves ~3.4-4.8 mm precision with 91% of USVs assigned to a source. This is approximately 3x better than prior systems and approaches the physical limits set by mouse snout diameter (~10 mm). However, the approach has important limitations: it requires specialized hardware (acoustic camera + mic array + calibration + video tracking), and temporally overlapping calls from mice in close proximity (<5 cm) remain ambiguous even at millimeter precision. Open source at github.com/benglitz/HyVL.

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[wearable miniature microphones achieve 90 percent USV attribution from amplitude alone]] — alternative hardware approach

Topics:
- [[detection]]
- [[experimental-methods]]
