---
description: "A pixel is classified as USV or not-USV with no concept of USV-1 vs USV-2 — overlapping calls in the same spectrogram window are merged into a single segment"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[detection]]"
---

# Spectrogram segmentation tools like SqueakOut and VocalMat are binary detectors that cannot separate overlapping USVs

Current spectrogram segmentation tools for USVs (SqueakOut, VocalMat, USVSEG) produce binary masks — each pixel is classified as "USV" or "not USV." When two calls overlap in time and frequency, they are merged into a single connected component with no mechanism to decompose them into individual sources. SqueakOut achieves excellent segmentation quality (Dice 90.22) but explicitly has no multi-source decomposition capability. VocalMat can detect that multiple calls are present using Hough Transform analysis but does not separate them. USVSEG's spectral peak tracking can follow individual frequency contours IF they occupy different frequency bands, but fails when calls cross in frequency. This binary limitation means that even the best segmentation tools lose information when mice vocalize simultaneously.

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] — our pipeline shares this binary detection limitation

Topics:
- [[detection]]
