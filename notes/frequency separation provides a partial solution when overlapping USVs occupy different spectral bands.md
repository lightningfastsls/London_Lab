---
description: "Mouse USVs often span different frequency ranges (30-50 kHz vs 60-80 kHz) — simple frequency-band masking can separate spectrally non-overlapping calls even without neural networks"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[signal-processing]]"
---

# frequency separation provides a partial solution when overlapping USVs occupy different spectral bands

Mouse USV syllables often span different frequency ranges — for example, one mouse might vocalize at 30-50 kHz while another calls at 60-80 kHz. When overlapping calls do not share spectral content, even simple frequency-band masking can separate them by splitting the spectrogram into frequency clusters. The hard case is same-frequency overlap, where calls cross in frequency space.

A practical heuristic for our pipeline: when the energy detector flags a wide-band region spanning more than ~20 kHz bandwidth, check if there are distinct spectral peaks at different frequencies. If so, split the candidate into separate detections per frequency cluster. This approach requires no neural network training and could be implemented as a post-processing step in the existing detection pipeline, using spectral peak analysis to identify separable frequency bands within detected energy regions.

USVSEG's spectral peak tracking already demonstrates this principle — it can follow individual frequency contours through time when they occupy different frequency bands, but fails when calls cross in frequency. The limitation is fundamental: when two USVs share the same frequency at the same time, spectral separation is impossible and only source separation methods like [[BioCPPNet U-Net architecture with permutation-invariant training enables single-channel bioacoustic source separation]] could help.

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[BioCPPNet U-Net architecture with permutation-invariant training enables single-channel bioacoustic source separation]] — needed when spectral separation fails
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] — this post-processing step would refine the permissive first stage

Topics:
- [[detection]]
- [[signal-processing]]
