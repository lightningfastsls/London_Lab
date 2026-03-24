---
description: "Mouse USVs span 30-110 kHz but detection uses 20-120 kHz — padded to catch edge cases without flooding with low-frequency noise"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[detection]]"
---

# 20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls

Mouse USVs fall roughly in the 30-110 kHz range. The detection pipeline uses a wider 20-120 kHz band, padded on both ends. Below ~25 kHz is mostly environmental noise and audible-range sounds — particularly problematic since [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]], which places a premium on keeping the lower bound high enough to reject environmental noise while still capturing edge-case calls. Above 110-120 kHz there is very little USV signal and microphone sensitivity drops off. The padding avoids clipping edge-case calls that may fall at the boundaries of the core USV range. It also ensures that harmonic structure is fully captured: since [[harmonics of a USV are treated as one call not multiple detections]], the band must be wide enough to contain both the fundamental (e.g. 45 kHz) and its harmonic (e.g. 90 kHz) within the detection window.

The exact boundaries aren't critical — what matters is capturing the full USV range without flooding the spectrogram with irrelevant low-frequency noise. Within this band, [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] gives approximately 171 frequency bins covering the range. After cropping to this band, [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] operates across those ~170 bins to produce CNN-ready input. The band also defines the frequency domain for the first-stage energy detector within the [[two-stage detection uses permissive energy detector followed by CNN precision filter]] architecture.

---

Source:
- Researcher brain-dump on preprocessing insights (2026-02-19)

Relevant Notes:
- [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] -- resolution within this band
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- Nyquist coverage of the upper bound
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- bandwidth filter operates within this detection range
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] -- energy detection mode choice operates across this frequency range
- [[electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs]] -- 60 kHz harmonics fall within this detection range
- [[harmonics of a USV are treated as one call not multiple detections]] -- the band must be wide enough to capture fundamental-harmonic relationships
- [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] -- normalization domain is the ~170 bins within this cropped band
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- this band defines the frequency domain for the first-stage energy scan
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- noisy environment motivates not lowering the 20 kHz floor further
- [[transient cage noises produce broadband vertical smears rejected by the minimum duration filter]] -- broadband cage noise spans below 20 kHz, supporting the lower bound choice
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- cross-tool convention differs slightly (25-125 kHz vs our 20-120 kHz); for Raven export the standard band is preferred

Topics:
- [[signal-processing]]
- [[detection]]
