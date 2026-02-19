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

Mouse USVs fall roughly in the 30-110 kHz range. The detection pipeline uses a wider 20-120 kHz band, padded on both ends. Below ~25 kHz is mostly environmental noise and audible-range sounds. Above 110-120 kHz there is very little USV signal and microphone sensitivity drops off. The padding avoids clipping edge-case calls that may fall at the boundaries of the core USV range. The exact boundaries aren't critical — what matters is capturing the full USV range without flooding the spectrogram with irrelevant low-frequency noise. Within this band, [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] gives approximately 171 frequency bins covering the range.

---

Source:
- Researcher brain-dump on preprocessing insights (2026-02-19)

Relevant Notes:
- [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] -- resolution within this band
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- Nyquist coverage of the upper bound

Topics:
- [[signal-processing]]
- [[detection]]
