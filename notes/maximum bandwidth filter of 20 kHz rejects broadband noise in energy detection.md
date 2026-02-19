---
description: "Candidates exceeding 20 kHz bandwidth are rejected as broadband noise rather than narrow-band USVs"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# Maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection

The energy detector includes a maximum bandwidth filter of 20,000 Hz (`max_bandwidth_hz = 20000`). Any candidate whose energy spans more than 20 kHz is rejected as broadband noise rather than a genuine USV. Mouse USVs are characteristically narrow-band signals -- even frequency-modulated sweeps occupy less than 20 kHz instantaneous bandwidth. This filter provides a complementary rejection mechanism to the CNN classifier, operating at the energy detection stage to reduce candidate volume. It particularly helps when [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] might be triggered by narrow-band noise artifacts.

---

Source:
- DECISIONS.md (ADR-003) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] -- peak mode's noise susceptibility mitigated by this filter
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- this filter operates within the first stage

Topics:
- [[detection]]
