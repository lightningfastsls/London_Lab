---
description: "Recordings are made in a shared lab, not a sound-attenuated room — background noise and electrical interference are expected"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[detection]]"
---

# Shared lab space without sound attenuation explains why noise robustness is a primary design constraint

USV recordings are made in a shared lab space, not a dedicated sound-attenuated room or acoustic chamber. This means background noise from other lab equipment, electrical interference, cage sounds, and environmental noise are expected in the recordings. This environmental reality is the root cause behind several design decisions in the detection pipeline: [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] because the training data inherently contains noisy recordings, and [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] because the shared environment produces diverse noise patterns. The [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] is also motivated by the need to detect faint USVs against this noisy background.

---

Source:
- Researcher brain-dump on lab conventions (2026-02-19)

Relevant Notes:
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- training policy responding to this environment
- [[CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure]] -- noise-induced failure mode
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- permissive threshold to catch faint signals in noise
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- bandwidth filter as noise mitigation

Topics:
- [[experimental-methods]]
- [[detection]]
