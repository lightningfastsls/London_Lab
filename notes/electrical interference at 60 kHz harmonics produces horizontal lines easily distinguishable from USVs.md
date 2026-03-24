---
description: "Power line harmonics appear as perfectly horizontal lines in the spectrogram — a known artifact pattern in the shared lab environment"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[detection]]"
---

# Electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs

In the shared lab recording environment, electrical interference from equipment produces harmonics at 60 kHz (and potentially its multiples). These appear as perfectly horizontal lines in the spectrogram — constant-frequency artifacts that persist across time. They are easily distinguishable from USVs, which have frequency modulation (sweeps, jumps, chevrons) and finite duration. Since [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]], these electrical artifacts are expected. The [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] would not reject these because they are narrowband — but their constant-frequency, infinite-duration pattern makes them unlikely to trigger the energy detector's temporal structure requirements.

---

Source:
- Researcher brain-dump on preprocessing insights (2026-02-19)

Relevant Notes:
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- the environment causing these artifacts
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- catches broadband but not narrowband artifacts
- [[transient cage noises produce broadband vertical smears rejected by the minimum duration filter]] -- the other major artifact type

Topics:
- [[signal-processing]]
- [[detection]]
