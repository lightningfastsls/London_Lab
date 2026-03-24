---
description: "Cage impacts span many frequencies in under 8 ms — caught by both the minimum duration filter and the 20 kHz max bandwidth filter, providing the functional rationale for the duration boundary"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[detection]]"
---

# Transient cage noises produce broadband vertical smears rejected by the minimum duration filter

Transient cage noises — from cage impacts, animal movements, or mechanical disturbances — appear as broadband vertical smears in the spectrogram, spanning many frequencies in a very short time. These are visually and acoustically distinct from USVs (which are narrowband and sustained). The minimum duration filter of approximately 8-10 ms rejects these transient artifacts because they are too brief to be USVs. This provides the functional rationale for the duration boundary discussed in [[whether very short USV signals near the 8-10 ms boundary should be included or excluded from training]] — the 8-10 ms threshold exists partly for artifact rejection, not just USV definition. These transients are also caught by the [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] because they span more than 20 kHz.

---

Source:
- Researcher brain-dump on preprocessing insights (2026-02-19)

Relevant Notes:
- [[whether very short USV signals near the 8-10 ms boundary should be included or excluded from training]] -- the duration boundary this artifact pattern motivates
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- also catches these broadband transients
- [[electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs]] -- the other major artifact type (narrowband vs broadband)
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- the recording environment that produces these cage noise artifacts
- [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]] -- the 20 kHz lower bound avoids admitting the worst of the broadband cage noise

Topics:
- [[signal-processing]]
- [[detection]]
