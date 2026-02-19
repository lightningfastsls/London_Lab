---
description: "Cage impacts produce broadband vertical smears spanning many frequencies — rejected by the 8-10 ms minimum duration filter"
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

Topics:
- [[signal-processing]]
- [[detection]]
