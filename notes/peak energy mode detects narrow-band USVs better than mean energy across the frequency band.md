---
description: "Using max energy per frame rather than mean prevents USV signal dilution by noise in non-USV frequencies"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[signal-processing]]"
---

# Peak energy mode detects narrow-band USVs better than mean energy across the frequency band

USVs are narrow-band signals occupying a small portion of the 25-110 kHz detection band. Computing mean energy across the entire band would dilute the USV signal with noise from non-USV frequencies, potentially dropping below threshold even when a strong USV is present. Peak mode (`energy_mode = "peak"`) takes the maximum energy value within the band for each time frame, capturing the strongest frequency component. This makes detection more sensitive to narrow-band signals. The tradeoff is slightly increased susceptibility to narrow-band noise, which is mitigated by [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]]. Mean mode remains available as an alternative for broadband signal detection scenarios.

---

Source:
- DECISIONS.md (ADR-012) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- threshold applied to peak energy
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- mitigates narrow-band noise susceptibility
- [[Hann window provides good sidelobe suppression for spectral analysis of USVs]] -- accurate spectral estimation supports peak mode

Topics:
- [[detection]]
- [[signal-processing]]
