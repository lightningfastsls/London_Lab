---
description: "Hop length 128 at n_fft=512 gives 75% frame overlap and 0.427 ms hop duration for smooth temporal coverage"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# 75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection

With n_fft=512, a hop_length of 128 gives 75% overlap between consecutive STFT frames. The hop duration is 128/300000 = 0.427 ms. This high overlap ensures smooth temporal coverage -- each time point is covered by multiple overlapping windows, reducing the chance of missing brief USV onsets or offsets. The 75% overlap is standard for spectral analysis applications where temporal smoothness matters.

---

Source:
- DECISIONS.md (ADR-002) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- parent STFT parameter decision
- [[segment continuity bridges brief amplitude dips that fragment single USVs]] -- downstream consumer of smooth temporal data
- [[AMVOC uses 2ms non-overlapping spectrogram windows giving 0.5 kHz frequency resolution at the expense of temporal smoothness]] -- opposite design choice: AMVOC uses 0% overlap and compensates with median filtering, trading smooth temporal coverage for computational speed and frame independence

Topics:
- [[signal-processing]]
