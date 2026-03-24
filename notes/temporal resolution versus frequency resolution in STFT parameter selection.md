---
description: "Fundamental tradeoff in spectral analysis -- n_fft=512 favors time resolution, n_fft=2048 favors frequency resolution"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# temporal resolution versus frequency resolution in STFT parameter selection

The STFT has a fundamental tradeoff: longer windows (larger n_fft) give finer frequency resolution but coarser temporal resolution, while shorter windows give finer temporal resolution but coarser frequency resolution. This is a consequence of the uncertainty principle in signal processing -- you cannot simultaneously have arbitrarily precise measurements of both time and frequency. In this project, the detection pipeline uses n_fft=512 (1.7 ms temporal, 586 Hz spectral) to prioritize capturing USV onset/offset precisely. The visualization pipeline uses n_fft=2048 (6.8 ms temporal, 61 Hz spectral) to prioritize spectral detail for human examination. Since [[visualization STFT uses different parameters than detection STFT by design]], this divergence is intentional and reflects different optimization targets for different use cases.

---

Source:
- DECISIONS.md (ADR-002) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- detection-optimized parameters
- [[visualization STFT uses different parameters than detection STFT by design]] -- the consequence of this tradeoff
- [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] -- adequacy at the chosen operating point
- [[chevron calls expose the STFT time-frequency tradeoff because they require simultaneous temporal and spectral precision]] -- concrete example: short calls (<15ms) get smeared but this is acceptable for binary detection

Topics:
- [[signal-processing]]
