---
description: "SpectrogramConfig uses n_fft=2048 with zero-padding to 4096 (61 Hz/bin) while detection uses n_fft=512 (586 Hz/bin)"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# visualization STFT uses different parameters than detection STFT by design

The visualization pipeline (`SpectrogramConfig`) and the detection pipeline (`DetectionConfig`) intentionally use different STFT parameters. Visualization uses n_fft=2048 with zero-padding to 4096, giving 61 Hz/bin frequency resolution -- optimized for display clarity. Detection uses n_fft=512 giving 586 Hz/bin -- optimized for temporal resolution needed to capture USV onset/offset precisely. This divergence is deliberate, since [[temporal resolution versus frequency resolution in STFT parameter selection]] means optimizing for one use case necessarily compromises the other.

---

Source:
- DECISIONS.md (ADR-002) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- the detection STFT parameters
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- the fundamental tradeoff driving this divergence

Topics:
- [[signal-processing]]
