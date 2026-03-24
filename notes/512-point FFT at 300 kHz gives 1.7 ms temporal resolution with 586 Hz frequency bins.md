---
description: "Core STFT parameter choice balancing temporal precision for short USVs against frequency resolution for subtype discrimination"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# 512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins

USVs are short (10-500 ms) and narrow-band, requiring good temporal resolution to capture onset/offset precisely. At 300 kHz sample rate, n_fft=512 yields a frame duration of 512/300000 = 1.707 ms and frequency resolution of 300000/512 = 585.9 Hz/bin. The frequency bins spanning 20-120 kHz give ~171 bins. This parameterization favors temporal resolution since [[temporal resolution versus frequency resolution in STFT parameter selection]] is a fundamental tradeoff -- you cannot optimize both simultaneously.

---

Source:
- DECISIONS.md (ADR-002) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- the sample rate that determines these derived values
- [[75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection]] -- companion hop parameter
- [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] -- adequacy assessment
- [[MUPET 2 ms frame duration with 80 percent overlap prioritizes temporal resolution for capturing rapid USV frequency modulations]] -- independent convergence on nearly identical parameters (512-point FFT, ~2 ms frames, ~500-600 Hz bins) validates this parameterization
- [[MUPET operates at 250 kHz sample rate with minimum 90 kHz requirement covering the 25-125 kHz USV band]] -- cross-tool context: our 300 kHz vs MUPET's 250 kHz yields slightly different frequency resolution from the same FFT size

Topics:
- [[signal-processing]]
