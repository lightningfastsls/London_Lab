---
description: "Standard Hann window chosen for STFT computation -- good sidelobe suppression with moderate main lobe width"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# Hann window provides good sidelobe suppression for spectral analysis of USVs

The Hann (Hanning) window is used for all STFT computation in the project. It provides a good balance between main lobe width and sidelobe suppression -- sidelobes are at -31 dB relative to the main lobe, which prevents spectral leakage from strong frequency components bleeding into adjacent bins. This is particularly important for USV analysis since [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] relies on accurate energy localization within the frequency band. Alternative windows (Hamming, Blackman, Kaiser) offer different tradeoffs but Hann is the standard choice for general spectral analysis.

---

Source:
- DECISIONS.md (ADR-002) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- companion STFT parameter
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] -- downstream consumer that benefits from clean spectral estimates
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- window function choice also affects the resolution tradeoff via main lobe width

Topics:
- [[signal-processing]]
