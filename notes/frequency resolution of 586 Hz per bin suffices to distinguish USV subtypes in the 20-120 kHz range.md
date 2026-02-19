---
description: "At 300000/512 = 586 Hz/bin, the ~171 bins spanning 20-120 kHz provide adequate granularity for USV subtype discrimination"
type: finding
confidence: likely
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range

The frequency resolution of 586 Hz/bin derived from n_fft=512 at 300 kHz provides ~171 frequency bins in the 20-120 kHz USV range. Mouse USV subtypes differ in their frequency characteristics -- flat calls, frequency-modulated sweeps, chevrons, and complex syllables all have distinct spectral signatures. A resolution of 586 Hz is sufficient to capture these distinctions because the frequency differences between subtypes are typically on the order of several kHz, well above the bin width. The visualization pipeline uses finer resolution (61 Hz/bin) when detailed spectral examination is needed.

---

Source:
- DECISIONS.md (ADR-002) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- the STFT parameters producing this resolution
- [[visualization STFT uses different parameters than detection STFT by design]] -- finer resolution available for display
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- 586 Hz adequacy is the consequence of accepting the temporal-resolution side of this tradeoff

Topics:
- [[signal-processing]]
