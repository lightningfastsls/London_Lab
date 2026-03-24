---
description: "DetectionConfig.auto_sample_rate=True reads actual sample rate from WAV files rather than assuming 300 kHz"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# auto sample rate reading from WAV headers prevents silent frequency miscalculation

While the canonical sample rate is 300 kHz, WAV files from different recording setups might have different rates. Hardcoding the sample rate would silently produce wrong STFT frequency bins -- a bin labeled as "50 kHz" might actually represent a different frequency. `DetectionConfig.auto_sample_rate = True` (the default) reads the actual sample rate from each WAV file header. The configured `sample_rate` field serves only as a fallback. This ensures STFT frequency bins are always correct for the actual data, since [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] is the expected but not guaranteed rate.

---

Source:
- DECISIONS.md (ADR-011) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- the expected canonical rate
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- derived values that depend on correct sample rate
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- wrong sample rate would invalidate the entire resolution calculation, making this safeguard critical

Topics:
- [[signal-processing]]
