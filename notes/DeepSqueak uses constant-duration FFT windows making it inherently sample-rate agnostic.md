---
description: "DeepSqueak specifies FFT windows in seconds not samples, so spectrograms from 250 kHz and 300 kHz recordings are directly comparable"
type: finding
confidence: proven
conditions:
  - verified in DeepSqueak documentation for mouse USV settings
meta_state: current
source: "inbox/deepsqueak-usv-syllable-classification-practical-guide.md"
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# DeepSqueak uses constant-duration FFT windows making it inherently sample-rate agnostic

DeepSqueak's internal spectrogram parameters are specified as time durations rather than sample counts. For mouse USVs, this means 3.2 ms FFT windows with 2.8 ms overlap. The documentation explicitly states that "spectrograms are created using FFT windows of constant duration, rather than constant sample numbers, so other sample rates are accepted."

This design means a 300 kHz WAV file works without any parameter changes. At 300 kHz, the 3.2 ms window translates to 960 samples and the 2.8 ms overlap to 840 samples — but DeepSqueak handles this conversion internally. The tool was primarily tested at 250 kHz sample rate, so our 300 kHz recordings represent a slightly higher-than-typical but fully supported configuration.

This is a good design pattern: duration-based STFT parameters make spectrograms comparable across recording setups, since the time-frequency resolution is held constant regardless of sample rate.

---

Source:
- DeepSqueak documentation and README
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md

Relevant Notes:
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] -- the specific parameter translation at our sample rate
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- our recording standard
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- the fundamental tradeoff DeepSqueak's parameters balance

Topics:
- [[signal-processing]]
- [[classification-tools]]
