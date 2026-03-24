---
description: "Recording hardware operates at 300 kHz, giving Nyquist coverage to 150 kHz -- well above the ~120 kHz USV upper bound"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# 300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz

Mouse USVs range up to ~120 kHz. Nyquist theorem requires at least 240 kHz to capture this range without aliasing. The recording hardware operates at 300 kHz, providing comfortable headroom up to 150 kHz. This is the canonical sample rate for all DSP code in the project (`DetectionConfig.sample_rate = 300_000`). Legacy `SpectrogramConfig.expected_sample_rate_hz = 250_000` is outdated from before the recording setup was finalized. All code must use `sr=300000` explicitly -- never rely on librosa's default sample rate.

---

Source:
- DECISIONS.md (ADR-001) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- derived STFT parameters depend on this rate
- [[auto sample rate reading from WAV headers prevents silent frequency miscalculation]] -- robustness measure for varying setups
- [[AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system]] -- the recording software configured to this sample rate
- [[MUPET operates at 250 kHz sample rate with minimum 90 kHz requirement covering the 25-125 kHz USV band]] -- MUPET uses 250 kHz (Nyquist 125 kHz), our 300 kHz provides 25 kHz more headroom
- [[frequency shifting USVs into the audible range could enable classification with standard audio foundation models]] -- the 300 kHz sample rate is 6-19x above the 16-48 kHz range foundation models expect, motivating pitch-shift strategies

Topics:
- [[signal-processing]]
