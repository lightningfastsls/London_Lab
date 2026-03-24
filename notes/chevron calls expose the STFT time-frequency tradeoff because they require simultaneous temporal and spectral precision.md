---
description: "Chevron calls (short up-then-down trajectory) need good time resolution to see brevity AND good frequency resolution to see the trajectory"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[signal-processing]]"
---

# Chevron calls expose the STFT time-frequency tradeoff because they require simultaneous temporal and spectral precision

Chevron calls — short USVs with an up-then-down frequency trajectory — are the concrete case that makes the abstract [[temporal resolution versus frequency resolution in STFT parameter selection]] tradeoff tangible. You need good time resolution to see that the call is short and good frequency resolution to see the up-then-down trajectory. With [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]], short calls (<15ms) get somewhat smeared, but this is acceptable for binary detection (USV vs noise). For classification tasks that require distinguishing chevron calls from other shapes, the smearing may be more problematic. The moderate n_fft=512 is a deliberate compromise — accepting some smearing on the shortest calls in exchange for adequate frequency resolution across the full repertoire.

---

Source:
- Researcher brain-dump on preprocessing insights (2026-02-19)

Relevant Notes:
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- the general tradeoff this example concretizes
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- the specific parameter compromise
- [[visualization STFT uses different parameters than detection STFT by design]] -- visualization n_fft=2048 gives finer frequency resolution that would render chevron trajectories more clearly
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- chevrons are one of the named discrete call types in the traditional taxonomy

Topics:
- [[signal-processing]]
