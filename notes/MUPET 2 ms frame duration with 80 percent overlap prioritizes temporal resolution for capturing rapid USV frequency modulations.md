---
description: "512-point FFT at 250 kHz: 500-sample frames (2 ms), 400-sample hop (1.6 ms, 80% overlap), Hamming window; ~0.5 kHz frequency resolution vs our 586 Hz"
type: baseline
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[signal-processing]]"
---

# MUPET 2 ms frame duration with 80 percent overlap prioritizes temporal resolution for capturing rapid USV frequency modulations

MUPET uses a 512-point STFT at 250 kHz with 500-sample frames (2 ms), 400-sample hop (1.6 ms, giving 80% overlap), and a Hamming window. The frequency resolution is approximately 0.5 kHz (250000/512), slightly better than our 586 Hz (300000/512).

The 80% overlap (vs our 75%) gives marginally smoother temporal coverage, while the 2 ms frame duration is comparable to our 1.7 ms. Since [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]], both systems make similar temporal-vs-spectral tradeoff decisions. MUPET's Hamming window differs from our Hann window — both have similar main lobe widths, but since [[Hann window provides good sidelobe suppression for spectral analysis of USVs]], the choice is largely equivalent for USV analysis.

The convergence of MUPET's parameters with ours — despite being designed independently — validates that ~2 ms frames with ~500 Hz frequency bins is close to optimal for the USV frequency modulation rates.

---

Source: mupet-sample-rate-usv-analysis-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- our comparable STFT parameters
- [[75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection]] -- our 75% vs MUPET's 80%
- [[Hann window provides good sidelobe suppression for spectral analysis of USVs]] -- Hann vs Hamming comparison
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- the fundamental tradeoff both systems navigate
- [[AMVOC uses 2ms non-overlapping spectrogram windows giving 0.5 kHz frequency resolution at the expense of temporal smoothness]] -- convergent 2 ms window duration but opposite overlap strategy: MUPET uses 80% overlap for smooth temporal coverage, AMVOC uses 0% overlap (independent frames) and compensates with a median filter; same frame size, fundamentally different temporal continuity approaches

Topics:
- [[signal-processing]]
