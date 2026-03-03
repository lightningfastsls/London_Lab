---
description: "CNN classifiers learn from frequency contour shape, so at least 512-point FFT (1024 preferred at 300 kHz for ~293 Hz resolution) is needed for classification"
type: finding
confidence: likely
conditions:
  - "sample_rate: 300000"
  - "task: CNN classification on spectrogram patches"
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# Fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches

A critical finding from recent USV classification work: for CNN-based syllable classification on spectrogram patches, **fine frequency resolution matters far more than time resolution**. The network learns from the frequency contour "skeleton" -- the shape of how frequency changes over time -- and resolving that shape requires sufficient frequency bins. The recommendation is at least 512 FFT points; **1024 is preferable at 300 kHz** for ~293 Hz frequency resolution.

This has direct implications for our STFT parameter choices. Our detection pipeline uses [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]], which is optimized for temporal precision to catch short transients. For a future classification stage, we may need a **separate STFT configuration** with longer windows (1024+ samples) to provide the finer frequency resolution that CNNs need for syllable typing. This follows the same logic as [[visualization STFT uses different parameters than detection STFT by design]] -- different tasks warrant different spectral parameters.

The spectrogram patch extraction recommendation for classification: extract detected USV audio with ~15 ms padding, compute STFT with 512-1024 point FFT, Hamming window, 75% overlap, restricted to 25-125 kHz, and resize to 128x128 or 224x224 for transfer learning.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)

Relevant Notes:
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- our detection FFT optimized for time resolution
- [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] -- detection-adequate but potentially insufficient for fine classification
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] -- DeepSqueak chose finer frequency resolution
- [[chevron calls expose the STFT time-frequency tradeoff because they require simultaneous temporal and spectral precision]] -- the fundamental tradeoff this finding resolves toward frequency
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- alternative spectral representation that may bypass STFT resolution limits

Topics:
- [[signal-processing]]
- [[classification-methodology]]
