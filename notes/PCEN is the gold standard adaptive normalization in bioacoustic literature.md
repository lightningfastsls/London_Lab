---
description: "Lostanlen et al 2019 PLOS ONE — operates at spectrogram level before the CNN, but requires model retraining, making it a next-iteration improvement rather than a drop-in for existing pipelines"
type: method
confidence: proven
conditions: []
meta_state: current
created: 2026-03-29
topics:
  - "[[detection]]"
  - "[[signal-processing]]"
---

# PCEN is the gold standard adaptive normalization in bioacoustic literature

Per-Channel Energy Normalization (PCEN) replaces the conventional log-magnitude spectrogram representation with an adaptive normalization scheme that adjusts gain independently for each frequency channel. The core mechanism applies automatic gain control per frequency bin, because different frequency bands experience different noise floor levels depending on recording conditions, equipment characteristics, and environmental factors. Therefore PCEN naturally compensates for non-stationary noise without requiring explicit noise estimation or manual threshold tuning.

The key advantage over log-magnitude spectrograms is that PCEN adapts to varying noise floors per channel rather than applying a uniform compression. Log compression treats all energy levels equally — a 60 dB signal in a quiet recording and a 60 dB signal in a noisy recording produce the same log-magnitude value, but they carry very different information content. PCEN's adaptive gain means the representation reflects signal-to-noise rather than absolute energy, which is fundamentally what a detector needs to distinguish vocalizations from background.

However, PCEN changes the input representation fed to the CNN, because the statistical properties of PCEN-normalized spectrograms differ substantially from log-magnitude spectrograms. A model trained on log-magnitude inputs cannot simply switch to PCEN inputs without retraining — the learned feature detectors in early convolutional layers would encounter unfamiliar input distributions. For our current pipeline, we therefore use [[per-recording normalization compensates for varying noise floors across recording sessions]] as a pragmatic alternative that operates within the existing model architecture. PCEN is recommended for the next model training iteration where retraining is already planned and the representation change can be incorporated from the ground up.

---

Source:
- archive/inbox/post-processing-pipeline-research.md (2026-03-27)

Relevant Notes:
- [[per-recording normalization compensates for varying noise floors across recording sessions]] -- the current pragmatic alternative that avoids model retraining
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- the sampling context within which PCEN would operate
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- the STFT parameters that define the spectrogram PCEN would normalize

Topics:
- [[detection]]
- [[signal-processing]]
