---
description: "BootSnap found gammatone spectrograms yield better CNN classification than standard STFTs, consistent with MUPET's gammatone filterbank choice"
type: finding
confidence: likely
conditions:
  - "task: supervised USV syllable classification"
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# Gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap

BootSnap (Abbasi et al., 2022) found that **gammatone spectrograms outperform standard STFTs** for CNN-based USV syllable classification. This finding is consistent with the independent choice by [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] to use gammatone filterbanks for USV feature extraction -- two different tools, from different research groups, arriving at the same spectral representation choice.

Gammatone filters model the auditory processing of the mammalian cochlea, providing logarithmic frequency spacing that naturally allocates more resolution to lower frequencies. For mouse USVs in the 25-125 kHz range, this means the filterbank provides non-uniform frequency resolution that may better capture the perceptually relevant features of the calls. Whether this biological inspiration is directly relevant to rodent auditory perception at ultrasonic frequencies is an open question, but the empirical performance advantage is documented.

This finding creates a practical consideration for our pipeline: our current STFT approach where [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] uses uniform frequency spacing. Adding a gammatone option for the classification stage could improve syllable typing accuracy. The tradeoff is increased implementation complexity versus [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]] which can be achieved with standard STFTs by using longer FFT windows.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Abbasi et al. (2022), *PLOS Computational Biology* -- BootSnap

Relevant Notes:
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- independent convergence on gammatone for USV analysis
- [[BootSnap snapshot ensemble CNN on gammatone spectrograms outperformed DeepSqueak classification with F1 67 percent on wild mice]] -- the tool that demonstrated this advantage
- [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]] -- STFT-based alternative to gammatone
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- our current uniform-spacing STFT for comparison
- [[Hann window provides good sidelobe suppression for spectral analysis of USVs]] -- STFT windowing that gammatone spectrograms bypass entirely

Topics:
- [[signal-processing]]
- [[classification-methodology]]
