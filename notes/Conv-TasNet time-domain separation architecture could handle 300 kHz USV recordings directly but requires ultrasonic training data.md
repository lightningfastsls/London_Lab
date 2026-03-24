---
description: "Fully convolutional time-domain architecture surpasses ideal time-frequency masking for speech separation — operates on raw waveforms so 300 kHz sample rate is natively supported"
type: method
confidence: speculative
meta_state: current
topics:
  - "[[detection-landscape]]"
  - "[[signal-processing]]"
---

# Conv-TasNet time-domain separation architecture could handle 300 kHz USV recordings directly but requires ultrasonic training data

Conv-TasNet (Luo & Mesgarani, IEEE 2019) is a fully convolutional time-domain architecture for source separation that surpasses ideal time-frequency masking for speech. Its architecture — linear encoder, temporal convolutional network with stacked dilated convolutions, mask estimation, decoder — operates directly on raw waveforms, meaning it can natively handle any sample rate including 300 kHz.

Unlike spectrogram-based methods where FFT parameters constrain resolution, Conv-TasNet learns its own encoding from the waveform. SepFormer (2021) extends this with dual-path Transformers for long-range temporal modeling. However, both approaches require substantial training data at the target sample rate.

Since no ultrasonic vocalization training data exists for these architectures, applying them to USVs would require training on [[synthetic mixture training is the standard approach for training bioacoustic source separation networks]] using single-animal USV recordings. The 300 kHz sample rate creates much higher computational cost for time-domain methods compared to speech (16 kHz), requiring roughly 19x more computation per second of audio. This contrasts with the STFT-based mask approach used by BioCPPNet, where the spectrogram dimensions can be controlled independently of sample rate.

The choice between Conv-TasNet (time-domain) and BioCPPNet (spectrogram-domain) for USV separation mirrors the broader tension identified in [[spectrogram-based SSL avoids the sample rate mismatch that limits waveform-based models for USV analysis]] — spectrogram approaches abstract away the sample rate at the cost of fixed time-frequency resolution.

---

Source:
- overlapping-usv-source-separation-state-of-art-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[synthetic mixture training is the standard approach for training bioacoustic source separation networks]] — how training data would be generated
- [[spectrogram-based SSL avoids the sample rate mismatch that limits waveform-based models for USV analysis]] — the complementary sample rate consideration
- [[BioCPPNet U-Net architecture with permutation-invariant training enables single-channel bioacoustic source separation]] — the spectrogram-domain alternative

Topics:
- [[detection]]
- [[signal-processing]]
