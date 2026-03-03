---
description: "Waveform-based SSL (CPC, wav2vec2, HuBERT) expects 16-48 kHz audio, creating a 10-19x gap with 300 kHz USV recordings — spectrograms abstract away the raw sample rate"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[bioacoustic-ssl]]"
---

# Spectrogram-based SSL avoids the sample rate mismatch that limits waveform-based models for USV analysis

Most SSL audio models operate at 16 kHz (speech) or 32 kHz (general audio). USV recordings at 300 kHz present a fundamental mismatch for waveform-based approaches, because the convolutional feature extractors in models like wav2vec2 and HuBERT are designed with specific kernel sizes and strides tuned for their expected sample rates. Three options exist for handling this gap: (1) downsample to the model's expected rate, which loses all ultrasonic content and therefore defeats the purpose; (2) retrain from scratch on 300 kHz data, which requires substantial compute and domain-specific pretraining data; or (3) train on spectrograms rather than raw waveforms.

Spectrogram-based approaches abstract away the raw sample rate, since the spectrogram can be computed at any resolution and treated as an image-like input. The STFT parameters (window size, hop length, frequency range) determine what acoustic content is captured, but the resulting spectrogram has a fixed dimensionality regardless of the original sample rate. This is why MAE-based methods that operate on spectrogram patches and our spectrogram-input transformer pipeline are better suited for USV analysis than waveform-based approaches like CPC and wav2vec2.

The practical consequence is that our pipeline's spectrogram-first design is not merely a convenient engineering choice but a necessary adaptation for working with ultrasonic audio data that most foundation models were never designed to handle.

---

Source:
- cpc-vs-mae-bioacoustic-representation-learning-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[no self-supervised foundation model has been applied to rodent USV data]] — the mismatch is a key reason for this gap
- [[DeepSqueak uses constant-duration FFT windows making it inherently sample-rate agnostic]] — same principle applied to detection

Topics:
- [[signal-processing]]
- [[bioacoustic-ssl]]
