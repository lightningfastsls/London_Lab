---
description: "Most audio foundation models expect 16-48 kHz input -- USVs at 25-120 kHz recorded at 300 kHz require frequency shifting, spectrogram-as-image treatment, or custom retraining"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[bioacoustic-ssl]]"
---

# the 300 kHz USV sample rate creates a domain shift challenge for applying audio foundation models

A practical barrier to using audio foundation models for USV classification is the sample rate mismatch. Perch 2.0, BEATs, AVES, and HuBERT are all pretrained on audio at 16-48 kHz sample rates, meaning they expect input where the highest representable frequency is 8-24 kHz. USVs occupy the 25-120 kHz range, recorded at 300 kHz with a Nyquist frequency of 150 kHz. Feeding 300 kHz audio directly into these models would either fail (wrong input dimensions) or produce meaningless embeddings (the model has never encountered these frequency patterns).

Three workarounds exist, each with distinct tradeoffs. First, the frequency-shift trick: pitch-shift USVs into the audible range before embedding, preserving relative spectral structure but losing absolute frequency information. This is theoretically sound if syllable types are defined by shape rather than absolute frequency. Second, treat mel-spectrograms as images and use vision-based few-shot methods -- CLIP embeddings on spectrogram images, for instance. This bypasses the audio domain entirely. Third, train a custom embedding model on ultrasonic audio, which produces the most faithful representations but requires significant labeled data and compute.

VocalMat and DeepSqueak both handle this challenge by operating on spectrogram images rather than raw audio, suggesting option two is the most practical near-term path. The spectrogram-as-image approach also aligns with our existing pipeline, which already generates spectrogram patches for CNN classification. Therefore, the most incremental path forward would be to embed our existing spectrogram patches through a vision model (or a vision-pretrained few-shot system) rather than attempting to adapt audio foundation models to our sample rate.

---

Source:
- few-shot-learning-animal-sound-classification-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[spectrogram-based SSL avoids the sample rate mismatch that limits waveform-based models for USV analysis]] -- the same challenge from the SSL perspective
- [[DeepSqueak uses constant-duration FFT windows making it inherently sample-rate agnostic]] -- spectrogram approach abstracts away sample rate
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- a fourth workaround: adapt foundation model weights to handle 300 kHz input via parameter-efficient fine-tuning
- [[QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost]] -- makes the "train a custom embedding model" workaround feasible on consumer hardware

Topics:
- [[classification-methodology]]
- [[signal-processing]]
- [[bioacoustic-ssl]]
