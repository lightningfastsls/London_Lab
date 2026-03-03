---
description: "DAS (Deep Audio Segmenter) uses temporal convolutional networks with TensorFlow, achieving near-perfect USV detection but only on raw audio streams"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input

DAS (Deep Audio Segmenter, `pip install das`) uses temporal convolutional networks and achieves **98% precision / 99% recall** on mouse USVs -- the highest reported detection metrics among Python USV tools. It is well-maintained Python using TensorFlow/Keras with both CLI and Python API.

However, DAS operates on **raw audio streams with frame-level annotation** and is not designed to classify pre-extracted segments. Integrating it into a pipeline that already has detected USV segments would require restructuring to feed raw audio rather than spectrogram patches. This limitation means that despite its superior detection metrics, DAS cannot serve as a drop-in classification stage after our energy detector + CNN pipeline, contributing to the reality that [[no Python USV tool cleanly accepts pre-detected segments for classification creating an integration gap]].

DAS's near-perfect metrics on lab mouse USVs should be interpreted cautiously in light of [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- performance on wild mouse recordings may differ substantially. The raw audio input requirement also contrasts with our spectrogram-based approach where [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] provides the representation for detection.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)

Relevant Notes:
- [[CNN baseline of 89.7 percent precision and 93.8 percent recall at threshold 0.05 validates the two-stage detection approach]] -- our detection performance for comparison (F1 91.7%)
- [[no Python USV tool cleanly accepts pre-detected segments for classification creating an integration gap]] -- DAS contributes to this gap by requiring raw audio
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- caution on reported metrics
- [[WhisperSeg adapts OpenAI Whisper transformer for animal vocalization segmentation with positive cross-species transfer]] -- another raw-audio tool with superior cross-species transfer

Topics:
- [[detection]]
- [[classification-tools]]
