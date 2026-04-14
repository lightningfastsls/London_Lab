---
description: "2022 hybrid architecture — CNN extracts spatial spectrogram features, bidirectional LSTM captures temporal context; outperforms DeepSqueak in recall and F1 under harsh conditions"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[detection-landscape]]"
---

# HybridMouse CNN plus BiLSTM first combined spatial and temporal features for USV detection outperforming DeepSqueak in low SNR

HybridMouse (2022) introduced the first hybrid architecture combining CNN spatial feature extraction with bidirectional LSTM (BiLSTM) temporal context modeling for USV detection. The CNN processes spectrogram patches to extract local features (frequency contours, harmonic structure), while the BiLSTM captures temporal dependencies across frames (call-to-call transitions, bout structure).

The hybrid outperformed DeepSqueak in recall and F1 score, particularly under "harsh experimental conditions" (low signal-to-noise ratio). This is relevant because since [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]], our recordings face similar low-SNR challenges. The temporal context provided by BiLSTM helps disambiguate noisy signals that look ambiguous in individual frames but become clearer when viewed in temporal sequence.

This architecture bridges the gap between spatial approaches (CNN/U-Net on single spectrograms) and temporal approaches (DAS on raw audio waveforms), suggesting that combining both modalities captures more information than either alone.

---

Source: usv-detection-methods-landscape-2024-2026-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- low SNR is our primary challenge; BiLSTM helps
- [[DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input]] -- temporal approach on raw audio; HybridMouse achieves temporal modeling on spectrograms
- [[self-attention provides O(1)-path global context from layer 1 while CNNs require many stacked layers to aggregate distant information]] -- the theoretical framing for why CNN alone needed BiLSTM: local receptive fields cannot reach distant temporal context without external help
- [[self-attention requires only O(1) sequential operations enabling full parallelization versus O(n) for RNNs]] -- the BiLSTM component pays O(n) sequential cost per sequence, which is why transformer-based alternatives can train dramatically faster on the same data

Topics:
- [[detection]]
