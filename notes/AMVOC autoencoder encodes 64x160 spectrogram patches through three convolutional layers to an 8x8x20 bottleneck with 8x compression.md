---
description: "reference architecture for USV autoencoder design — 3 conv layers with MaxPool halving dimensions each stage, transposed convolutions for decoding, code discrepancy between 8-filter and 4-filter variants"
type: method
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification-methodology]]"
---

# AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression

AMVOC's convolutional autoencoder (Stoumpou et al. 2022) takes single-channel spectrogram patches of shape `(1, 64, 160)` — 64 time frames (128 ms at 2 ms/frame) by 160 frequency bins (30–110 kHz at 0.5 kHz resolution). The encoder uses three Conv2d layers (64 → 32 → 8 filters, all 3×3 with same-padding) each followed by 2×2 MaxPool with stride 2, halving spatial dimensions at each stage:

```
Input:  1 × 64 × 160  = 10,240 values
conv1:  64 × 32 × 80   (after pool)
conv2:  32 × 16 × 40   (after pool)
conv3:  8 × 8 × 20     (after pool) = 1,280 values  → 8× compression
```

The decoder mirrors the encoder using ConvTranspose2d with 2×2 kernels and stride 2 for upsampling (not nearest-neighbor + conv). All activations are ReLU except the final decoder layer which uses sigmoid, because [[BCE loss with sigmoid output treats spectrogram pixels as independent probabilities requiring input normalization to 0-1 range]].

The bottleneck is flattened to a 1,280-dimensional vector per USV for downstream processing — no global average pooling, just pure flatten. This 8× compression ratio was a deliberate choice: the paper tested 2, 4, and 8 bottleneck filters. Two filters lost significant information; 4 was sufficient for basic shapes; 8 was selected to capture all USV shape variations.

There is a **code discrepancy** between `training_task.py` (8 filters, matches the paper's final design) and `conv_autoencoder.py` (4 filters). The 4-filter version may be used for the semi-supervised retraining pipeline, but this is not documented clearly.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] — the high-level overview this note details architecturally
- [[SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines]] — different autoencoder architecture (MobileNetV2, 4.6M params) for detection rather than feature extraction
- [[Best et al 2023 showed learned audio embeddings match species-specific models for vocalization clustering across six species]] — 256-dim bottleneck with perceptual loss, deeper architecture
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] — our CNN also uses 3 conv blocks but adds BatchNorm and GlobalAvgPool that AMVOC's autoencoder lacks; both validate that 3-layer depth is sufficient for USV feature extraction
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] — the 1,280-dim AMVOC bottleneck feeds downstream PCA; our VQ-VAE codebook (K=64) performs an analogous dimensionality reduction but with learned discrete structure rather than linear projection
- [[perceptual loss outperforms pixel-level MSE for autoencoder spectrogram representation learning]] — the loss function paired with this architecture is BCE (sigmoid output), but Best 2023 showed perceptual loss produces better-clustering embeddings; swapping BCE for perceptual loss is an orthogonal upgrade to the architecture described here
- [[symmetric zero-padding for short USVs and center-cropping for long ones standardizes variable-duration inputs to fixed dimensions]] — how variable-length USVs are coerced into the fixed (1, 64, 160) input shape this architecture requires

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
