---
description: "architectural coupling constrains preprocessing choices — choosing BCE over MSE forces sigmoid activation and per-max or min-max normalization, ruling out z-score or log-scale inputs"
type: decision
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[signal-processing]]"
  - "[[unsupervised-usv-discovery]]"
---

# BCE loss with sigmoid output treats spectrogram pixels as independent probabilities requiring input normalization to 0-1 range

Binary Cross-Entropy loss (`nn.BCELoss()`) for spectrogram autoencoders treats each spectrogram pixel as an independent Bernoulli probability: L(y, y-hat) = -y*log(y-hat) - (1-y)*log(1-y-hat). This is a deliberate architectural choice over MSE, and it creates a cascading constraint chain:

1. **BCE requires outputs in [0, 1]** → the decoder's final activation must be sigmoid (not ReLU)
2. **BCE requires targets in [0, 1]** → input spectrograms must be normalized to this range
3. **Per-max normalization** (divide by max value) is the simplest way to satisfy this constraint, because [[per-spectrogram max normalization is the simplest effective preprocessing for BCE-based spectrogram reconstruction]]

The choice of BCE over MSE is not arbitrary. MSE treats the reconstruction error as Gaussian noise, spending model capacity equally on all pixel values. BCE instead treats each pixel as a probability of "energy present" — which better matches the sparse, binary-ish nature of USV spectrograms where most of the spectrogram is background and the USV trace occupies a small fraction of pixels. This is conceptually similar to how binary segmentation works.

However, the independence assumption (each pixel independent) means BCE ignores spatial structure — it cannot penalize, say, a blurred USV contour differently from a sharp one at the same average pixel values. This is where [[perceptual loss outperforms pixel-level MSE for autoencoder spectrogram representation learning]] offers an alternative that captures structural similarity rather than pixel-level probability.

For our VQ-VAE pipeline, the loss function choice cascades through the entire preprocessing and architecture: BCE+sigmoid, MSE+linear, or perceptual loss each imply different normalization schemes and output activations.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[perceptual loss outperforms pixel-level MSE for autoencoder spectrogram representation learning]] — alternative to both BCE and MSE
- [[MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction]] — the MSE side of this design choice; BCE, MSE, perceptual loss, and GMM heads form a four-way design space for spectrogram reconstruction/prediction
- [[per-spectrogram max normalization is the simplest effective preprocessing for BCE-based spectrogram reconstruction]] — the preprocessing this loss function requires
- [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]] — the architecture that instantiates this BCE+sigmoid choice, with sigmoid as the decoder's final activation

Topics:
- [[signal-processing]]
- [[unsupervised-usv-discovery]]
