---
description: "Best et al 2023 used VGG-based perceptual loss instead of MSE for autoencoder training on spectrograms — produced 256-dim embeddings that clustered better across 6 species"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
  - "[[signal-processing]]"
---

# Perceptual loss outperforms pixel-level MSE for autoencoder spectrogram representation learning

Best et al. (2023) trained convolutional autoencoders on spectrogram representations using perceptual loss (VGG-based rather than pixel-level MSE), producing 256-dimensional bottleneck representations that clustered effectively across eight datasets spanning six species. The perceptual loss encourages the autoencoder to preserve high-level structural features rather than pixel-exact reconstruction.

This matters for spectrogram learning because pixel-level MSE treats all frequency bins equally, but the perceptually important features of a vocalization — fundamental frequency contour, harmonic structure, amplitude envelope — are distributed across many pixels. A model optimizing MSE will spend capacity on reconstructing noise and background, while perceptual loss focuses on the features that a pretrained vision network considers structurally important.

Since [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]], perceptual loss offers an alternative path: rather than making MSE work better with a more complex output head, use a loss function that already knows what matters. For our VQ-VAE pipeline, this suggests perceptual loss could improve the encoder's spectrogram representations before quantization.

---

Source: unsupervised-clustering-bioacoustic-vocalizations-2025-research-2026-02-27 (archived to archive/inbox/)

Relevant Notes:
- [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]] -- perceptual loss as alternative to MSE workarounds
- [[MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction]] -- perceptual loss is a third option in this tradeoff
- [[diffusion models factorize generation into many small denoising steps each narrowing the possibility space]] -- diffusion/flow matching provides a fourth approach: sidestep mode-averaging by iterative refinement rather than changing the loss

Topics:
- [[bioacoustic-ssl]]
- [[signal-processing]]
