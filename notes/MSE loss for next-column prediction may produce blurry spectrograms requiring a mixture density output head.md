---
description: MSE averages over multimodal futures, producing blurry predictions; a GMM output head with K=5-10 components is the planned fallback if blurriness is observed.
type: finding
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head

MSE loss minimizes expected squared error, which means the optimal MSE prediction is the mean over all possible continuations weighted by their probability. When multiple continuations are plausible — for example, a USV could rise or fall in frequency after a flat segment — the MSE-optimal prediction averages these possibilities, producing a blurry, diffuse prediction that corresponds to no single real spectrogram. This is a known pathology of regression losses applied to multimodal prediction targets.

The practical consequence for spectrogram prediction: if a transition point in a vocalization admits several possible acoustic futures (different syllable types, pauses, frequency jumps), the model will predict an averaged smear across those futures rather than committing to any one. This blurriness would propagate into the hidden state representations that VQ-VAE later discretizes — representations encoding "ambiguous between several things" are harder to cluster into clean discrete codes than representations encoding "this specific acoustic pattern."

The planned response is staged: start with MSE (simpler implementation, well-understood loss landscape, faster iteration) and monitor prediction sharpness qualitatively by visualizing predicted vs actual spectrograms every 10 epochs. If predictions are blurry, upgrade to a Gaussian Mixture Model output head with K=5-10 components, where the model predicts mixture weights, means, and variances rather than a single point estimate. This connects to [[transformer-first then VQ-VAE avoids forcing premature discretization]] — the discretization quality depends on representation quality, which depends on prediction sharpness. It also connects to [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]], since multimodal prediction is intrinsic to the sequential structure of USV streams.

A radically different approach to the multimodality problem comes from generative modeling: [[diffusion models factorize generation into many small denoising steps each narrowing the possibility space]], where instead of predicting the full output in one step (which forces mode-averaging under MSE), the model commits to one mode early and refines within it across many small steps. This is why [[whether flow matching could replace VQ-VAE for unsupervised USV representation learning]] considers flow matching as a potential alternative for spectrogram generation.

---

Source: [ROADMAP](../ROADMAP.md), Phase 8
