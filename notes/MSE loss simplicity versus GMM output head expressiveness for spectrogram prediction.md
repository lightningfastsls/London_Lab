---
description: Tension between MSE simplicity and GMM expressiveness for spectrogram column prediction when USV transitions may be multimodal.
type: finding
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction

A core tension in the transformer output head design is whether the prediction target is unimodal or multimodal. MSE loss is simple, well-understood, and mathematically equivalent to maximum likelihood under a Gaussian assumption — it is sufficient whenever predictions are unimodal. However, spectrogram prediction at USV transitions may be genuinely multimodal: a call could rise or fall in frequency, pause or continue, and the next column distribution is not a single Gaussian but a mixture.

A Gaussian Mixture Model head with K=5–10 components captures this multimodality by outputting mixture weights, per-component means, and per-component variances. The output dimensionality grows by a factor of 2K+1 relative to plain regression, and the loss becomes negative log-likelihood under the mixture. This adds complexity in both implementation and interpretation: the loss landscape is non-convex, and training is sensitive to initialization of the mixture weights.

The roadmap resolves this tension pragmatically: start with MSE and upgrade only if predictions appear blurry. Blurriness in spectrogram prediction is a visible symptom of multimodality — the model averages over possible futures rather than committing to one. If qualitative inspection of predicted columns shows smearing at call transitions, that is the signal to upgrade to a GMM head. This approach avoids over-engineering before evidence of the problem.

This connects to [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]], which documents the failure mode, and [[transformer-first then VQ-VAE avoids forcing premature discretization]], which notes that the same pragmatic staging applies to the broader architecture choice. The blurriness check is built into the monitoring protocol of [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]], where predicted vs actual spectrogram visualizations every 10 epochs provide the diagnostic signal for upgrading to GMM.

---

Source: [[ROADMAP.md]]

Relevant Notes:
- [[MSE loss for next-column prediction may produce blurry spectrograms requiring a mixture density output head]] -- the specific failure mode this tension addresses
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- same pragmatic staging philosophy applies to loss function choice
- [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]] -- predicted vs actual visualizations provide the diagnostic signal for upgrading
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- training stability is a prerequisite before loss function complexity matters

Topics:
- [[classification]]
