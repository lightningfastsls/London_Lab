---
description: "finding that a vocal repertoire lives on a low-dim manifold is itself produced BY autoencoders — so using low-dim findings to reject autoencoders inverts the evidence chain"
type: claim
confidence: likely
conditions:
  - "applies to any argument that 'intrinsic dimensionality is low, so learned features cannot help'"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[classification]]"
  - "[[unsupervised-usv-discovery]]"
---

# low-dimensional intrinsic manifold argues for learned features rather than against them because bottleneck compression is how you find low-dim structure

A common misstep: "our manifold is low-dimensional (Goffinet VAE k≤2, our HDBSCAN 3 clusters), so high-dim autoencoder features won't help — the signal is simple." This inverts the evidence chain.

**The low-dim finding was produced by autoencoders.** Goffinet's 2021 result that mouse USVs lie on a low-dimensional manifold came from training a VAE and inspecting its latent space. The method produced the conclusion. Using the conclusion to argue against the method confuses output with input.

Bottleneck compression is precisely the tool for discovering low-dim structure when you don't know its dimensionality in advance. If the data has 2 intrinsic dimensions, a well-trained bottleneck will learn to use only 2 of its available dimensions effectively — you read off the low-dim structure *from* the learned representation.

**The correct inference:** Low intrinsic dimensionality argues *for* autoencoders, not against them. It means the method will work efficiently: the bottleneck doesn't need to preserve 80 dimensions if the data only has 3. What it argues *against* is running a classifier with 1280 independent features — that's the overparameterization concern, and it's solved by the PCA-after-bottleneck step.

**General principle:** When a finding was produced by a method, don't use the finding to reject the method. Either the method is valid (and produced a valid finding) or the finding is also suspect.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] — the VAE-produced low-dim finding this note addresses
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] — our own low-dim finding, produced by (UMAP) learned features
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] — the autoencoder method sometimes rejected via this inverted argument
- [[autoencoder bottleneck plus PCA extracts concepts because reconstruction forces the model to preserve axes of variation that matter]] — the mechanism that makes bottlenecks effective at finding low-dim structure
- [[four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization]] — this defense protects hypothesis 3 (learned features) from being eliminated by the inverted evidence chain

Topics:
- [[classification]]
- [[unsupervised-usv-discovery]]
