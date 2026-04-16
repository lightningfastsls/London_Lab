---
description: "DeepSqueak v3.1 introduced VAE clustering alongside k-means and ARTwarp, better capturing the continuous nature of mouse USV variation"
type: finding
confidence: proven
conditions:
  - DeepSqueak v3.1+
meta_state: current
source: "inbox/deepsqueak-usv-syllable-classification-practical-guide.md"
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# DeepSqueak v3.1 added VAE-based contour-invariant clustering as upgrade over k-means for continuous USV variation

DeepSqueak version 3.1 introduced Variational Autoencoder (VAE)-based contour-invariant clustering as a significant upgrade to its unsupervised classification capabilities. This sits alongside the existing k-means clustering (which operates on contour shape derivatives, frequency contours, and duration, all z-score normalized) and ARTwarp methods.

The VAE approach is particularly well-suited to USV analysis because it learns a continuous latent space rather than forcing hard cluster assignments. This aligns with the finding from Goffinet et al. (2021) that mouse USVs form a continuous manifold rather than discrete categories. Where k-means must commit to exactly k clusters (the original DeepSqueak paper found k=20 via elbow method), the VAE captures gradients of variation between syllable types.

t-SNE visualization is built into DeepSqueak for inspecting the resulting cluster distributions from any method.

This represents a broader trend in USV analysis: moving from discrete taxonomies (Holy & Guo 2005, Scattoni 2008) toward continuous representation approaches — the same direction our VQ-VAE pipeline explores, though with a different architecture and purpose.

---

Source:
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md

Relevant Notes:
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- the predecessor clustering approach
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- the paradigm shift this feature responds to
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] -- evidence for continuous structure
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- our strategy that parallels this approach
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- another unsupervised approach using handcrafted features vs VAE learned features
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- practical use of DeepSqueak clustering before custom VQ-VAE is ready
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] -- parallel autoencoder-based unsupervised clustering tool: AMVOC uses a standard AE + k-means while DeepSqueak v3.1 uses VAE; same design space, different regularization strategies for the same continuum problem

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-tools]]
