---
description: "Goffinet et al. 2021 AVA tool found GMM clustering on 32-dim VAE latent space only supported k<=2 for mice, vs clean clustering in zebra finches"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs

Goffinet et al. (2021, *eLife*) applied Variational Autoencoders to mouse USV spectrograms using their "Autoencoded Vocal Analysis" (AVA) tool (`github.com/pearsonlab/autoencoded-vocal-analysis`). The VAE learned a 32-dimensional latent space, and critically: **Gaussian mixture model clustering on this latent space was only supported for k<=2 clusters in mice** -- a stark contrast to zebra finch syllables, which cluster cleanly.

This quantitative result is the strongest evidence that [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]]. While the continuum finding is often cited qualitatively (smooth interpolations exist between syllable types), the GMM k<=2 result provides a concrete, model-selection-based quantification: formal statistical clustering finds at most two groups in the mouse USV space. Traditional taxonomies with 10-20 categories (like DeepSqueak's k=20 from [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]]) are imposing structure that the data does not naturally support.

Our own lab data independently corroborates this: [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] — using density-based HDBSCAN (not GMM) on 7,864 calls from 5970/lmt_034, 96.6% collapsed into a single cluster. Cross-method convergence (GMM model selection AND HDBSCAN density estimation) on different datasets strengthens the continuum claim considerably.

This directly motivates our VQ-VAE approach: rather than clustering a learned latent space (which fails for mice), we let the codebook discover its own discretization of the continuum through the training objective. The [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] doesn't claim 64 natural categories -- it provides 64 reference points along a continuum.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Goffinet et al. (2021), *eLife* -- AVA tool, `github.com/pearsonlab/autoencoded-vocal-analysis`

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the qualitative continuum finding this quantifies
- [[Goffinet 2021 found 64 to 95 percent of traditional USV feature information captured in VAE latent space]] -- the same VAE retains acoustic information despite not supporting clustering
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- k-means finds k=20 where GMM finds k<=2 on different representations
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our approach that avoids explicit clustering
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- the tension this finding fuels
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- Holy & Guo's ~10-15 discrete types are directly contradicted by GMM finding k<=2
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the k<=2 result justifies the unsupervised branch as primary discovery tool, with supervised categories retained only for literature comparability
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- if formal clustering finds k<=2, comparing distributions in the continuous space is more justified than comparing categorical proportions

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
