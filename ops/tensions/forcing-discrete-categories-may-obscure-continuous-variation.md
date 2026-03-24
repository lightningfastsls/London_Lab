---
status: archived
created: 2026-02-23
archived: 2026-03-02
archived_by: rethink-2026-03-02
---

# Forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations

Traditional USV analysis forces calls into discrete syllable types (Holy & Guo, Scattoni, DeepSqueak clusters, etc.) then compares type proportions between populations. But if USVs form a continuous manifold as [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] demonstrated, then the category boundaries are arbitrary -- and the continuous variation within categories is lost. Two populations might differ in subtle frequency modulation patterns that fall within the same "chevron" category, making the difference invisible to categorical analysis.

This tension is more than theoretical: [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]], meaning formal statistical clustering does not support the 10-20 categories commonly used. Yet nearly all published USV comparisons use discrete categories -- and our planned pipeline includes a supervised classification component for literature comparability.

## Conflicting Notes
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- our planned approach uses categories AND continuous methods
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- continuous alternative
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- imposes k=20 categories
- [[VocalMat represents supervised classification with predefined categories achieving 86 percent accuracy on 11 USV types]] -- imposes k=11 categories
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- evaluates categories by predictive power, partially dissolving the tension

## Possible Resolution
The dual approach may dissolve this: use categorical analysis for literature comparability, but treat continuous latent-space analysis as the primary scientific measure. VQ-VAE codebook entries are explicitly "reference points along a continuum" not "natural categories" (see resolved tension: [[VQ-VAE imposes discrete codes on a continuum that Goffinet showed resists discrete categorization]]). The SIS method from Hertz et al. can validate whether any categorization captures meaningful structure.
