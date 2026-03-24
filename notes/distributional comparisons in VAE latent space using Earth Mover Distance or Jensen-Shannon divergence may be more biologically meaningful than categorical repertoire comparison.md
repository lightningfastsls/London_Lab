---
description: "If USVs are continuous, comparing population distributions in latent space (EMD, JSD) captures variation that forcing calls into categories would obscure"
type: method
confidence: experimental
conditions:
  - "VAE or VQ-VAE latent space trained"
  - "sufficient samples per population"
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[experimental-methods]]"
---

# Distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison

If mouse USVs form a continuous spectrum as [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]], then the traditional approach of computing proportions of discrete syllable types per animal and testing for differences may be fundamentally flawed. **Distributional comparisons in latent space** -- using Earth Mover's Distance (EMD) or Jensen-Shannon divergence (JSD) on VAE/VQ-VAE embeddings -- may be more biologically meaningful because they compare the full continuous distributions rather than forcing USVs into categorical bins.

This is a methodological alternative to the standard framework for [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]]:
- **Traditional**: Classify each USV -> compute per-animal type proportions -> PERMANOVA on Bray-Curtis dissimilarity
- **Latent space**: Embed each USV in VAE space -> compare population-level embedding distributions -> EMD or JSD

The latent space approach preserves continuous variation that discrete categorization loses. Two USVs that differ subtly within a "chevron" category would be indistinguishable in categorical analysis but distinguishable in latent space. This is especially relevant because [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] -- if formal clustering fails, forcing categories for comparison is doubly problematic.

The tradeoff: latent space comparisons are harder to interpret ("population A's USV distribution is shifted by 0.3 EMD units") versus categorical comparisons ("population A produces 20% more chevron calls"). The [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] strategy accommodates both.

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)

Relevant Notes:
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the continuum finding motivating this method
- [[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]] -- quantitative evidence categories are artificial
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question this method serves
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- the tension this method resolves
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the broader strategy this method fits within
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- VQ-VAE provides discrete codes that could also be compared distributionally
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- the categorical JSD counterpart to latent-space JSD; comparing both approaches on the same data tests whether categories lose information
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- the standard categorical test this latent-space approach aims to complement or surpass
- [[whether flow matching could replace VQ-VAE for unsupervised USV representation learning]] -- flow matching produces continuous trajectories that are naturally suited to distributional comparison, bypassing the discretization tension entirely

Topics:
- [[unsupervised-usv-discovery]]
- [[experimental-methods]]
