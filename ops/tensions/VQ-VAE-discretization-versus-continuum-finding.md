---
status: resolved
created: 2026-02-19
resolved: 2026-02-19
---

# VQ-VAE imposes discrete codes on a continuum that Goffinet showed resists discrete categorization

The fundamental methodological tension: Goffinet et al. (2021) showed that USVs form a continuum in acoustic feature space rather than discrete clusters. Yet our VQ-VAE architecture imposes discrete codes on this continuum. The justification is that VQ-VAE discovers data-driven codes rather than imposing predefined categories — the discretization is learned, not assumed. But this does not fully resolve the tension: if the underlying reality is truly continuous, is any discretization justified? Or does the VQ-VAE simply find a useful approximation?

## Conflicting Notes
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]]
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]]
- [[transformer-first then VQ-VAE avoids forcing premature discretization]]

## Resolution
**Resolved: pragmatic discretization of a continuum, not a categorical claim.**

Three factors dissolve the tension:
1. The v2 architecture ([[transformer-first then VQ-VAE avoids forcing premature discretization]]) means the VQ-VAE is a post-hoc interpretability lens, not a generative assumption. The transformer learns continuous representations first.
2. A codebook of K=64 doesn't claim 64 discrete "types" — it provides 64 reference points along the continuum, analogous to naming colors on a continuous spectrum. Useful labels ≠ ontological categories.
3. The empirical test ([[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]]) will validate whether the discretization captures meaningful structure.

Goffinet's finding remains correct (the continuum is real), AND the VQ-VAE is a valid tool (discretization is useful). No contradiction.
