---
status: pending
created: 2026-02-19
---

# VQ-VAE imposes discrete codes on a continuum that Goffinet showed resists discrete categorization

The fundamental methodological tension: Goffinet et al. (2021) showed that USVs form a continuum in acoustic feature space rather than discrete clusters. Yet our VQ-VAE architecture imposes discrete codes on this continuum. The justification is that VQ-VAE discovers data-driven codes rather than imposing predefined categories — the discretization is learned, not assumed. But this does not fully resolve the tension: if the underlying reality is truly continuous, is any discretization justified? Or does the VQ-VAE simply find a useful approximation?

## Conflicting Notes
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]]
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]]
- [[transformer-first then VQ-VAE avoids forcing premature discretization]]

## Resolution Path
May resolve empirically: if the VQ-VAE codebook entries are interpretable, well-utilized, and discriminate between wild and lab populations, the discretization is justified pragmatically even if the underlying reality is continuous. The [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] experiment will provide evidence.
