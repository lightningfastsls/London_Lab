---
description: v = alpha_t * epsilon - sigma_t * x_0 enables algebraic conversion between parameterizations while each exhibits different stability properties.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# the three diffusion prediction targets epsilon x0 and velocity form a triangular relationship where any two determine the third

Given a noisy sample x_t = alpha_t * x_0 + sigma_t * epsilon, three prediction targets are available:

- **Epsilon (noise)**: predict the Gaussian noise epsilon that was added
- **x_0 (signal/data)**: predict the clean data directly
- **Velocity**: predict v = alpha_t * epsilon - sigma_t * x_0

These form a closed algebraic system. From velocity: x_0 = alpha_t * x_t - sigma_t * v and epsilon = sigma_t * x_t + alpha_t * v. From epsilon: x_0 = (x_t - sigma_t * epsilon) / alpha_t. Any one target can be algebraically recovered from any other given the noise schedule parameters.

This equivalence is mathematically exact — the three networks have identical expressivity. The differences are entirely about *stability* and *implicit loss weighting*:

- Epsilon prediction has unstable gain O(1/b(t)) near clean data, as shown in [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]]
- x_0 prediction has unstable gain O(1/b(t)^2) near pure noise (implicit 1/SNR weighting over-emphasizes high-noise timesteps)
- Velocity prediction has bounded gain nu(t)=1, balanced variance across all timesteps

The practical implication: since [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]], the "right" target depends on what matters for your application. But for stability, [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] is the unambiguous winner.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Salimans & Ho 2022, Dieleman 2024, arXiv:2602.18428)

Relevant Notes:
- [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]] -- the loss weighting consequence
- [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]] -- epsilon's instability
- [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] -- velocity's stability
- [[EDM signal prediction achieves stability through exponentially vanishing estimator error not through bounded gain]] -- x0's gain O(1/b(t)^2) is the strongest singularity of the three, yet achieves stability through a qualitatively different mechanism
- [[progressive distillation introduced velocity prediction because noise prediction becomes unstable with few sampling steps]] -- empirical discovery of velocity's advantage years before the triangular relationship was formally understood

Topics:
- [[generative-modeling]]
