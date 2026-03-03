---
description: Anderson's theorem — every SDE has a deterministic ODE with identical marginals. Enables DDIM (eta=0), latent interpolation, and likelihood.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# the probability flow ODE enables deterministic sampling and exact likelihood computation from any diffusion SDE

Anderson's theorem establishes that any stochastic differential equation (SDE) has a corresponding ordinary differential equation (ODE) that produces the same marginal distributions at every time step:

dx = [f(x, t) - 1/2 * g(t)^2 * nabla_x log p_t(x)] dt

This probability flow ODE removes all stochastic noise from the sampling process while preserving the distribution of outputs. The result is a deterministic mapping from noise to data — given the same initial noise, you always get the same output.

The practical consequences are significant:
1. **Deterministic sampling** — DDIM (Song et al. 2020) is exactly this ODE with eta=0. Same trained network as DDPM, different sampling procedure.
2. **Fewer steps** — deterministic trajectories can be followed with fewer integration steps (50-200 vs DDPM's 1000) because there is no stochastic noise to fight.
3. **Exact likelihood** — the ODE admits exact log-probability computation via the instantaneous change of variables formula, enabling density estimation.
4. **Latent interpolation** — deterministic encoding maps data to a unique noise vector, enabling meaningful interpolation in latent space.

This result bridges the SDE and ODE formulations of diffusion models and is foundational to [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]], which works directly with the ODE formulation. The score function nabla_x log p_t(x) — the gradient of the log-density — is the key quantity that connects both formulations.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Song et al. 2021, arXiv:2011.13456; Anderson's theorem)

Relevant Notes:
- [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]] -- builds on the ODE formulation
- [[diffusion models factorize generation into many small denoising steps each narrowing the possibility space]] -- the SDE formulation this provides an alternative to
- [[DDPM requires around 1000 sampling steps while flow matching achieves comparable quality in 10-100 steps through straighter paths]] -- step efficiency from ODE sampling
- [[under Gaussian assumptions flow matching and diffusion are mathematically identical differing only in numerical stability]] -- the ODE formulation bridges both frameworks, making their equivalence visible
- [[consistency models map any ODE trajectory point directly to clean data enabling single-step generation]] -- learns to shortcut entire ODE trajectories to single-step generation

Topics:
- [[generative-modeling]]
