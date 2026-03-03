---
description: nu(t)=1 constant throughout the trajectory — posterior uncertainty absorbed into smooth geometric drift rather than amplified.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable

Where noise prediction amplifies errors through a 1/b(t) gain singularity, velocity prediction has a gain of exactly nu(t) = 1 — constant across all noise levels. The autonomous velocity field v_aut(u, t) = mu(t) * u + nu(t) * f*(u) absorbs posterior uncertainty into its drift term rather than amplifying it through the gain.

This is not just "better" than noise prediction — it is qualitatively different. The bounded gain means the system is self-correcting: estimation errors at any point along the trajectory produce bounded perturbations in the sampling dynamics. The Geometry of Noise paper proves this formally, showing that the velocity parameterization's dynamics "absorb posterior uncertainty into a smooth geometric drift."

The practical consequence is that velocity-based models can operate without explicit noise-level conditioning (autonomous/blind models) and still produce high-quality samples, because [[in high dimensions noise level reveals itself through concentration of measure making noise-agnostic models viable]]. This theoretical result explains why [[SD3 and Flux adopted rectified flow with velocity prediction replacing DDPM-style noise prediction in production]] — the stability advantage is not just empirical preference but mathematical necessity.

Velocity is defined as v = alpha_t * epsilon - sigma_t * x_0, capturing the instantaneous direction and speed at which the noisy sample should move toward the data distribution. This geometric interpretation — "which way to go" rather than "what to subtract" — is inherently better-conditioned because direction remains well-defined even when the magnitude of noise approaches zero.

---

Source: [[diffusion-flow-matching-stability-research-2026-03-02]] (arXiv:2602.18428, Salimans & Ho 2022)

Relevant Notes:
- [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]] -- the instability this resolves
- [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]] -- the general principle
- [[progressive distillation introduced velocity prediction because noise prediction becomes unstable with few sampling steps]] -- historical discovery

Topics:
- [[generative-modeling]]
