---
description: Signal prediction has stronger singularity 1/b(t)^2 than noise prediction but rapid estimator convergence near data counteracts it.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# EDM signal prediction achieves stability through exponentially vanishing estimator error not through bounded gain

The Geometry of Noise analysis reveals three distinct stability profiles for the three prediction targets:

- **Epsilon (noise)**: gain O(1/b(t)), estimation error O(1) near data → **unstable** (error × gain diverges)
- **x0 (signal/data)**: gain O(1/b(t)^2) — a *stronger* singularity — but estimation error vanishes exponentially fast near data → **stable** (error vanishes faster than gain grows)
- **Velocity**: gain O(1) bounded → **stable** (gain itself bounded)

The surprise is that x0-prediction (as used in EDM) has a *worse* gain singularity than epsilon-prediction, yet produces stable flows. The mechanism is qualitatively different from velocity prediction: near the data manifold, the optimal signal estimator D_t*(u) converges rapidly to the true data point. This exponential convergence rate overcomes the polynomial divergence of the gain, producing a bounded product.

This means there are (at least) two distinct paths to stability in iterative refinement:
1. **Bounded gain** — keep sensitivity bounded regardless of error magnitude (velocity prediction)
2. **Vanishing error** — allow unbounded sensitivity if the error rate drops fast enough to compensate (signal prediction)

Both are valid, but velocity prediction's bounded-gain stability is more robust because it does not depend on the quality of the estimator near the boundary. The EDM approach works because the estimator *happens* to converge fast enough — but this is a property of the specific problem (denoising), not a general guarantee. Since [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]], the velocity approach is the safer general-purpose choice.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (arXiv:2602.18428)

Relevant Notes:
- [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]] -- the unstable case
- [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] -- bounded-gain stability
- [[EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations]] -- the framework exhibiting this mechanism
- [[the three diffusion prediction targets epsilon x0 and velocity form a triangular relationship where any two determine the third]] -- x0 (signal) prediction is the target EDM uses; its gain O(1/b(t)^2) is the strongest singularity of the three
- [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]] -- x0 prediction's implicit 1/SNR^2 weighting over-emphasizes high-noise timesteps, and the gain singularity is the stability consequence of this weighting
- [[in high dimensions noise level reveals itself through concentration of measure making noise-agnostic models viable]] -- at high D, the rapid estimator convergence that gives EDM x0-prediction its stability is strengthened by concentration of measure

Topics:
- [[generative-modeling]]
