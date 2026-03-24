---
description: Applies to any system where gain diverges as the state approaches its target — denoising, optimization convergence, control systems, numerical integration.
type: pattern
confidence: likely
topics:
  - "[[generative-modeling]]"
  - "[[signal-processing]]"
---

# error amplification near targets is a general instability pattern in iterative refinement systems beyond diffusion models

The Geometry of Noise paper's stability analysis identifies a principle that extends far beyond diffusion models: when an iterative refinement system uses a parameterization where the gain (sensitivity to estimation errors) diverges as the system approaches its target, the system is structurally unstable. Conversely, when the gain remains bounded, the system is self-correcting.

The diffusion case makes the principle concrete: epsilon prediction has gain O(1/b(t)) which diverges as noise vanishes (t→0). Velocity prediction has constant gain nu(t)=1. The same x_t, the same estimation quality, but fundamentally different error propagation characteristics.

But the pattern applies broadly:
- **Any denoising system** where noise level approaches zero — including signal processing pipelines that iteratively filter toward a target
- **Iterative optimization** near convergence — gradient-based methods can amplify errors when step sizes are not appropriately bounded
- **Control systems** approaching a set point — PID controllers with unbounded gain near the target oscillate
- **Numerical integration** near singularities — adaptive step size methods exist precisely to handle unbounded sensitivity regions

The design principle: **prefer parameterizations where sensitivity to estimation error remains bounded throughout the operating range.** In diffusion, this means velocity over epsilon. In optimization, this means adaptive learning rates. In control, this means gain scheduling. The specific mechanism differs but the principle is the same: [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]].

For the USV pipeline, this connects to iterative refinement in detection and classification — any two-stage pipeline where the second stage amplifies errors from the first stage is vulnerable to this pattern.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (arXiv:2602.18428, Section 11)

Relevant Notes:
- [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]] -- the concrete diffusion case
- [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]] -- the abstracted design pattern
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- USV detection's two-stage pipeline is potentially subject to this

Topics:
- [[generative-modeling]]
- [[signal-processing]]
