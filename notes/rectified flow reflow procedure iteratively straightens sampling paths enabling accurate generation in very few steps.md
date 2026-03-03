---
description: Train, generate (x_0,x_1) pairs from learned flow, retrain on paired trajectories. After sufficient iterations single Euler step suffices.
type: method
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps

Rectified flow (Liu et al. 2023) learns an ODE velocity field v_theta to match straight-line paths between noise and data. The training loss is simply E[|| v_theta(x_t, t) - (x_1 - x_0) ||^2] where x_t = (1-t)*x_0 + t*x_1 — the network learns to predict the direction from noise to data.

The key innovation is the *reflow* procedure: after initial training, use the learned model to generate synthetic (x_0, x_1) pairs by running the ODE forward from noise to data. Then retrain on these paired trajectories. Each reflow iteration straightens the paths further — the model's own outputs become better-aligned training targets, bootstrapping toward perfectly straight paths.

After sufficient iterations, the paths are so straight that a single Euler step (one function evaluation) produces high-quality samples. This is qualitatively different from [[consistency models map any ODE trajectory point directly to clean data enabling single-step generation]], which achieves single-step generation through a different mechanism (trajectory-to-origin mapping rather than path straightening).

Straight paths are preferred because they: (1) are shortest between two points, (2) can be simulated exactly without time discretization error, and (3) yield computationally efficient inference. This is complementary to [[optimal transport coupling reduces path crossings and produces straighter flow matching trajectories]], which achieves straightness through coupling rather than retraining. In practice, both can be combined: OT coupling for initial training, reflow for further refinement.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Liu et al. 2023, arXiv:2209.03003)

Relevant Notes:
- [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]] -- the foundation this builds on
- [[optimal transport coupling reduces path crossings and produces straighter flow matching trajectories]] -- complementary straightening via coupling
- [[SD3 and Flux adopted rectified flow with velocity prediction replacing DDPM-style noise prediction in production]] -- production adoption of this approach

Topics:
- [[generative-modeling]]
