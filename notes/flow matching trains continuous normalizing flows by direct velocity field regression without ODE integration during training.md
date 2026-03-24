---
description: Simulation-free training with unbiased mini-batch estimates. Lipman et al. 2022 showed this matches diffusion quality with simpler optimization.
type: method
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training

Flow Matching (Lipman et al. 2022/2023) trains a neural ODE dX_t/dt = u_theta(t, X_t) by directly regressing the velocity field u_theta against target vector fields — no likelihood computation, no divergence estimation, and critically no ODE integration during training. This "simulation-free" property is the key practical advantage: previous CNF training methods required expensive ODE solves at every training step.

The training algorithm is straightforward: sample t ~ U[0,1], data x_1, noise x_0, compute the linear interpolation x_t = (1-t)*x_0 + t*x_1, evaluate the conditional target velocity, and minimize the MSE. At inference, solve the ODE from t=0 to t=1 using a standard numerical solver (Euler, Heun). The velocity field u_theta directly gives the update direction at each step.

The simplicity of this formulation — compared to DDPM's noise schedule design, variance reduction techniques, and careful loss weighting — is itself an advantage. Since [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]], flow matching sidesteps the schedule design problem entirely by working directly with linear interpolation paths. Combined with [[optimal transport coupling reduces path crossings and produces straighter flow matching trajectories]], the resulting paths need far fewer integration steps.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Lipman et al. 2022, arXiv:2210.02747)

Relevant Notes:
- [[conditional flow matching replaces intractable marginal loss with a conditional version having identical gradients]] -- the mathematical trick that enables this
- [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps]] -- further path straightening
- [[DDPM requires around 1000 sampling steps while flow matching achieves comparable quality in 10-100 steps through straighter paths]] -- the efficiency payoff

Topics:
- [[generative-modeling]]
