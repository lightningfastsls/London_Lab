---
description: Mini-batch Sinkhorn OT coupling minimizes Wasserstein distance, reducing gradient variance from crossing conditional paths.
type: method
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# optimal transport coupling reduces path crossings and produces straighter flow matching trajectories

A fundamental challenge in flow matching: with independent coupling q(x_1, x_0) = p_data(x_1) * p_init(x_0), conditional paths from different data points can cross each other. When paths cross, the marginal velocity field becomes multi-valued — the model must average over conflicting directions, producing high-variance gradients and curved marginal paths that require many ODE steps to follow accurately.

Mini-batch Optimal Transport (OT) coupling addresses this by solving a transport problem within each mini-batch using the Sinkhorn algorithm. Instead of randomly pairing noise samples with data samples, OT finds the pairing that minimizes the total transport cost (Wasserstein distance). The result: paths from nearby noise samples go to nearby data samples, dramatically reducing crossings.

Straighter paths are preferred for three reasons: (1) they are the shortest distance between two points, (2) they can be simulated with minimal time discretization error, and (3) they yield models that generate accurately with very few ODE steps. This connects directly to why [[DDPM requires around 1000 sampling steps while flow matching achieves comparable quality in 10-100 steps through straighter paths]].

The OT coupling is complementary to the [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps]], which achieves straighter paths through iterative refinement of the flow itself. Both approaches target the same goal — path straightness — through different mechanisms (coupling vs retraining).

---

Source: [[diffusion-flow-matching-stability-research-2026-03-02]] (Lipman et al. 2022, Cambridge MLG blog)

Relevant Notes:
- [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]] -- the framework this improves
- [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps]] -- alternative straightening method
- [[DDPM requires around 1000 sampling steps while flow matching achieves comparable quality in 10-100 steps through straighter paths]] -- the efficiency payoff

Topics:
- [[generative-modeling]]
