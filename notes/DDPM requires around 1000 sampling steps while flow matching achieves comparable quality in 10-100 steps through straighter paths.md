---
description: DDPM ~1000, DDIM ~50-200, flow matching ~10-100 steps. Path curvature determines step requirements — OT-coupled straight paths are most efficient.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# DDPM requires around 1000 sampling steps while flow matching achieves comparable quality in 10-100 steps through straighter paths

The efficiency gap between diffusion model variants is primarily explained by the geometry of their sampling trajectories:

**DDPM** (Ho et al. 2020): implements the reverse SDE with stochastic noise injection (eta=1). The inherently curved diffusion paths, combined with stochastic sampling, require ~1000 steps for high-quality generation. Each step only removes a small amount of noise, and the stochasticity introduces variance that needs many steps to average out.

**DDIM** (Song et al. 2020): implements [[the probability flow ODE enables deterministic sampling and exact likelihood computation from any diffusion SDE|the probability flow ODE]] (eta=0). Same trained network as DDPM but deterministic sampling. Removing stochasticity allows larger steps — typically 50-200 steps — because the deterministic trajectory can be followed more efficiently. However, the underlying paths are still curved (inherited from the diffusion formulation).

**Flow Matching** (Lipman et al. 2023): trains directly on velocity fields with linear interpolation paths. With [[optimal transport coupling reduces path crossings and produces straighter flow matching trajectories|OT coupling]], paths become approximately straight, requiring only 10-100 steps. Combined with [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps|reflow]], paths can be straightened further to enable near-single-step generation.

The reduction from 1000 to 10-100 steps is a 10-100× speedup in inference time, making real-time generation practical. This efficiency advantage, combined with [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable|stability advantages]], explains why [[SD3 and Flux adopted rectified flow with velocity prediction replacing DDPM-style noise prediction in production]].

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Ho et al. 2020, Song et al. 2020, Lipman et al. 2023)

Relevant Notes:
- [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]] -- the efficient framework
- [[optimal transport coupling reduces path crossings and produces straighter flow matching trajectories]] -- why paths are straighter
- [[diffusion models factorize generation into many small denoising steps each narrowing the possibility space]] -- why many steps exist in the first place
- [[consistency models map any ODE trajectory point directly to clean data enabling single-step generation]] -- the most extreme acceleration: single-step generation by learning trajectory-to-origin mapping
- [[the probability flow ODE enables deterministic sampling and exact likelihood computation from any diffusion SDE]] -- DDIM (eta=0) implements this ODE, reducing steps from 1000 to 50-200 as the intermediate regime

Topics:
- [[generative-modeling]]
