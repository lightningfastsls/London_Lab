---
description: Diffusion models, flow matching, and generative frameworks — stability analysis, prediction targets, acceleration methods, and production adoption
type: moc
---

# generative-modeling

Generative models that learn to produce data by reversing a corruption process or following learned vector fields. The central theoretical finding is that prediction target choice determines stability: noise prediction has unbounded gain near clean data, while velocity prediction has bounded gain throughout. This explains the industry shift from DDPM to flow matching in production systems.

## Stability Analysis (Geometry of Noise)
- [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]] -- gain O(1/b(t)) diverges at t→0
- [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] -- nu(t)=1 constant, the theoretical winner
- [[EDM signal prediction achieves stability through exponentially vanishing estimator error not through bounded gain]] -- a third path: error convergence outpaces gain divergence
- [[in high dimensions noise level reveals itself through concentration of measure making noise-agnostic models viable]] -- why blind models work despite instability in some parameterizations

## Prediction Targets and Loss Design
- [[the three diffusion prediction targets epsilon x0 and velocity form a triangular relationship where any two determine the third]] -- v = alpha_t * epsilon - sigma_t * x_0
- [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]] -- parameterization, weighting, and schedule are interchangeable
- [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]] -- Dieleman's "schedules considered harmful"
- [[logit-normal timestep sampling concentrates training on the critical SNR transition improving diffusion quality]] -- SD3's importance sampling over timesteps

## Core Methods
- [[diffusion models factorize generation into many small denoising steps each narrowing the possibility space]] -- the foundational principle: coarse-to-fine
- [[the probability flow ODE enables deterministic sampling and exact likelihood computation from any diffusion SDE]] -- Anderson's theorem bridges SDE and ODE
- [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]] -- simulation-free training (Lipman et al. 2022)
- [[conditional flow matching replaces intractable marginal loss with a conditional version having identical gradients]] -- the mathematical trick enabling practical flow matching

## Efficiency and Acceleration
- [[DDPM requires around 1000 sampling steps while flow matching achieves comparable quality in 10-100 steps through straighter paths]] -- the efficiency comparison
- [[optimal transport coupling reduces path crossings and produces straighter flow matching trajectories]] -- OT coupling via mini-batch Sinkhorn
- [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps]] -- iterative path straightening (Liu et al. 2023)
- [[consistency models map any ODE trajectory point directly to clean data enabling single-step generation]] -- trajectory-to-origin mapping (Song et al. 2023)
- [[progressive distillation introduced velocity prediction because noise prediction becomes unstable with few sampling steps]] -- historical origin of v-prediction

## Design Space and Unification
- [[EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations]] -- Karras et al. 2022
- [[under Gaussian assumptions flow matching and diffusion are mathematically identical differing only in numerical stability]] -- the equivalence result

## Production Adoption
- [[SD3 and Flux adopted rectified flow with velocity prediction replacing DDPM-style noise prediction in production]] -- 12-32B parameter models using flow matching

## Transferable Principles
- [[error amplification near targets is a general instability pattern in iterative refinement systems beyond diffusion models]] -- applies to denoising, optimization, control, numerical integration
- [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]] -- the abstracted design pattern

## Open Questions
- [[whether flow matching could replace VQ-VAE for unsupervised USV representation learning]] -- continuous paths vs discrete tokens for USV analysis

## Related Areas
- [[representation-learning]] -- VQ-VAE pipeline for unsupervised USV representation; potential synergies with flow matching
- [[bioacoustic-ssl]] -- SSL models that could serve as encoders in diffusion/flow matching pipelines
- [[signal-processing]] -- bounded gain principle transfers to iterative detection pipelines
- [[model-adaptation]] -- LoRA and fine-tuning approaches for adapting generative models
- [[transformer-architecture]] -- pre-norm stability parallels bounded gain; the USV transformer's training stability is the same principle applied to gradient flow
- [[detection]] -- two-stage coarse-to-fine detection is an iterative refinement system where the bounded gain principle applies

---

Topics:
- [[index]]
