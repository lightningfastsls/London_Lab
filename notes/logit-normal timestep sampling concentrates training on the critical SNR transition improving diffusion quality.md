---
description: Used in SD3. Biases toward logSNR equals zero where the prediction task is most challenging, instead of uniform sampling over time.
type: method
confidence: likely
topics:
  - "[[generative-modeling]]"
---

# logit-normal timestep sampling concentrates training on the critical SNR transition improving diffusion quality

Standard diffusion training samples timesteps uniformly from [0, T]. But not all timesteps are equally important — the model needs to learn different things at different noise levels. At very high noise, only coarse structure matters; at very low noise, only fine details. The most challenging prediction tasks occur around logSNR = 0, where signal and noise have roughly equal magnitude and the model must simultaneously handle structure and detail.

Logit-normal timestep sampling (used in SD3, Esser et al. 2024) biases the training distribution toward this critical transition point. Instead of t ~ U[0,1], it samples t from a logit-normal distribution centered near the middle of the trajectory. This gives the model more training signal in the regime where prediction is hardest, at the cost of less signal in the easy extremes.

This is a direct application of the insight that [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]] — since parameterization, weighting, and timestep distribution are interchangeable, shifting the timestep distribution is equivalent to changing the loss weighting. Logit-normal sampling effectively implements a form of importance sampling that allocates training compute to where it matters most.

The approach is complementary to other schedule improvements: since [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]], logit-normal sampling is one way to directly control the effective weighting without modifying the noise schedule itself.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Esser et al. 2024, arXiv:2403.03206)

Relevant Notes:
- [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]] -- the theoretical basis for this approach
- [[SD3 and Flux adopted rectified flow with velocity prediction replacing DDPM-style noise prediction in production]] -- production use of this technique
- [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]] -- the broader principle
- [[EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations]] -- EDM's sigma(t)=t was an early step toward schedule-free thinking that logit-normal sampling completes
- [[in high dimensions noise level reveals itself through concentration of measure making noise-agnostic models viable]] -- at high D, noise level is identifiable from the observation alone; logit-normal sampling optimizes training compute around the SNR=0 transition where this identification is hardest

Topics:
- [[generative-modeling]]
