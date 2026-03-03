---
description: Karras et al. 2022 derived preconditioning from first principles — unit-variance inputs and targets, sigma(t)=t. CIFAR-10 FID 1.79 with 35 evaluations.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations

The Elucidating the Design Space of Diffusion-Based Generative Models paper (Karras et al. 2022) argued that diffusion model theory had become "unnecessarily convoluted" — conflating mathematical framework choices with practical design decisions. EDM untangled these by deriving concrete recommendations from first principles.

The core insight: preconditioning functions should ensure that network inputs and training targets both have unit variance. This minimizes the amplification of estimation errors through the network — a principle that, in hindsight, connects to [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]]. Setting sigma(t) = t eliminates the time variable everywhere in favor of the noise level directly, simplifying the mathematical framework.

The results speak for themselves: CIFAR-10 FID 1.79 (class-conditional) and 1.97 (unconditional) with only 35 network evaluations during sampling. This demonstrated that careful design space analysis — rather than more compute or bigger models — was the key bottleneck.

EDM anticipated the "schedule-free" thinking later formalized by Dieleman: since [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]], EDM's direct parameterization by noise level was a step toward eliminating the schedule abstraction entirely. Additionally, [[EDM signal prediction achieves stability through exponentially vanishing estimator error not through bounded gain]] reveals that EDM's x0-prediction variant achieves stability through a qualitatively different mechanism than velocity prediction.

---

Source: [[diffusion-flow-matching-stability-research-2026-03-02]] (Karras et al. 2022, arXiv:2206.00364)

Relevant Notes:
- [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]] -- the schedule-free direction EDM anticipated
- [[EDM signal prediction achieves stability through exponentially vanishing estimator error not through bounded gain]] -- EDM's unique stability mechanism
- [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]] -- the weighting principle EDM exploits

Topics:
- [[generative-modeling]]
