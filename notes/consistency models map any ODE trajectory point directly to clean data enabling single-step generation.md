---
description: Song et al. 2023 trained models to map trajectory points to origin via distillation or from scratch. CIFAR-10 FID 3.55 one-step.
type: method
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# consistency models map any ODE trajectory point directly to clean data enabling single-step generation

Consistency models (Song et al. 2023) take a fundamentally different approach to fast generation. Rather than straightening the ODE paths so fewer steps suffice (as in [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps]]), they learn a function that maps any point on the ODE trajectory directly to the trajectory's origin — the clean data.

The key property: for any two points x_t1 and x_t2 on the same ODE trajectory, the model produces the same output (the origin). This "consistency" constraint is enforced during training. The model can be trained either by distillation from a pre-trained diffusion model (using the teacher's ODE to generate trajectory pairs) or from scratch using a self-consistency objective.

The practical result is single-step generation: given any noise level, the model jumps directly to the clean data estimate without iterating through intermediate steps. CIFAR-10 FID of 3.55 for one-step generation demonstrates viability, though this is worse than multi-step diffusion methods like [[EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations]] (FID 1.79). The trade-off is speed vs quality — consistency models sacrifice some quality for maximum inference efficiency.

Consistency models also allow multi-step refinement: starting from the one-step estimate, add noise, then denoise again for improved quality. This "progressive refinement" bridges the gap between one-step and multi-step methods, offering a flexible quality-speed knob.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Song et al. 2023, arXiv:2303.01469)

Relevant Notes:
- [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps]] -- alternative few-step approach via path straightening
- [[DDPM requires around 1000 sampling steps while flow matching achieves comparable quality in 10-100 steps through straighter paths]] -- the efficiency problem this addresses
- [[the probability flow ODE enables deterministic sampling and exact likelihood computation from any diffusion SDE]] -- the ODE whose trajectories consistency models learn to shortcut
- [[EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations]] -- EDM achieves FID 1.79 with 35 steps; consistency models trade quality (FID 3.55) for single-step speed

Topics:
- [[generative-modeling]]
