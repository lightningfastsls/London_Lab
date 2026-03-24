---
description: Dieleman 2024 showed the schedule adds no expressivity — the weighting w(t)*p(t) over noise levels is the true degree of freedom.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters

Sander Dieleman (2024) made the provocative argument in "Noise schedules considered harmful" that the noise schedule — long considered a fundamental design choice in diffusion models — adds no expressivity to the model. The schedule (linear, cosine, VP, VE, sub-VP) is just a nonlinear function mapping time t to a signal-to-noise ratio. Different schedules trace different paths through logSNR space but the final model quality depends only on how the loss weights different noise levels, not on the path taken through them.

The effective weighting decomposes as: w_eff(t) = parameterization_weight × explicit_weight(t) × timestep_distribution(t). Since [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]], any desired weighting can be achieved through any combination of the three factors. This makes the noise schedule redundant — a convenience rather than a constraint.

Dieleman's summary: "the weighting function is the most important part of the loss." This explains why seemingly different formulations (cosine schedule with epsilon prediction vs linear schedule with v-prediction) can produce similar results — they may have similar effective weightings despite different surface-level parameterizations. The implication: when comparing diffusion methods, compare their effective weighting profiles over logSNR, not their nominal design choices.

This perspective was anticipated by [[EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations]], where sigma(t) = t already eliminated the schedule abstraction, and helps explain why [[under Gaussian assumptions flow matching and diffusion are mathematically identical differing only in numerical stability]].

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Dieleman 2024, sander.ai; Hang et al. 2023)

Relevant Notes:
- [[changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss]] -- the mechanism underlying schedule equivalence
- [[EDM separates concrete diffusion design choices from theory achieving strong results with minimal network evaluations]] -- anticipated schedule-free thinking
- [[logit-normal timestep sampling concentrates training on the critical SNR transition improving diffusion quality]] -- a practical application of this insight

Topics:
- [[generative-modeling]]
