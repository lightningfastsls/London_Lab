---
description: Parameterization, explicit weighting w(t), and timestep distribution p(t) are mathematically interchangeable via importance sampling (Dieleman 2024).
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# changing the diffusion prediction target implicitly changes how noise levels are weighted in the loss

Dieleman (2024) clarified a subtle but important relationship: three independent-looking design choices in diffusion training are actually mathematically interchangeable:

1. **Model parameterization** (epsilon vs x0 vs velocity)
2. **Explicit loss weighting** w(t)
3. **Timestep sampling distribution** p(t)

The connection: E[(x0_hat - x_0)^2] = E[(sigma_t^2 / alpha_t^2) * (epsilon_hat - epsilon)^2] = E[1/SNR(t) * (epsilon_hat - epsilon)^2]. Switching from epsilon to x0 prediction is equivalent to multiplying the loss by 1/SNR(t) — a massive upweighting of high-noise timesteps.

This means you can achieve any desired effective weighting w_eff(t) = w(t) * p(t) * implicit_weight(parameterization) through any combination of the three factors. The implication: arguments about "which prediction target is best" are really arguments about which implicit weighting is desirable. The noise schedule adds no expressivity — it is just a reparameterization.

This insight has practical consequences for [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]]: rather than carefully designing noise schedules, one can use a simple schedule and adjust the explicit weighting or timestep distribution to achieve the desired emphasis. SD3's [[logit-normal timestep sampling concentrates training on the critical SNR transition improving diffusion quality]] is one such approach.

---

Source: [[diffusion-flow-matching-stability-research-2026-03-02]] (Dieleman 2024, sander.ai)

Relevant Notes:
- [[the three diffusion prediction targets epsilon x0 and velocity form a triangular relationship where any two determine the third]] -- the algebraic relationship underlying this
- [[the noise schedule is an arbitrary reparameterization while the effective loss weighting over signal-to-noise ratio is what matters]] -- the schedule-free implication
- [[MSE loss simplicity versus GMM output head expressiveness for spectrogram prediction]] -- analogous loss design trade-off in the USV pipeline

Topics:
- [[generative-modeling]]
