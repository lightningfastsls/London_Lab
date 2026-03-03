---
description: Salimans and Ho 2022 empirically found v-prediction necessary for low-step regimes. The Geometry of Noise later formalized the mechanism as the Jensen Gap.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# progressive distillation introduced velocity prediction because noise prediction becomes unstable with few sampling steps

Progressive distillation (Salimans & Ho 2022) is a method for accelerating diffusion model sampling: iteratively distill a teacher model (many steps) into a student model (half the steps), repeatedly halving the step count. During this work, the authors discovered that noise prediction (epsilon-parameterization) becomes unstable when the step count drops below a threshold — the student model fails to learn useful denoising at extreme step-reduction ratios.

Their solution was to introduce the velocity target v = alpha_t * epsilon - sigma_t * x_0, which exhibited stable training even at very low step counts. At the time, this was an empirical finding without formal theoretical justification — it simply worked better.

Three years later, the Geometry of Noise paper (2025) provided the formal explanation: [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]]. The 1/b(t) gain singularity means that with fewer sampling steps, each step covers a larger range of noise levels, and the errors from the high-gain region near t=0 have a larger impact on the overall trajectory. In the extreme case of very few steps, the instability dominates.

This is a compelling example of theory catching up to practice — practitioners discovered the fix (velocity prediction) years before theoreticians explained why it works ([[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]]). The pattern itself — that [[error amplification near targets is a general instability pattern in iterative refinement systems beyond diffusion models]] — is now understood as a general principle.

---

Source: [[diffusion-flow-matching-stability-research-2026-03-02]] (Salimans & Ho 2022, arXiv:2202.00512)

Relevant Notes:
- [[noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data]] -- the instability they encountered
- [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] -- the fix they discovered empirically
- [[SD3 and Flux adopted rectified flow with velocity prediction replacing DDPM-style noise prediction in production]] -- industry adoption of their insight

Topics:
- [[generative-modeling]]
