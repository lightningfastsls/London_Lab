---
description: When sensitivity to estimation errors remains bounded as the system approaches its target, the system is self-correcting. Unbounded sensitivity diverges.
type: pattern
confidence: likely
topics:
  - "[[generative-modeling]]"
  - "[[signal-processing]]"
---

# bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain

This is the abstracted design pattern from the Geometry of Noise stability analysis, stated independently of the diffusion model context where it was derived.

**The pattern:** In any iterative refinement system that approaches a target state through successive approximations, the *gain* — the system's sensitivity to estimation errors at each step — determines whether errors accumulate or dissipate.

- **Bounded gain** (sensitivity stays below a fixed constant): errors at each step produce bounded perturbations. The system self-corrects. Even imperfect estimates lead to convergent trajectories.
- **Unbounded gain** (sensitivity grows without limit as target is approached): errors get amplified, potentially without bound. The system is structurally unstable — it may work in practice but fails under perturbation.

**The diffusion instantiation:** velocity prediction has nu(t) = 1 (bounded) and produces stable flows. Noise prediction has nu(t) = O(1/b(t)) (unbounded near t=0) and produces unstable flows. Same model capacity, same estimation quality, radically different error propagation.

**A third path** exists: if the estimation error itself vanishes fast enough to counteract growing gain, stability can be achieved — as [[EDM signal prediction achieves stability through exponentially vanishing estimator error not through bounded gain]]. But this is problem-specific, while bounded gain is a universal guarantee.

**Design implication:** when choosing parameterizations for iterative systems, prefer those with bounded sensitivity throughout the operating range. This applies to denoising, optimization (learning rate scheduling), control (gain scheduling), signal processing (two-stage detection filtering where second-stage errors are amplified by first-stage false positive rates), and neural network training (since [[pre-norm transformer architecture improves training stability for spectrogram prediction]] uses LayerNorm before attention and FFN to ensure bounded gradient flow -- the same "bounded gain" principle applied to the optimization trajectory rather than the sampling trajectory).

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (arXiv:2602.18428, Section 11)

Relevant Notes:
- [[error amplification near targets is a general instability pattern in iterative refinement systems beyond diffusion models]] -- the concrete observation this generalizes
- [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] -- the diffusion instantiation
- [[two-stage coarse-to-fine filtering is effective for imbalanced detection tasks]] -- USV pipeline instance of iterative refinement
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- bounded gradient flow as the training-time analogue of bounded sampling gain

Topics:
- [[generative-modeling]]
- [[signal-processing]]
