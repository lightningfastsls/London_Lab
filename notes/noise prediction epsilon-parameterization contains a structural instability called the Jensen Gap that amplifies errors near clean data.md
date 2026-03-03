---
description: The gain scales as 1/b(t) near t=0 creating unbounded error amplification. Geometry of Noise shows blind DDIM FID 40.90 vs flow matching 2.61 on CIFAR-10.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# noise prediction epsilon-parameterization contains a structural instability called the Jensen Gap that amplifies errors near clean data

The epsilon-parameterization — where the network predicts the Gaussian noise that was added — has been the default in diffusion models since DDPM (Ho et al. 2020). But it carries a structural flaw. To convert a noise estimate into a score function or data prediction, you divide by sigma_t (the noise standard deviation). As the sample approaches clean data (t→0), sigma_t shrinks toward zero, and any estimation error gets amplified by a factor of 1/sigma_t. This is the "high-gain amplifier for estimation errors."

The Geometry of Noise paper (Sahraee-Ardakan, Delbracio, Milanfar 2025) formalizes this as the "Jensen Gap" — the mismatch between the harmonic mean of posterior noise levels and the true noise level. For noise-agnostic (autonomous) models, this gap is finite and fixed, but the 1/b(t) gain singularity amplifies it without bound as t→0. The error diverges: lim Δv → ∞.

Experimentally, blind DDIM (noise prediction without conditioning on noise level) achieves catastrophic FID 40.90 on CIFAR-10, while blind Flow Matching achieves FID 2.61 — a 15× difference attributable entirely to the parameterization's stability properties, since [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]].

This matters beyond diffusion models because [[error amplification near targets is a general instability pattern in iterative refinement systems beyond diffusion models]] — any system with unbounded gain near its target state exhibits this structural instability.

---

Source: [[diffusion-flow-matching-stability-research-2026-03-02]] (arXiv:2602.18428)

Relevant Notes:
- [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] -- the stable counterpart that avoids this instability
- [[EDM signal prediction achieves stability through exponentially vanishing estimator error not through bounded gain]] -- a third path to stability via error convergence
- [[the three diffusion prediction targets epsilon x0 and velocity form a triangular relationship where any two determine the third]] -- structural equivalence despite different stability

Topics:
- [[generative-modeling]]
