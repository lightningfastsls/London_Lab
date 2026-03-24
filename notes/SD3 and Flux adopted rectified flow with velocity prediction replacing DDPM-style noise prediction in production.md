---
description: SD3 uses MMDiT plus logit-normal sampling. Flux scales to 12-32B params. Industry shift validates theoretical stability advantage.
type: finding
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# SD3 and Flux adopted rectified flow with velocity prediction replacing DDPM-style noise prediction in production

The theoretical advantages of velocity prediction and flow matching are not just academic — they drove a measurable paradigm shift in production generative models between 2023-2025.

**Stable Diffusion 3** (Esser et al. 2024) replaced the DDPM-style noise prediction used in SD1.x/SD2.x/SDXL with rectified flow and velocity prediction. Key innovations included MMDiT (Multi-Modal Diffusion Transformer) with separate weights for image and text tokens with bidirectional information flow, and [[logit-normal timestep sampling concentrates training on the critical SNR transition improving diffusion quality]]. Training predicts velocity v = noise - data and minimizes MSE, producing straighter inference paths that need fewer sampling steps.

**Flux** (Black Forest Labs 2024-2025) scaled this approach to 12 billion parameters (FLUX.1) and 32 billion parameters (FLUX.2). The architecture uses flow matching in latent space — similar to latent diffusion but with velocity prediction instead of noise prediction. The velocity field defines direct paths from noise to data, eliminating iterative denoising.

This industry shift validates the theory: since [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] and [[under Gaussian assumptions flow matching and diffusion are mathematically identical differing only in numerical stability]], the adoption is driven by practical stability and efficiency, not increased expressivity. The fact that the largest and most capable image generation models all moved to velocity/flow matching in the same 2-year window is strong evidence that the stability advantage matters at scale.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Esser et al. 2024, arXiv:2403.03206; Black Forest Labs 2024-2025)

Relevant Notes:
- [[velocity prediction satisfies a bounded-gain condition making diffusion sampling inherently stable]] -- the theoretical basis for this shift
- [[rectified flow reflow procedure iteratively straightens sampling paths enabling accurate generation in very few steps]] -- the path straightening methodology
- [[progressive distillation introduced velocity prediction because noise prediction becomes unstable with few sampling steps]] -- the empirical predecessor
- [[under Gaussian assumptions flow matching and diffusion are mathematically identical differing only in numerical stability]] -- the equivalence that enabled reusing diffusion architectures with flow matching training
- [[logit-normal timestep sampling concentrates training on the critical SNR transition improving diffusion quality]] -- SD3's specific training innovation

Topics:
- [[generative-modeling]]
