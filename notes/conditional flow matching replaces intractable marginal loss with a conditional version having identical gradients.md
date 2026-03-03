---
description: Conditioning on individual data points x_1 makes the target vector field tractable while preserving gradient equivalence to the marginal loss.
type: method
confidence: proven
topics:
  - "[[generative-modeling]]"
---

# conditional flow matching replaces intractable marginal loss with a conditional version having identical gradients

The marginal flow matching loss L_FM requires knowing the target vector field u(t, x) that induces the desired probability path — but this field depends on the marginal distribution p_t, which is intractable. The conditional flow matching (CFM) trick resolves this by conditioning on individual data points x_1:

L_CFM = E_{t, x_1, x_t} [|| u_theta(t, x_t) - u_t(x_t | x_1) ||^2]

The conditional vector field u_t(x|x_1) for Gaussian paths is analytically available and equals (sigma_dot_t / sigma_t) * (x - mu_t(x_1)) + mu_dot_t(x_1). For the standard linear interpolation with mu_t = t*x_1 and sigma_t = (1-t) + t*sigma_min, this simplifies to a closed-form expression.

The key mathematical result: the gradients of L_CFM and L_FM with respect to network parameters are identical. This means optimizing the tractable conditional loss is equivalent to optimizing the intractable marginal loss. The proof exploits the linearity of expectation and the fact that the conditional fields average to the marginal field.

This trick is analogous to how denoising score matching replaces the intractable score function with a tractable conditional version — both exploit the same decomposition principle. The difference is that flow matching conditions on the endpoint (data) while score matching conditions on the noise level. Since [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]], this tractability result is what makes the entire framework practical.

---

Source: diffusion-flow-matching-stability-research-2026-03-02 (Lipman et al. 2022, Cambridge MLG blog)

Relevant Notes:
- [[flow matching trains continuous normalizing flows by direct velocity field regression without ODE integration during training]] -- the framework this enables
- [[optimal transport coupling reduces path crossings and produces straighter flow matching trajectories]] -- further improvement built on this foundation
- [[under Gaussian assumptions flow matching and diffusion are mathematically identical differing only in numerical stability]] -- the Gaussian conditional paths used here are precisely what establishes the equivalence with diffusion

Topics:
- [[generative-modeling]]
