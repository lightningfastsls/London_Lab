---
description: "Deep survey of diffusion model parameterizations and flow matching stability: Jensen Gap in noise prediction vs bounded-gain velocity prediction, with practical implications for SD3/Flux"
source_type: article
url: "multiple -- see source log"
author: "multiple sources"
date_accessed: "2026-03-02"
status: unprocessed
research_tool: "web-search"
research_query: "diffusion models flow matching velocity prediction stability analysis Jensen Gap"
research_depth: "deep"
---

# Diffusion Models and Flow Matching: Stability Analysis and Modern Parameterization Choices

This survey covers the mathematical foundations of diffusion-based generative models, the three prediction target parameterizations (epsilon, x0, velocity), the stability analysis proving velocity prediction's superiority, and the practical adoption of flow matching in state-of-the-art systems. The central finding is that noise prediction (epsilon) contains a structural instability -- a "Jensen Gap" that amplifies errors near clean data -- while velocity prediction satisfies a bounded-gain condition that makes it inherently stable. This theoretical result from the "Geometry of Noise" paper provides rigorous justification for the empirical shift toward flow matching in production systems like Stable Diffusion 3 and Flux.

---

## 1. Diffusion Model Fundamentals

### Forward Process

Diffusion models work by defining a forward process that gradually corrupts data with Gaussian noise over T steps. The forward transition at each step is:

q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

where beta_t is a variance schedule (linear, cosine, or learned). The joint forward process is a Markov chain:

q(x_{1:T} | x_0) = product_{t=1}^{T} q(x_t | x_{t-1})

The critical closed-form shortcut allows sampling x_t directly from x_0 without iterating:

q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

or equivalently: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon, where epsilon ~ N(0, I), alpha_t = 1 - beta_t, and alpha_bar_t = product_{i=1}^{t} alpha_i. At t=0, x_0 is clean data; at t=T, x_T is approximately pure Gaussian noise.

### Reverse Process

The reverse process learns to denoise step by step:

p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), Sigma_theta(x_t, t))

Given knowledge of x_0, the true reverse conditional is tractable:

q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_tilde_t, beta_tilde_t * I)

where beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t and mu_tilde_t = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_t).

The simplified training loss (Ho et al. 2020) is:

L_simple = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]

where the network predicts the noise that was added.

### Why Small Steps Work

Each denoising step narrows the possibility space. At high noise levels, the model only needs to resolve coarse structure (which direction is data). At low noise levels, it refines fine details. By factoring the generation into many small, manageable steps, the model avoids having to make large jumps in probability space. This is analogous to gradually focusing a lens rather than trying to snap into focus all at once.

### Continuous-Time Formulation (Song et al. 2021)

The discrete process generalizes to a stochastic differential equation (SDE):

Forward SDE: dx = f(x, t) dt + g(t) dw

For Variance-Preserving (VP-SDE), matching DDPM: dx = -beta(t)/2 * x dt + sqrt(beta(t)) dw

For Variance-Exploding (VE-SDE), matching NCSN: dx = sqrt(d[sigma^2(t)]/dt) dw

The reverse SDE (Anderson's theorem): dx = [f(x, t) - g(t)^2 * nabla_x log p_t(x)] dt + g(t) dw_bar (solved backward in time)

The score function nabla_x log p_t(x) is the key quantity to estimate.

### Probability Flow ODE

Any SDE has a deterministic ODE with identical marginal distributions:

dx = [f(x, t) - 1/2 * g(t)^2 * nabla_x log p_t(x)] dt

This is DDIM with eta=0. The ODE enables exact likelihood computation, deterministic sampling, and interpolation in latent space.

---

## 2. Three Prediction Targets

Given the noisy sample x_t = alpha_t * x_0 + sigma_t * epsilon (where alpha_t and sigma_t define the noise schedule), the network can predict three equivalent quantities:

### Epsilon-Prediction (Noise Prediction)

Target: the Gaussian noise epsilon that was added. Loss: L_epsilon = E[|| epsilon - epsilon_theta(x_t, t) ||^2]. Used in DDPM (Ho et al. 2020), the original and historically dominant parameterization. The noise prediction relates to the score function: s_theta(x_t, t) = -epsilon_theta(x_t, t) / sqrt(1 - alpha_bar_t).

Stability: Problematic at low noise levels (near clean data). When sigma_t is small, the noise is hard to estimate accurately, and errors are amplified by the 1/sigma_t factor in converting to score or data predictions. According to the Geometry of Noise analysis, the effective gain scales as nu(t) proportional to 1/b(t), creating a "high-gain amplifier for estimation errors."

### x0-Prediction (Data/Signal Prediction)

Target: the clean data x_0 directly. Loss: L_x0 = E[|| x_0 - x0_theta(x_t, t) ||^2]. The network directly predicts what the clean image looks like.

Stability: Problematic at high noise levels (near pure noise). Predicting sharp data details from nearly pure noise is extremely difficult. The implicit loss weighting is 1/SNR(t), which over-emphasizes high-noise timesteps where estimation quality is worst.

Relationship: epsilon_hat = (x_t - alpha_t * x0_hat) / sigma_t and x0_hat = (x_t - sigma_t * epsilon_hat) / alpha_t. The two are related by a simple linear transformation.

### Velocity Prediction (v-Prediction)

Target: v = alpha_t * epsilon - sigma_t * x_0 (introduced by Salimans & Ho 2022 in "Progressive Distillation for Fast Sampling").

In the continuous formulation from the MIT 6.S184 lecture notes: v_target(x|z) = alpha'_t * z - beta'_t * epsilon, where alpha'_t and beta'_t are time derivatives of the schedule.

Stability: Balanced across all timesteps. The velocity target has more consistent variance than either epsilon or x0 across the full time range. It weights the loss uniformly over [0, 1] and avoids the extreme behaviors at either end of the noise spectrum.

Geometric interpretation: v-prediction captures the instantaneous "velocity" of the probability path in data space -- the direction and speed at which the noisy sample should move toward the data distribution.

Relationship to others: v = alpha_t * epsilon - sigma_t * x_0, so x_0 = alpha_t * x_t - sigma_t * v and epsilon = sigma_t * x_t + alpha_t * v. All three form a "triangular" relationship where any two determine the third.

### Implicit Loss Weighting

Dieleman (2024) clarified that changing the prediction target implicitly changes how noise levels are weighted in the loss:

E[(x0_hat - x_0)^2] = E[(sigma_t^2 / alpha_t^2) * (epsilon_hat - epsilon)^2] = E[1/SNR(t) * (epsilon_hat - epsilon)^2]

Three factors jointly determine relative noise-level importance: (1) model parameterization, (2) explicit weighting w(t), and (3) timestep distribution p(t). These are mathematically interchangeable through importance sampling.

---

## 3. "The Geometry of Noise" (Sahraee-Ardakan, Delbracio, Milanfar -- Google, 2025/2026)

This paper (arXiv:2602.18428) provides the core theoretical analysis of why velocity prediction is more stable than noise prediction. The central question: autonomous (noise-agnostic) models like Equilibrium Matching learn a single, time-invariant vector field without noise-level conditioning. How can this work?

### Marginal Energy

Definition: E_marg(u) = -log p(u), where p(u) = integral p(u|t) p(t) dt integrates the conditional density over all noise levels. The marginal energy represents the landscape being optimized when noise level is treated as a random variable.

Gradient: nabla_u E_marg(u) = E_{t|u}[(u - a(t) * D_t*(u)) / b(t)^2], where D_t*(u) is the optimal conditional denoiser, connected to the score via Tweedie's formula: nabla_u log p(u|t) = (a(t) * D_t*(u) - u) / b(t)^2.

The marginal energy gradient has a 1/t^p singularity near the data manifold (as t approaches 0), creating an infinitely deep potential well.

### Jensen Gap: Why Noise Prediction Fails

For noise prediction (epsilon-parameterization), the effective gain nu(t) scales as 1/b(t), where b(t) is the noise standard deviation. The drift perturbation error is:

Delta v(u, t) = |nu(t)| * || f*(u) - f_t*(u) ||

where f*(u) is the autonomous field and f_t*(u) is the noise-conditional optimal field.

The "Jensen Gap" is the mismatch between the harmonic mean of posterior noise levels and the true noise level. As t approaches 0 (near clean data), this singularity amplifies the finite Jensen Gap, causing the error to diverge: lim Delta v approaches infinity.

Result: Autonomous noise prediction is structurally unstable. The paper shows that "DDPM Blind exhibits structural instability and noise" in experiments. On CIFAR-10, blind DDIM achieves FID 40.90 (catastrophic failure) vs Flow Matching blind at FID 2.61.

### Bounded-Gain Condition: Why Velocity Prediction Works

For velocity-based models (Flow Matching), the gain is bounded: nu(t) = 1 (constant). The autonomous velocity field:

v_aut(u, t) = mu(t) * u + nu(t) * f*(u)

With bounded nu(t), posterior uncertainty is absorbed into the drift, yielding stability. The dynamics "absorb posterior uncertainty into a smooth geometric drift."

### EDM (Signal Prediction) -- Stable via Different Mechanism

For EDM's signal prediction (x0), the gain contains a stronger singularity: nu(t) proportional to 1/b(t)^2. However, the error in the signal estimator vanishes exponentially fast near the data manifold. This rapid convergence counteracts the polynomial divergence of the gain, resulting in a stable flow -- but through a different mechanism than velocity prediction's inherent boundedness.

### Concentration of Measure: Noise Self-Reports in High Dimensions

The paper proves that in high dimensions, noise-agnostic models work because the noise level reveals itself:

For data on a d-dimensional manifold in R^D with codimension k = D - d > 2: as the observation u approaches the data support, the posterior p(t|u) concentrates sharply on t approaching 0 (Lemma 6). The noise shells for different noise levels become "effectively disjoint" -- the observation is a "deterministic proxy for the noise level."

For discrete data with ambient dimension D > 2: the posterior evidence scales as Z_j(epsilon) = O(epsilon^{2-D}), so the posterior converges weakly to delta(t) (Lemma 5).

The practical implication: in a space like 32x32x3 images (D=3072), the network can implicitly estimate the noise level from the corrupted observation without being told -- the geometry of high-dimensional space provides the information for free.

Progressive dimensionality experiments confirm three regimes:
- D=2: Both blind models fail (ambiguous posterior)
- D in {8, 32}: Flow Matching succeeds; DDPM Blind shows instability due to O(1/b(t)) gain
- D=128: Absolute concentration -- even structurally unstable DDPM Blind produces clean samples because estimation error is forced to zero

### Energy-Aligned Decomposition

The learned vector field decomposes into three components:

f*(u) = lambda_bar(u) * nabla E_marg(u) [Natural Gradient] + Transport Correction + c_bar_scale(u) * u [Linear Drift]

The posterior noise variance acts as a local conformal metric that preconditions and counteracts the geometric singularity. The effective gain lambda(t) = b(t)/a(t) * (d(t)*a(t) - c(t)*b(t)) acts as a "perfect preconditioner."

---

## 4. Flow Matching Framework

### Core Formulation (Lipman et al. 2022/2023)

Flow Matching trains Continuous Normalizing Flows (CNFs) by directly regressing vector fields rather than maximizing likelihood. The ODE:

dX_t/dt = u_theta(t, X_t), X_0 ~ p_init (Gaussian noise)

Goal: make X_1 approximate p_data.

The flow matching loss: L_FM(theta) = E_{t ~ U[0,1], x_t ~ p_t} [|| u_theta(t, x_t) - u(t, x_t) ||^2]

where u(t, x) is the target vector field inducing the interpolation path.

### Conditional Flow Matching Trick

The intractable marginal loss is replaced by a conditional version with identical gradients:

L_CFM(theta) = E_{t, x_1 ~ p_data, x_t ~ p_t(.|x_1)} [|| u_theta(t, x_t) - u_t(x_t | x_1) ||^2]

The conditional vector field for Gaussian paths p_t(x|x_1) = N(mu_t(x_1), sigma_t(x_1)^2 I):

u_t(x|x_1) = (sigma_dot_t / sigma_t) * (x - mu_t(x_1)) + mu_dot_t(x_1)

For the standard linear interpolation: mu_t(x_1) = t * x_1, sigma_t = (1-t) + t * sigma_min, giving:

u_t(x|x_1) = (x_1 - (1 - sigma_min) * x) / (1 - (1 - sigma_min) * t)

### Training Algorithm

1. Sample t ~ U[0,1], x_1 ~ p_data, x_0 ~ N(0, I)
2. Optionally compute OT coupling via mini-batch Sinkhorn
3. Compute x_t = (1-t) * x_0 + t * x_1
4. Evaluate conditional target u_t(x_t | x_1)
5. Gradient step: minimize || u_theta(t, x_t) - u_t(x_t | x_1) ||^2

Advantages: No ODE integration during training (simulation-free), unbiased mini-batch estimates, reduced variance via OT coupling, no divergence estimation needed.

### Sampling

Solve the ODE from t=0 to t=1 using standard numerical ODE solver (e.g., Euler, Heun). The velocity field u_theta directly gives the update direction.

### Optimal Transport Paths

The path crossing problem: with independent coupling q(x_1, x_0) = p_data(x_1) * p_init(x_0), conditional paths from different data points can cross, causing high-variance gradients and curved marginal paths requiring more ODE steps.

Solution: Mini-batch Optimal Transport coupling minimizes Wasserstein distance, dramatically reducing path crossings. OT paths are straighter (more efficient) than diffusion paths.

---

## 5. Rectified Flow (Liu et al. 2023)

Rectified flow learns an ODE to follow straight paths connecting noise and data:

Training: Learn velocity field v_theta to match the linear interpolation direction x_1 - x_0.
Loss: E[|| v_theta(x_t, t) - (x_1 - x_0) ||^2] where x_t = (1-t) * x_0 + t * x_1.

Reflow procedure: After training, use the learned model to generate (x_0, x_1) pairs from the trained flow, then retrain on these paired trajectories. Each reflow iteration straightens the paths further. After sufficient iterations, nearly straight paths allow accurate generation with a single Euler step.

Straight paths are preferred because they are (1) shortest between two points, (2) can be simulated exactly without time discretization, and (3) yield computationally efficient models.

---

## 6. DDPM vs DDIM vs Flow Matching: Comparison

### DDPM (Ho et al. 2020)

- Implements reverse SDE with stochastic noise injection (eta=1)
- Noise prediction (epsilon) parameterization
- Typically 1000 steps for sampling
- High diversity but slow

### DDIM (Song et al. 2020)

- Implements probability flow ODE (eta=0)
- Same trained network as DDPM, different sampling
- Deterministic, enabling fewer steps (50-200)
- Enables latent space interpolation

### Flow Matching (Lipman et al. 2023)

- Direct velocity field regression via ODE
- Linear interpolation paths (simpler than diffusion paths)
- Typically 10-100 sampling steps with higher-order solvers
- Training is simulation-free (no ODE solving during training)
- Provably more stable than noise prediction

### Key Differences

1. Training objective: DDPM predicts noise; Flow Matching predicts velocity
2. Paths: DDPM/DDIM paths are inherently curved; Flow Matching paths can be made straight via OT coupling
3. Stability: Noise prediction has O(1/b(t)) gain singularity; velocity prediction has bounded gain
4. Steps: DDPM needs ~1000; DDIM ~50-200; Flow Matching ~10-100
5. Schedule: DDPM requires careful noise schedule tuning; Flow Matching with rectified flow is simpler

### Equivalence Under Certain Conditions

As shown by the "Diffusion Meets Flow Matching" analysis: under Gaussian assumptions, flow matching and diffusion are mathematically identical. Flow matching weighting equals diffusion weighting of v-MSE loss with cosine noise schedule. The difference is in numerical stability, not expressivity.

---

## 7. Practical Adoption: Stable Diffusion 3 and Flux

### Stable Diffusion 3 (Esser et al. 2024, arXiv:2403.03206)

SD3 uses Rectified Flow with velocity prediction instead of DDPM-style noise prediction used in SD1.x/SD2.x/SDXL. Key innovations:
- MMDiT (Multi-Modal Diffusion Transformer): separate weights for image and text tokens with bidirectional information flow
- Logit-normal timestep sampling: biases training toward middle of trajectory (more challenging prediction tasks)
- Training: samples time, noise, computes linear interpolation x_t, predicts velocity v = noise - data, minimizes MSE
- Result: straighter inference paths enabling fewer sampling steps

### Flux (Black Forest Labs, 2024)

FLUX.1 is a 12-billion parameter rectified flow transformer. FLUX.2 (2025) scales to 32B parameters. Key: velocity prediction defines direct paths from noise to data, eliminating iterative denoising. Uses flow matching in latent space (similar to latent diffusion but with flow matching instead of DDPM).

### Why Industry Shifted

The empirical benefits align with theory: velocity prediction is more stable during training, enables fewer sampling steps, produces straighter paths, and the theoretical guarantees from the Geometry of Noise paper explain why -- bounded gain prevents error amplification near the data manifold.

---

## 8. EDM: Elucidating the Design Space (Karras et al. 2022)

Karras et al. argued that diffusion model theory was "unnecessarily convoluted" and separated concrete design choices:
- Preconditioning functions derived from first principles: network inputs and training targets should have unit variance, amplifying errors as little as possible
- Setting sigma(t) = t eliminates the time variable everywhere in favor of noise level directly
- Achieved CIFAR-10 FID 1.79 (class-conditional), 1.97 (unconditional) with only 35 network evaluations

The EDM perspective anticipated the schedule-free thinking later formalized by Dieleman: the noise schedule is just a reparameterization; what matters is the effective weighting over signal-to-noise ratios.

---

## 9. SNR, Loss Weighting, and Schedule Design

### Signal-to-Noise Ratio

SNR(t) = alpha(t)^2 / sigma(t)^2

The logSNR provides a unified measure across formulations. Different noise schedules (linear, cosine, VE, VP, sub-VP) all trace different paths through logSNR space but can be compared on this common axis.

### Schedule Types

- Linear (DDPM): beta_t increases linearly
- Cosine (iDDPM): smoother, SNR approaches zero more gradually
- Sub-VP / Flow Matching: alpha(t) = 1 - sigma(t), produces straighter paths
- Logit-normal (SD3): concentrates training around logSNR = 0, the critical transition point

### Loss Weighting Is What Matters

Dieleman (2024): "the weighting function is the most important part of the loss." The noise schedule doesn't add expressivity -- it's an arbitrary nonlinear function. What matters is the effective weighting w(t) * p(t) over noise levels. Min-SNR weighting (Hang et al. 2023) seeks Pareto-optimal direction for different prediction tasks.

---

## 10. Consistency Models and Other Acceleration Methods

### Consistency Models (Song et al. 2023)

Map any point on the ODE trajectory to the trajectory's origin (clean data), enabling single-step generation. Can be trained via distillation from pre-trained diffusion models or from scratch. CIFAR-10 FID 3.55 for one-step generation.

### Progressive Distillation (Salimans & Ho 2022)

Iteratively distill a teacher (many steps) into a student (half the steps). The v-prediction parameterization was introduced in this work specifically because noise prediction becomes unstable with very few steps -- precisely the Jensen Gap problem that the Geometry of Noise paper later formalized.

---

## 11. Broader Pattern: Error Amplification Near Targets

The stability analysis from the Geometry of Noise paper identifies a general principle that extends beyond diffusion models: when an iterative refinement system uses a parameterization where the gain (sensitivity to estimation errors) diverges as the system approaches its target, the system is structurally unstable. Conversely, when the gain remains bounded, the system is self-correcting.

This pattern applies to:
- Any denoising system where the noise level approaches zero
- Iterative optimization algorithms near convergence
- Control systems approaching a set point
- Numerical integration near singularities

The velocity parameterization works because it reframes the problem: instead of estimating what to subtract (noise, which vanishes and makes the estimation ill-conditioned), it estimates which direction to move (velocity, which remains well-defined throughout).

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | https://arxiv.org/abs/2602.18428 | fetched (abstract) | critical | Geometry of Noise paper -- Jensen Gap, bounded-gain, concentration of measure |
| 2 | https://arxiv.org/html/2602.18428v1 | fetched (full HTML) | critical | Complete technical details: all theorems, CIFAR-10 FID comparison (DDIM blind 40.90 vs FM blind 2.61) |
| 3 | https://arxiv.org/abs/2412.06264 | fetched (abstract) | high | Meta FAIR Flow Matching Guide and Code |
| 4 | https://arxiv.org/abs/2506.02070 | fetched (abstract) | high | MIT 6.S184 lecture notes on Flow Matching and Diffusion |
| 5 | https://arxiv.org/html/2506.02070 | fetched (full HTML) | high | Three prediction targets with formulas, stability comparison, CFG |
| 6 | https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ | fetched | high | Complete DDPM mathematical formulation, ELBO, DDIM equations |
| 7 | https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html | fetched | high | Cambridge MLG flow matching introduction: CFM trick, OT paths, training algorithm |
| 8 | https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/ | fetched | high | ICLR 2025 blogpost: visual CFM explanation, conditional paths, connection to diffusion |
| 9 | https://yang-song.net/blog/2021/score/ | fetched | high | Score-based generative models: score function, NCSN, SDE/ODE, VP/VE, controllable generation |
| 10 | https://blog.sotaaz.com/post/sde-vs-ode-en | fetched | high | SDE vs ODE mathematical foundations, VP-SDE, VE-SDE, Anderson's theorem |
| 11 | https://sander.ai/2023/07/20/perspectives.html | fetched | medium | Sander Dieleman's perspectives on diffusion: prediction targets, SNR, weighting |
| 12 | https://sander.ai/2024/06/14/noise-schedules.html | fetched | high | Noise schedules considered harmful: SNR analysis, implicit weighting, schedule-free formulation |
| 13 | https://diffusionflow.github.io/ | fetched | high | Unified perspective: diffusion = flow matching under Gaussian assumptions, weighting equivalence |
| 14 | https://arxiv.org/abs/2210.02747 | fetched (abstract) | high | Original Flow Matching paper by Lipman et al. |
| 15 | https://arxiv.org/abs/2202.00512 | fetched (abstract) | high | Progressive Distillation, v-prediction introduction (Salimans & Ho 2022) |
| 16 | https://arxiv.org/abs/2403.03206 | fetched (abstract) | high | SD3 paper: Scaling Rectified Flow Transformers (Esser et al. 2024) |
| 17 | https://learnopencv.com/stable-diffusion-3/ | attempted, JS-only | medium | SD3 architecture overview (metadata only) |
| 18 | https://en.wikipedia.org/wiki/Diffusion_model | search result | low | General overview, not fetched |
| 19 | https://arxiv.org/abs/2209.03003 | search result | medium | Rectified Flow (Liu et al. 2023) -- used search summary |
| 20 | https://github.com/black-forest-labs/flux | search result | medium | Flux official repo -- confirmed velocity-based architecture |
| 21 | https://huggingface.co/black-forest-labs/FLUX.1-schnell | search result | medium | 12B param rectified flow transformer |
| 22 | https://arxiv.org/abs/2303.01469 | search result | medium | Consistency Models (Song et al. 2023) |
| 23 | https://arxiv.org/abs/2206.00364 | search result | medium | EDM paper (Karras et al. 2022) |
| 24 | https://arxiv.org/abs/2011.13456 | search result | high | Score-Based Generative Modeling via SDE (Song et al. 2021) |
| 25 | https://arxiv.org/abs/2505.22935 | search result | medium | Noise conditioning not necessary for graph diffusion |
| 26 | https://www.marktechpost.com/2024/08/02/black-forest-labs-open-source-flux-1/ | search result | low | Flux announcement |
| 27 | https://www.marktechpost.com/2025/11/25/flux-2/ | search result | medium | FLUX.2 32B flow matching transformer |
| 28 | https://hunterheidenreich.com/notes/machine-learning/generative-models/flow-matching-for-generative-modeling/ | search result | medium | Flow matching notes |
| 29 | https://medium.com/@pietrobolcato/stable-diffusion-3-explained-84fd085934cb | search result | medium | SD3 rectified flow explanation |
| 30 | https://arxiv.org/html/2507.09595v1 | search result | medium | Demystifying Flux architecture |

## Research Context

- **Query**: Diffusion models and flow matching -- stability analysis and modern parameterization choices
- **Depth**: deep (auto-detected based on multi-faceted theoretical topic with specific papers)
- **Existing vault knowledge**: No existing notes on diffusion models, flow matching, or generative model parameterization
- **Knowledge gap addressed**: Entire topic is new to the vault. Covers foundational theory, stability proofs, and practical implications for production systems. The stability analysis pattern (error amplification near targets) is a transferable principle.
