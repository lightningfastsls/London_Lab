---
description: "Dherin et al 2025 showed context tokens are processed like SGD steps with h = 1/||A(x)||² as implicit learning rate — an analogy/resemblance rather than exact equivalence"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# sequential ICL context processing follows dynamics resembling online stochastic gradient descent with learning rate determined by attention magnitude

Dherin et al. (2025) show that the sequential processing of context tokens in ICL follows a dynamic that resembles online stochastic gradient descent. The implicit update rule takes the form W_i = W_{i-1} - h · grad_W L_i(W_{i-1}), where the learning rate h = 1/||A(x)||² is determined by the attention output magnitude, and L_i(W) = trace(delta_i^T · W) is an implicit per-token loss.

Each new context token is like a new training example processed by one step of gradient descent. The learning rate is not fixed but depends on the attention output — when the model attends strongly (large ||A(x)||), the learning rate is smaller, creating a natural adaptive rate. Gradient updates decay as context converges, consistent with gradient descent dynamics approaching a minimum.

This is framed as an analogy/resemblance rather than an exact equivalence — the paper demonstrates the connection mathematically for the simplified single-block case but appropriately hedges on whether the same dynamics hold in full multi-layer models. The resemblance is strongest for the first generated token and weakens for autoregressive generation where each new token changes the context.

The SGD interpretation has a powerful implication: it suggests that the quality of ICL depends on the same factors that affect SGD quality — learning rate (attention magnitude), number of steps (number of context examples), and gradient quality (how informative each example is). This may explain why example selection and ordering matter for ICL, since [[ICL performance degrades with excessive context because the issue is attention quality not token capacity]].

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- the weight update this SGD produces
- [[ICL performance degrades with excessive context because the issue is attention quality not token capacity]] -- quality degradation that SGD dynamics may explain
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- ICL's implicit SGD makes the parallel to LoRA's explicit gradient descent even more direct
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- ICL's SGD dynamics connect to LoRA's gradient descent via a shared optimization interpretation

Topics:
- [[transformer-architecture]]
