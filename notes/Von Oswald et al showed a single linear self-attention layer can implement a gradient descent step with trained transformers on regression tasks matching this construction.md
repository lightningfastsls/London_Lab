---
description: "ICML 2023 — proved equivalence for linear self-attention on regression, then empirically showed trained models converge to this GD-equivalent configuration, limited to simple tasks and linear attention"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# Von Oswald et al showed a single linear self-attention layer can implement a gradient descent step with trained transformers on regression tasks matching this construction

Von Oswald et al. (2023, "Transformers Learn In-Context by Gradient Descent," ICML) demonstrated a precise mathematical correspondence between linear self-attention and gradient descent. They provided a weight construction showing that a single linear self-attention layer can implement one step of gradient descent on a regression loss, and then empirically showed that transformers trained on regression tasks converge to weight configurations matching this construction.

The result is significant because it establishes that the transformer architecture can naturally discover optimization algorithms through training — the trained model becomes an optimizer learned within the weights of an outer optimization process. The construction is explicit: given the right weight matrices, the attention operation literally computes a gradient descent update.

Important scope limitations:

**Linear self-attention only**: The proof requires removing the softmax nonlinearity from attention, using linear attention instead. Standard softmax attention introduces nonlinearities that break the exact GD equivalence, though empirical results suggest approximate equivalence persists.

**Simple regression tasks**: The mathematical proof and primary experiments use linear regression tasks. The paper itself acknowledges: "our findings are restricted to small Transformers and simple regression problems." Extension to the complex, nonlinear settings of real LLMs remains an open question.

**Convergence is empirical, not guaranteed**: While the paper shows that trained models match the GD construction, this convergence is demonstrated empirically rather than proven theoretically for all cases. Subsequent work (Ahn et al. 2023, Zhang et al. 2023) has provided stronger theoretical guarantees, but for limited settings.

This complements since [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]], providing converging evidence from a different mathematical angle that ICL implements optimization-like dynamics.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- complementary result from attention+MLP perspective
- [[the ICL-as-implicit-weight-update analysis covers only single transformer blocks and final token output not full autoregressive generation]] -- shared scope limitations
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- LoRA is the explicit version of the gradient descent that this paper proves attention can implement implicitly

Topics:
- [[transformer-architecture]]
