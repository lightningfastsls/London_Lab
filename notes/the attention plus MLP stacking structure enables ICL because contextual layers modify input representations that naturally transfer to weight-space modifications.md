---
description: "Dherin et al's key insight — ICL is less about attention internals and more about how any contextual layer's input-space modifications propagate to implicit weight-space changes in downstream MLPs"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# the attention plus MLP stacking structure enables ICL because contextual layers modify input representations that naturally transfer to weight-space modifications

Dherin et al. (2025) provide a surprising architectural insight about why ICL works: the mechanism is not primarily about the internals of self-attention but about the stacking structure of a contextual layer (like attention) with an MLP. The paper's key formulation: "ICL is less about the internals of self-attention, but rather about the fact that regular neural networks can transfer modification of input space to their weight structure."

The mechanism works because attention modifies the input representation to the MLP based on context. The MLP then naturally transfers these input-space modifications to weight-space modifications — the MLP behaves as if its weights were temporarily adjusted for the current input. This is a general property of the contextual-layer + MLP stacking pattern, not specific to attention.

This means that in principle, any layer that modifies representations based on context could enable ICL-like behavior. Attention is particularly effective because it can selectively aggregate relevant context — choosing which parts of the context to attend to — but the implicit weight update mechanism would work with other contextual architectures too.

The architectural implication is significant: since [[attention alone cannot compute nonlinear features and MLP alone cannot communicate across positions making both necessary for the complete transformer compute primitive]], ICL emerges not from either component alone but from their composition. Attention provides context-dependent input modification; MLP provides the nonlinear weight-like computation that turns that modification into useful behavior change.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- the specific weight modification this enables
- [[attention alone cannot compute nonlinear features and MLP alone cannot communicate across positions making both necessary for the complete transformer compute primitive]] -- the complementary roles that compose into ICL
- [[transformers implement a gather-then-transform cycle where attention moves information between positions and MLP transforms it independently]] -- the gather-transform cycle that produces ICL
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- LoRA makes permanent the implicit weight modifications this stacking structure enables
- [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]] -- the process of converting these stacking-enabled implicit modifications into explicit permanent weight changes

Topics:
- [[transformer-architecture]]
