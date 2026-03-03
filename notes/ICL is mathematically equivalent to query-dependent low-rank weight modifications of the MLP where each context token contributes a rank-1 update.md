---
description: "Dherin et al 2025 'Learning Without Training' — attention+MLP stacking implicitly creates delta_W(Y) as rank-1 per context token, but the update varies per query input (not a fixed single delta_W)"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update

Dherin et al. (2025, "Learning Without Training: The Implicit Dynamics of In-Context Learning") provide a mathematical framework showing that in-context learning is equivalent to low-rank weight modifications of the MLP layer. The attention layer produces a difference vector representing the change in attention output caused by including context element Y, and this gets transformed into a rank-1 update to the MLP's first-layer weight matrix.

Each context token contributes one rank-1 update. Multiple context tokens create a low-rank update with rank at most equal to the number of context tokens. This elegantly explains why ICL performance generally improves with more demonstrations — each additional example adds another rank-1 component to the implicit weight modification.

A critical nuance: the weight update is **query-dependent** — it varies for each input token being processed. The context does not reduce to a single fixed delta_W that could be precomputed and applied universally. This means the "weight modification" is better understood as a virtual, per-query adaptation rather than a true weight change. The MLP *behaves as if* its weights were modified, but the modification is entangled with the specific query.

Experimental validation shows training/validation losses match between direct context processing and weight-modified inference, and gradient updates decay as context converges, consistent with optimization dynamics. However, since [[the ICL-as-implicit-weight-update analysis covers only single transformer blocks and final token output not full autoregressive generation]], the full picture for multi-layer models remains an open theoretical question.

The explicit counterpart to this implicit mechanism is LoRA (Hu et al. 2021), which performs actual gradient descent to learn a low-rank weight update B*A. Since [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]], the parallel is direct: ICL creates implicit rank-1 updates per context token via attention, while LoRA creates explicit low-rank updates through gradient optimization. Both operate in the same space of low-rank weight modifications — ICL's are virtual and per-query, LoRA's are permanent and per-task.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[sequential ICL context processing follows dynamics resembling online stochastic gradient descent with learning rate determined by attention magnitude]] -- the SGD interpretation of this mechanism
- [[the attention plus MLP stacking structure enables ICL because contextual layers modify input representations that naturally transfer to weight-space modifications]] -- why this mechanism exists architecturally
- [[the ICL-as-implicit-weight-update analysis covers only single transformer blocks and final token output not full autoregressive generation]] -- important limitations
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- the explicit counterpart making real what ICL simulates
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- LoRA's low-rank constraint mirrors the rank structure of ICL's implicit updates
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- this finding anchors the implicit end of the spectrum
- [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]] -- the explicit process of converting these virtual weight modifications into real ones
- [[pre-trained language models have low intrinsic dimension with larger models having even lower intrinsic dimension after pre-training]] -- low intrinsic dimension means ICL's rank-limited updates may suffice for many tasks

Topics:
- [[transformer-architecture]]
