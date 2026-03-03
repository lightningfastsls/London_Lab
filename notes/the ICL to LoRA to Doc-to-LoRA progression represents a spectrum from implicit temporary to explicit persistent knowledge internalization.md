---
description: "ICL (implicit, per-query, no weight change) → LoRA (explicit, per-task, gradient-trained weights) → Doc-to-LoRA (instant, per-document, hypernetwork-generated weights) — same goal, different persistence"
type: pattern
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
  - "[[agent-cognition]]"
---

# the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization

There is a clean conceptual progression in how LLMs absorb task-specific or document-specific knowledge, moving along two axes simultaneously: from implicit to explicit, and from temporary to persistent.

**In-Context Learning (ICL)**: Knowledge lives in the prompt. Every query re-reads the full context, paying quadratic attention cost. Temporary, per-query, no weight changes. Since [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]], ICL simulates weight changes through attention without actually modifying anything.

**LoRA Fine-Tuning**: Knowledge is explicitly baked into low-rank weight updates. Requires gradient-based training (minutes to hours per adapter). Persistent, per-task, explicit weight changes. Since [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]], LoRA takes what ICL does virtually and makes it real.

**Doc-to-LoRA**: Since [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]], Doc-to-LoRA automates LoRA generation. Document in, adapter out, sub-second, no gradients needed. This combines ICL's speed with LoRA's persistence — knowledge internalization at the speed of ICL with the durability of fine-tuning.

Between ICL and LoRA sits since [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]], which makes the intermediate step explicit.

The implication for agent architecture: rather than choosing between context-based and weight-based knowledge, the progression suggests these are points on a continuum. Future systems may dynamically shift knowledge between context and weights based on access frequency and persistence requirements.

---

Source: [[lora-doc-to-lora-hypernetworks-research-2026-03-02]]

Relevant Notes:
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- the implicit end of the spectrum
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- the ICL-LoRA bridge
- [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]] -- the intermediate step
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- the automated end of the spectrum
- [[ICL fails on specification-heavy tasks reaching less than half of fine-tuned performance due to inadequate schema comprehension]] -- the ICL ceiling that motivates movement along the spectrum toward weight-based internalization
- [[Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights]] -- the practical memory benefit of moving along the spectrum
- [[sequential ICL context processing follows dynamics resembling online stochastic gradient descent with learning rate determined by attention magnitude]] -- the SGD dynamics connecting ICL's implicit optimization to LoRA's explicit optimization

Topics:
- [[model-adaptation]]
- [[agent-cognition]]
