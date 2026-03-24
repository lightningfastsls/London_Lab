---
description: "Snell et al 2022 — model conditioned on [instructions+input] generates output; same model fine-tuned to produce output from input alone, internalizing the instructions into weights"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present

Context distillation (Snell et al., 2022, "Learning by Distilling Context") occupies the conceptual middle ground between in-context learning and explicit fine-tuning. The process: a model conditioned on [instructions + task-input] generates [scratch-pad + final answer]. Then the same model is fine-tuned to produce [final answer] from [task-input] alone, effectively internalizing the instructions into the model's weights.

This makes explicit the transfer from context to weights that since [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] describes at the mechanistic level. Context distillation is the *process* of performing that transfer: take knowledge that lives in context (ICL-style), and compress it into weight modifications (fine-tuning-style).

The limitation is cost: context distillation requires per-task optimization, making it expensive for applications requiring rapid adaptation to new documents or tasks. This is exactly the bottleneck that since [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] addresses — by training a hypernetwork to perform context distillation in a single forward pass.

The conceptual progression is therefore: ICL (implicit, per-query) → context distillation (explicit transfer process) → LoRA (explicit, per-task) → Doc-to-LoRA (automated distillation at ICL speed). Context distillation is the theoretical bridge; Doc-to-LoRA is its practical automation.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- the full progression this bridges
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- the mechanistic link between ICL and weights
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- the automation of this process
- [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]] -- hypernetworks amortize context distillation's per-task optimization cost
- [[ICL is mathematically equivalent to query-dependent low-rank weight modifications of the MLP where each context token contributes a rank-1 update]] -- the virtual weight modifications that context distillation makes permanent

Topics:
- [[model-adaptation]]
- [[agent-cognition]]
- [[context-management]]
