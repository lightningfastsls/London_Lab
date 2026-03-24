---
description: "Kalajdzievski 2023 — theoretically optimal scaling factor that decouples adaptation magnitude from rank hyperparameter, eliminating rank-dependent retuning"
type: method
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# rsLoRA rank-stabilized scaling uses alpha over sqrt(r) instead of alpha over r preventing adaptation strength from depending on rank choice

Standard LoRA scales the adapter output by alpha/r, where alpha is a fixed hyperparameter and r is the rank. This creates a subtle coupling: changing rank changes the effective adaptation strength, requiring alpha retuning for each rank value. rsLoRA (Kalajdzievski 2023) replaces alpha/r with alpha/sqrt(r), which provides theoretically optimal scaling that keeps the adaptation magnitude independent of rank.

The practical consequence: when experimenting with different rank values (a common workflow since [[adapting multiple LoRA weight matrices with lower rank outperforms single-matrix adaptation at higher rank for the same parameter budget]] suggests rank allocation matters), rsLoRA eliminates one confound. Without rank-stabilized scaling, performance differences between rank values reflect both the rank's capacity and the scaling's side effects. With rsLoRA, rank differences reflect only capacity.

The standard LoRA heuristic of alpha = 2*rank partially compensates — it scales alpha linearly with rank, which helps but is not the theoretically correct relationship. rsLoRA's sqrt(r) scaling is mathematically derived to maintain constant expected output magnitude regardless of rank.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the base scaling formula this corrects
- [[adapting multiple LoRA weight matrices with lower rank outperforms single-matrix adaptation at higher rank for the same parameter budget]] -- the rank experimentation context where this matters
- [[DoRA weight decomposition into magnitude and direction consistently outperforms standard LoRA by 1-4 points across model sizes]] -- another LoRA variant addressing different scaling limitations

Topics:
- [[model-adaptation]]
