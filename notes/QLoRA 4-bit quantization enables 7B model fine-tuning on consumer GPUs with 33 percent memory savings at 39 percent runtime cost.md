---
description: "Dettmers et al 2023 — quantizes base model to 4-bit then trains LoRA on top; 21.33 GB to 14.18 GB VRAM at cost of 1.85h to 2.79h training time on 7B LLaMA"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# QLoRA 4-bit quantization enables 7B model fine-tuning on consumer GPUs with 33 percent memory savings at 39 percent runtime cost

QLoRA (Dettmers et al. 2023) combines model quantization with LoRA to make large model fine-tuning accessible on consumer hardware. The approach quantizes the pre-trained base model to 4-bit precision while training the LoRA adapter in full precision. The concrete trade-off on 7B LLaMA:

| Metric | Standard LoRA | QLoRA |
|--------|--------------|-------|
| VRAM | 21.33 GB | 14.18 GB |
| Training time | 1.85h | 2.79h |

The 33% memory reduction (7+ GB savings) is what makes 7B model fine-tuning feasible on a single consumer GPU like an RTX 3090 or 4090 rather than requiring datacenter hardware. The 39% runtime increase comes from the dequantization overhead during the forward pass, but this is a worthwhile trade-off for accessibility.

This directly addresses the challenge of since [[HPC dependency for transformer training versus local-only development capability]], enabling local LoRA fine-tuning without HPC infrastructure. Combined with the finding that since [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]], a researcher with a single consumer GPU and a small curated dataset can achieve competitive fine-tuning results.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the base method QLoRA extends
- [[DoRA weight decomposition into magnitude and direction consistently outperforms standard LoRA by 1-4 points across model sizes]] -- combines with QLoRA in QDoRA
- [[HPC dependency for transformer training versus local-only development capability]] -- the accessibility barrier QLoRA addresses
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- small curated datasets + consumer GPU = accessible fine-tuning

Topics:
- [[model-adaptation]]
