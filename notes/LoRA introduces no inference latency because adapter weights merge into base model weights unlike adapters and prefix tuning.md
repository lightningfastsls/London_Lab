---
description: "W_new = W_0 + B*A computed once at deployment — no sequential adapter layers or prefix token budget consumed, making LoRA transparent at inference time"
type: finding
confidence: proven
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# LoRA introduces no inference latency because adapter weights merge into base model weights unlike adapters and prefix tuning

A critical practical advantage of LoRA over other PEFT methods: the low-rank adapter weights can be merged into the base model weights at deployment time. Since W_new = W_0 + B*A, the merged weight matrix is computed once and served as a standard model layer. There is no additional computation during inference — no sequential adapter bottleneck layers (Houlsby et al. 2019) and no virtual prefix tokens consuming attention budget (Li & Liang 2021).

This matters because alternative PEFT methods impose runtime costs:
- **Adapters** insert bottleneck layers between transformer layers, adding sequential computation per forward pass
- **Prefix tuning** prepends learnable virtual tokens that consume attention budget and add latency proportional to prefix length
- **LoRA** merges cleanly: compute B*A once, add to W_0, discard A and B, serve the fused model

The merge-ability also enables since [[multi-LoRA serving enables hundreds of concurrent adapters from a single base model with millisecond switching in production]], because switching adapters means swapping a small delta rather than modifying the model architecture. This property — that the adaptation is mathematically equivalent to a weight change, not an architectural change — is what makes LoRA the foundation for production PEFT serving infrastructure.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the method that enables this merge property
- [[multi-LoRA serving enables hundreds of concurrent adapters from a single base model with millisecond switching in production]] -- the production consequence
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- hypernetwork-generated adapters inherit this merge property

Topics:
- [[model-adaptation]]
