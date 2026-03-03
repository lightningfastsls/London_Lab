---
description: "Documents split into contiguous chunks each producing rank-r adapter; chunks compose by rank concatenation (effective rank r*K for K chunks), enabling near-perfect NIAH at ~40K tokens"
type: method
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
  - "[[context-management]]"
---

# Doc-to-LoRA chunk composition concatenates along rank dimension enabling extrapolation from 256 training tokens to 32K-plus context

For documents exceeding training sequence length, Doc-to-LoRA partitions them into contiguous chunks processed independently by the same hypernetwork. Each chunk produces a rank-r LoRA adapter. The key innovation: chunks are composed by **concatenating along the rank dimension**, yielding an effective rank of r*K for K chunks.

This mechanism enables remarkable extrapolation far beyond training distribution. Trained only on sequences up to 256 tokens, Doc-to-LoRA achieves near-perfect accuracy on contexts up to 32K tokens — a 125x extrapolation. For needle-in-a-haystack evaluation, the haystack is segmented into 1,024-token chunks and composed into a single adapter. Despite training on only up to 8 chunks, evaluation reached approximately 40K tokens with near-perfect accuracy.

The rank concatenation is elegant because it is additive: each chunk contributes its own low-rank subspace to the combined adapter, and the combined adapter's rank grows linearly with the number of chunks. Since [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]], the growing rank simply means more capacity to encode more information — the information from different document sections lives in orthogonal subspaces of the adapter.

Long-context QA evaluation shows 85% relative accuracy at up to 32K tokens, far beyond the training length of 2,344 tokens. This demonstrates that the hypernetwork learns a general document→adapter mapping that generalizes compositionally, not just a memorized mapping for specific input lengths.

---

Source: [[lora-doc-to-lora-hypernetworks-research-2026-03-02]]

Relevant Notes:
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- the system this scaling mechanism serves
- [[Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights]] -- the memory benefit of this approach
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- why rank concatenation works for composition
- [[adapting multiple LoRA weight matrices with lower rank outperforms single-matrix adaptation at higher rank for the same parameter budget]] -- parallel finding: distributed rank captures more information than concentrated rank
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- chunk composition extends the persistent end of the spectrum to arbitrarily long documents

Topics:
- [[model-adaptation]]
- [[context-management]]
