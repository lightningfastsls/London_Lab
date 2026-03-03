---
description: "Fundamentally different from cache compression — information lives in adapter weights not context; KV-cache for 128K tokens drops from 12+ GB to <50 MB with constant scaling"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
  - "[[context-management]]"
---

# Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights

Doc-to-LoRA achieves a qualitative shift in memory scaling by moving document information from the context window (where it consumes KV-cache linearly with length) into adapter weights (where the cost is constant regardless of document size):

| Metric | Full Context | Doc-to-LoRA |
|--------|-------------|-------------|
| KV-cache (128K tokens) | 12+ GB | <50 MB |
| Update latency | N/A (per-query) | <1 second |
| Oracle context distillation VRAM | 7+ GB | Sub-GB |
| Memory scaling | Linear with doc length | Constant |

This is fundamentally different from since [[KV cache compression techniques extend effective context by 3-32x with trade-offs between memory reduction and information preservation]], which reduces cache size but retains the linear-in-length scaling. Doc-to-LoRA's memory cost is constant because the information is *in the weights*, not in cached key-value pairs. Adding more document content increases adapter rank (since [[Doc-to-LoRA chunk composition concatenates along rank dimension enabling extrapolation from 256 training tokens to 32K-plus context]]) but the total adapter size remains under 50 MB even for very long documents.

The trade-off is different too: cache compression loses some information from the compression process but preserves the full attention mechanism. Doc-to-LoRA loses some information from the document→adapter distillation (83.5% of full-context quality on SQuAD) but eliminates the per-query cost of re-reading the document entirely. For applications where the same document is queried many times, the amortization strongly favors the weight-based approach.

---

Source: [[lora-doc-to-lora-hypernetworks-research-2026-03-02]]

Relevant Notes:
- [[KV cache compression techniques extend effective context by 3-32x with trade-offs between memory reduction and information preservation]] -- the alternative approach this contrasts with
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- the mechanism producing these memory savings
- [[infinite context architectures combine compressive memory with standard attention to handle arbitrarily long sequences]] -- another approach to the same scaling problem
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- both approaches reduce per-query context cost, but Doc-to-LoRA moves information to weights while subagents compress summaries
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the context limitation that motivates moving information to weights

Topics:
- [[model-adaptation]]
- [[context-management]]
