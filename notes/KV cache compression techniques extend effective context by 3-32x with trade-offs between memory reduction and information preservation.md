---
description: "Active 2025-2026 research area — FreqKV (frequency-domain), ChunkKV (semantic chunks), KVzip (query-agnostic 3-4x), token eviction (permanent loss) — all trade compression for fidelity"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# KV cache compression techniques extend effective context by 3-32x with trade-offs between memory reduction and information preservation

KV (Key-Value) cache compression is an active research area (2025-2026) with multiple approaches, all sharing a fundamental trade-off: reducing memory enables longer effective context, but every compression scheme loses some information.

The major approaches differ in what they sacrifice:
- **FreqKV**: Frequency-domain compression that transforms KV cache into frequency space and discards high-frequency components. Extended LLaMA-2-7B from 8K to 256K context (32x) with stable perplexity. Preserves low-frequency patterns (gradual semantic relationships) at the cost of fine-grained positional specificity.
- **ChunkKV**: Semantic chunk-based compression that preserves linguistic structures by compressing within meaningful boundaries rather than arbitrary token windows.
- **KVzip**: Query-agnostic compression reducing KV cache by 3-4x with negligible performance loss. Works regardless of what the model will be asked, making it deployment-friendly.
- **Token eviction** (SnapKV, PyramidKV, FastKV): Evict tokens by attention score, keeping only the most-attended tokens in cache. Simple and fast, but information from evicted tokens is permanently lost — if a future query needs an evicted token, there is no recovery.

The fundamental constraint: since [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]], middle-position tokens already receive less attention and are the most likely candidates for eviction or aggressive compression — compounding the existing positional bias.

For agent architecture, KV cache compression is a deployment-level optimization rather than a design-level strategy. The architectural mitigations (subagent isolation, JIT retrieval, Fresh Context) remain primary; compression enables those patterns to scale to larger individual windows.

A fundamentally different alternative: since [[Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights]], the Doc-to-LoRA approach eliminates KV-cache scaling entirely by transferring document information into adapter weights. Rather than compressing the cache, this approach bypasses it — the information lives in the model weights, not in cached key-value pairs. The trade-off profile is different: 83.5% of full-context quality on reading comprehension, but constant memory cost regardless of document length.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]] -- the bias that interacts with eviction policies
- [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]] -- the application-level analog
- [[infinite context architectures combine compressive memory with standard attention to handle arbitrarily long sequences]] -- the end goal these techniques support
- [[Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights]] -- orthogonal approach that eliminates cache entirely
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- KV cache compression optimizes within the context-based (ICL) paradigm; the spectrum shows the alternative of moving toward weight-based knowledge

Topics:
- [[agent-cognition]]
