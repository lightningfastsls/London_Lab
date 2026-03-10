---
description: "four RAG papers converge on the retrieve-or-not decision as the primary bottleneck — systems with good retrieval still fail when activation triggers are absent or poorly timed"
type: finding
confidence: likely
created: 2026-03-07
meta_state: current
---

# activation timing matters as much as retrieval quality in agent knowledge systems

The key lesson across Self-RAG, FLARE, CRAG, and Adaptive RAG is that retrieval quality alone does not determine system effectiveness. The decision of *when to retrieve* — and whether to retrieve at all — is equally critical. A system with excellent retrieval infrastructure but no activation triggers will underperform a system with mediocre retrieval but well-timed triggers.

This maps directly to knowledge vault operation. The vault has good retrieval: qmd provides keyword and semantic search, topic maps organize navigation, and dense wiki links enable spreading activation. But since [[spreading activation models how agents should traverse]], activation only works once traversal is triggered. The gap is entirely in activation — the agent must voluntarily decide to search, and under cognitive load that decision is the first to be dropped.

The four papers each address a different aspect of timing:
- **Self-RAG** (Asai et al., ICLR 2024): trains the model to decide retrieve/no-retrieve via reflection tokens
- **FLARE** (Jiang et al., EMNLP 2023): triggers retrieval when the model's confidence drops during generation
- **CRAG** (Yan et al., 2024): evaluates retrieval results before surfacing them, preventing noise
- **Adaptive RAG** (Jeong et al., 2024): routes queries to different retrieval depths by complexity

Together they argue that activation is a design problem, not a willpower problem. Since [[schema validation hooks externalize inhibitory control that degrades under cognitive load]], the same principle applies to retrieval decisions: the agent cannot be trusted to remember to search under pressure, so the system must create activation triggers that fire automatically or procedurally.

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[spreading activation models how agents should traverse]] — covers HOW to traverse but not WHEN to trigger; this note fills the timing gap
- [[schema validation hooks externalize inhibitory control that degrades under cognitive load]] — the same externalization principle applied to retrieval decisions
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] — the storage-side analogue: timing of forgetting matters as much as timing of retrieval

Topics:
- [[agent-memory]]
