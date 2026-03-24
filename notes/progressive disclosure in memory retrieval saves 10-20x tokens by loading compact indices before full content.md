---
description: "Multiple MCP memory servers implement two-stage retrieval: compact indices (50-100 tokens) for filtering, then full details (500-1000 tokens per result) only for relevant items — a memory-specific application of progressive disclosure"
type: pattern
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content

Multiple MCP memory servers converge on progressive disclosure for token efficiency:

- **claude-mem**: Compact indices (~50-100 tokens) before full details (~500-1000 tokens/result), approximately 10x savings.
- **Memorix**: 3-tier disclosure (search overview, timeline, full detail), approximately 10x savings.
- **SimpleMem**: Intent-aware retrieval planning reduces unnecessary full-content loads, contributing to overall 30x token reduction.

The pattern: present search results as lightweight summaries first, then load full content only for items the agent selects as relevant. This is a memory-specific application of since [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] — the same principle that drives this vault's discovery layers (title -> description -> full note body).

The token savings matter because memory competes for the same context window as the current task. Loading 50 full memories at 1000 tokens each consumes 50K tokens — potentially half the effective context window since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]]. Progressive disclosure keeps memory's footprint at ~5K tokens for the same 50 memories until specific details are needed.

This is not just efficiency — it directly affects reasoning quality. Since [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]], the context freed by progressive disclosure is disproportionately valuable when the agent needs to reason rather than retrieve.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] -- the general principle this applies
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- why token savings matter for reasoning
- [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]] -- why freed context is valuable

Topics:
- [[agent-memory]]
