---
description: "Auto-memory stores build commands and preferences but not the diagnostic trail — which approaches failed, why, and under what conditions — so each new session rediscovers rather than builds on prior reasoning"
type: finding
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Claude Code auto-memory captures configuration not learning because it preserves workspace patterns but loses diagnostic reasoning paths

Claude Code's auto-memory (MEMORY.md) captures build commands, debugging insights, architecture notes, code style preferences, and workflow habits. What it does not capture is the diagnostic path — which approaches were tried, why some failed, what specific conditions made one approach work over another. That reasoning disappears when a session ends.

Peterson (2026) frames this as "configuration, not learning": auto-memory trains Claude to operate in a specific workspace but does not build cumulative understanding. The distinction matters because configuration is static (run this command, use this pattern) while learning is dynamic (this approach failed because X, try Y when Z). An agent with only configuration repeats the same investigation each session; an agent with learning compounds its effectiveness.

This explains why structured knowledge systems like wiki-link graphs and MCP memory servers have emerged as complements to auto-memory. Since [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]], the ideal is not to load everything at session start but to have retrievable reasoning available on demand. Auto-memory handles the "what to do" layer; the reasoning layer requires something richer.

The practical consequence is a three-tier architecture: CLAUDE.md for stable rules, auto-memory for learned patterns, and structured knowledge (whether vault notes or MCP memory) for diagnostic reasoning and cross-session learning.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] -- the progressive disclosure pattern that complements auto-memory
- [[context compaction quality degrades cumulatively with multiple compressions regardless of implementation]] -- auto-memory's 200-line limit forces a form of compaction
- [[RAG-based memory provides only marginal improvement versus intent resolution demonstrating retrieval is not equivalent to resolving intent]] -- retrieval alone is insufficient even with richer memory

Topics:
- [[agent-memory]]
