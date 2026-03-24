---
description: "The vault uses markdown files (ops/last-session.md, MEMORY.md) for session continuity — MCP memory servers offer richer semantic search, forgetting, and diagnostic path preservation but add infrastructure complexity"
type: open-question
confidence: speculative
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# whether MCP memory servers would improve this vault's session continuity beyond the current ops handoff approach

This vault's cross-session continuity relies on three mechanisms: ops/last-session.md (explicit handoff), MEMORY.md (auto-memory patterns), and the notes/ knowledge graph (persistent reasoning). This combination works but has known limitations — since [[Claude Code auto-memory captures configuration not learning because it preserves workspace patterns but loses diagnostic reasoning paths]], the diagnostic trail of each session is lost.

MCP memory servers offer capabilities the current approach lacks: semantic search over session history, automatic consolidation of recurring patterns, forgetting of stale information, and cross-tool portability. Since [[eight practical criteria for evaluating cross-session memory tools span retrieval precision token efficiency scope isolation forgetting quality setup friction portability auditability and consolidation]], a rigorous evaluation would test whether the additional infrastructure cost yields measurable continuity improvement.

Key considerations:
- The vault's wiki-link graph already provides structured knowledge retrieval — would MCP memory duplicate or complement this?
- Progressive disclosure since [[progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content]] could reduce the orient hook's token footprint
- Since [[complementary memory architecture uses CLAUDE.md for stable rules auto-memory for learned patterns and MCP for structured cross-session knowledge]], MCP would fill the "diagnostic reasoning" layer without replacing existing mechanisms
- Setup friction (Docker, databases) is a real barrier for a single-developer research project

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[Claude Code auto-memory captures configuration not learning because it preserves workspace patterns but loses diagnostic reasoning paths]] -- the gap MCP could address
- [[complementary memory architecture uses CLAUDE.md for stable rules auto-memory for learned patterns and MCP for structured cross-session knowledge]] -- the three-tier pattern

Topics:
- [[agent-memory]]
