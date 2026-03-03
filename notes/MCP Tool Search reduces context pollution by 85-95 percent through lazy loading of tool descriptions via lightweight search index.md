---
description: "With 7 MCP servers active, tool definitions consumed 67K tokens (33.7 percent of 200K budget) before any conversation — Tool Search replaces this with a 3K token search index that loads 3-5 relevant tools on demand"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
---

# MCP Tool Search reduces context pollution by 85-95 percent through lazy loading of tool descriptions via lightweight search index

MCP Tool Search shipped in Claude Code 2.1.7 (January 14, 2026) to solve a critical scaling problem: with 7 MCP servers active, tool definitions consumed 67,300 tokens — 33.7% of the 200K context budget — before any conversation began. This left barely half the effective context for actual work, since [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]].

The mechanism: when MCP tool descriptions exceed 10K tokens total, Tool Search marks tools with `defer_loading`, injects a lightweight search index instead (~3K tokens), and selectively loads 3-5 relevant tools per query based on the index. Token reduction: 77K → 8.7K (85-95% savings). Now enabled by default for all users.

This is the critical enabler for running many MCP servers simultaneously. Before Tool Search, each additional MCP server directly consumed context budget, creating a hard scaling ceiling. After Tool Search, the marginal cost of an additional MCP server is near-zero for irrelevant tools and a few hundred tokens for relevant ones.

Since [[progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content]], Tool Search applies the same progressive disclosure pattern to tool definitions: present a lightweight index, then load full definitions only when needed. The principle is identical; the domain (tool descriptions vs memory retrieval) is different.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the context budget Tool Search protects
- [[progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content]] -- the same pattern applied to memory

Topics:
- [[context-management]]
- [[agent-memory]]
