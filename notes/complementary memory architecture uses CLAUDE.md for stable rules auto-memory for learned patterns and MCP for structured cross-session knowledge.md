---
description: "Three-tier pattern emerging in practice where each layer handles different knowledge volatility — stable project rules versus learned workspace patterns versus rich diagnostic reasoning and cross-tool sharing"
type: pattern
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# complementary memory architecture uses CLAUDE.md for stable rules auto-memory for learned patterns and MCP for structured cross-session knowledge

The memory architecture emerging in Claude Code practice is not a single system but a three-tier complementary pattern, each layer handling knowledge of different volatility and scope:

1. **CLAUDE.md** — Human-written, version-controlled project instructions. Stable rules, conventions, and architecture decisions. Loaded every session, shared via git. This is the "constitution" layer.

2. **Auto-memory (MEMORY.md)** — Machine-written learned patterns. Build commands, debugging insights, code style preferences. Per-project, machine-local, 200-line limit. This is the "muscle memory" layer.

3. **MCP memory servers** — Structured cross-session knowledge. Semantic search, typed relations, forgetting mechanisms, cross-tool sharing. This is the "reasoning memory" layer that preserves diagnostic paths.

Since [[Claude Code auto-memory captures configuration not learning because it preserves workspace patterns but loses diagnostic reasoning paths]], the MCP layer fills the gap that auto-memory leaves. And since [[auto-memory 200-line hard-coded limit and lack of automatic consolidation creates growing redundancy without manual intervention]], MCP servers offer managed lifecycle as an alternative.

This vault implements a variant of this pattern: CLAUDE.md for rules, MEMORY.md for session patterns, and the notes/ knowledge graph (with qmd semantic search) for structured reasoning. The notes/ graph is not an MCP memory server, but it serves the same architectural role — persistent, searchable, structured knowledge with explicit connections.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[Claude Code auto-memory captures configuration not learning because it preserves workspace patterns but loses diagnostic reasoning paths]] -- the gap that motivates the three-tier pattern
- [[auto-memory 200-line hard-coded limit and lack of automatic consolidation creates growing redundancy without manual intervention]] -- the scaling limitation of tier 2

Topics:
- [[agent-memory]]
