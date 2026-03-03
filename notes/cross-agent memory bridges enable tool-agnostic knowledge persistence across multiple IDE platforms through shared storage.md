---
description: "Memorix supports 8 IDE agents (Cursor, Claude Code, Windsurf, Codex, Copilot, Kiro, OpenCode, Gemini CLI) through shared filesystem memory — knowledge persists across tool switches"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# cross-agent memory bridges enable tool-agnostic knowledge persistence across multiple IDE platforms through shared storage

Memorix (146 stars, v0.9.29) implements a cross-agent memory bridge supporting 8 IDE agents: Cursor, Claude Code, Windsurf, Codex, Copilot, Kiro, OpenCode, and Gemini CLI. All agents read and write to the same shared filesystem (`~/.memorix/data/`), auto-scoped by git remote per project.

The architecture: Orama search engine (BM25 + optional vector), 25 MCP tools across 5 categories, 9 observation types (gotchas, decisions, discoveries). Progressive 3-tier disclosure (search, timeline, detail) saves approximately 10x tokens.

This addresses a practical problem: developers frequently switch between AI coding tools as capabilities evolve. Without cross-agent memory, each tool switch means starting from zero. With a memory bridge, project knowledge — debugging insights, architectural decisions, user preferences — persists regardless of which tool accesses it.

The shared-filesystem approach is the simplest interop mechanism. More sophisticated approaches (shared databases, API servers) offer better concurrency handling but higher setup friction. Since [[complementary memory architecture uses CLAUDE.md for stable rules auto-memory for learned patterns and MCP for structured cross-session knowledge]], the memory bridge fills the cross-tool gap that CLAUDE.md (git-shared but read-only) and auto-memory (machine-local, tool-specific) cannot.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[complementary memory architecture uses CLAUDE.md for stable rules auto-memory for learned patterns and MCP for structured cross-session knowledge]] -- the three-tier pattern this extends
- [[memory scoping by project agent and task prevents cross-project contamination in multi-context agent systems]] -- scoping within the bridge

Topics:
- [[agent-memory]]
