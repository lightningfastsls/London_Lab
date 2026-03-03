---
description: "Code graph indexes every function, dependency, and historical change enabling cross-file multi-module analysis — runs on Claude Opus 4.5 with prompt caching, v3 uses Agent SDK for autonomous investigation"
type: baseline
confidence: likely
created: 2026-03-02
meta_state: current
---

# Greptile full codebase indexing with code graph achieves 82 percent bug catch rate through multi-hop investigation

Greptile builds a comprehensive knowledge graph of the entire repository, indexing every function, dependency, and historical change. This code graph enables multi-hop investigation — tracing issues across files, modules, and dependency chains rather than reviewing files in isolation.

In Greptile's own benchmark (July 2025): 82% catch rate, 41% higher than Cursor's 58%. CodeRabbit achieved 44%. v3 (late 2025) uses the Anthropic Claude Agent SDK for autonomous investigation. Released Claude Code plugin: pull down and auto-address Greptile comments. MCP integration for Cursor, Windsurf, Claude Desktop.

The architectural insight: code graph indexing enables review quality that line-by-line analysis cannot achieve. Cross-file bugs (where a change in one module breaks invariants in another) require following dependency chains that single-file reviewers miss. This is why CodeRabbit, which reviews PRs without full codebase context, catches fewer bugs.

However, since [[vendor self-evaluation bias means every AI code review benchmark vendor wins their own evaluation]], the 82% figure comes from Greptile's own benchmark — Augment Code evaluated Greptile at only 45% on the same repositories. The true catch rate is likely between these extremes.

The code graph approach parallels this vault's own architecture: since [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]], structured relationships enable reasoning that flat retrieval cannot.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[vendor self-evaluation bias means every AI code review benchmark vendor wins their own evaluation]] -- the caveat on all vendor benchmarks
- [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]] -- the parallel pattern in memory

Topics:
- [[agent-governance]]
