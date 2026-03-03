---
description: "Isolating memory by project (git remote), agent identity, and task type prevents knowledge leakage between contexts — multiple implementations converge on this pattern including namespace hierarchies and access control tables"
type: pattern
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# memory scoping by project agent and task prevents cross-project contamination in multi-context agent systems

Memory scoping along three dimensions prevents knowledge leakage in multi-context agent systems:

1. **Project scope** — Memories are isolated by project, typically keyed on git remote URL or directory path. Memorix auto-scopes by git remote. OpenMemory implements per-project scoping with access control tables (allow/deny rules between apps and memories).

2. **Agent scope** — Different agents working in the same project may need different knowledge subsets. Agent-Recall uses namespace-based scope hierarchy for data isolation. Custom agents in Claude Code can maintain their own auto-memory.

3. **Task scope** — Different task types (user preferences vs troubleshooting vs architecture) benefit from different memory subsets. OpenMemory categorizes memories into user_preferences, implementation, troubleshooting, component_context, project_overview, incident_rca.

Since [[single-scope MCP memory causes cross-project contamination when agent contexts are not separated]], scoping is not optional for production deployments. The AIMultiple benchmark found that single-project implementations performed adequately, but cross-project scenarios failed.

This pattern is the memory-layer analogue of since [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] — both address the problem of context pollution through isolation boundaries. The difference is temporal: subagent isolation is within-session, memory scoping is cross-session.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[single-scope MCP memory causes cross-project contamination when agent contexts are not separated]] -- the failure mode this prevents
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the within-session analogue

Topics:
- [[agent-memory]]
