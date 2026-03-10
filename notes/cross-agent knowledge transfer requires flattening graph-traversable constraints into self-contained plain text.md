---
description: "agents without vault access cannot follow wiki links or run semantic search — constraints must be serialized into handoff documents as standalone statements with source attribution"
type: pattern
confidence: likely
created: 2026-03-07
meta_state: current
---

# cross-agent knowledge transfer requires flattening graph-traversable constraints into self-contained plain text

A knowledge vault's power comes from its graph structure: wiki links enable spreading activation, topic maps provide navigation, and semantic search discovers non-obvious connections. But these capabilities are a barrier for agents that don't have vault access.

When Claude Code generates task specs for Codex (which operates without MCP, without qmd, and without the vault's linking infrastructure), architectural constraints must be *flattened* from their graph representation into self-contained plain text. The constraint "since [[saved-previous ghost detections current editable and saved-current form three aligned detection state tiers in the app]], never merge saved-previous and current detection lists" becomes a standalone sentence in the handoff: "Ghost detections (saved_previous) must remain separate from current editable detections — merging them causes duplicate rendering."

The flattening operation loses the graph's navigability but preserves the constraint's actionable content. Since [[spreading activation models how agents should traverse]], the receiving agent cannot traverse — it can only read what was serialized. This means the handoff author must anticipate which constraints the receiving agent will need, since the receiving agent has no way to discover constraints it wasn't given.

This is a general pattern for multi-agent systems with heterogeneous capabilities. Since [[activation timing matters as much as retrieval quality in agent knowledge systems]], the activation happens at handoff-generation time (by the agent WITH vault access), not at task-execution time (by the agent WITHOUT it). The vault-aware agent activates on behalf of the vault-unaware agent.

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — activation must happen at handoff time since the receiving agent cannot self-activate
- [[spreading activation models how agents should traverse]] — describes the graph traversal that flattening sacrifices
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] — related pattern of information compression across agent boundaries
- [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]] — constraint flattening for handoffs is a form of information compression at delegation boundaries, where the handoff author's activation budget must cover what the receiver cannot activate independently

Topics:
- [[agent-memory]]
