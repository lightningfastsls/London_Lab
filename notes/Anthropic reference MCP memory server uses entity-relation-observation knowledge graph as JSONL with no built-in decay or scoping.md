---
description: "The canonical @modelcontextprotocol/server-memory provides 9 tools over entities/relations/observations in JSONL — simple reference baseline that all production implementations improve upon"
type: baseline
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Anthropic reference MCP memory server uses entity-relation-observation knowledge graph as JSONL with no built-in decay or scoping

Anthropic's `@modelcontextprotocol/server-memory` provides the canonical MCP memory pattern. It stores a knowledge graph as JSONL (default `memory.jsonl`) with three primitives: entities (nodes with name, type, and observations as atomic fact strings), relations (directed edges in active voice between entities), and observations (discrete facts attached to entities, independently addable and removable).

Nine tools: create_entities, create_relations, add_observations, delete_entities, delete_observations, delete_relations, read_graph, search_nodes, open_nodes. No built-in decay, expiration, or automatic consolidation — memory persists until explicitly deleted. No scoping mechanism beyond using separate JSONL files.

This simplicity is deliberate: as a reference implementation, it establishes the minimal viable pattern. But it means since [[single-scope MCP memory causes cross-project contamination when agent contexts are not separated]], production deployments must add their own scoping. The entity-relation-observation model itself is well-suited to knowledge graphs — entities as nodes, relations as directed edges, observations as properties — but the flat JSONL storage lacks the query capabilities of graph databases like Neo4j that power more sophisticated implementations.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[single-scope MCP memory causes cross-project contamination when agent contexts are not separated]] -- the failure mode this baseline doesn't prevent
- [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]] -- the paradigm this implementation uses

Topics:
- [[agent-memory]]
