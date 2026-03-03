---
description: "The Glama registry lists 96 knowledge-and-memory MCP servers spanning three architectural camps — landscape baseline showing the breadth of approaches to cross-session agent knowledge"
type: baseline
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# 96 MCP memory servers exist as of March 2026 fragmented across unstructured markdown knowledge graph and hybrid vector-graph paradigms

The MCP memory server landscape as of March 2026 has fragmented into three architectural camps, with at least 96 knowledge-and-memory servers listed on Glama alone:

1. **Unstructured markdown**: Claude Code auto-memory, simple file-based approaches. Lowest friction, human-readable, but no semantic search or relationship tracking.

2. **Structured knowledge graphs**: Zep/Graphiti (20K stars), Memento (Neo4j backend, 404 stars), mcp-neuralmemory (Neo4j + Gemini). Best for multi-hop reasoning and temporal coherence, but higher setup cost and resource requirements.

3. **Hybrid vector + graph**: Memorix (Orama BM25 + optional vector, 8 IDEs), mcp-memory-service (SQLite-vec + typed graph edges, 1.4K stars), SimpleMem (LanceDB multi-view). The convergent pattern since [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]].

The fragmentation reflects genuine architectural trade-offs rather than immaturity — different use cases favor different paradigms. This parallels the broader observation that since [[graph-based memory reduces hallucination by grounding agent outputs in structured verifiable content]], structured approaches offer reliability benefits but at a complexity cost that simple markdown avoids.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]] -- the convergence trend within this fragmentation
- [[Anthropic reference MCP memory server uses entity-relation-observation knowledge graph as JSONL with no built-in decay or scoping]] -- the canonical baseline

Topics:
- [[agent-memory]]
