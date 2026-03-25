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

2. **Structured knowledge graphs**: Zep/Graphiti (20K stars), Memento (Neo4j backend, 404 stars), mcp-neuralmemory (Neo4j + Gemini). Best for multi-hop reasoning and temporal coherence since [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]], but higher setup cost and resource requirements.

3. **Hybrid vector + graph**: Memorix (Orama BM25 + optional vector, 8 IDEs), mcp-memory-service (SQLite-vec + typed graph edges, 1.4K stars), SimpleMem (LanceDB multi-view). The convergent pattern since [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]].

The fragmentation reflects genuine architectural trade-offs rather than immaturity — since [[agent memory is a distinct discipline from LLM memory RAG and context engineering with formation evolution and retrieval as core dynamics]], the three-dimensional taxonomy of forms, functions, and dynamics means different use cases favor different paradigms. This parallels the broader observation that since [[graph-based memory reduces hallucination by grounding agent outputs in structured verifiable content]], structured approaches offer reliability benefits but at a complexity cost that simple markdown avoids.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]] -- the convergence trend within this fragmentation
- [[Anthropic reference MCP memory server uses entity-relation-observation knowledge graph as JSONL with no built-in decay or scoping]] -- the canonical baseline
- [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]] -- empirical evidence for the structured KG camp's strengths and scaling limitation
- [[agent memory is a distinct discipline from LLM memory RAG and context engineering with formation evolution and retrieval as core dynamics]] -- the theoretical framework explaining why fragmentation reflects genuine architectural diversity
- [[8600 MCP servers and 6500 Claude Code plugin repositories exist as of March 2026 reflecting rapid open ecosystem growth]] -- the broader ecosystem denominator (96 memory servers out of 8600+ total)
- [[complementary memory architecture uses CLAUDE.md for stable rules auto-memory for learned patterns and MCP for structured cross-session knowledge]] -- the three camps map to different tiers of a practical memory stack

Topics:
- [[agent-memory]]
