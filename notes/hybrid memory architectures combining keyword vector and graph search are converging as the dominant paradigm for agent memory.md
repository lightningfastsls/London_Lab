---
description: "No single memory paradigm suits all scenarios — production systems converge on BM25 keyword plus vector embedding plus typed graph edges because each addresses a different retrieval failure mode"
type: finding
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory

The graph-based memory taxonomy survey (Feb 2026) concludes that no single paradigm suits all agent memory scenarios — success depends on aligning memory structure with specific application requirements. In practice, the leading implementations all combine multiple retrieval paradigms:

- **mcp-memory-service**: BM25 + vector search + typed graph edges (causes, fixes, contradicts)
- **SimpleMem**: LanceDB with multi-view indexing (dense embeddings, BM25 sparse, SQL metadata)
- **Memorix**: Orama BM25 + optional vector + knowledge graph layer
- **Memento**: Neo4j graph + vector embeddings in same database

Each paradigm addresses a different retrieval failure mode. Keyword search (BM25) excels at exact-match precision but misses semantic equivalents. Vector search catches conceptual similarity but cannot perform logical joins. Graph traversal enables multi-hop reasoning but scales poorly. Combining them means no single failure mode blocks retrieval.

This mirrors the dual-discovery principle in knowledge management: since [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]], hybrid retrieval reduces the risk that relevant memory is unfindable. This vault's own architecture implements a version of this: wiki links (explicit graph) plus qmd semantic search (implicit vector), as documented in CLAUDE.md's "Semantic Search" section.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]] -- the trade-off that drives hybridization
- [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] -- the progressive disclosure pattern complementary to hybrid search

Topics:
- [[agent-memory]]
