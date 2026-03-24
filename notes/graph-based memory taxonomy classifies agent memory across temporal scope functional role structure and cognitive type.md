---
description: "Feb 2026 survey proposes four-dimensional classification with five graph structures compared — knowledge graphs for factual relations, hierarchical for compressed experience, temporal for dynamic events, hypergraphs for n-ary interactions, and hybrid architectures"
type: method
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# graph-based memory taxonomy classifies agent memory across temporal scope functional role structure and cognitive type

The graph-based memory taxonomy survey (Feb 2026) proposes a four-dimensional classification system for agent memory:

1. **Temporal scope**: Short-term (within-session working memory) vs long-term (persistent cross-session knowledge).
2. **Functional role**: Knowledge memory (factual relations) vs experience memory (interaction patterns, outcomes).
3. **Structure**: Non-structural (flat, unconnected) vs structural (graph-organized with explicit relations).
4. **Cognitive type**: Semantic (facts), procedural (how-to), associative (related concepts), working (active context), episodic (specific events), sentiment (emotional valence).

Five graph structures are compared for implementing structured memory:
- **Knowledge graphs** — Best for factual relations, but struggle with n-ary relationships (relations involving more than two entities).
- **Hierarchical graphs** — Compressed experience representation, good for summarization.
- **Temporal graphs** — Dynamic event tracking, good for time-ordered knowledge.
- **Hypergraphs** — Complex n-ary interactions, but higher computational cost.
- **Hybrid architectures** — Combine multiple structures for different memory types.

Since [[agent memory is a distinct discipline from LLM memory RAG and context engineering with formation evolution and retrieval as core dynamics]], this taxonomy provides the structural vocabulary for designing memory systems. The four dimensions help determine which graph structure fits: factual knowledge benefits from knowledge graphs, temporal reasoning from temporal graphs, and complex multi-entity interactions from hypergraphs.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[agent memory is a distinct discipline from LLM memory RAG and context engineering with formation evolution and retrieval as core dynamics]] -- the broader framing this taxonomy serves
- [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]] -- the empirical results behind one dimension

Topics:
- [[agent-memory]]
