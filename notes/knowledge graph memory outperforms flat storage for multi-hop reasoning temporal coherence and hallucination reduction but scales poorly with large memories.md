---
description: "Graph-structured memory enables relationship traversal and temporal tracking that flat key-value stores cannot — but graph traversal cost grows with memory size and NER extraction quality limits graph accuracy"
type: finding
confidence: likely
created: 2026-03-02
meta_state: current
---

# knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories

Graph-based memory approaches — entities connected by typed relations — outperform traditional flat storage (JSONL, markdown, SQLite rows) across four dimensions: multi-hop reasoning (traversing explicit relational links rather than searching unstructured text), temporal coherence (preventing logical hallucinations about time sequences), personalization (capturing interaction patterns as graph structure), and hallucination reduction (grounding outputs in structured, verifiable content).

The evidence comes from multiple implementations. Zep/Graphiti's formal graph G = (N, E, phi) with bi-temporal modeling achieved 94.8% accuracy on the DMR benchmark versus MemGPT's 93.4%, and 18.5% accuracy improvement on LongMemEval with 90% latency reduction. Since [[graph-based memory reduces hallucination by grounding agent outputs in structured verifiable content]], the reliability benefit is not just theoretical.

However, graph traversal scales poorly with memory size, extraction quality depends on NER accuracy (garbage entities create noisy graphs), and real-time graph updates are resource-intensive. This is why since [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]] — production systems increasingly combine graph structure for reasoning with vector search for discovery and keyword search for precision.

This vault's own architecture — wiki-link graph with atomic notes — implements a manual version of the knowledge graph paradigm. The manual curation avoids NER quality issues but cannot scale to the volume an automated system handles.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[graph-based memory reduces hallucination by grounding agent outputs in structured verifiable content]] -- the mechanism behind the reliability benefit
- [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]] -- the convergence driven by these trade-offs

Topics:
- [[agent-memory]]
