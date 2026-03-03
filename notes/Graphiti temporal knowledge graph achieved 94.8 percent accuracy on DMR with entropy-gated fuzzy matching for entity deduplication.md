---
description: "Zep's Graphiti (20K stars) uses bi-temporal modeling and deterministic IR front-ends before LLM fallback for entity matching — 94.8 percent on DMR versus MemGPT 93.4 percent and 18.5 percent improvement on LongMemEval"
type: baseline
confidence: proven
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# Graphiti temporal knowledge graph achieved 94.8 percent accuracy on DMR with entropy-gated fuzzy matching for entity deduplication

Zep/Graphiti (20K GitHub stars, MCP Server 1.0) is a temporally-aware knowledge graph engine with formal graph G = (N, E, phi) supporting bi-temporal modeling — tracking both when facts were true in the world and when they were recorded in the system.

Benchmark results: 94.8% accuracy on DMR (vs MemGPT's 93.4%), 18.5% accuracy improvement on LongMemEval with 90% latency reduction. These benchmarks test temporal reasoning — the ability to correctly answer questions about time-ordered events and knowledge evolution.

The key technical innovation is entropy-gated fuzzy matching for entity deduplication. The problem: knowledge graphs accumulate duplicate entities ("React.js", "ReactJS", "React") that fragment the graph. Graphiti addresses this by using deterministic IR front-ends (cheap, fast) before falling back to LLMs (expensive, accurate) — the entropy gate determines when the cheap method is sufficient versus when LLM judgment is needed. This efficiency improvement enables real-time graph updates that would otherwise be prohibitively expensive.

Since [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]], Graphiti's benchmarks validate the graph approach while the entropy-gated matching addresses the scaling concern.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[knowledge graph memory outperforms flat storage for multi-hop reasoning temporal coherence and hallucination reduction but scales poorly with large memories]] -- the paradigm this validates
- [[hybrid memory architectures combining keyword vector and graph search are converging as the dominant paradigm for agent memory]] -- the broader convergence trend

Topics:
- [[agent-memory]]
