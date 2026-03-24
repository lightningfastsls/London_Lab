---
description: "Synthesized evaluation framework from research and implementations — retrieval precision and token efficiency are necessary but forgetting quality and auditability differentiate production-ready systems"
type: method
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# eight practical criteria for evaluating cross-session memory tools span retrieval precision token efficiency scope isolation forgetting quality setup friction portability auditability and consolidation

From the convergence of research benchmarks (MemBench, MemoryAgentBench, LOCOMO, DMR, AIMultiple) and practical implementation patterns, eight evaluation criteria emerge for cross-session agent memory:

1. **Retrieval precision** — Does it return what is relevant, not everything? Hybrid search (keyword + vector + graph) correlates with higher precision.
2. **Token efficiency** — Progressive disclosure vs loading all memories. The difference between ~5K and ~50K tokens for the same memory set.
3. **Scope isolation** — Project separation, agent separation, task separation. The AIMultiple benchmark showed this is a hard failure mode.
4. **Forgetting quality** — Does it manage stale and contradicted knowledge? Systems without active forgetting degrade over time.
5. **Setup friction** — Zero-config (auto-memory) vs Docker + databases (OpenMemory). Inversely correlates with capability.
6. **Cross-tool portability** — Does memory survive IDE switches? Only shared-storage approaches (Memorix, OpenMemory) achieve this.
7. **Auditability** — Can humans inspect and edit what is stored? JSONL and markdown are auditable; opaque vector stores are not.
8. **Consolidation strategy** — How does memory scale over months? Dream-inspired cycles, threshold-based compression, or hierarchical promotion.

Since [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]], criteria 4 (forgetting quality) and 8 (consolidation strategy) are the differentiators between demo-quality and production-quality systems. Many MCP memory servers score well on retrieval but lack any forgetting mechanism.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] -- why forgetting quality is a differentiator
- [[progressive disclosure in memory retrieval saves 10-20x tokens by loading compact indices before full content]] -- criterion 2 in practice
- [[Letta sleep-time compute pairs a primary agent with a sleep-time agent that processes memory during idle periods]] — concrete architecture implementing criterion 8 (consolidation strategy): sleep-time agents process memory during idle gaps, the most explicit paired-agent approach to consolidation
- [[between-session observation accumulation is directed dreaming that produces patterns no individual session contained]] — the vault's implementation of criterion 8: observation accumulation + threshold-triggered rethink IS a consolidation strategy that produces generative recombination, not just compression
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — criterion 1 (retrieval precision) is necessary but not sufficient: timing the retrieve-or-not decision matters equally, and no criterion currently captures this activation dimension

Topics:
- [[agent-memory]]
