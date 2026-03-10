---
description: "not all modifications need the same retrieval investment — files that caused regressions warrant deep search while new standalone code needs none, matching Adaptive RAG's complexity-based routing"
type: pattern
confidence: likely
created: 2026-03-07
meta_state: current
---

# Adaptive RAG routes retrieval depth by query complexity which maps to file modification risk in coding agents

Adaptive RAG (Jeong et al., 2024) routes queries to different retrieval depths based on predicted complexity. Simple factual lookups get shallow retrieval. Multi-step reasoning gets deep retrieval. The system learns to predict which queries need which depth, avoiding both under-retrieval (missing critical context) and over-retrieval (polluting context with irrelevant material).

For coding agents operating over a knowledge vault, the complexity dimension maps to modification risk. The relevant question is not "how complex is this query?" but "how likely is this modification to violate an architectural constraint?" Three risk tiers emerge:

- **HIGH (deep retrieval):** Files that have caused regressions, implement architectural invariants, or have known constraint dependencies. These warrant both static references (canary comments) and dynamic search (/kcheck). Examples: detection app state management, export adapters.
- **MEDIUM (shallow retrieval):** Files with non-obvious design decisions that aren't fragile. A static reference (canary comment) is sufficient — the agent sees the note title and can choose to load it.
- **LOW (no retrieval):** Standalone utilities, test files, new files with no constraint history. Adding retrieval triggers here creates noise that dilutes the signal from high-risk files.

The principle is that retrieval investment should be proportional to regression risk, not uniform. Since [[activation timing matters as much as retrieval quality in agent knowledge systems]], the depth dimension adds precision to the timing dimension — not just WHEN to retrieve but HOW DEEPLY. Since [[fewer well-placed activation triggers outperform many ignored ones because noise teaches agents to skip gates]], the risk classification ensures retrieval density stays focused where it matters.

---

Source: knowledge-activation-architecture-phase1-2026-03-07 (archived inbox)

Relevant Notes:
- [[activation timing matters as much as retrieval quality in agent knowledge systems]] — timing + depth together define the activation architecture
- [[fewer well-placed activation triggers outperform many ignored ones because noise teaches agents to skip gates]] — why LOW-risk files get no triggers
- [[CRAG's retrieval evaluator prevents noise-induced gate fatigue through relevance thresholds]] — the quality filter that complements depth routing
- [[simple retrieval tasks tolerate far more context than reasoning tasks requiring latent inference]] — the empirical basis for risk-tiered depth: HIGH-risk modifications are reasoning-intensive (need deep retrieval) while LOW-risk modifications are retrieval-tolerant (adding context wastes budget)

Topics:
- [[agent-memory]]
