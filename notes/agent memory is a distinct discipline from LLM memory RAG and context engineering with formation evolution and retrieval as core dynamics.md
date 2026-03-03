---
description: "Dec 2025 survey establishes agent memory as separate from model-level memory, RAG pipelines, and prompt engineering — with a three-dimensional taxonomy of forms, functions, and dynamics spanning five research frontiers"
type: finding
confidence: likely
created: 2026-03-02
meta_state: current
topics: "[[agent-memory]]"
---

# agent memory is a distinct discipline from LLM memory RAG and context engineering with formation evolution and retrieval as core dynamics

The "Memory in the Age of AI Agents" survey (Dec 2025) formally establishes agent memory as a distinct research discipline, separate from three related but different concepts:

- **LLM memory** — What the model learned during pretraining. Static, parametric, not modifiable at runtime.
- **RAG** — Retrieval-augmented generation. Provides external context at inference time but has no lifecycle, no forgetting, no learning.
- **Context engineering** — Managing what goes into the context window. Tactical, session-scoped, no persistence.

Agent memory differs because it has **dynamics**: formation (how memories are created from agent experience), evolution (how memories change over time — consolidation, contradiction, decay), and retrieval (how memories are accessed for current tasks). These dynamics make it more analogous to biological memory than to database storage.

The survey proposes a three-dimensional taxonomy: forms (token-level, parametric, latent), functions (factual, experiential, working), dynamics (formation, evolution, retrieval). Five research frontiers: memory automation, RL integration, multimodal memory, multi-agent memory coordination, trustworthiness.

Since [[RAG-based memory provides only marginal improvement versus intent resolution demonstrating retrieval is not equivalent to resolving intent]], the distinction between "retrieval from storage" and "memory with lifecycle" is empirically validated. RAG provides context; memory provides understanding that evolves.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[RAG-based memory provides only marginal improvement versus intent resolution demonstrating retrieval is not equivalent to resolving intent]] -- empirical evidence that retrieval is not memory
- [[agents using add-all memory strategies exhibit sustained performance decline making active forgetting essential not optional]] -- why the dynamics dimension matters
- [[graph-based memory taxonomy classifies agent memory across temporal scope functional role structure and cognitive type]] -- a complementary classification scheme

Topics:
- [[agent-memory]]
