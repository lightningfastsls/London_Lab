---
description: "AIMultiple benchmark found MCP memory servers could not separate contexts of two projects when using single-file storage — architectural decisions from one project leaked into another"
type: finding
confidence: proven
created: 2026-03-02
meta_state: current
---

# single-scope MCP memory causes cross-project contamination when agent contexts are not separated

The AIMultiple MCP Memory Benchmark tested 4 MCP servers with a LangChain ReAct agent + GPT-4, measuring operation accuracy (percentage of turns with correct memory operations) and testing read-on-resume and read-before-write behaviors.

The critical finding: cross-project context separation failed. When an agent worked across two different projects using a single-scope memory system, architectural decisions and conventions from one project leaked into the other. Single-project implementations performed adequately.

This failure mode is analogous to the context interference problems in since [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]] — structured knowledge from one domain actively interferes with reasoning about another domain because the model cannot cleanly partition its context.

The reference implementation since [[Anthropic reference MCP memory server uses entity-relation-observation knowledge graph as JSONL with no built-in decay or scoping]] has no scoping mechanism beyond separate JSONL files, making it particularly vulnerable. Production implementations address this through namespace hierarchies (Agent-Recall), git-remote-based auto-scoping (Memorix), or explicit access control tables (OpenMemory).

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[memory scoping by project agent and task prevents cross-project contamination in multi-context agent systems]] -- the solution pattern
- [[structured coherent text creates more context interference than shuffled unstructured text across all tested models]] -- the analogous context interference mechanism

Topics:
- [[agent-memory]]
