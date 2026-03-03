---
description: "Agent governance contracts from CLAUDE.md define behavioral boundaries but do not address memory boundaries — cross-project memory sharing could violate project-specific constraints that contracts are designed to enforce"
type: open-question
confidence: speculative
created: 2026-03-02
meta_state: current
---

# how memory scoping interacts with behavioral contracts when agents share cross-project knowledge

Agent governance through behavioral contracts (CLAUDE.md, tiered instructions) defines what an agent should and should not do within a project. But since [[cross-agent memory bridges enable tool-agnostic knowledge persistence across multiple IDE platforms through shared storage]], agents increasingly share knowledge across project boundaries.

The interaction creates potential conflicts:
- A behavioral contract in Project A may prohibit certain patterns that Project B's memory freely stores
- Cross-project memory contamination since [[single-scope MCP memory causes cross-project contamination when agent contexts are not separated]] could import assumptions that violate a project's architectural constraints
- Since [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]], adding memory-scope constraints further taxes the instruction budget

The question is whether memory scoping should be treated as a governance concern (enforced by behavioral contracts) or an infrastructure concern (enforced by memory system architecture). Since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]], the infrastructure approach — building scope isolation into the memory system itself — may be more reliable than adding "do not use memories from other projects" to behavioral contracts.

No existing system formally addresses this interaction. OpenMemory's access control tables come closest, but they control which apps can access which memories, not which behavioral constraints apply to shared knowledge.

---

Source: mcp-memory-servers-cross-session-knowledge-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[cross-agent memory bridges enable tool-agnostic knowledge persistence across multiple IDE platforms through shared storage]] -- the capability creating this interaction
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- why adding memory constraints to contracts is problematic
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- the infrastructure approach

Topics:
- [[context-management]]
- [[agent-governance]]
