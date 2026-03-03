---
description: "Context isolation by approved stack — searches restricted to project-specific approved technologies rather than the full 9000-plus library documentation index, applying the scoping pattern from memory to documentation"
type: method
confidence: likely
created: 2026-03-02
meta_state: current
---

# Docfork Cabinets project-specific context isolation locks agents to verified technology stacks preventing irrelevant search results

Docfork (9,000+ libraries, MIT license) introduced Cabinets — project-specific context isolation that restricts documentation searches to an approved technology stack. Rather than searching across all indexed libraries and returning potentially irrelevant results, agents are locked to a verified set of technologies for each project.

The pattern is architecturally identical to memory scoping. Since [[memory scoping by project agent and task prevents cross-project contamination in multi-context agent systems]], Cabinets apply the same principle to documentation context: isolation by project prevents knowledge from one technology stack contaminating reasoning about another.

This addresses a specific failure mode: when an agent queries documentation without stack constraints, it may retrieve documentation for the wrong framework version, a similarly-named API from a different library, or patterns that apply to a technology the project does not use. These results consume context tokens and can actively mislead the agent.

Since [[context quality over quantity is the defining trend with tools optimizing for less context that matters more not more context that dilutes]], Cabinets represent the documentation-specific implementation of precision-over-volume: fewer, more relevant results from a constrained search space.

---

Source: claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[memory scoping by project agent and task prevents cross-project contamination in multi-context agent systems]] -- the same isolation pattern applied to memory
- [[context quality over quantity is the defining trend with tools optimizing for less context that matters more not more context that dilutes]] -- the broader trend

Topics:
- [[context-management]]
