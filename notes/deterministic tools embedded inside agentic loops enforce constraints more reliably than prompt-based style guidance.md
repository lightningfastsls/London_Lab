---
description: "Van Eyck 2026 — linters, formatters, type systems, architectural tests, and vulnerability scanners should run INSIDE the agent loop, not as post-submission gates — 'speed vs velocity' distinction"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance

Van Eyck (2026) articulates six practical guardrails for maintaining quality as agent autonomy increases, with a core insight: embed guardrails inside agentic loops, not as post-submission gates. The six guardrails are: (1) real continuous integration (trunk-based, merge in hours), (2) static type systems with domain types, (3) deterministic tools over prompting (linters, formatters as hard constraints), (4) architectural unit tests (ArchUnit-style structural constraints), (5) high-quality automated tests around scenarios not implementation, (6) vulnerability and code quality scanners within agent loops.

The "speed vs velocity" distinction captures it: agents generate code quickly (speed) but need track boundaries to make progress (velocity). Deterministic tools provide those boundaries with 100% reliability — a linter catches every formatting violation, a type system prevents every type error, an architectural test catches every structural violation. Prompt-based instructions to "follow the style guide" provide probabilistic compliance that degrades with session length.

The "inside the loop" placement is critical. Post-submission gates catch errors but require rework. In-loop enforcement prevents errors from being generated in the first place. When a linter runs after each agent action, the agent sees violations immediately and corrects before moving on, rather than accumulating technical debt that a final review must catch. Since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]], the principle extends: any constraint expressible as a deterministic check should be implemented as a tool, not an instruction.

CodeRabbit provides production-scale validation of this pattern: it runs 30+ static analyzers BEFORE prompting the LLM, uses AST and symbol lookups for context identification, and applies filters based on past review learnings. Their infrastructure runs "tools in jail" using Jailkit within Cloud Run instances (8 vCPUs, 32 GiB per instance, 200+ instances at peak). This demonstrates the pattern working not just in theory but at massive scale — deterministic analysis handles the high-confidence checks, freeing the LLM to focus on judgment calls that require reasoning rather than rule-matching.

The Claude Code ecosystem now provides a concrete three-level governance hierarchy that implements this principle at scale: since [[three-level tool governance layers gateway enforcement hook enforcement and contract enforcement in decreasing reliability but increasing flexibility]], the most reliable enforcement (MCP gateways controlling tool access) is deterministic infrastructure, the middle layer (16 lifecycle hooks including PreToolUse block/modify/escalate) is code-enforced, and the least reliable (CLAUDE.md contracts) is prompt-dependent. This hierarchy instantiates Van Eyck's principle as an architecture: deterministic enforcement at every possible layer, with prompt-level guidance reserved for judgment calls.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/), ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/), claude-code-mcp-ecosystem-march-2026-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- the broader principle
- [[test integrity truth table prevents the most dangerous agent failure mode of modifying tests to match bugs]] -- deterministic test constraints
- [[one-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses]] -- scope as a structural constraint
- [[behavioral contract effectiveness degrades beyond approximately 150-200 instructions requiring progressive disclosure]] -- offloading to deterministic tools directly reduces contract instruction burden
- [[active enforcement is necessary because passive monitoring cannot prevent all behavioral drift in self-evolving agents]] -- in-loop tools are a form of active enforcement

Topics:
- [[agent-governance]]
