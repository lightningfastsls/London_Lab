---
description: "Vass hierarchy — (1) user learning, (2) quality over speed, (3) integrity always — makes the agent a teaching partner rather than a code-generation pipeline"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Learning-first priority reframes the agent-user relationship from task execution to collaborative teaching

The Vass contract establishes a priority hierarchy: (1) user learning first, (2) quality over speed, (3) integrity always. This ordering is deliberate — it reframes the agent-user relationship from a task-execution pipeline (user requests, agent delivers) to a collaborative teaching interaction (user learns, agent explains reasoning).

The practical effect is that the contract requires agents to explain the "why" behind decisions, teach concepts when touching specialized domains, surface trade-offs when multiple approaches exist, and connect changes to the bigger picture. This adds overhead but produces a fundamentally different outcome: instead of receiving code they don't understand, users receive understanding they can apply beyond the immediate task.

The learning-first priority also serves as a natural check on the premature commitment problem. When agents must explain their reasoning before executing, weak reasoning becomes visible. Since [[externalized reasoning at approval gates forces agents to improve their plans before executing them]], the teaching requirement is also a quality gate — agents that must teach their approach are less likely to pursue approaches they cannot explain.

The counterargument is efficiency: users who know what they want may find the teaching overhead frustrating. This is addressed by the tiered contract system — minimal contracts for experienced users on routine tasks omit the learning-first requirement while retaining integrity constraints.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- teaching as quality gate
- [[tiered behavioral contracts must scale with project complexity because instruction-following degrades with instruction count]] -- how the teaching requirement scales
- [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]] -- teaching pushes work toward the "thought" end

Topics:
- [[agent-governance]]
