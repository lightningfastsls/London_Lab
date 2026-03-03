---
description: "ABC compositionality theorem gives necessary conditions (interface compatibility, assumption discharge, governance consistency, recovery independence) but assumes recovery mechanism independence — real systems may violate this"
type: open-question
confidence: speculative
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# How behavioral norms propagate across agent boundaries in multi-agent systems

When a contracted agent delegates work to sub-agents, what happens to the behavioral norms? The ABC framework's Compositionality Theorem provides necessary conditions for safe multi-agent composition: interface compatibility, assumption discharge, governance consistency, and recovery independence. The Agent Contracts framework provides conservation laws ensuring resource bounds propagate. But neither fully addresses the question of behavioral norm propagation.

The gap is clear in practice: since [[Liza blackboard architecture coordinates multi-agent work through shared state files without inter-agent conversation]], the blackboard pattern implicitly propagates norms through the shared state structure — the state file's schema constrains what agents can express and therefore what they can do. But this is implicit rather than explicit norm propagation.

Several sub-questions remain open. First, do sub-agents inherit parent contracts or operate under their own? If inherited, the contract may not apply to the sub-agent's specific task. If independent, norms may conflict. Second, the ABC compositionality theorem assumes recovery mechanism independence — but in practice, multiple agents recovering simultaneously may compete for the same resources, creating interference. Since [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]], resource conservation is addressed, but behavioral norm conservation is not.

Third, the Agents.md Standard proposes manifest signing, standardized telemetry, and human-approval gates as inter-agent governance mechanisms. But these are interoperability standards, not norm propagation mechanisms.

This question is particularly relevant for the vault's own pipeline, where /reduce, /reflect, and /reweave operate as separate agents — behavioral norms from CLAUDE.md propagate to each through the system prompt, but skill-specific instructions may override or conflict.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[Liza blackboard architecture coordinates multi-agent work through shared state files without inter-agent conversation]] -- implicit norm propagation through state structure
- [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]] -- resource conservation solved, behavioral conservation open
- [[ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps]] -- the compositionality theorem's assumptions

Topics:
- [[agent-governance]]
