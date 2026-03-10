---
description: "Ye and Tan 2026 formalize three allocation strategies — proportional, equal, negotiated — with zero observed conservation violations in multi-agent delegation and 90 percent token reduction"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[context-management]]"
---

# Conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget

The Agent Contracts framework (Ye and Tan, 2026) defines contracts as seven-tuples C = (I, O, S, R, T, Phi, Psi) with a critical constraint: total resource consumption across delegated sub-agents must not exceed the parent agent's budget. This is the conservation law — resources are bounded at the delegation boundary, preventing cascading resource exhaustion in multi-agent hierarchies.

Three allocation strategies are formalized: proportional (allocate based on estimated task complexity), equal (uniform distribution), and negotiated (sub-agents bid for resources). The evaluation shows zero conservation violations in multi-agent delegation, 90% token reduction with 525x lower variance, and measurable quality-resource trade-offs through contract modes (BALANCED: 86% success vs URGENT: 70% success at lower resource cost).

The conservation law provides formal grounding for practical patterns like since [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] — the subagent pattern works partly because each subagent operates within a bounded resource envelope. Without conservation constraints, a subagent could consume unlimited tokens chasing a dead end, starving other subagents.

The lifecycle states (DRAFTED → ACTIVE → {FULFILLED, VIOLATED, EXPIRED, TERMINATED}) are also valuable — they provide a formal vocabulary for tracking agent task completion that maps to the practical state machines in tools like the Vass contract and Liza system.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the pattern conservation laws formalize
- [[Liza blackboard architecture coordinates multi-agent work through shared state files without inter-agent conversation]] -- practical multi-agent system using bounded delegation
- [[ABC framework defines probabilistic compliance where hard constraints hold with high probability and soft violations recover within bounded steps]] -- the complementary compliance framework
- [[agent team token costs scale linearly with teammates making 3-5 the recommended size before coordination overhead dominates]] -- concrete instance: linear token scaling constrains practical team size to 3-5 before conservation limits bite
- [[three parallelization strategies in Claude Code serve different isolation needs subagents for focused results teams for collaborative discussion worktrees for filesystem safety]] -- the three strategies all operate under conservation constraints but with different cost profiles
- [[cross-agent knowledge transfer requires flattening graph-traversable constraints into self-contained plain text]] — constraint flattening at delegation boundaries is a form of information compression under conservation constraints: the handoff author's activation budget must cover what the receiver cannot activate independently

Topics:
- [[agent-governance]]
