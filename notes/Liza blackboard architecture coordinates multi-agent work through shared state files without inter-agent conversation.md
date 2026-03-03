---
description: "Three agents (Planner, Coder, Reviewer) read/write a shared YAML state file — emerged from 6 months of single-agent contract iteration as the natural multi-agent extension"
type: method
confidence: experimental
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Liza blackboard architecture coordinates multi-agent work through shared state files without inter-agent conversation

In the follow-up to his behavioral contract articles, Vass describes Liza, a multi-agent system using three roles (Planner, Coder, Reviewer) coordinating through a blackboard architecture — a shared YAML file defining current state. No conversation between agents; they read state, do work, and write state. The Coder/Reviewer dynamic operates like adversarial PR review in a loop: the Coder submits, the Reviewer examines against specs and falsifiable "done_when" criteria, then approves or rejects with specific feedback.

A critical governance invariant underpins this architecture: "The Coder can't merge their own work — ever. The Reviewer can't implement code — ever." This strict role separation prevents the confirmation bias that since [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]]. Tasks have explicit falsifiable `done_when` conditions (e.g., "python -m hello --name Alice prints 'Hello, Alice!' to stdout and exits 0"), and TDD is used as a structural constraint — tests define the specification before code is written. The human steers between sprints based on what emerges, not micromanaging each step. Liza represents what comes after vibe coding: multiple agents coordinating autonomously with peer review replacing human approval for routine work.

The architecture emerged naturally from six months of contract iteration. Once behavioral norms are established for single agents, multi-agent coordination becomes tractable because each agent operates within known constraints. The blackboard pattern avoids the multi-turn degradation that would afflict conversational agent coordination — since [[context window and multi-turn degradation have distinct root causes but compound when both occur in long multi-turn sessions]], eliminating conversation between agents removes both degradation sources.

This is structurally similar to how since [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]], except the shared state file replaces the summary compression step with explicit structured state. The blackboard serves as the compression medium — agents read only what they need from the shared state rather than receiving summaries.

The conservation laws from the Agent Contracts framework (Ye and Tan, 2026) provide formal grounding for this pattern — since [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]], the blackboard architecture naturally enforces resource bounds through explicit state tracking. The Coder/Reviewer invariant also implements the same principle as metaswarm: since [[no instruction path from failure to commit is the critical safety invariant in automated code pipelines]], the "Coder can't merge, Reviewer can't implement" constraint prevents any single agent from bypassing the review gate.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[subagent architecture isolates context degradation by giving each task a focused window and compressing summaries upward]] -- the related isolation pattern
- [[context window and multi-turn degradation have distinct root causes but compound when both occur in long multi-turn sessions]] -- why avoiding inter-agent conversation matters
- [[conservation laws for agent delegation constrain total sub-agent resource consumption to not exceed parent budget]] -- formal grounding for resource bounds
- [[no instruction path from failure to commit is the critical safety invariant in automated code pipelines]] -- Liza's "Coder can't merge" implements the same structural safety invariant
- [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]] -- Liza's role separation prevents this confirmation bias

Topics:
- [[agent-governance]]
