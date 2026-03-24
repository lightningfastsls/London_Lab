---
description: "Anthropic's two-agent harness pattern — Initializer plus Coding Agent — uses single-feature focus, JSON feature lists, mandatory e2e tests, and explicit 'never edit tests' instruction"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# One-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses

Anthropic's documented harness for long-running agents uses a two-agent architecture (Initializer for environment setup, Coding Agent for incremental progress) with a critical constraint: one feature per session. Additional constraints include JSON-based feature lists (preventing inappropriate modification that markdown allows), mandatory end-to-end testing before completion, clean exit state with git commits and progress updates, and the explicit instruction "It is unacceptable to remove or edit tests."

The one-feature constraint works because it converts an open-ended session into a bounded task with clear completion criteria. Since [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]], limiting scope means errors in one feature don't contaminate another. It also directly addresses the multi-turn degradation problem: since [[even two conversational turns trigger multi-turn degradation regardless of task complexity]], shorter focused sessions outperform longer multi-feature sessions even when total work is the same.

The one-feature constraint directly addresses one of the four integrity failures: since [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]], scope creep is a predictable RLHF-driven tendency, and the structural constraint is more reliable than an instruction to "stay focused."

The JSON feature list is a subtle but important detail. Markdown feature lists are human-readable but agent-modifiable — agents can "helpfully" reword, reorder, or extend them. JSON lists resist casual modification because the format is strict enough that changing content requires deliberate action. This is a micro-example of how since [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]], data format constraints enforce behavior that instructions alone cannot.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]] -- why scope limitation matters
- [[even two conversational turns trigger multi-turn degradation regardless of task complexity]] -- why shorter sessions outperform
- [[boundary-level enforcement via type systems and linters is more robust than prompt-level behavioral instructions]] -- why JSON over markdown
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- scope creep is a predictable RLHF-driven failure mode

Topics:
- [[agent-governance]]
