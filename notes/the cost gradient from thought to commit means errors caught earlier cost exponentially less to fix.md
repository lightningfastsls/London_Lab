---
description: "Vass's artifact sequence — thought → words → specs → code → tests → docs → commits — each rightward step is more expensive to change, motivating 'leftward' work"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# The cost gradient from thought to commit means errors caught earlier cost exponentially less to fix

Vass articulates a cost gradient for development artifacts: thought → words → specs → code → tests → docs → commits. Each step to the right is more expensive to change. An error caught at the thought level costs nearly nothing — just revise the reasoning. An error caught at the commit level requires rollbacks, potentially broken CI, team confusion, and rework across multiple files.

The contract encourages "leftward" work — spending more time in thought and words before reaching code. This is why the Approval Request format demands intent, context, scope, plan with "why," assumptions, risks, and validation approach: all of these are "thought" and "words" artifacts that catch errors before they become code. Since [[externalized reasoning at approval gates forces agents to improve their plans before executing them]], the contract structurally pushes work leftward on the gradient.

This maps directly to the broader pattern of preferring reversible actions over irreversible ones. The cost gradient is really an irreversibility gradient — thoughts are infinitely reversible, commits are expensive to reverse. The practical implication for agent workflows is that plan-file-first, approval-gated execution, and mandatory validation all serve to shift error detection leftward where corrections are cheap.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- the mechanism that pushes work leftward
- [[one-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses]] -- another leftward-pushing pattern
- [[struggle protocol over silent failure requires agents to surface uncertainty rather than fabricate confidence]] -- catching errors at the "words" level

Topics:
- [[agent-governance]]
