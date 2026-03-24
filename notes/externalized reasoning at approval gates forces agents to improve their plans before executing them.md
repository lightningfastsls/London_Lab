---
description: "The state machine's psychological mechanism — agents trained to appear competent resist writing plans like 'I'll try random things' so surfacing reasoning improves it without better models"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Externalized reasoning at approval gates forces agents to improve their plans before executing them

The Tangi Vass behavioral contract centers on a state machine (IDLE → ANALYSIS → APPROVAL_PENDING → EXECUTION → VALIDATION → DONE) with forbidden transitions — most critically, ANALYSIS → EXECUTION (skipping approval). The insight is not the state machine itself but its psychological effect: agents must externalize their reasoning at each gate. The Approval Request format forces agents to state intent, context, scope, plan (with "why" for each step), assumptions, risks, and validation approach.

The mechanism works because agents trained to appear competent resist writing plans like "I'll try random things until something works." Surfacing the reasoning improves it without requiring better models or different prompting. This is essentially using the agent's own training bias (appearing competent) against its other training bias (premature action). Since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]], the approval gate interrupts the premature commitment cycle by forcing a pause between analysis and execution.

The strongest counterargument is that this adds latency — every action requires a round-trip through approval. But since [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]], the upfront cost of externalized reasoning pays for itself in avoided rework.

The approval gate directly prevents two of the four integrity failures: since [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]], the forbidden ANALYSIS->EXECUTION transition structurally prevents scope creep (the agent must declare scope before acting) and false completion (the agent must state validation criteria before claiming success). The structural scope bound pairs with since [[one-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses]] — the per-session constraint bounds the outer scope, while the approval gate bounds the per-action scope within a session.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training bias that approval gates interrupt
- [[the cost gradient from thought to commit means errors caught earlier cost exponentially less to fix]] -- why the latency trade-off is worthwhile
- [[struggle protocol over silent failure requires agents to surface uncertainty rather than fabricate confidence]] -- complementary mechanism for when agents get stuck
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- the approval gate prevents false completion and scope creep directly
- [[one-feature-per-session constraint prevents scope creep and enables clean validation in long-running agent harnesses]] -- outer-scope constraint complementing per-action approval gates

Topics:
- [[agent-governance]]
