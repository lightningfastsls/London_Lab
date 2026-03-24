---
description: "ops/goals.md and session state files bridge the memory gap between sessions — structured handoff substitutes for the persistent memory agents lack"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-external-cognition]]"
  - "[[agent-memory]]"
---

# session handoff creates continuity without persistent memory

Agents lose all context between sessions. The vault's session handoff pattern (ops/goals.md, session-relevance.md, MEMORY.md) creates continuity by externalizing the minimal state needed to resume work. This is not persistent memory — it is structured handoff. The distinction matters: persistent memory implies the agent remembers; structured handoff acknowledges it does not and compensates.

Since [[session boundaries simultaneously limit agents and enable between-session processing making the limitation the precondition]], the handoff design acknowledges that full continuity is impossible and instead optimizes for efficient re-orientation. Since [[fresh context per task preserves quality better than chaining phases]], the clean break between sessions is actually beneficial — it prevents multi-turn degradation while the handoff prevents cold-start waste. The handoff is deliberately minimal: only what the next session needs to orient, not a full transcript of the previous session.

---

Relevant Notes:
- [[session boundaries simultaneously limit agents and enable between-session processing making the limitation the precondition]] — session gaps as feature not bug
- [[fresh context per task preserves quality better than chaining phases]] — why clean breaks help
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] — prompt design for fresh starts
