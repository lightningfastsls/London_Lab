---
description: "spawning fresh subagents per task phase prevents multi-turn degradation — the /ralph pattern of isolated context per phase outperforms long conversational chains"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[multi-turn-degradation]]"
  - "[[agent-external-cognition]]"
---

# fresh context per task preserves quality better than chaining phases

Multi-turn degradation (~39% average per Laban et al.) means that chaining multiple task phases in a single conversation accumulates errors. The Fresh Context Pattern — spawning a new agent per phase with only the relevant inputs — avoids this degradation. Each fresh agent operates at peak single-turn quality rather than inheriting the accumulated noise of prior turns.

Since [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]], each fresh agent gets optimal single-turn conditions. Since [[session handoff creates continuity without persistent memory]], the pattern works across session boundaries too: each session starts fresh with handoff state rather than trying to maintain an impossibly long conversation. The /ralph skill implements this by spawning isolated subagents per queue task. The cost is coordination overhead; the benefit is that each phase gets the model's best work rather than its degraded tail.

---

Relevant Notes:
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] — prompt design for fresh starts
- [[session handoff creates continuity without persistent memory]] — cross-session application
- [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]] — quantified degradation motivating this pattern
