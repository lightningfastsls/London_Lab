---
description: "Gradual sharding experiment on GPT-4o and GPT-4o-mini shows degradation starts at 2 shards — caused by temporal distribution not information volume"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# Even two conversational turns trigger multi-turn degradation regardless of task complexity

Laban et al. expanded 31 instructions into 7 variants each, varying shard count from 2 to 8 while fixing task complexity. Testing on GPT-4o and GPT-4o-mini, both models degraded starting with just two shards. The degradation curve is gradual but begins immediately — there is no safe threshold of turns below which multi-turn interaction preserves single-turn performance.

This finding is critical because it rules out the intuitive hypothesis that degradation is caused by excessive information volume or conversation length. The problem is temporal distribution itself — splitting information across time, even minimally, triggers the failure mode. This suggests the degradation is structural to how autoregressive models process conversation history, not a resource limitation.

The practical implication for agent design: even a single follow-up clarification degrades performance. The optimal interaction pattern is a single, fully-specified prompt whenever possible, which is what the Concat strategy achieves.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]] -- the aggregate finding this refines
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the mitigation that exploits this finding
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- why even 2 turns fails

Topics:
- [[agent-cognition]]
