---
description: "Aptitude drops only 16% but unreliability more than doubles — all models converge to ~50 percentile-point variability regardless of base capability"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# Multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss

Laban et al. decompose multi-turn performance into three metrics from N=10 simulations per instruction: P-bar (average performance), A90 (aptitude — 90th percentile, representing best-case capability), and U10-90 (unreliability — the gap between 90th and 10th percentiles). In single-turn settings, higher aptitude correlates with lower unreliability — top models are both more capable and more consistent.

In multi-turn settings, this relationship breaks down. Aptitude drops modestly (16% average decline), meaning models CAN still solve the tasks at their best. But unreliability more than doubles (112% increase), meaning they rarely DO solve them consistently. All models converge to approximately 50 percentile-point unreliability spread regardless of their base capability. A frontier model with 92% single-turn accuracy becomes as unreliable as a small open-source model in multi-turn — they differ in ceiling but not in floor.

This is the paper's most striking finding because it reframes the problem: multi-turn conversation does not make models dumber, it makes them unpredictable. The practical implication is that reliability engineering (reducing variance) matters more than capability improvements for multi-turn applications.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]] -- the headline finding this decomposes
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- a root cause of the unreliability
- [[reasoning models produce longer responses and additional test-time compute does not solve multi-turn unreliability]] -- even reasoning can't fix variance

Topics:
- [[agent-cognition]]
