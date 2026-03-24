---
description: "Liu et al. 2026 found that most multi-turn performance loss is architecture-independent — contradicts the hypothesis that more capable models will outgrow the problem"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# Approximately 60 percent of relative multi-turn degradation is constant across model sizes suggesting scaling alone cannot solve it

Liu et al. (2026) analyzed the relationship between model scale and multi-turn degradation and found that approximately 60% of the relative performance degradation is constant across model sizes. This contradicts the optimistic hypothesis that scaling — building larger, more capable models — will eventually solve the multi-turn problem.

The implication is that multi-turn degradation has a structural component that is architecture-independent. Whether the model has 8B or 400B+ parameters, it loses roughly the same fraction of performance when information is distributed across turns. The remaining ~40% does vary with capability, which explains why frontier models still perform somewhat better in absolute terms. But the floor of the problem is baked into something more fundamental — likely the autoregressive generation process combined with RLHF training incentives that reward premature helpfulness.

This finding motivates architectural mitigations (like the Mediator-Assistant pattern) rather than waiting for scale to solve the problem. Since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]], the fix may require changing training objectives rather than increasing model size.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs lose approximately 25 percentage points average performance when tasks are distributed across conversational turns]] -- the aggregate finding
- [[multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss]] -- the reliability decomposition
- [[Mediator-Assistant framework separates intent inference from task execution recovering approximately 20 percentage points]] -- an architectural fix rather than scaling

Topics:
- [[agent-cognition]]
