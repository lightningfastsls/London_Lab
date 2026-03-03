---
description: "Each turn layers assumptions on prior (often wrong) answers rather than replacing errors — the self-reinforcing anti-pattern of multi-turn conversation"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# Answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions

Laban et al. identify answer bloat as the second root cause of multi-turn degradation. Responses grow increasingly verbose across turns as models rely on their prior (often incorrect) answer attempts, adding new assumptions rather than replacing incorrect content. New shards rarely invalidate prior guesses — instead, each response layers on more content without pruning errors from earlier turns.

This creates a self-reinforcing failure loop: longer responses from turn N become expensive input context for turn N+1, both in tokens and in anchoring weight. The model treats its own prior output as partially-reliable context, weighting it alongside the user's actual requirements. Since the prior output contains errors, subsequent turns inherit and compound those errors.

The anti-pattern is compounded by the fact that since [[reasoning models produce longer responses and additional test-time compute does not solve multi-turn unreliability]], more capable models may actually bloat *faster* — producing longer, more detailed (but potentially wrong) initial solutions that create stronger anchoring effects.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the cause that drives the bloat
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the cascading failure pattern
- [[LLMs over-adjust based on the last turn of conversation disproportionately weighting recent information]] -- the recency bias that interacts with bloat
- [[Slow Thinking training templates are prone to self-reinforcing reasoning collapse where think tag frequency correlates with reward creating positive feedback loops]] -- parallel mechanism: both describe systems that generate increasingly verbose output because evaluation rewards length over substance
- [[RLHF-trained models exhibit sycophancy verbosity bias and confident nonsense as systematic reward hacking manifestations]] -- verbosity bias in RLHF is the training-time cause of multi-turn answer bloat

Topics:
- [[agent-cognition]]
