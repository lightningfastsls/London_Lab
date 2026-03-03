---
description: "o3 and Deepseek-R1 produce longer reasoning chains but degrade similarly — more thinking does not compensate for the structural problem of premature commitment"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# Reasoning models produce longer responses and additional test-time compute does not solve multi-turn unreliability

Laban et al. tested reasoning-enhanced models including o3 and Deepseek-R1 alongside standard models. These models generate longer reasoning chains through extended test-time compute, yet their multi-turn degradation percentages (o3: 38.4%, R1: 39.2%) fall squarely within the same 30-46% range as non-reasoning models. More thinking does not compensate for the structural problem of premature commitment and answer bloat.

In fact, reasoning models may exacerbate the problem. Longer responses produce more verbose initial solutions that create stronger anchoring effects and contribute more context tokens to subsequent turns. Since [[answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions]], the additional reasoning content becomes additional surface area for error compounding.

This counter-intuitive finding challenges the assumption that chain-of-thought and extended reasoning will eventually solve multi-turn limitations. The problem is not insufficient reasoning depth — it is the structural interaction between training incentives (premature helpfulness), conversation architecture (temporal distribution), and autoregressive generation (each token conditioned on prior, potentially wrong, tokens).

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss]] -- reasoning models have high aptitude but still high unreliability
- [[approximately 60 percent of relative multi-turn degradation is constant across model sizes suggesting scaling alone cannot solve it]] -- scaling argument extends to reasoning
- [[answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions]] -- longer reasoning = more bloat potential

Topics:
- [[agent-cognition]]
