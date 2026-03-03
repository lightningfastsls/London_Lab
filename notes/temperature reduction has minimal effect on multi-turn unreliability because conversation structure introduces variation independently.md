---
description: "At T=0.0 single-turn unreliability drops 50-80% but multi-turn unreliability remains ~30 — deterministic decoding cannot prevent cascading deviations from conversation structure"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# Temperature reduction has minimal effect on multi-turn unreliability because conversation structure introduces variation independently

Laban et al. tested the effect of reducing temperature on multi-turn reliability. At T=0.0, single-turn unreliability decreased by 50-80% — as expected, since deterministic decoding eliminates sampling randomness. However, multi-turn unreliability remained approximately 30 even at T=0.0 (compared to approximately 40 at T=1.0). This represents at best a 15-20% improvement in multi-turn reliability from making the model fully deterministic.

The explanation: in multi-turn conversation, the conversation structure itself introduces variation that deterministic decoding cannot eliminate. One early token difference — from the user's phrasing, the model's first response, or the interaction between them — compounds across turns. Even with greedy decoding, different conversation paths produce different outcomes because the conversation history is itself a source of randomness.

This finding challenges the intuition that making models more deterministic will improve multi-turn reliability. The variation is not in the model's decoding process — it is in the interaction dynamics between model and conversation context.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[multi-turn degradation is primarily a 112 percent increase in unreliability rather than capability loss]] -- the unreliability that temperature can't fix
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the cascading mechanism that creates structural variation
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the mitigation that actually addresses the structural problem

Topics:
- [[agent-cognition]]
