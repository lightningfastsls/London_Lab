---
description: "Stop conditions trigger at 3+ assumptions on critical path, same approach tried twice, evidence contradicting hypothesis, or uncertainty about whether code or test is wrong"
type: pattern
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Struggle protocol over silent failure requires agents to surface uncertainty rather than fabricate confidence

When agents hit difficulty, Vass's contract specifies a struggle protocol rather than allowing silent failure or fabrication. Stop conditions trigger when: assumption count reaches 3 or more on the critical path, the same approach has been tried twice without new rationale, evidence contradicts the hypothesis, or it is uncertain whether code or the test expectation is wrong.

The protocol works because it addresses the specific failure mode created by RLHF training. Since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]], agents are biased toward confident-sounding wrong answers over honest uncertainty. The struggle protocol creates an explicit permission structure for uncertainty — transforming "I'm stuck" from a penalized response into the correct response. This is structurally similar to how the SELAUR framework uses uncertainty as a natural measure of model confidence: high uncertainty should trigger alternative strategies, not forced continuation.

The practical value is in what it prevents: without the protocol, agents that hit difficulty tend to either fabricate results, make silent assumptions, or retry the same failing approach repeatedly. Each of these wastes resources and compounds errors. The protocol converts what would be a silent failure spiral into an explicit signal that surfaces to the user.

The weakest point is enforcement — the stop conditions depend on the agent self-monitoring its own assumption count and recognizing when evidence contradicts its hypothesis, which requires the same metacognitive capability that RLHF training undermines.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- the training bias the protocol counteracts
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the failure pattern the protocol prevents
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- the complementary pre-execution gate

Topics:
- [[agent-governance]]
