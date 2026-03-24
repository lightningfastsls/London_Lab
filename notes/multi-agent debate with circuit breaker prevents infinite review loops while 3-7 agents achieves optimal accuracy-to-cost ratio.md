---
description: "Nielsen's adversarial-review orchestrates Claude vs GPT Codex in 4-phase debate with ~21 API calls worst-case — circuit breaker detects stalled convergence via three trigger conditions"
type: method
confidence: experimental
created: 2026-03-01
meta_state: current
topics:
  - "[[code-review-governance]]"
---

# Multi-agent debate with circuit breaker prevents infinite review loops while 3-7 agents achieves optimal accuracy-to-cost ratio

Alec Nielsen's open-source adversarial-review system implements a concrete engineering solution to the multi-agent review problem: how many agents, how many rounds, and when to stop. The system orchestrates a four-phase debate loop between Claude and GPT Codex: independent reviews, cross-review (each critiques the other's findings), meta-review (defending or revising positions), and synthesis (determining consensus).

The key engineering contribution is the circuit breaker — a pattern borrowed from distributed systems that prevents infinite loops when agents cannot converge. Three conditions trigger the breaker: 3 consecutive zero-fix iterations (progress has stalled), 5+ persistent disagreement iterations (agents are stuck in a loop), or 3+ identical unfixable issues (the same problem keeps recurring without resolution). Without this, multi-agent debate can degenerate into expensive cycles of mutual criticism.

The research basis for the architecture is that "multi-agent debate reduces hallucinations and false positives" with "3-7 agents offering the best accuracy-to-cost ratio." This finding maps to the broader principle in agent architecture: too few agents lacks the diversity to catch blind spots (since [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]]), while too many agents adds cost without proportional quality improvement. The ~21 API call ceiling at worst case provides a predictable cost bound.

The debate format — where agents explicitly critique and then defend positions — creates a form of structured deliberation that mirrors the dialectical pattern in Block g3, though at the inter-model rather than intra-model level.

---

Source: ai-code-review-optimization-multi-agent-architectures-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[same-model generation and review creates confirmation bias producing 8x duplicated code and 72 percent Java security failures]] -- why multi-model diversity matters
- [[3-5 actor-critic review rounds eliminate over 90 percent of issues at under 2 dollars per feature]] -- the cost-effectiveness reference point
- [[memory wipe per review turn prevents attention degradation treating each attempt as fresh start guided by coach feedback]] -- dialectical autocoding at intra-model level

Topics:
- [[agent-governance]]
