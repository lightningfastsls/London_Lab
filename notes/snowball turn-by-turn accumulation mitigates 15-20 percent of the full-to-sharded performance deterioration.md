---
description: "Each turn re-presents all previous shards before the new one — practical for production where Concat is not possible, recovering partial performance"
type: method
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# Snowball turn-by-turn accumulation mitigates 15-20 percent of the full-to-sharded performance deterioration

Laban et al. tested a "Snowball" condition where each conversational turn includes all previously revealed shards before presenting the new one, creating an accumulating context. Results: GPT-4o-mini improved from 50.4% (Sharded) to 61.8% (Snowball), GPT-4o from 59.1% to 65.3% — a 6-7 absolute percentage point improvement, mitigating approximately 15-20% of the full-to-sharded deterioration.

The Snowball strategy has a key advantage over the Recap strategy (which recovered 16-17pp but requires knowing when the conversation ends): Snowball is realistic for production use. Each turn self-contains all prior context, so the model always has the full specification available at each step. It does not require predicting conversation length or having a special "final turn."

The limitation is that Snowball only partially mitigates degradation — it does not approach Concat's ~95% recovery. This is because Snowball preserves the temporal structure (information still arrives incrementally) even while re-presenting it. The model still makes premature assumptions on early turns, and re-presenting context does not fully undo the anchoring since [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]].

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[concatenating all requirements into a single prompt restores approximately 95 percent of single-turn performance]] -- the gold standard mitigation
- [[LLMs over-adjust based on the last turn of conversation disproportionately weighting recent information]] -- Snowball counteracts recency by always presenting the full history
- [[Mediator-Assistant framework separates intent inference from task execution recovering approximately 20 percentage points]] -- a different mitigation approach

Topics:
- [[agent-cognition]]
