---
description: "Appendix F.3 documents over-adjustment to the most recent turn, causing earlier requirements to be deprioritized or forgotten"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# LLMs over-adjust based on the last turn of conversation disproportionately weighting recent information

Laban et al. document in Appendix F.3 ("Over-adjust based on Last Turn of Conversation") that LLMs disproportionately weight the most recent turn when generating responses, at the expense of information revealed in earlier turns. This is a recency bias operating at the conversation level: requirements stated in turn 2 may be deprioritized or effectively forgotten when the model focuses on satisfying turn 5's specifications.

This is conceptually related to but distinct from the "lost in the middle" phenomenon documented in long-context retrieval (Liu et al. 2023). The retrieval finding concerns positional bias within a single input document. The conversation-level finding concerns how models weight information distributed across interaction turns — a temporal rather than spatial distribution. However, both may share an underlying attention mechanism that favors recent and early positions.

The practical implication: in multi-turn workflows, the most recent instruction has outsized influence. This explains why correction-based workflows can oscillate — fixing one issue in the latest turn causes the model to deprioritize earlier constraints, creating new problems from previously-satisfied requirements.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[answer bloat compounds multi-turn errors as responses grow verbose without pruning incorrect assumptions]] -- the bloat that interacts with recency bias
- [[LLMs prematurely commit to incorrect solutions in early turns and fail to revise them producing cascading errors]] -- the anchoring that recency bias compounds
- [[snowball turn-by-turn accumulation mitigates 15-20 percent of the full-to-sharded performance deterioration]] -- a mitigation that counteracts recency by re-presenting all prior information

Topics:
- [[agent-cognition]]
