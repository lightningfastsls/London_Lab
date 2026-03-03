---
description: "Liu et al. 2026 information-theoretic framing: multi-turn degradation occurs because the mapping from intent to utterance is many-to-one, creating systematic ambiguity"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
---

# User utterances are a lossy compression of high-dimensional intent into low-dimensional surface forms

Liu et al. (2026) provide an information-theoretic reframing of multi-turn degradation. Their core argument: user utterances are a lossy projection of high-dimensional intent (what the user actually wants) into low-dimensional surface forms (what they actually say). The mapping is many-to-one — the same utterance can represent disparate underlying intentions depending on the user.

Formally, the paper decomposes task success into Execution (can the model do the task given clear intent?) times Inference (can the model correctly infer intent from the utterance?). Multi-turn degradation is primarily an Inference failure, not an Execution failure — models can execute tasks well when intent is clear, but they systematically misinterpret intent in multi-turn settings.

Users exhibit systematic individual variation in communication style, and general models default to population-level priors rather than individual pragmatics. Since [[the principle of least effort drives conversational underspecification making ambiguity a fundamental feature not a bug]], the lossy compression is not a correctable user behavior — it is a fundamental property of human communication. The fix must come from the model side, through explicit intent resolution rather than assuming surface forms are complete specifications.

This connects to Zipf's law: users naturally provide minimal information, optimizing for their own communication effort rather than for model comprehension.

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[Mediator-Assistant framework separates intent inference from task execution recovering approximately 20 percentage points]] -- the architectural solution to the compression problem
- [[the principle of least effort drives conversational underspecification making ambiguity a fundamental feature not a bug]] -- Zipf's law framing
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- models guess intent instead of resolving it

Topics:
- [[agent-cognition]]
