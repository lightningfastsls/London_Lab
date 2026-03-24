---
description: "Zipf's law applied to conversation: users naturally provide minimal information, so underspecification is not fixable user behavior but a structural property"
type: finding
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[multi-turn-degradation]]"
---

# The principle of least effort drives conversational underspecification making ambiguity a fundamental feature not a bug

Both Laban et al. and Liu et al. cite Zipf's principle of least effort as applied to conversation. Users naturally provide the minimum information needed to communicate, optimizing for their own effort rather than for the listener's (or model's) comprehension. This makes conversational underspecification not a bug in user behavior but a fundamental, immutable feature of human communication.

This framing reorients the multi-turn degradation problem. If ambiguity is structural, then solutions that assume users will "just be more specific" are doomed. The burden of disambiguation must fall on the model side — through explicit clarification-seeking, intent inference, or architectural patterns that separate understanding from execution.

The implication for AI system design is that multi-turn interfaces must be designed to tolerate and resolve ambiguity, not to require specification-complete inputs. Since [[user utterances are a lossy compression of high-dimensional intent into low-dimensional surface forms]], every conversational turn is an incomplete specification. Systems that treat incomplete specifications as complete will systematically fail — which is exactly what current RLHF-trained models do when they since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]].

---

Source: multi-turn-conversation-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[user utterances are a lossy compression of high-dimensional intent into low-dimensional surface forms]] -- the information-theoretic formalization
- [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] -- why models treat ambiguity as specification
- [[whether RLHF can be modified to reward clarification-seeking over premature helpfulness in multi-turn settings]] -- the open question about fixing this

Topics:
- [[agent-cognition]]
