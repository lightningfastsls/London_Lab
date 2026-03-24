---
description: "Context Rot (Chroma 2025) — counterintuitively, coherent haystacks hurt retrieval more than random ones because structured content competes for attention more effectively"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Structured coherent text creates more context interference than shuffled unstructured text across all tested models

The Context Rot evaluation (Chroma, July 2025) tested 18 models across NIAH variants, repeated words, and LongMemEval tasks. One of its most counterintuitive findings: shuffled (unstructured) haystacks produced better model performance than coherent ones across all 18 models tested.

This is surprising because one might expect models to handle structured text better — after all, they were trained on structured language. But the mechanism makes sense in light of how attention works. Since [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]], and attention is a competition for probability mass, coherent text creates stronger distractors. Structured content has internal relationships that attract attention away from the target, while random shuffled tokens are easily dismissed by the attention mechanism. Non-uniform distractor impact amplifies with input length.

The evaluation also revealed model-family behavioral differences in long-context scenarios: Claude models showed the lowest hallucination rates with conservative abstention (preferring to say "I don't know"), GPT models showed the highest hallucination rates with confident-but-incorrect responses, and Gemini models exhibited random word generation starting around 500-750 words into their responses.

The practical implication for context engineering is clear: the content surrounding your target information matters as much as the target's position. Filling context with semantically related but irrelevant material (as happens naturally in codebase exploration or document analysis) is worse than filling it with random padding. This reinforces why [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] — preloading creates exactly the kind of structured interference that hurts retrieval.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]] -- the underlying attention mechanism that explains this
- [[NoLiMa found 11 of 12 models dropped below 50 percent baseline at 32K when lexical shortcuts were removed]] -- semantic distractors cause even worse degradation
- [[just-in-time context retrieval via lightweight identifiers outperforms preloading data into context]] -- the architectural response to this finding

Topics:
- [[agent-cognition]]
