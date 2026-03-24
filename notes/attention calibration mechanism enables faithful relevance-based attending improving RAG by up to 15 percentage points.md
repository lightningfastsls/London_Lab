---
description: "Found in the Middle (Hsieh et al. ACL Findings 2024) — a calibration method that counteracts the U-shaped attention bias allowing models to attend according to actual relevance"
type: method
confidence: likely
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Attention calibration mechanism enables faithful relevance-based attending improving RAG by up to 15 percentage points

"Found in the Middle" (Hsieh et al., ACL Findings 2024) proposes a calibration mechanism that allows models to attend to contexts faithfully according to their relevance rather than their position. The method improves RAG (Retrieval-Augmented Generation) performance by up to 15 percentage points.

The technique directly targets the core problem described by [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]]. Without calibration, attention is biased toward initial tokens (since [[initial tokens attract disproportionate attention regardless of semantic relevance due to autoregressive causal masking during training]]) and recent tokens, regardless of where the relevant information actually sits. Calibration reweights attention to make it position-agnostic.

The 15-percentage-point improvement is specific to RAG scenarios where retrieved documents are concatenated into context at arbitrary positions. For non-RAG use cases (direct conversation, tool use), the improvement would be different because the attention patterns differ. However, the principle — that models can be calibrated to overcome positional bias — suggests that the U-shaped curve is not an immutable architectural constraint but a learned bias that can be corrected.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent]] -- the bias this method corrects
- [[initial tokens attract disproportionate attention regardless of semantic relevance due to autoregressive causal masking during training]] -- one component of the bias

Topics:
- [[agent-cognition]]
