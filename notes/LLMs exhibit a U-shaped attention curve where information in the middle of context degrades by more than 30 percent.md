---
description: "Liu et al. 2023 (TACL 2024) established the primacy-recency bias in transformer attention — softmax normalization spreads probability mass across n-squared token pairs, starving relevant middle tokens"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# LLMs exhibit a U-shaped attention curve where information in the middle of context degrades by more than 30 percent

Liu et al. (2023, TACL 2024) established the seminal finding in long-context LLM research: models reliably access information at the beginning and end of their context window, but information positioned in the middle suffers significant degradation — more than 30% on multi-document QA and key-value retrieval tasks when relevant content moves from boundaries to middle positions.

The mechanism is architectural. Transformer attention creates n-squared pairwise token relationships. As context lengthens, the softmax normalization constraint forces attention weights to sum to 1, which means irrelevant tokens literally steal attention probability from relevant ones. A single relevant sentence becomes statistically insignificant against millions of distractor tokens. This primacy-recency bias mirrors human cognitive patterns and has been replicated across virtually every subsequent benchmark, including [[RULER benchmark showed only half of long-context models maintained performance at 32K despite claiming 32K-plus support]] and [[NoLiMa found 11 of 12 models dropped below 50 percent baseline at 32K when lexical shortcuts were removed]].

The U-shaped curve operates within a single forward pass (positional bias), which distinguishes it from [[LLMs over-adjust based on the last turn of conversation disproportionately weighting recent information]] (temporal bias across turns). However, both may share an underlying attention mechanism that favors recent and initial positions — the question is whether the recency at conversation level reduces to recency at token level. Since [[initial tokens attract disproportionate attention regardless of semantic relevance due to autoregressive causal masking during training]], the primacy component at least has a clear architectural explanation.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[initial tokens attract disproportionate attention regardless of semantic relevance due to autoregressive causal masking during training]] -- explains the primacy component of the U-shape
- [[LLMs over-adjust based on the last turn of conversation disproportionately weighting recent information]] -- the conversation-level analog of positional recency
- [[attention calibration mechanism enables faithful relevance-based attending improving RAG by up to 15 percentage points]] -- a direct mitigation

Topics:
- [[agent-cognition]]
