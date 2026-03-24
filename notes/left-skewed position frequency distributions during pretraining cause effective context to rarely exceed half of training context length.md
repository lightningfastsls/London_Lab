---
description: "arXiv 2410.18745 — position frequency follows f(i) = L-i, severely undertraining long-range positions — in SlimPajama-627B positions above 1536 represent less than 5 percent of exposures"
type: finding
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-cognition]]"
  - "[[context-management]]"
---

# Left-skewed position frequency distributions during pretraining cause effective context to rarely exceed half of training context length

Research (arXiv:2410.18745, 2024) identified a fundamental mathematical cause for why models consistently fail at their claimed context lengths: during pretraining, the frequency of position indices follows f(i) = L - i, where L is the training context length and i is the position index. This creates a dramatically left-skewed distribution that severely undertrains long-range dependencies.

The numbers make the problem concrete. In SlimPajama-627B with 2K training context: positions ≤1024 account for more than 80% of training exposures, while positions ≥1536 represent less than 5%. This means the model has seen 16x more examples of tokens in the first half than in the last quarter of its context window. Most failure cases occur within the first L/3 of the position range — equivalently, the last third of the input sequence.

This explains why [[RULER benchmark showed only half of long-context models maintained performance at 32K despite claiming 32K-plus support]] and why [[maximum effective context window can differ from claimed context by as much as 99 percent and shifts by problem type]] — the models literally have not been trained on the positions they claim to support. The effective context is bounded by the training distribution, not the architectural maximum.

The proposed fix, StRing (Shifted Rotary Position Embedding), drops infrequent position indices and shifts well-trained positions into their slots, achieving an 18-point average improvement across seven models on NIAH 4-needle tests without additional training. Since [[StRing shifted rotary position embedding improves long-context performance by 18 points without additional training]], this confirms the root cause is position frequency, not architectural capacity — the model has the capacity but lacks the training exposure.

---

Source: context-window-degradation-llms-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[initial tokens attract disproportionate attention regardless of semantic relevance due to autoregressive causal masking during training]] -- a related training-time root cause for the primacy component
- [[StRing shifted rotary position embedding improves long-context performance by 18 points without additional training]] -- the targeted fix for this problem
- [[practical guidance converges on 60-70 percent of maximum context window as effective usable capacity]] -- the practical response to this limitation

Topics:
- [[agent-cognition]]
