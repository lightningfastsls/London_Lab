---
description: Mutual information between past and future halves of sequences quantifies long-range dependency complexity beyond what pairwise bigram analysis captures.
type: method
confidence: experimental
topics:
  - "[[classification]]"
---

# excess entropy measures long-range structure complexity in discrete code sequences

Excess entropy (also called "complexity" or "effective measure complexity") is defined as the mutual information between the semi-infinite past and semi-infinite future of a stationary process: E = I(past; future) = sum over n of [H(Cn | past n-gram) - h], where h is the true entropy rate. Practically, approximate by estimating I(C_1..C_L; C_(L+1)..C_(2L)) for increasing window length L and observing convergence.

Higher excess entropy indicates more complex long-range dependencies — knowing the past reduces uncertainty about the future substantially, and this reduction persists over long contexts. Natural language has high excess entropy because word choice at position t constrains word choice at t+100 (topic, grammar, narrative coherence). A simple i.i.d. source has zero excess entropy. A Markov chain has finite, small excess entropy. A process with power-law correlations may have divergent excess entropy.

The measure goes fundamentally beyond pairwise transition analysis (bigrams, which only capture 1-step memory) to quantify multi-step, multi-scale dependencies. For USV code sequences, high excess entropy would suggest that code sequences encode bout-level organization — e.g., an introductory phrase followed by a main vocalization followed by a closing pattern — where knowing the opening constrains the entire structure. This analysis connects to [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]], which measures conditional entropy at each context length (a related but distinct quantity), and to [[separating representation learning from discretization enables richer feature discovery]], where the hypothesis is that the transformer's representations will reveal structure that simpler methods miss.

---

Source: [[ROADMAP.md]], Phase 8
