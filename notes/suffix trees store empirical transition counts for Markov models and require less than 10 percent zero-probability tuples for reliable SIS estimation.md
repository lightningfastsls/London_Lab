---
description: "Hertz 2020 validity criterion: fraction of zero-probability (D+1)-tuples must be below 10% — limits Nc≤64 at depth 1 and Nc≤16 at depth 2 on their 346K-syllable dataset"
type: method
confidence: proven
conditions:
  - "dataset size approximately 346K syllables"
meta_state: current
topics:
  - "[[classification-methodology]]"
---

# suffix trees store empirical transition counts for Markov models and require less than 10 percent zero-probability tuples for reliable SIS estimation

Hertz et al. (2020) implement SIS using suffix tree data structures that store empirical visit counts for each (D+1)-tuple of syllable labels. The three probability distributions needed for SIS computation are all estimated from the suffix tree:

- **p(y)** = p(X_{n-1},...,X_{n-D}) — probability of each suffix (leaf visit counts / total)
- **p(x,y)** = p(X_n, X_{n-1},...,X_{n-D}) — computed via law of total probability as p(X_n | suffix) × p(suffix)
- **p(x)** = p(X_n) — 0th-order marginal

**Validity criterion:** For the Markov model to be reliable, the fraction of conditional probabilities with value 0 (i.e., (D+1)-tuples never observed in the dataset) must be **less than 10%**. Never-observed transitions introduce zeros that make the entropy rate undefined or heavily biased.

**Practical limits on their 346K-syllable / 33K-sequence dataset:**
- Depth 1: Nc ≤ 64 labels satisfies the <10% criterion
- Depth 2: Nc ≤ 16 labels required; higher Nc produces too many unobserved triplets

**Critical implication for small datasets:** With ~8,000 USVs (43× smaller), the same criterion will bind much more aggressively. At depth 1 with 7 Scattoni labels, the criterion is likely satisfied; at depth 2, even 7 labels may produce >10% zero-probability triplets. SIM (which uses depth 1 SIS as its objective with K=8) may still be tractable, but depth-2 analyses should be validated against this criterion.

The suffix tree structure is also the implementation used to compute the stationary distribution μ_i in the entropy rate formula H_m = -Σ_{i,j} μ_i P_{ij} log P_{ij}. Reference: Cover & Thomas (2005), Theorem 4.2.4.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context]] -- the formula that suffix trees compute
- [[Hertz et al 2020 dataset is 346K syllables across 385 sessions making our 8K dataset 43 times smaller]] -- the dataset size context for these limits
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- our analogous entropy rate computation; Miller-Madow correction addresses similar finite-sample issues
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- the parent method this data structure supports

Topics:
- [[classification-methodology]]
