---
description: Approximating entropy rate with increasing n-gram context (1 through 8-gram) and observing a decreasing curve signals long-range sequential structure in code sequences.
type: method
confidence: experimental
topics:
  - "[[classification]]"
---

# entropy rate decreasing with context length indicates sequential predictability in USV code streams

The entropy rate h(C) of a stochastic process is the per-symbol uncertainty in the limit of infinite context: h = lim H(Cn | Cn-1, ..., C1) as n grows. Approximate this by estimating conditional entropies for n-gram models of increasing order (1-gram through 8-gram) using empirical code frequencies from all bout sequences. Plot H(Cn | Cn-1,...,C1) as a function of n (context length in frames).

A decreasing curve is the key signature: each additional context frame reduces uncertainty about the next code, implying that past codes carry information about future codes — long-range sequential structure. The curve should plateau as it approaches the true entropy rate; the plateau value quantifies irreducible uncertainty. Compare the plateau to maximum entropy log2(K) (where K is codebook size, e.g., log2(64) = 6 bits for K=64) to quantify predictability: a plateau at 3 bits with maximum 6 bits means roughly half the uncertainty is resolved by context.

A flat curve (no decrease with context) would indicate codes are approximately independent — no sequential structure, just a bag of codes. This would be a negative result for the hypothesis that USV sequences have language-like temporal organization. The rate of decrease (fast vs slow convergence) indicates the effective memory length — short memory suggests local structure only, long memory suggests global bout-level organization. This connects to [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] (marginal distribution) and [[excess entropy measures long-range structure complexity in discrete code sequences]] (integrated long-range mutual information), forming a complementary triple of sequential structure analyses.

---

Source: [[ROADMAP.md]], Phase 8
