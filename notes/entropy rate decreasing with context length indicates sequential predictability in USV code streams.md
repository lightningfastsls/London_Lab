---
description: Approximating entropy rate with increasing n-gram context (1 through 8-gram) and observing a decreasing curve signals long-range sequential structure in code sequences.
type: method
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# entropy rate decreasing with context length indicates sequential predictability in USV code streams

The entropy rate h(C) of a stochastic process is the per-symbol uncertainty in the limit of infinite context: h = lim H(Cn | Cn-1, ..., C1) as n grows. Approximate this by estimating conditional entropies for n-gram models of increasing order (1-gram through 8-gram) using empirical code frequencies from all bout sequences. Plot H(Cn | Cn-1,...,C1) as a function of n (context length in frames).

A decreasing curve is the key signature: each additional context frame reduces uncertainty about the next code, implying that past codes carry information about future codes — long-range sequential structure. The curve should plateau as it approaches the true entropy rate; the plateau value quantifies irreducible uncertainty. Compare the plateau to maximum entropy log2(K) (where K is codebook size, e.g., log2(64) = 6 bits for K=64) to quantify predictability: a plateau at 3 bits with maximum 6 bits means roughly half the uncertainty is resolved by context.

The plugin estimator used by the existing implementation should be supplemented with Miller-Madow bias correction: H_corrected = H_plugin + (m-1)/(2N*ln2), where m = number of non-zero bins and N = sample size. This correction is critical because the plugin estimator is systematically negatively biased, and the bias grows worse as context length increases (more bins from K^n contexts with smaller sample counts per bin). Phase 14.1's `entropy_rate()` adds this correction.

A flat curve (no decrease with context) would indicate codes are approximately independent — no sequential structure, just a bag of codes. This would be a negative result for the hypothesis that USV sequences have language-like temporal organization. The rate of decrease (fast vs slow convergence) indicates the effective memory length — short memory suggests local structure only, long memory suggests global bout-level organization. This connects to [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] (marginal distribution) and [[excess entropy measures long-range structure complexity in discrete code sequences]] (integrated long-range mutual information), forming a complementary triple of sequential structure analyses. The fourth member of this test battery is [[bigram productivity ratio measures compositionality of USV code sequences]], which probes the joint (two-code) structure that bridges between marginal frequencies and full sequential dependencies. Together, the four analyses test language-likeness at increasing levels of structural complexity: marginal frequencies, pairwise transitions, conditional entropy curves, and long-range mutual information.

---

Source: [ROADMAP](../ROADMAP.md), Phase 8; vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- empirical evidence that USV sequences contain predictable structure, motivating this analysis
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- context-dependent syntax implies sequential structure exists to be measured
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- marginal frequency analysis in the same test battery
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- long-range mutual information in the same test battery
- [[bigram productivity ratio measures compositionality of USV code sequences]] -- pairwise transition analysis in the same test battery
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- K=64 determines the maximum entropy baseline for this analysis
- [[Miller-Madow correction compensates for finite sample bias in entropy rate estimation]] -- the specific correction formula and when it matters most

Topics:
- [[classification]]
