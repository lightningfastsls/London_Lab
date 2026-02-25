---
description: Fitting a power law to code frequency vs rank and comparing the exponent to natural language (alpha ~1.0) tests whether USV code sequences show language-like statistical structure.
type: method
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution

Count the frequency of each codebook entry across all bout sequences, rank entries by frequency (rank 1 = most common), plot log(frequency) vs log(rank), and fit a power law: frequency proportional to rank^(-alpha). Natural language word frequency distributions follow Zipf's law with alpha approximately 1.0 — a few words are extremely common, most words are rare, and the distribution follows a straight line on a log-log plot. If USV code sequences show a similar distribution, this is evidence of language-like statistical structure in the learned vocabulary.

The interpretation of the alpha value carries nuance. Alpha close to 1.0 suggests rich combinatorial structure similar to human language. Alpha much larger than 1.0 (steeper slope) suggests a highly skewed distribution where a few codes dominate and most are rarely used — potentially indicating insufficient codebook diversity or dataset bias toward common vocalizations. Alpha much smaller than 1.0 (flatter slope) suggests a more uniform distribution, inconsistent with Zipf's law and potentially indicating that codes are not organized along a frequency-of-use axis. A minimum count threshold of 5 excludes rare noise codes that might distort the tail of the distribution.

The existing OLS log-log approach should be complemented by two superior methods: (1) Clauset et al. 2009 MLE (maximum likelihood estimation of alpha, xmin, and KS goodness-of-fit p-value) which avoids the systematic bias of OLS on small datasets, and (2) Shannon entropy equivalence for PLC estimation, which provides an independent cross-validation particularly robust for datasets under 10K tokens. These are implemented in `information_theory.py` (Phase 14.1).

This analysis is one of four sequential structure tests forming a language-likeness test battery. It captures the marginal frequency distribution but says nothing about transition patterns. The complementary analyses — [[bigram productivity ratio measures compositionality of USV code sequences]] (pairwise transition freedom), [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] (context-dependent conditional entropy), and [[excess entropy measures long-range structure complexity in discrete code sequences]] (long-range mutual information) — probe increasingly complex structural properties. Whether Zipf's law holds depends critically on [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] — if the codebook is too small, a few codes will dominate regardless of true structure; if too large, the distribution will be artificially flattened by dead codes.

---

Source: [[ROADMAP.md]], Phase 8; [[vacation-master-plan-v2]]

Relevant Notes:
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- empirical evidence that sequence-level statistics are meaningful
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- context-dependent usage implies some codes are used more frequently in certain contexts
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- K=64 determines the vocabulary size over which Zipf's law is tested
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- if VQ-VAE codes show Zipf-like distribution, the learned discretization may be more natural than imposed taxonomies
- [[Clauset et al 2009 MLE produces the gold standard power law fit for Zipf exponent estimation]] -- provides the specific MLE algorithm for rigorous fitting
- [[entropy-based Zipf estimation cross-validates MLE on small datasets where power law fitting is unreliable]] -- independent cross-check via Shannon entropy

Topics:
- [[representation-learning]]
