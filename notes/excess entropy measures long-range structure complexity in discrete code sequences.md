---
description: Mutual information between past and future halves of sequences quantifies long-range dependency complexity beyond what pairwise bigram analysis captures.
type: method
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# excess entropy measures long-range structure complexity in discrete code sequences

Excess entropy (also called "complexity" or "effective measure complexity") is defined as the mutual information between the semi-infinite past and semi-infinite future of a stationary process: E = I(past; future) = sum over n of [H(Cn | past n-gram) - h], where h is the true entropy rate. Practically, approximate by estimating I(C_1..C_L; C_(L+1)..C_(2L)) for increasing window length L and observing convergence.

Higher excess entropy indicates more complex long-range dependencies — knowing the past reduces uncertainty about the future substantially, and this reduction persists over long contexts. Natural language has high excess entropy because word choice at position t constrains word choice at t+100 (topic, grammar, narrative coherence). A simple i.i.d. source has zero excess entropy. A Markov chain has finite, small excess entropy. A process with power-law correlations may have divergent excess entropy.

The specific computational method for excess entropy is Crutchfield & Feldman (2003) block entropy extrapolation. The procedure fits H(L) = h*L + E in the linear regime (large L), where h is the entropy rate and E is the excess entropy. The existing implementation in `sequence_analysis.py` uses entropy rate convergence (H_1 - h_infinity), which is an equivalent formulation. The block entropy approach provides a complementary way to compute and validate the same quantity.

The measure goes fundamentally beyond pairwise transition analysis (bigrams, which only capture 1-step memory) to quantify multi-step, multi-scale dependencies. For USV code sequences, high excess entropy would suggest that code sequences encode bout-level organization — e.g., an introductory phrase followed by a main vocalization followed by a closing pattern — where knowing the opening constrains the entire structure. This analysis connects to [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]], which measures conditional entropy at each context length (a related but distinct quantity), and to [[separating representation learning from discretization enables richer feature discovery]], where the hypothesis is that the transformer's representations will reveal structure that simpler methods miss. The complementary analyses in this test battery also include [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] (marginal distribution) and [[bigram productivity ratio measures compositionality of USV code sequences]] (pairwise transition structure). High excess entropy with low bigram productivity would suggest that long-range dependencies arise from rigid sequential patterns rather than free combinatorial composition. Whether this long-range structure is genuinely learned depends on [[whether attention patterns in the trained transformer attend beyond the immediately preceding frame]] -- if attention is purely local, the codes may lack the long-range dependencies that excess entropy is designed to detect.

---

Source: [ROADMAP](../ROADMAP.md), Phase 8; vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- empirical evidence that sequence statistics are informative, motivating this formal measure
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- context-dependent syntax implies long-range structure exists
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- related conditional entropy measure in the same test battery
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- marginal frequency complement
- [[bigram productivity ratio measures compositionality of USV code sequences]] -- pairwise transition complement
- [[whether attention patterns in the trained transformer attend beyond the immediately preceding frame]] -- long-range attention is a prerequisite for learning the dependencies this metric detects
- [[Crutchfield and Feldman 2003 block entropy extrapolation is the standard method for computing excess entropy]] -- the specific algorithm and mathematical foundation

Topics:
- [[classification]]
