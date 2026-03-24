---
description: "Without null model baselines, metrics like Zipf exponent, entropy rate, and excess entropy cannot distinguish real sequential structure from statistical artifacts"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
---

# null models are essential for interpreting information-theoretic metrics on USV code sequences

A shuffled random sequence drawn from a non-uniform distribution will exhibit a Zipf exponent, because Zipf's law describes the rank-frequency relationship of the marginal distribution, which shuffling preserves. Similarly, a first-order Markov process will show a decreasing entropy rate curve, because conditioning on even one previous token reduces uncertainty when transitions are non-uniform. Therefore, observing these signatures in real USV code sequences does not, by itself, demonstrate meaningful structure — it may simply reflect trivial statistical properties that any sequence with the same marginal frequencies or pairwise transitions would exhibit.

The solution is a null model hierarchy that systematically tests increasingly complex hypotheses about what generates the observed patterns. The simplest null model — random shuffling — preserves unigram frequencies but destroys all temporal order. If a metric (e.g., excess entropy) is significantly higher in the real data than in shuffled surrogates, we can attribute it to at least pairwise temporal structure. However, a Markov null model (preserving bigram transition probabilities) tests whether the structure exceeds what first-order dependencies explain. A renewal process null model (preserving inter-event interval distributions) tests whether timing patterns alone account for the observed statistics. An HMM null model tests whether hidden state switching is sufficient. Phase-randomized surrogates (preserving the power spectrum but destroying phase relationships) test whether the structure lies in phase correlations rather than spectral properties.

Each level in the hierarchy asks: "Can a simpler generative process produce the same metric values?" Only when the real data significantly exceeds the null at a given level can we attribute structure to mechanisms more complex than that null. This is why [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] needs comparison against Markov baselines, why [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] needs comparison against shuffled surrogates, and why [[excess entropy measures long-range structure complexity in discrete code sequences]] needs comparison against HMM surrogates to confirm that the long-range structure is genuinely beyond what hidden state models capture.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- entropy rate curves require Markov null comparison to confirm structure beyond first-order dependencies
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- Zipf exponents persist under shuffling, so null models distinguish real rank-frequency structure from trivial marginal effects
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- excess entropy needs HMM null baselines to confirm structure exceeds hidden state switching
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- empirical evidence motivating formal null model testing of USV sequential structure

Topics:
- [[representation-learning]]
