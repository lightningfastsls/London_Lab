---
description: "Random permutation of code sequences preserves unigram frequencies while destroying all temporal order, providing the simplest baseline for sequential structure claims"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# shuffled null model preserves code frequencies but destroys all sequential structure

The shuffled null model is the simplest and most fundamental baseline in the null model hierarchy for USV code sequences. The procedure is straightforward: take the observed sequence of VQ-VAE codes and apply a random permutation, which means every code remains in the dataset but its position is randomized. This preserves the unigram frequency distribution exactly — every code appears the same number of times — but destroys all temporal relationships between codes, because any ordering is equally likely under this model.

This baseline answers the question: "What if the codes were statistically independent?" Because the shuffled sequence retains the same marginal distribution, any metric that depends only on code frequencies (such as the Zipf exponent or Shannon entropy of the unigram distribution) will be identical between real and shuffled data. Therefore, these metrics alone cannot demonstrate sequential structure. But metrics that depend on temporal order — entropy rate, excess entropy, n-gram idiom frequencies, burstiness coefficients — should differ significantly between real and shuffled sequences if genuine sequential structure exists.

The practical implementation generates multiple shuffled surrogates (typically 100-1000) and computes each metric on every surrogate, which means we obtain a null distribution for each metric. The real data's metric value is then compared against this null distribution using a z-score or percentile rank. If the real value falls outside the 95th or 99th percentile of the null distribution, we can reject the hypothesis that the observed pattern arises from independent codes with non-uniform frequencies.

This is the first rung in the null model ladder, because it tests the weakest form of structure. If a metric on real data does not significantly exceed the shuffled baseline, there is no sequential structure to explain — the pattern is fully accounted for by the marginal distribution alone. Only metrics that pass this first test proceed to more stringent null models like Markov-k surrogates, which preserve increasingly complex temporal dependencies.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- this note describes one specific level in the null model hierarchy outlined there

Topics:
- [[representation-learning]]
- [[experimental-methods]]
