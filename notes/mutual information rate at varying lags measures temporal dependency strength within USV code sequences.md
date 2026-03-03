---
description: "I(X_t; X_{t+lag}) quantifies shared information between tokens at distance, complementing entropy rate (contiguous history) and conditional entropy by lag (remaining uncertainty)"
type: method
confidence: likely
meta_state: current
topics:
  - "[[representation-learning]]"
---

# mutual information rate at varying lags measures temporal dependency strength within USV code sequences

The mutual information rate I(X_t; X_{t+lag}) measures how much information one VQ-VAE code token carries about another token at a given temporal distance. Unlike [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]], which conditions on the entire contiguous history X_{n-1},...,X_{n-k}, the mutual information rate isolates the pairwise relationship between two tokens separated by a specific lag. And unlike [[conditional entropy by lag probes single-token influence at distance unlike entropy rate which uses contiguous history]], which measures remaining uncertainty H(X_t|X_{t-lag}), this metric measures the complementary quantity — shared information I(X_t; X_{t+lag}) = H(X_t) - H(X_t|X_{t+lag}).

The distinction matters because mutual information is symmetric and bounded between 0 and min(H(X_t), H(X_{t+lag})), making it directly comparable across lags. A plot of I(X_t; X_{t+lag}) versus lag reveals the temporal reach of sequential dependencies: if MI decays exponentially, the process has a characteristic memory timescale; if it decays as a power law, the process has long-range correlations. For USV code sequences, comparing this decay profile against null models — particularly the [[Markov order-k null model generates surrogates preserving k-step transition dependencies]] — tests whether real sequences carry information at distances beyond what short-range Markov structure can explain.

This metric is computed at varying lags (typically 1 through 20 or more) to produce a temporal dependency profile. The profile shape is itself informative: language-like sequences tend to show slowly decaying MI (long-range dependencies), while simple Markov processes show sharp exponential decay. Comparing MI decay profiles between wild and lab mouse populations could reveal differences in sequential complexity even when simpler metrics like [[bigram productivity ratio measures compositionality of USV code sequences]] show no difference.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[conditional entropy by lag probes single-token influence at distance unlike entropy rate which uses contiguous history]] — measures the complementary quantity H rather than I at the same lag distances
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] — uses contiguous history rather than isolated lag, capturing different aspect of sequential structure
- [[excess entropy measures long-range structure complexity in discrete code sequences]] — captures total long-range information as a single scalar, while MI rate decomposes it by lag

Topics:
- [[representation-learning]]
