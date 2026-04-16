---
description: "SIS = H(X_n) - H(X_n | X_{n-1},...,X_{n-D}) — the mutual information between the next syllable and its D-depth suffix; exactly equivalent to three formulations"
type: method
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[representation-learning]]"
---

# SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context

The Syntax Information Score (SIS) at depth D is defined as the mutual information between the next syllable X_n and its preceding D-syllable context (X_{n-D}, ..., X_{n-1}):

SIS = I(X_n; Y) = H(X_n) - H(X_n | X_{n-1},...,X_{n-D})

where H(X_n) is the 0th-order entropy (no context) and H(X_n | ...) is the entropy rate of the Dth-order Markov model. **In practice: SIS = (entropy rate at depth 0) − (entropy rate at depth D).**

Three equivalent formulations:
1. **Entropy difference form:** I(X;Y) = H(X) - H(X|Y)
2. **KL-divergence form:** I(X;Y) = D_KL(p(x,y) || p(x)p(y))
3. **Sum form:** I(X;Y) = Σ_{x,y} p(x,y) log p(x,y)/[p(x)p(y)]

The entropy rate of the mth-order Markov model used as the inner component is:
H_m = -Σ_{i,j} μ_i P_{ij} log P_{ij}
where μ_i is the stationary probability of visiting suffix i and P_{ij} is the conditional probability of label j given suffix i. (Cover & Thomas 2005, Theorem 4.2.4)

**Key design property:** SIS is insensitive to the 0th-order distribution itself. An algorithm that assigns all USVs the same label gets SIS = 0 even though entropy rate = 0. An algorithm assigning random labels also gets SIS = 0 (entropy stays high at all orders). This makes SIS superior to raw entropy rate for comparing algorithms — it measures what structure the labels ADD over marginal frequencies alone.

SIS can be decomposed into per-pair contributions. For depth 1, each pair (x_{n-1}, x_n) contributes:
p(x_n, x_{n-1}) log p(x_n, x_{n-1}) / [p(x_n) · p(x_{n-1})]
This is pointwise KL divergence: pairs where the observed joint equals the independence product contribute 0.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- the broader SIS finding; this note provides the exact formula
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- our implementation of this entropy rate approach; SIS is exactly H_0 - H_D
- [[mutual information rate at varying lags measures temporal dependency strength within USV code sequences]] -- our MI at lag 1 = 0.093 bits is directly comparable to SIS depth 1
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- the higher-level finding this formula operationalizes
- [[self-repetition is the dominant pairwise contributor to SIS in mouse courtship vocalizations]] -- which pairs drive the SIS score

Topics:
- [[classification-methodology]]
- [[representation-learning]]
