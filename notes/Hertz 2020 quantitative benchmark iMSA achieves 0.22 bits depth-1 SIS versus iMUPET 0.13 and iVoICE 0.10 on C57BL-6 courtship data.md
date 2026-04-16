---
description: "iMSA ~0.22 bits (depth-1 SIS), iMUPET ~0.13, iVoICE ~0.10, SIM ~0.23 — C57BL/6 courtship, 8 labels, 346K syllables; our wild-mice 7-type result is 0.093 bits"
type: baseline
confidence: proven
conditions:
  - "C57BL/6 lab strain"
  - "8 labels"
  - "346K syllables"
  - "courtship context"
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[wild-lab-vocal-comparison]]"
---

# Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data

Quantitative SIS values from Hertz et al. (2020), 8 labels, C57BL/6 courtship data (385 sessions, 346K syllables, 33K sequences):

| Algorithm | SIS depth 1 (bits/sym) | SIS depth 2 (bits/sym) |
|-----------|----------------------|----------------------|
| iMSA (rule-based pitch jumps) | ~0.22 | ~0.27 |
| iVoICE (spectral similarity) | ~0.10 | ~0.14 |
| iMUPET (gammatone k-means) | ~0.13 | ~0.17 |
| **SIM (iMUPET + sequential optimization)** | ~0.23 | ~0.25 |

(Values approximate, read from Fig. 4b and 7d)

**Entropy rates (depth 0):**
- iMSA: ~2.45 bits (skewed distribution, >50% in "Simple" types)
- iVoICE: ~2.75 bits
- iMUPET: ~2.90 bits (near-uniform, close to log2(8) = 3.0)

**Key finding:** iMSA achieves the highest SIS despite the lowest 0th-order entropy. Pitch jumps are an unusually informative feature for sequential structure. The rule-based approach wins because pitch-jump categories have predictive power beyond purely acoustic similarity.

**Comparison to our data:**
- Our 7-type Scattoni MI at lag 1 = 0.093 bits (directly comparable to SIS depth 1)
- Our conditional entropy reduction = 3.7% vs Hertz's ~7-10% from depth 0 to depth 1
- Our lower values are expected: fewer categories, wild-derived mice (not C57BL/6), 43× smaller dataset

**Predictable extension:** SIM starting from DeepSqueak's 27 centroids should be able to improve SIS from iMUPET-level baseline toward or above iMSA-level, if the dataset is large enough to evaluate (this may not apply to our 8K-call dataset).

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context]] -- the formula producing these numbers
- [[SIS normalized by log2 of cluster count removes dependency on number of labels enabling cross-Nc comparisons]] -- normalization makes iMSA 0.22/3 = 0.073 vs our 0.093/2.81 = 0.033
- [[Hertz et al 2020 dataset is 346K syllables across 385 sessions making our 8K dataset 43 times smaller]] -- the denominator for understanding these baselines
- [[self-repetition is the dominant pairwise contributor to SIS in mouse courtship vocalizations]] -- what drives iMSA's high SIS
- [[iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy]] -- the surprising winner
- [[Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences]] -- SIM reaches iMSA's score starting from iMUPET
- [[mutual information rate at varying lags measures temporal dependency strength within USV code sequences]] -- our measure that produces the 0.093-bit value

Topics:
- [[classification-methodology]]
- [[wild-lab-vocal-comparison]]
