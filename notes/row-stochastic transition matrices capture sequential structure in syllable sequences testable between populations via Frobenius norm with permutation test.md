---
description: "Per-animal P(type_{t+1}|type_t) matrices averaged within populations, compared via ||M_wild - M_lab||_F with permutation test for significance"
type: method
confidence: likely
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test

Beyond comparing which syllable types each population uses (proportions), transition matrices capture how syllables follow each other — the sequential structure of vocal behavior. If wild mice produce type A → type C transitions more frequently than lab mice, this reveals differences in vocal syntax that proportions alone would miss.

The method proceeds in four steps:

1. **Per-animal transition matrix:** From time-ordered detections, compute P(type_{t+1} | type_t) — a row-stochastic matrix where each row sums to 1, and entry (i,j) gives the probability that syllable type j follows type i.

2. **Population averaging:** Average all per-animal matrices within each population to get M_wild and M_lab. This accounts for individual variability.

3. **Distance metric:** The Frobenius norm ||M_wild - M_lab||_F = sqrt(sum of squared element-wise differences) gives a single scalar measuring how different the two population-level transition structures are.

4. **Permutation test:** Shuffle population labels (wild/lab) across animals, recompute the Frobenius distance, and repeat 999–9999 times to build a null distribution. The p-value is the fraction of permuted distances exceeding the observed distance.

This method can also identify specific transitions that differ most strongly between populations (the largest element-wise differences in the matrices), providing biological interpretability beyond the overall significance test.

The transition matrix is effectively a first-order Markov model of syllable sequences. The [[Markov order-k null model generates surrogates preserving k-step transition dependencies]] in the information-theoretic analysis stream formalizes this connection: if the Markov-1 null model fully explains observed sequential structure, then the transition matrix captures all the sequence information that exists. If higher-order null models are needed, the first-order transition matrix is an incomplete picture of syllable syntax. The complementary [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] analysis can quantify how much sequential information exists beyond what first-order transitions capture.

---

Source:
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23)

Relevant Notes:
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question; mentions transition matrix comparison briefly
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- evidence that syllable sequences carry meaningful information
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- establishes that syntax varies with context, motivating population comparison
- [[Markov order-k null model generates surrogates preserving k-step transition dependencies]] -- formalizes the connection between transition matrices and Markov models
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- complementary measure of sequential structure beyond first-order transitions
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- complementary compositional test (proportions vs syntax)
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- complementary diversity metric
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- complementary distributional distance
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] -- SIS evaluates whether classification labels capture meaningful sequential structure; transition matrices operationalize that structure

Topics:
- [[experimental-methods]]
- [[classification]]
