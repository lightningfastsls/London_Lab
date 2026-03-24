---
description: "H = -sum(p_i * log2(p_i)) applied to syllable type proportions measures vocal diversity — prediction: wild mice show higher H than lab mice"
type: method
confidence: likely
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage

Shannon entropy H = -sum(p_i * log2(p_i)), where p_i is the proportion of syllable type i in a recording or animal's repertoire, provides a single scalar measure of vocal diversity. The measure has clear boundary conditions: H = 0 when the repertoire consists of only one syllable type (minimum diversity), and H = log2(K) when all K types are equally represented (maximum diversity). This makes it directly interpretable: higher H means more diverse, more evenly distributed syllable usage.

Applied to the wild-vs-lab comparison, the specific prediction is that wild mice should show higher H than lab mice, reflecting more diverse repertoires. This prediction follows from the courtship degradation hypothesis: if [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]], one measurable consequence should be reduced vocal diversity (lower H) in lab populations.

This is distinct from the vault's existing use of Shannon entropy in the information-theoretic analysis of VQ-VAE code sequences, where [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] measures sequential predictability. Here, entropy is applied to categorical syllable type proportions from DeepSqueak classification — a population-level diversity metric, not a sequential structure metric.

Shannon entropy complements the other repertoire comparison methods: [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] tests multivariate compositional differences, [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] measures pairwise distributional distance, and [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] captures sequential syntax. Shannon entropy adds a per-population scalar diversity dimension that these three methods do not directly provide.

---

Source:
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23)

Relevant Notes:
- [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]] -- the finding this metric would quantify
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- distinct application of entropy to sequential codes, not repertoire diversity
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- complementary multivariate test
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- complementary pairwise distributional distance
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- complementary sequential syntax comparison
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- related but distinct: entropy rate on code sequences vs Shannon entropy on syllable type proportions
- [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] -- the causal hypothesis predicting lower H in lab populations
- [[Zala et al 2020 showed wild-derived mice modulate USVs with social context producing 9 types during interaction versus 6 during introduction]] -- the 9 vs 6 type split maps to concrete entropy differences; social context modulates H

Topics:
- [[experimental-methods]]
- [[classification-methodology]]
