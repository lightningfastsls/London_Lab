---
description: "All null models x all metrics matrix with z-scores and rank-based p-values — the main table proving whether USV structure exceeds statistical baselines"
type: method
confidence: likely
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# null model comparison framework produces z-scores rank-based p-values and effect sizes as the publishable statistical output

The comparison framework is the statistical machinery that turns raw metric values and null model surrogates into publishable results. For each combination of information-theoretic metric M and null model N, the procedure is: compute M on the real VQ-VAE code sequence, compute M on all N surrogates (typically 100-1000 per null model), then derive three statistics. The **z-score** is (real - mean_null) / std_null, which measures how many standard deviations the real data's metric falls from the null distribution mean. The **rank-based p-value** is the fraction of surrogates whose metric value equals or exceeds the real data's value, which is distribution-free and robust to non-Gaussian null distributions. The **effect size** via Cohen's d provides a standardized measure of the magnitude of the difference, not just its statistical significance.

The full analysis produces a matrix where rows are null models (shuffled, Markov-1, Markov-2, Markov-3, HMM, renewal process, phase-randomized) and columns are metrics (Zipf exponent, entropy rate, excess entropy, bigram productivity, burstiness coefficient, and others). This matrix is the core publishable result, because the pattern of significance across null models reveals the nature of the structure. A metric that is significant against the shuffled null model but not against the Markov-1 null model means that bigram transition probabilities fully explain the observed pattern — therefore the structure is locally predictable but not deeply sequential. A metric significant against all Markov orders but not against the HMM null model means that hidden behavioral state switching accounts for the structure — consistent with the Chabout et al hypothesis. A metric significant against ALL null models, including the HMM, constitutes the strongest evidence for language-like structure that exceeds simple statistical patterns.

This hierarchical interpretation is the scientific payoff of maintaining a full null model ladder rather than testing against a single baseline. Each null model preserves progressively more structure, which means significance against higher-order null models is progressively harder to achieve and progressively more meaningful. The comparison framework makes this hierarchy quantitative and rigorous, because every claim about USV sequential structure is backed by explicit statistical tests against specific alternative hypotheses rather than vague assertions of "complexity."

The framework also enables direct comparison between wild and lab mouse populations. If wild mice show significant excess entropy against all null models but lab mice only show significance against the shuffled baseline, that constitutes quantitative evidence that domestication reduced vocal sequential complexity — a testable prediction of the courtship degradation hypothesis.

---

Source:
- [[vacation-master-plan-v2]]

Relevant Notes:
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- the comparison framework operationalizes the null model hierarchy into concrete statistical output
- [[analytically verifiable test cases validate information-theoretic metric implementations]] -- ground-truth test cases ensure the metrics feeding this framework are correctly implemented
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- shares the permutation-based statistical philosophy; PERMANOVA also generates p-values via label shuffling

Topics:
- [[representation-learning]]
- [[experimental-methods]]
