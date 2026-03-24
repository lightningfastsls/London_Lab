---
description: "Known-answer test cases with analytical solutions provide ground-truth sanity checks for entropy rate, burstiness, n-gram idiom, and periodicity metric implementations"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# analytically verifiable test cases validate information-theoretic metric implementations

Information-theoretic metrics are notoriously easy to implement incorrectly — off-by-one errors in n-gram counting, incorrect normalization, wrong log base, finite-sample bias — and the results often look plausible even when wrong, because there is no obvious "crash" that signals a bug. Therefore, analytically verifiable test cases with known solutions are essential for validating implementations before applying them to real USV code sequences where the ground truth is unknown.

The key test cases exploit sequences whose information-theoretic properties can be computed by hand. First, a shuffled uniform sequence with K=64 symbols has entropy rate exactly equal to log2(64) = 6.0 bits at all context orders, because every symbol is equally likely and independent of context. If the implementation reports anything other than 6.0 (within finite-sample noise), it has a bug in entropy estimation or normalization. Second, a sequence generated from a known first-order Markov chain should show entropy rate convergence at exactly order 1: the conditional entropy H(C_n | C_{n-1}) should equal the theoretical Markov entropy rate, and conditioning on longer context should not reduce it further. If the curve continues decreasing beyond order 1, the implementation is not correctly estimating conditional entropies or there is a sample size artifact.

Third, a Poisson process (exponentially distributed inter-event intervals) has coefficient of variation (CV) exactly equal to 1, which means the burstiness coefficient should equal zero. This validates the CV/burstiness calculation. Fourth, a sequence with a planted idiom (a specific n-gram appearing at 10x its expected frequency under independence) should be detected by the n-gram idiom analysis with a z-score proportional to the excess frequency. Fifth, a perfectly periodic sequence (e.g., repeating [1, 2, 3, 1, 2, 3, ...]) has CV < 1 and entropy rate of 0 at context order equal to the period length, because the sequence is fully deterministic given sufficient context.

These test cases are computationally cheap and should be run as unit tests before every analysis pipeline execution. Any implementation that fails even one of them cannot be trusted on real data, because the analytical solutions leave no room for ambiguity about what the correct answer should be.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- entropy rate convergence test case directly validates the implementation used for this analysis
- [[burstiness coefficient via coefficient of variation of inter-event intervals distinguishes Poisson from bursty temporal patterns]] -- Poisson CV=1 test case validates the burstiness implementation

Topics:
- [[representation-learning]]
- [[experimental-methods]]
