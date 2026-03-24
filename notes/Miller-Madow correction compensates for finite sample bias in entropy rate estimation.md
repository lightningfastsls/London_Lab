---
description: "Plugin entropy estimators systematically underestimate true entropy; the correction adds (m-1)/(2N ln2) where m is the number of non-empty bins"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
---

# Miller-Madow correction compensates for finite sample bias in entropy rate estimation

When estimating Shannon entropy from finite samples, the plugin estimator — which computes entropy directly from observed frequency counts — is negatively biased. It systematically underestimates the true entropy because some probability bins are empty or undersampled in any finite dataset. Bins with zero observed counts contribute zero to the estimated entropy but may have nonzero true probability, and bins with few observations have their probabilities estimated with high variance, which on average reduces the entropy estimate.

The Miller-Madow correction compensates for this bias by adding (m-1)/(2*N*ln(2)) bits to the plugin estimate, where m is the number of bins with nonzero observed probability and N is the total sample size. This first-order bias correction is derived from a Taylor expansion of the bias of the plugin estimator and is accurate when N is large relative to m. The correction is small when the sample size greatly exceeds the number of bins, but becomes substantial when sample size is comparable to or smaller than the number of possible outcomes.

This correction is critical for entropy rate estimation on USV code sequences, because the effective sample size shrinks dramatically as the conditioning context grows. For a codebook of size K=64, the number of possible n-gram contexts grows as K^(n-1) = 64^(n-1). At order n=3, there are 64^2 = 4096 possible bigram contexts, meaning that even a dataset of 50K tokens has only about 12 observations per context on average — and the distribution is highly uneven, so many contexts have far fewer observations. Without the Miller-Madow correction, entropy rate curves may show artificial decreases at higher orders that reflect sampling bias rather than genuine sequential structure. This directly affects the interpretation of results from [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]], where distinguishing real predictability from estimation artifact is the central scientific question.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- the primary analysis that requires bias-corrected entropy estimation to avoid artificial structure detection
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- null model comparison provides a complementary defense against artifacts, but does not substitute for bias correction
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- K=64 determines the growth rate of the context space and thus the severity of the finite-sample bias

Topics:
- [[representation-learning]]
