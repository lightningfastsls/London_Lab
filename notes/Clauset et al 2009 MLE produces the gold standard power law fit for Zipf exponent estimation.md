---
description: "Maximum likelihood estimation with KS goodness-of-fit test avoids the log-log OLS bias that plagues small-sample Zipf analysis"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
---

# Clauset et al 2009 MLE produces the gold standard power law fit for Zipf exponent estimation

The standard approach to fitting power laws — ordinary least squares (OLS) regression on a log-log plot of frequency versus rank — is fundamentally biased, because the logarithmic transformation distorts the error distribution and gives disproportionate weight to rare events in the tail. For small samples (which USV code datasets often are, with perhaps 10K-50K tokens), this bias can produce spurious power-law fits or wildly inaccurate exponent estimates. The Clauset, Shalizi, and Newman (2009) methodology addresses this through a principled statistical framework.

The method works in three stages. First, it estimates xmin — the lower bound of power-law behavior — by minimizing the Kolmogorov-Smirnov (KS) distance between the empirical distribution and the fitted power law for all candidate xmin values. This is critical because many empirical distributions only follow a power law in their tail, not across the entire range. Second, it estimates the exponent alpha via maximum likelihood estimation (MLE) for all observations greater than or equal to xmin, which is unbiased and asymptotically efficient. Third, a semi-parametric bootstrap procedure generates synthetic datasets from the fitted power law, refits each, and computes a p-value for goodness-of-fit. A p-value above 0.1 indicates the power law is a plausible model for the data; below 0.1, the power law should be rejected.

Additionally, a likelihood ratio test compares the power-law fit against alternative distributions — particularly the exponential distribution, which is the most common competitor. This is important because a log-normal or stretched exponential may fit the data equally well or better than a power law, and claiming Zipf-like behavior without testing alternatives is scientifically unsound. For USV code sequences, this rigorous approach ensures that claims about language-like frequency distributions in [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] are statistically defensible rather than artifacts of naive fitting.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- this method provides the statistically rigorous implementation for measuring the Zipf exponent
- [[entropy-based Zipf estimation cross-validates MLE on small datasets where power law fitting is unreliable]] -- independent cross-validation approach when MLE alone may be insufficient
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- K=64 determines the vocabulary size and thus the number of frequency ranks available for fitting

Topics:
- [[representation-learning]]
