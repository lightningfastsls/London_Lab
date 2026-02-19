---
description: "Statistical properties of USV sequences contain information beyond individual call features, supporting sequence-level modeling"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information

Hertz et al. (2020) showed that the statistical properties of USV sequences — transition probabilities, sequence patterns, temporal statistics — carry predictive information about experimental conditions. This complements [[Chabout et al 2015 established that male mice change syllable syntax with social context]] by demonstrating that not just individual calls but their sequential organization is informative. This directly supports our autoregressive transformer approach where [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] — if sequence statistics are predictive, then a model that learns to predict future calls from past context should capture behaviorally meaningful patterns.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- Hertz et al. (2020)

Relevant Notes:
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- complementary evidence for sequential structure
- [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] -- our approach to capturing this structure
- [[bigram productivity ratio measures compositionality of USV code sequences]] -- one way to quantify sequence statistics
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- quantifies long-range mutual information in the sequences Hertz showed are informative
- [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] -- tests whether code frequency distributions follow language-like patterns
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- measures how much predictive information each additional context frame adds

Topics:
- [[classification]]
