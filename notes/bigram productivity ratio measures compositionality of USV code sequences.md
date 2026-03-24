---
description: Unique observed bigrams divided by K-squared quantifies whether codes combine freely (language-like) or follow rigid fixed sequences (formulaic, not compositional).
type: method
confidence: experimental
topics:
  - "[[representation-learning]]"
---

# bigram productivity ratio measures compositionality of USV code sequences

Bigram productivity is computed as: (number of unique observed code pairs) / K^2, where K is the codebook size. For K=64, the maximum possible unique bigrams is 64^2 = 4096. If the observed data contains 2500 unique bigrams, productivity = 2500/4096 ≈ 0.61. A high ratio (e.g., > 0.5) indicates codes combine freely — most pairs appear in the data, suggesting combinatorial organization where codes are contextually independent building blocks. A low ratio (e.g., < 0.1) indicates codes follow rigid sequences — only a small fixed set of transitions occurs, suggesting the vocabulary encodes holistic patterns rather than compositional elements.

The basic productivity ratio (unique bigrams / K^2) should be accompanied by bootstrap confidence intervals for statistical inference. Phase 14.1 extends this with `ngram_productivity()` which generalizes to arbitrary n-gram lengths (not just bigrams) and provides 95% bootstrap CIs from n_bootstrap=1000 resamples. Additionally, statistical significance of specific n-grams can be tested via `ngram_idioms()` which detects "idioms" -- n-grams occurring significantly above chance frequency (z-score > 3, FDR-corrected) compared to shuffled surrogates.

The compositionality test extends this: take code pairs that were never observed in training data (held-out bigrams), decode each pair using the VQ-VAE decoder, and assess whether the decoded spectrogram is coherent — does it resemble a plausible USV segment? If unseen bigrams decode to coherent spectrograms, this suggests the codes maintain independent acoustic identity regardless of context, a stronger form of compositionality. If unseen bigrams decode to noise or artifacts, codes are context-dependent and compositional combination fails.

This connects to [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] — a codebook that is too small will artificially inflate productivity (all K^2 pairs observed by necessity if K is tiny relative to data volume) while a codebook that is too large will deflate it (many codes never co-occur). The productivity ratio and compositionality test together address whether the learned vocabulary is genuinely combinatorial or merely a lookup table of fixed sequences. Compare with [[Zipf's law exponent reveals whether VQ-VAE code sequences have language-like frequency distribution]] for the marginal (single-code) frequency structure — compositionality is the joint (two-code) generalization. This analysis forms part of a four-part sequential structure test battery that also includes [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] (conditional entropy curves) and [[excess entropy measures long-range structure complexity in discrete code sequences]] (long-range mutual information). Each measures a different facet of language-likeness: frequency distribution, pairwise composition, context-dependent predictability, and long-range dependency.

---

Source: [ROADMAP](../ROADMAP.md), Phase 8; vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- transition probabilities (which bigrams probe) are part of the sequence statistics Hertz showed are informative
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- syntax changes imply flexible combination of syllable types, which bigram productivity would detect
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- K=64 defines the denominator K^2=4096 for the productivity ratio
- [[n-gram idiom detection identifies compositional phrases exceeding chance frequency in USV code sequences]] -- extends productivity from ratio to statistical significance testing

Topics:
- [[classification]]
