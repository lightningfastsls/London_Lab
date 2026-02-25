---
description: "Two distinct research questions: (1) courtship degradation via repertoire comparison (achievable now) and (2) language-like sequential structure (requires VQ-VAE)"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# VQ-VAE investigation of language-like sequential structure in USVs is a separate deeper question from courtship degradation

The research explicitly separates two distinct questions: (1) Have lab mice degraded their courtship USV repertoire compared to wild mice? This can be answered with simpler classification tools since [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]]. (2) Do USV sequences have language-like properties — predictable sequential structure, compositionality, context-dependent syntax? This requires the full transformer/VQ-VAE pipeline and is a deeper, more ambitious question. The separation matters because question 1 is the publishable core finding while question 2 is a longer-term investigation. Evidence from [[Chabout et al 2015 established that male mice change syllable syntax with social context]] and [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] motivates question 2, but the existing metrics ([[entropy rate decreasing with context length indicates sequential predictability in USV code streams]], [[bigram productivity ratio measures compositionality of USV code sequences]]) will only become applicable after the VQ-VAE codebook is trained.

---

Source:
- Researcher brain-dump on scientific hypotheses (2026-02-19)

Relevant Notes:
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- how question 1 gets answered first
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the bridge format that enables question 1 using our detections + DeepSqueak classification
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- motivation for question 2
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- motivation for question 2
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- question 1's primary statistical test
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- question 1's diversity metric

Topics:
- [[classification]]
