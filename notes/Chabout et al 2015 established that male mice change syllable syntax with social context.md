---
description: "USV syllable sequences vary by social context (female present vs urine only), establishing that syntax carries behavioral information"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# Chabout et al 2015 established that male mice change syllable syntax with social context

Chabout et al. (2015) demonstrated that male mice alter their syllable syntax depending on social context — for example, producing different sequence patterns when a female is present versus when only female urine is present. This finding establishes that USV sequences are not random assemblages of calls but carry behavioral information in their sequential structure. This motivates our sequence analysis approach: since [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]], and since [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]], we expect to find learnable sequential structure in USV sequences that reflects behavioral context.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- Chabout et al. (2015)

Relevant Notes:
- [[causal attention in autoregressive transformer matches the scientific question of predicting what comes next in USV streams]] -- our architectural response to the sequence structure finding
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- metric for measuring the structure Chabout identified
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- complementary evidence for sequential structure
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- formalizes syllable syntax comparison between populations as transition probability matrices

Topics:
- [[classification-methodology]]
