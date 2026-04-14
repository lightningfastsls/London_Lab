---
description: "Already observed: wild mice produce a richer variety of USV call types than lab mice, supporting the degradation hypothesis"
type: finding
confidence: experimental
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# Wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation

The researcher has already observed that wild mice show much more diverse USV repertoires than lab mice. This is preliminary evidence supporting the directional hypothesis that [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]]. The observation has not yet been formally quantified — the planned analyses (both DeepSqueak-based classification comparison and VQ-VAE codebook analysis) will provide statistical backing. The goal is to show that wild mice use more call types, with different distributions and potentially more complex temporal patterning, consistent with courtship behavioral degradation in lab strains.

---

Source:
- Researcher brain-dump on scientific hypotheses (2026-02-19)

Relevant Notes:
- [[inbreeding and absence of courtship selection pressure in captivity caused lab mice to degrade courtship vocal competence]] -- the hypothesis this evidence supports
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the analytical framework for testing this
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- the specific metric that would quantify "more diverse" as higher H for wild mice
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- the multivariate test for formalizing this observation statistically
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- caveat: the diversity observation depends on how calls are categorized; categorical analysis may understate or mischaracterize the true difference
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- continuous distributional comparison would formalize this observation without relying on discrete categories
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- the analytical strategy that will quantify this preliminary observation through both categorical and data-driven lenses

Topics:
- [[experimental-methods]]
- [[classification]]
