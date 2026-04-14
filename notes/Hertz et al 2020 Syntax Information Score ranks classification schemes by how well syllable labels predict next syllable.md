---
description: "SIS measures temporal predictiveness of category labels -- higher SIS means labels better capture sequential structure; found no one-to-one mapping between schemes"
type: method
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification]]"
---

# Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable

Hertz et al. (2020, *Communications Biology*) developed the **Syntax Information Score (SIS)** to evaluate and rank different USV classification schemes. The SIS measures how well syllable labels predict the next syllable in a sequence -- essentially using temporal structure to validate whether categories capture meaningful biological variation. A higher SIS means the classification scheme produces labels that carry more sequential predictive information.

A key finding: different classification schemes (Holy & Guo, MUPET, DeepSqueak) produce **no one-to-one mapping between labels**. Categories from one system do not cleanly correspond to categories from another, which means the "syllable types" are partially artifacts of the classification method rather than purely biological categories. This reinforces [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- if discrete categories were natural, different methods should converge on the same types.

The SIS is directly applicable to evaluating our VQ-VAE codebook: we can compute SIS for the codebook-assigned labels and compare against traditional classification schemes. If our learned codes achieve higher SIS than predefined taxonomies, that would validate the data-driven approach. This connects to our information-theoretic analysis where [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] and [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]].

---

Source:
- inbox/deepsqueak-usv-syllable-classification-practical-guide.md (Compass artifact, 2026-02-23)
- Hertz et al. (2020), *Communications Biology*

Relevant Notes:
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- the broader finding this method operationalizes
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- our analogous measure of sequential structure
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the continuum that prevents scheme convergence
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- one of the schemes Hertz compared
- [[MUPET uses gammatone filterbank and unsupervised k-means to discover 100-140 data-driven USV types]] -- another compared scheme
- [[bigram productivity ratio measures compositionality of USV code sequences]] -- complementary sequential structure metric
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- SIS evaluates whether labels capture sequential structure; transition matrices operationalize that structure as testable population-level differences
- [[traditional Holy and Guo 2005 USV taxonomy defines discrete types but Goffinet 2021 showed USVs form a continuum]] -- SIS found no one-to-one mapping between Holy & Guo, MUPET, and DeepSqueak schemes, undermining the assumption of objectively "correct" discrete categories
- [[forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations]] -- SIS partially dissolves this tension by offering a principled criterion for evaluating any categorization
- [[dual supervised plus unsupervised classification addresses the USV taxonomy problem from both directions]] -- SIS could evaluate both the supervised (Scattoni categories) and unsupervised (HDBSCAN cluster labels) branches of the dual approach
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- complementary evaluation axis: SIS measures sequential predictiveness of categories; EMD/JSD measures distributional separation in continuous space

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
