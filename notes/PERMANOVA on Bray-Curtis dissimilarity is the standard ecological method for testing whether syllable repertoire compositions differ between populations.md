---
description: "Non-parametric permutational test borrowed from community ecology (Anderson 2001) that compares multivariate syllable compositions without distributional assumptions"
type: method
confidence: proven
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations

PERMANOVA (Permutational Multivariate Analysis of Variance; Anderson 2001) is the standard method for testing whether the composition of syllable repertoires differs between mouse populations. It is borrowed from community ecology, where it was developed to compare species compositions across sites — syllable type proportions map directly onto species abundance proportions.

The method works by:
1. Computing per-animal syllable proportion vectors (one entry per syllable type)
2. Building a Bray-Curtis dissimilarity matrix between all pairs of animals
3. Testing whether within-group distances are smaller than between-group distances via permutation of group labels

The key advantages for USV repertoire comparison are:
- **No distributional assumptions** — proportional data often violate normality, making parametric MANOVA inappropriate
- **Works with any distance metric** — Bray-Curtis is standard for compositional data, but could also use Jensen-Shannon divergence
- **Handles high-dimensional compositions** — works whether DeepSqueak yields 5, 20, or 50 syllable types
- **Null hypothesis is clear** — no difference in syllable composition between wild and lab populations

Available in Python via `scikit-bio` (`skbio.stats.distance.permanova`). The input is a `DistanceMatrix` object and a grouping vector. Typical permutation count is 999 or 9999.

PERMANOVA provides one dimension of repertoire comparison — overall compositional differences. It is complemented by [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] for per-population diversity (a univariate measure), [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] for pairwise distributional distance, and [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] for sequential syntax differences. Together, these four methods form a comprehensive statistical toolkit for repertoire comparison. The permutation-based statistical philosophy is shared with the broader [[null model comparison framework produces z-scores rank-based p-values and effect sizes as the publishable statistical output]] used in the information-theoretic analysis stream.

---

Source:
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23)
- Anderson, M.J. (2001). "A new method for non-parametric multivariate analysis of variance."

Relevant Notes:
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question this method directly tests
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- complementary latent-space approach vs this categorical approach
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- the syllable types whose proportions feed the dissimilarity matrix
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- complementary per-population diversity measure
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- alternative pairwise distributional distance
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- sequential syntax comparison complements this compositional test
- [[null model comparison framework produces z-scores rank-based p-values and effect sizes as the publishable statistical output]] -- shared permutation-based statistical philosophy
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] -- the population-specific differences PERMANOVA is designed to detect and quantify

Topics:
- [[experimental-methods]]
- [[classification-methodology]]
