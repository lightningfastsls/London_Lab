---
description: "JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) applied to syllable type proportions gives a [0,1] distance between population repertoires"
type: method
confidence: likely
meta_state: current
topics:
  - "[[experimental-methods]]"
  - "[[classification]]"
---

# Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations

Jensen-Shannon Divergence (JSD) applied to syllable type proportions measures how different two populations' repertoire distributions are. The formula is JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q) is the mixture distribution. When using log2, JSD is bounded in [0, 1]: JSD = 0 means identical distributions, JSD = 1 means completely non-overlapping.

The advantages over raw KL divergence are symmetry (KL(P||Q) != KL(Q||P), but JSD is symmetric) and boundedness (KL diverges when support sets differ). For syllable proportions, JSD directly answers "how different are the wild and lab repertoire distributions?"

This is a complementary application to the existing note on [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]], which discusses JSD in the VAE latent space. Here, JSD is applied to categorical syllable type proportions from DeepSqueak classification — a simpler, more interpretable application that works with the pre-VQ-VAE bridge pipeline. The two uses can be compared: does JSD on categorical types and JSD in latent space agree about the magnitude of wild-vs-lab differences?

---

Source:
- inbox/raven-deepsqueak-classification-bridge-plan.md (2026-02-23)

Relevant Notes:
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- latent-space JSD (complementary approach)
- [[PERMANOVA on Bray-Curtis dissimilarity is the standard ecological method for testing whether syllable repertoire compositions differ between populations]] -- alternative multivariate test
- [[Shannon entropy quantifies USV repertoire diversity with higher values indicating more evenly distributed syllable usage]] -- per-population diversity vs between-population distance
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- captures sequential structure that proportional measures miss
- [[wild versus lab mouse USV comparison tests whether domestication altered vocal repertoires]] -- the research question this metric serves
- [[DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method]] -- the syllable categories whose proportions become the P and Q distributions

Topics:
- [[experimental-methods]]
- [[classification-methodology]]
