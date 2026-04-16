---
description: "RF proximity = proportion of times two calls land in same leaf — nonlinear similarity that revealed family-level vocal conventions in marmosets (Oren 2024)"
type: method
confidence: proven
conditions:
  - requires trained random forest; proximity emerges naturally from ensemble structure
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# random forest proximity from leaf co-occurrence provides a nonlinear vocalization similarity measure sensitive to family-level vocal conventions

Random forest proximity is defined as the proportion of times two data points (calls) land in the same leaf node across all trees in the ensemble. This is a **nonlinear similarity measure** that emerges naturally from the random forest structure — it captures the forest's learned notion of "similar" without requiring an explicit distance metric.

In Oren et al. (2024), proximity was computed between all caller pairs using only calls directed at the same receiver, revealing family-level clustering:

1. Train all-monkeys classifier (100 models x 150 trees)
2. For each pair of callers, compute average proximity between their calls to the same receiver
3. Construct proximity matrix
4. Apply MDS to (1 - proximity) dissimilarity matrix -> family clusters emerge

Statistical validation was strong: within-family proximity was significantly higher than across-family for all three families (Wilcoxon rank sum; z values 47-110, all P < 0.0001). Same-receiver proximity was higher than different-receiver within families (z values 17-976, all P < 0.0001), ruling out nonspecific family convergence.

The families included **unrelated adults paired as mature adults** (families A and C), showing the same within-family similarity as a parent-offspring family (B). This strongly implies vocal learning among adults, not genetic predisposition.

For mouse USV analysis, RF proximity offers an interesting alternative to:
- Cosine similarity on feature vectors
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison|EMD/JSD in latent space]]
- Euclidean distance after PCA

The advantage is that RF proximity is nonlinear and task-adapted — it reflects the classifier's learned feature importance rather than raw feature geometry.

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003.

Relevant Notes:
- [[distributional comparisons in VAE latent space using Earth Mover Distance or Jensen-Shannon divergence may be more biologically meaningful than categorical repertoire comparison]] -- alternative continuous similarity measures
- [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]] -- density-based approach to the same question of vocal structure
- [[Jensen-Shannon divergence on categorical syllable proportions provides a symmetric bounded measure for comparing repertoire distributions between populations]] -- categorical distributional distance
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- vocal convergence between dyad partners (testable via RF proximity) complements Chabout's syntax-level context modulation

Topics:
- [[classification]]
- [[experimental-methods]]
