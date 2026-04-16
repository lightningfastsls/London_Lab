---
description: "three-loss combination with gamma weights 0.5 BCE plus 0.2 KL plus 0.001 pairwise — uncertain USVs prioritized for annotation, an early 2022 implementation of active learning for mouse USVs"
type: method
confidence: proven
created: 2026-04-15
meta_state: current
topics:
  - "[[unsupervised-usv-discovery]]"
  - "[[classification-methodology]]"
---

# AMVOC semi-supervised retraining combines reconstruction KL divergence and pairwise constraint losses with uncertainty-based annotation priority

AMVOC implements a semi-supervised retraining loop where human pairwise constraints (must-link/cannot-link between USV pairs) refine the autoencoder's learned representation. The combined loss function during retraining:

- **BCE reconstruction loss** (gamma_1 = 0.5): maintains autoencoder's ability to reconstruct spectrograms
- **KL divergence clustering loss** (gamma_2 = 0.2): soft cluster assignments pulled toward a sharpened target distribution (similar to DEC — Deep Embedded Clustering)
- **Pairwise constraint loss** (gamma_3 = 0.001): penalizes Euclidean distance between must-link pairs in the latent space

During retraining, K-means cluster centers are updated via gradient descent simultaneously with the autoencoder weights. The small gamma_3 reflects that pairwise constraints are sparse (humans annotate few pairs) and should nudge rather than dominate the representation.

The annotation priority strategy is an early implementation of uncertainty-based active learning: USVs with the lowest maximum probability of belonging to any cluster (most uncertain assignment) are presented first for human annotation. This ensures each human judgment provides maximum information to the model, since [[active learning annotation workflows are the frontier in bioacoustic tools]].

The combined loss is notable because it bridges three objectives that are usually separate: reconstruction quality (autoencoder), cluster structure (K-means/DEC), and human knowledge (constraints). The risk is that the three losses may conflict — reconstruction wants faithful spectrograms while clustering loss wants separable embeddings — but the weight hierarchy (0.5 > 0.2 > 0.001) ensures reconstruction dominance.

For our pipeline, this semi-supervised approach could complement the uncertainty revealed by [[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]]: rather than forcing discrete clusters, we could use pairwise constraints to ask whether specific boundary USVs belong together.

---

Source: [[amvoc-stoumpou-2022-deep-read-2026-04-15]]

Relevant Notes:
- [[active learning annotation workflows are the frontier in bioacoustic tools]] — the broader trend AMVOC's loop exemplifies
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] — our own implementation of the active learning principle
- [[AMVOC 4-stage feature pipeline reduces 1280 bottleneck features through variance thresholding StandardScaler and PCA to cluster-ready dimensions]] — the features the semi-supervised loop refines
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] — both address stable representation update during iterative refinement: AMVOC uses weighted multi-loss (0.5/0.2/0.001) to balance competing objectives, while VQ-VAE uses four-mechanism defense to prevent codebook degeneration; the challenge is the same — preventing catastrophic drift when updating learned representations

Topics:
- [[unsupervised-usv-discovery]]
- [[classification-methodology]]
