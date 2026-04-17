---
description: "a label-space search algorithm that wins regardless of starting features means sequential structure lives in the label boundary, not the representation — different scientific conclusion than feature engineering"
type: claim
confidence: likely
conditions:
  - "assumes SIM converges to comparable SIS from multiple feature starting points"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[classification-methodology]]"
---

# SIM optimization is structurally feature-independent so if it wins the finding is that labels matter more than features for sequential prediction

SIM (Syntax Information Maximization) is structurally different from every other SIS maximization approach. It doesn't produce new features or new rules — it takes an existing labeling and iteratively perturbs labels to maximize SIS on the training sequences.

**Why this matters for interpretation:** If SIM started from iMUPET features (0.13 bits) and reached iMSA's 0.22 bits — as Hertz 2020 showed — then the iMSA-vs-iMUPET gap was not a feature-engineering gap. It was a label-boundary-search gap that either side could have found.

**The structural claim:** SIM is feature-agnostic. It perturbs labels directly in the label space, using only the sequence structure as fitness signal. The initial feature vector determines which clusters exist; SIM then moves examples between clusters. If SIM reliably reaches the same SIS from multiple feature starting points, the feature space was sufficient — the features just weren't producing the best boundary.

**If SIM wins our benchmark on ANY starting labeling:** The finding is "labels matter more than features for sequential USV prediction." This is a *different* scientific conclusion than "iMSA's features are the right ones." It means the difficulty of SIS maximization on mouse USVs is not representation learning — it is label-space search. The bottleneck is finding the right *partition*, not the right *features*.

**If SIM fails to reach high SIS from any starting point:** The finding is the opposite — sequential structure is genuinely limited regardless of how you slice the label space, and feature engineering alone cannot help.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences]] — the algorithm itself
- [[Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data]] — the result showing SIM from iMUPET matched iMSA
- [[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]] — the metric SIM optimizes
- [[four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization]] — SIM is the direct-optimization branch (hypothesis 4); this note specifies what a SIM-wins outcome would mean
- [[decision-gate methodology requires computing free SIS baselines before committing to feature engineering]] — if the gate passes and SIM wins, the conclusion redirects investment from features to label-space search

Topics:
- [[classification-methodology]]
