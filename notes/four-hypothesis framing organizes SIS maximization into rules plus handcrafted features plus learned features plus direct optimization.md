---
description: "organizing method comparison around distinct mechanisms — not around papers — surfaces which axis of the problem each method actually tests"
type: method
confidence: likely
conditions:
  - "applies when multiple candidate methods claim to solve the same problem via different mechanisms"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[classification-methodology]]"
  - "[[unsupervised-usv-discovery]]"
---

# four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization

Maximizing SIS for mouse USVs has four structurally distinct mechanisms — each a testable hypothesis about where sequential structure lives:

1. **Rules** — pitch-jump heuristics (iMSA): sequential structure lives in biologically motivated discontinuity detection
2. **Handcrafted features** — FM+AM ridge vectorization (Oren): sequential structure lives in continuous pitch-contour shape
3. **Learned features** — autoencoder bottleneck + PCA (AMVOC): sequential structure lives in reconstruction-relevant axes
4. **Direct optimization** — SIM label-space search: sequential structure is discoverable by iterative label reassignment independent of features

**Why this framing matters:** Paper-anchored planning (pick one paper, implement it) smuggles in the implicit claim that its mechanism is the right one. Hypothesis-anchored planning treats each mechanism as a falsifiable axis of the problem. The four branches are not competing for adoption — they are complementary tests that triangulate *which* axis carries the signal.

This prevents a common failure mode where the most recently ingested paper wins by recency bias. It also prevents collapse of the comparison: if only iMSA is built and scores 0.15 bits, we learn what iMSA can do but not whether that was the right hypothesis. Running all four lets us distinguish "rules are the right abstraction" from "mouse USVs simply have 0.15 bits of sequential structure regardless of method."

**General principle:** When multiple methods purport to solve the same problem, frame each by the *mechanism* it tests, not by its paper of origin. The benchmark ranks mechanisms; the papers are implementations.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy]] — the rules hypothesis
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] — the handcrafted-features hypothesis
- [[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]] — the learned-features hypothesis
- [[Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences]] — the direct-optimization hypothesis
- [[decision-gate methodology requires computing free SIS baselines before committing to feature engineering]] — the gate that determines whether the four-hypothesis benchmark is worth building at all
- [[Oren marmoset ridge vectorization requires re-engineering not parameter tuning when adapted to mouse USVs because duration frequency band harmonics SNR and absolute-pitch relevance all differ]] — concretizes hypothesis 2: porting the handcrafted-features mechanism to mouse data is itself substantial work
- [[autoencoder bottleneck plus PCA extracts concepts because reconstruction forces the model to preserve axes of variation that matter]] — the mechanism by which hypothesis 3 produces clusterable features
- [[low-dimensional intrinsic manifold argues for learned features rather than against them because bottleneck compression is how you find low-dim structure]] — defends hypothesis 3 from premature rejection on intrinsic-dimensionality grounds
- [[SIM optimization is structurally feature-independent so if it wins the finding is that labels matter more than features for sequential prediction]] — interpretation rule for hypothesis 4: a SIM win means representation matters less than partition

Topics:
- [[classification-methodology]]
- [[unsupervised-usv-discovery]]
