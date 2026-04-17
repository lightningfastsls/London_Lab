---
description: "SIM: multiplicative perturbation of iMUPET centroids in [0.9,1.1] range, accept if ΔSIS>0, force-accept after 5 failures; converges in ~24K iterations to match or exceed iMSA"
type: method
confidence: proven
conditions:
  - "requires fixed-length feature vectors per syllable"
  - "computationally expensive at scale"
meta_state: current
topics:
  - "[[classification-methodology]]"
---

# Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences

The Syntax Information Maximization (SIM) algorithm (Hertz et al. 2020) refines cluster centroids to maximize SIS on a training set. It starts from iMUPET centroids (chosen because iMUPET had second-best SIS, leaving room for improvement).

**Input:** iMUPET centroids (K=8), all syllables vectorized via gammatone filter (16 filters → 2016-dim vectors), labeled sequences split 50/50 into train/test.

**Per-iteration procedure:**
1. Draw perturbation vector V: each of 2016 dimensions ~ Uniform[0.9, 1.1]
2. For each centroid k (1 to K):
   - Apply V element-wise to C[k] → C_perturbed[k]
   - Reassign all training syllables to nearest centroid (with C[k] replaced)
   - Compute ΔSIS = SIS(new labels) - SIS(current labels)
3. Accept the centroid perturbation that maximizes ΔSIS, if ΔSIS > 0
4. If no improvement: increment failure counter; if counter ≥ 5, **force-accept** the best perturbation (even negative) and reset counter

**Convergence:** SIS on training set stabilizes; from Fig. 7a, this occurs around 16,000–24,000 iterations.

**Key design decisions:**
- **Multiplicative perturbation:** scales centroid dimensions, not additive shifts. Same V applied to whichever centroid is being evaluated in a given iteration.
- **Force-accept after 5 failures:** escape mechanism for local optima, analogous to simulated annealing forced acceptance but without temperature schedule.
- **Train/test split by sequences (50/50):** SIM optimizes only on train; test SIS evaluated post-hoc by replaying the exact accepted perturbation chain.
- **One centroid at a time:** evaluates each centroid independently, picks the best ΔSIS, accepts/rejects. Not gradient-based.

**Results:** SIM depth-1 SIS reached ~0.23 bits, surpassing iMSA (~0.22 bits). At depth 2, SIM (~0.25 bits) approached but remained slightly below iMSA (~0.27 bits). Improvement generalized to the test set, confirming new centroids capture real structure.

**Computational cost:** Very high. Each iteration requires full reassignment of ~173K training syllables + suffix tree rebuild + SIS computation. ~24K iterations. Authors acknowledge it is "computationally suboptimal" and presented as proof-of-concept.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context]] -- the objective SIM maximizes
- [[Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data]] -- SIM reaches ~0.23 bits, surpassing iMSA
- [[iMUPET adapted for Hertz 2020 uses 16 gammatone filters producing 2016-dimensional feature vectors per syllable]] -- the feature space SIM perturbs centroids in
- [[suffix trees store empirical transition counts for Markov models and require less than 10 percent zero-probability tuples for reliable SIS estimation]] -- the data structure SIM rebuilds each iteration
- [[Hertz et al 2020 dataset is 346K syllables across 385 sessions making our 8K dataset 43 times smaller]] -- SIM may not be viable on 8K calls; data size constraint
- [[four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization]] -- SIM is hypothesis 4 (direct optimization): the only branch that searches label-space rather than engineering features
- [[SIM optimization is structurally feature-independent so if it wins the finding is that labels matter more than features for sequential prediction]] -- interpretation rule for what a SIM win means in the four-hypothesis benchmark

Topics:
- [[classification-methodology]]
