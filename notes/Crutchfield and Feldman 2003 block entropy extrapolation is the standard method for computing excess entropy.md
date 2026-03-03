---
description: "Excess entropy E = I(past;future) is computed by extrapolating the linear regime of block entropy H(L) versus block length L to intercept"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
---

# Crutchfield and Feldman 2003 block entropy extrapolation is the standard method for computing excess entropy

Excess entropy quantifies the total amount of structure in a stationary stochastic process — formally, it is the mutual information between the semi-infinite past and the semi-infinite future: E = I(past; future). This single scalar captures all the predictive structure in the sequence, integrating across all time scales. A memoryless (i.i.d.) process has E = 0, a k-th order Markov process has finite E proportional to k, and processes with long-range correlations can have large or even divergent excess entropy.

Crutchfield and Feldman (2003) established the standard computational approach for estimating excess entropy from finite data. The procedure is: compute block entropies H(L) — the Shannon entropy of blocks of L consecutive symbols — for increasing block lengths L. For sufficiently large L, the block entropy enters a linear regime where H(L) approximately equals h*L + E, where h is the entropy rate and E is the excess entropy. The entropy rate h appears as the slope of this linear region, because each additional symbol adds h bits of uncertainty once all transient correlations have been accounted for. The excess entropy E appears as the y-intercept, representing the total "overhead" of structure that goes beyond the per-symbol entropy rate.

In practice, one fits a line to H(L) in the regime where it appears linear and extrapolates to L=0 to obtain E. The challenge is identifying where the linear regime begins — too-short blocks still contain transient correlations, while too-long blocks suffer from finite-sample estimation noise. The existing `excess_entropy` function in `sequence_analysis.py` implements a related approach via entropy rate convergence, but it lacks the specific Crutchfield and Feldman extrapolation methodology and the explicit block entropy fitting procedure. Implementing the canonical method would strengthen the scientific rigor of the excess entropy analysis described in [[excess entropy measures long-range structure complexity in discrete code sequences]].

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[excess entropy measures long-range structure complexity in discrete code sequences]] -- the analysis framework that this computational method serves
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- entropy rate (the slope h) is a byproduct of the same block entropy computation
- [[Miller-Madow correction compensates for finite sample bias in entropy rate estimation]] -- block entropy estimation at large L requires bias correction due to sparse sampling of long blocks
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- excess entropy values must be compared against null model baselines to confirm meaningful structure

Topics:
- [[representation-learning]]
