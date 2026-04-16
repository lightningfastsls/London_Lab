---
description: "Same-label transitions contribute the most to SIS in all three algorithms — short follows short, long follows long — suggesting duration autocorrelation is the main sequential signal"
type: finding
confidence: proven
conditions:
  - "C57BL/6 courtship"
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[wild-lab-vocal-comparison]]"
---

# self-repetition is the dominant pairwise contributor to SIS in mouse courtship vocalizations

In Hertz et al. (2020), per-pair SIS contributions (pointwise KL divergence between observed and independence-assumed distributions) reveal that **self-repetitions** — the same syllable label following itself — contribute the most SIS for all three algorithms (iMSA, iVoICE, iMUPET).

For iMSA specifically (Fig. 4c):
- "Simple-long" → "Simple-long" and "Simple-short" → "Simple-short" are above the independence baseline
- Cross-duration transitions ("Simple-long" → "Simple-short" and vice versa) are below independence
- This pattern suggests **duration autocorrelation** is the primary sequential signal: consecutive syllables tend to have similar duration

The finding has an earlier pre-labeling confirmation in **Fig. 1g**: adjacent syllable duration correlation r = 0.44 (p < 0.001) at the raw feature level, before any labeling. Short follows short, long follows long. This pre-labeling correlation is the first evidence of temporal structure in the dataset, and SIS formalizes what fraction of that correlation each labeling scheme captures.

**Implication for your data:** Your self-repetition rate in the 7-type Scattoni analysis is likely also the dominant driver of your MI = 0.093 bits. Examining per-pair contributions will show whether any cross-type transitions are informative beyond self-repetition, which would indicate richer structure than simple duration autocorrelation.

**Implication for SIM:** When SIM optimizes centroids to maximize SIS, it will largely be optimizing to better capture duration autocorrelation — a relatively simple acoustic feature. If the main structure is self-repetition, SIM's improvements may be modest on limited data.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data]] -- the full SIS results
- [[SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context]] -- the formula; self-repetitions dominate the sum
- [[iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy]] -- iMSA captures self-repetition most clearly via duration split
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- transition matrices will reveal the same self-repetition dominance

Topics:
- [[classification-methodology]]
