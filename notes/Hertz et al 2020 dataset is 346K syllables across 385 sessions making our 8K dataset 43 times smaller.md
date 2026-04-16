---
description: "346,632 syllables / 33,481 sequences / 385 C57BL/6 courtship sessions; our ~8K detections are 43x smaller — limits viable depth, Nc, and SIM convergence"
type: baseline
confidence: proven
conditions:
  - "C57BL/6 strain"
  - "male-female courtship interaction"
meta_state: current
topics:
  - "[[wild-lab-vocal-comparison]]"
  - "[[classification-methodology]]"
---

# Hertz et al 2020 dataset is 346K syllables across 385 sessions making our 8K dataset 43 times smaller

The Hertz et al. (2020) dataset specification:

| Property | Value |
|----------|-------|
| Mouse strain | C57BL/6 (inbred lab strain) |
| Context | Male-female courtship interaction |
| Recording system | Avisoft UltraSoundGate, 250 kHz, 16-bit |
| Sessions | 385 (349 London lab + 36 mouseTube) |
| Total recording time | ~78 hours |
| Total syllables | 346,632 |
| Total sequences | 33,481 |
| ISI threshold for sequences | 160 ms |

**Our current scale:** ~7,575 (5970) + ~456 (3452) + 9252 batch = roughly 8,000–10,000 detections total. This is approximately **43× smaller** than Hertz's dataset.

**Practical consequences of the size difference:**
1. **Suffix tree validity:** The <10% zero-probability criterion (needed for reliable SIS) limits Hertz to Nc≤64 at depth 1. Our data will hit this limit at a much lower Nc — possibly Nc≤8-12 at depth 1, and Nc≤4-6 at depth 2.
2. **SIM viability:** SIM converged on 173K training syllables (50% of 346K). With ~4K training syllables (50% of 8K), SIM's perturbation-rejection cycles will have high variance — each reassignment changes labels for a tiny fraction of the training data, making ΔSIS noisy. SIM may require aggregating more recordings.
3. **Depth-2 analysis:** Depth-2 SIS requires observing (label, label, label) triplets. With 7 labels and ~8K sequences, expected count per triplet = 8000 / 7^3 ≈ 23. Marginal; many triplets will be unobserved.
4. **Error bars:** Hertz uses 25 × 60% bootstrap. Our 60% subsamples would have ~4,800 syllables each, still barely adequate for depth 1.

**Important strain difference:** Hertz's C57BL/6 data (inbred lab strain) vs our wild-derived mice. The lower SIS values we observe (0.093 bits vs 0.22 bits) partly reflect this difference — wild-derived mice have more variable vocalization patterns, which reduces sequential predictability. See [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]].

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[suffix trees store empirical transition counts for Markov models and require less than 10 percent zero-probability tuples for reliable SIS estimation]] -- the data criterion that this size difference makes binding
- [[Syntax Information Maximization SIM algorithm iteratively perturbs cluster centroids to maximize SIS on training sequences]] -- SIM needs ~173K training syllables; 4K is marginal
- [[ISI threshold of 160ms defines sequence boundaries in Hertz 2020 mouse courtship vocalization analysis]] -- this parameter determines sequence count from the syllable total
- [[wild mice show more diverse USV repertoires than lab mice as preliminary evidence for courtship vocal degradation]] -- strain difference alongside size difference explains lower SIS
- [[Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data]] -- the values our dataset is compared against

Topics:
- [[wild-lab-vocal-comparison]]
- [[classification-methodology]]
