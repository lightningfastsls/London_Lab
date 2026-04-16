---
description: "Inter-syllable intervals >160ms mark sequence boundaries in Hertz 2020; produces 33,481 sequences from 346K syllables in C57BL/6 courtship (~10 syllables/sequence on average)"
type: method
confidence: proven
conditions:
  - "C57BL/6 courtship interaction"
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[experimental-methods]]"
---

# ISI threshold of 160ms defines sequence boundaries in Hertz 2020 mouse courtship vocalization analysis

Hertz et al. (2020) define sequence boundaries using an inter-syllable interval (ISI) threshold of **160 ms**: any gap between consecutive syllables exceeding 160 ms marks the end of one sequence and the start of the next.

**Rationale from Fig. 1:** The ISI distribution is bimodal with peaks at ~20 ms (within-sequence transitions) and ~70 ms (another common within-sequence gap). The threshold at 160 ms falls after the main body of short ISIs, cleanly separating within-sequence syllables from between-sequence gaps.

**Resulting statistics:** 346,632 syllables / 33,481 sequences ≈ **~10.3 syllables per sequence on average**.

**Custom parser:** The sequence parser was custom-developed and made available at the GitHub repository (https://github.com/london-lab/MouseUSVs).

**Relevance to our analysis:** Our sequence analysis uses sequences as the statistical unit for SIS/entropy rate computation. The 160 ms threshold is a reasonable starting point, but our wild-derived mice may have different ISI distributions. The bimodal structure (if present in our data) would guide appropriate threshold selection. Different thresholds will produce different sequence counts and can affect entropy rate estimates.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[Hertz et al 2020 dataset is 346K syllables across 385 sessions making our 8K dataset 43 times smaller]] -- the full dataset context
- [[row-stochastic transition matrices capture sequential structure in syllable sequences testable between populations via Frobenius norm with permutation test]] -- transition matrices also depend on sequence boundary definition
- [[entropy rate decreasing with context length indicates sequential predictability in USV code streams]] -- entropy rate is computed over sequences; boundary choice affects the results

Topics:
- [[classification-methodology]]
- [[experimental-methods]]
