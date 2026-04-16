---
description: "SIS_norm = SIS / log2(Nc) makes SIS flat across cluster counts from 4 to 64 in Hertz 2020 — raw SIS increase with Nc is proportional to label bits, not extra structure"
type: method
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
---

# SIS normalized by log2 of cluster count removes dependency on number of labels enabling cross-Nc comparisons

Raw SIS increases with the number of clusters Nc because more labels allow finer-grained sequential distinctions, and the maximum possible SIS is bounded by H(X_n) which grows with Nc. To compare classification schemes with different Nc, Hertz et al. define:

SIS_norm = SIS / log2(Nc)

On their C57BL/6 courtship data, normalized SIS showed **no clear dependency on Nc** across the range Nc = 4 to 64 (Fig. 5). This means the raw SIS increase with more clusters is roughly proportional to log2(Nc) — the bits needed to encode the labels — rather than reflecting proportionally more sequential structure being captured.

**Implication:** When comparing our 7-type Scattoni (MI = 0.093 bits) to Hertz's 8-label results (~0.22 bits iMSA), normalizing is important. SIS_norm with 7 types: 0.093 / log2(7) ≈ 0.093 / 2.81 ≈ 0.033. Hertz iMSA: 0.22 / log2(8) = 0.22 / 3 ≈ 0.073. Our normalized SIS is roughly half the iMSA result, which is meaningful given we're on wild-derived mice and a 43× smaller dataset.

**Corollary:** When iMUPET reaches Nc = 32 clusters, it surpasses iMSA at Nc = 8 in raw SIS — but at 4× label complexity. The normalized comparison shows this is partly an artifact of encoding more bits, not purely better sequential structure capture.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context]] -- the raw SIS formula being normalized
- [[mutual information rate at varying lags measures temporal dependency strength within USV code sequences]] -- our MI = 0.093 bits at lag 1 is the SIS depth-1 equivalent for our data
- [[Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data]] -- the absolute values before normalization
- [[Hertz et al 2020 dataset is 346K syllables across 385 sessions making our 8K dataset 43 times smaller]] -- the comparison context

Topics:
- [[classification-methodology]]
