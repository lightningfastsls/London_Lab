---
description: "iMSA detects frequency discontinuities (pitch jumps) and splits into 8 types via median duration — achieves ~0.22 bits SIS depth-1, highest of three compared algorithms"
type: finding
confidence: proven
conditions:
  - "C57BL/6 courtship"
  - "8 labels"
meta_state: current
topics:
  - "[[classification-methodology]]"
---

# iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy

The iMSA (Mouse Song Analyzer v1.3, Chabout et al. 2015, adapted by Hertz et al.) uses a deterministic rule system:

1. Detect **pitch jumps** = frequency discontinuities in the spectrogram (rapid transitions in the dominant frequency)
2. Classify each syllable by jump count and direction: **Simple** (no jump), **Up**, **Down**, **Multiple**
3. Split each of the 4 categories by median duration → **8 labels** total
4. Preprocessing: gap removal — short silence gaps within a syllable are bridged

No vectorization or clustering is needed. It is the only algorithm that is entirely rule-based and produces deterministic labels.

**Why iMSA wins SIS despite biased label distribution:** iMSA has the most skewed distribution (>50% of USVs fall in "Simple" categories, H₀ ≈ 2.45 bits vs 2.90 for iMUPET). Yet it achieves the highest SIS (~0.22 bits depth-1). The reason: **pitch jumps are a feature that captures sequential dependencies that acoustic similarity alone does not capture**. Duration clustering (median split) also helps: the "Simple-long / Simple-short" pair shows strong same-duration self-repetition, contributing significantly to SIS.

**Important implication:** Rules based on biologically meaningful features (like frequency modulation structure) can outperform unsupervised clustering on sequential structure metrics. This is not obvious — one might expect uniform distributions (like iMUPET at ~2.90 bits) to allow more sequential structure. The SIS metric reveals that label informativeness is about sequential predictiveness, not distributional evenness.

**Per-pair contribution:** Self-repetition (same label following same label) is the dominant SIS contributor for all algorithms. Within iMSA, the same-duration sub-pairs (Simple-long→Simple-long, Simple-short→Simple-short) show above-independence joint probability; cross-duration pairs are below independence.

---

Source:
- hertz_2020_deep_read.md (direct paper reading, 2026-04-15)
- Hertz et al. (2020), *Communications Biology* 3, 333. DOI: 10.1038/s42003-020-1053-7

Relevant Notes:
- [[Hertz 2020 quantitative benchmark iMSA achieves 0.22 bits depth-1 SIS versus iMUPET 0.13 and iVoICE 0.10 on C57BL-6 courtship data]] -- the numbers for this algorithm
- [[self-repetition is the dominant pairwise contributor to SIS in mouse courtship vocalizations]] -- what makes iMSA win
- [[SIS equals entropy rate at depth zero minus entropy rate at depth D giving information gained from sequential context]] -- formula iMSA maximizes by accident
- [[Chabout et al 2015 established that male mice change syllable syntax with social context]] -- iMSA (Mouse Song Analyzer v1.3) is the same tool used in Chabout 2015
- [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]] -- iMSA's pitch-jump rules implicitly perform ridge extraction; ridge detection formalizes the algorithmic step that makes iMSA's features effective for SIS

Topics:
- [[classification-methodology]]
