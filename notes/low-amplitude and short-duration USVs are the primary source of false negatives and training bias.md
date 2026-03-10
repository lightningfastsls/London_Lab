---
description: "Quiet USVs near the noise floor and very short calls are hardest to detect — they were the original source of training bias in earlier CNN iterations"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# Low-amplitude and short-duration USVs are the primary source of false negatives and training bias

Quiet USVs (low amplitude, near the noise floor) and very short calls are the hardest to detect and were the original source of training bias in earlier CNN iterations. The energy detector created this selection bias by filtering out quiet USVs, which then biased CNN training — since [[CNN trained only on energy-detector candidates classifies everything as USV because it never sees normal audio]], the upstream filtering silently shaped the downstream training distribution. Early training sets under-represented faint/short signals, causing the CNN to learn a biased distribution skewed toward louder, longer, more obvious calls. This bias was a key driver behind the active learning approach where [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] — specifically, the "mine" step targets regions where the current model is uncertain, which disproportionately surfaces these faint/short signals. The [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] exists specifically to catch these low-amplitude signals before CNN filtering. This training bias has downstream implications beyond the CNN: since the transformer for VQ-VAE training operates on bout-level spectrograms assembled from detection output, systematically missed faint USVs are absent from representation learning — and since [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]], the missed calls aren't a separate category but the quiet tail of a continuous distribution that the codebook should represent.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- deliberately permissive to catch faint signals
- [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]] -- iterative process to correct the initial training bias
- [[recall versus precision tradeoff in two-stage USV detection]] -- false negatives from faint signals directly affect recall
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- faint USVs at the quiet end of the continuum are underrepresented in VQ-VAE training if detection misses them
- [[codebook collapse prevention requires simultaneous EMA updates plus dead code reset plus k-means init plus L2 normalization]] -- underrepresented call types from detection bias may contribute to codebook collapse, since entries for rare/faint calls receive fewer training updates
- [[end-to-end VQ-VAE on animal vocalizations remains an open research gap as of February 2026]] -- an end-to-end approach trained directly on raw audio could sidestep detection-stage bias, but no published system exists yet
- [[PCEN normalization is more robust than log-mel spectrograms for few-shot bioacoustic scenarios]] -- PCEN's adaptive gain control could improve detection of low-amplitude USVs by normalizing per-channel energy, reducing the amplitude-dependent bias that causes these false negatives
- [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]] -- entropy-based detection may be less sensitive to absolute amplitude since it measures spectral concentration rather than energy magnitude
- [[DCASE few-shot bioacoustic detection improved from F1 40 percent to 70 percent across 2021-2024 challenge editions]] -- few-shot detection from minimal examples could address the training bias by not requiring large labeled datasets that reflect the full amplitude distribution

Topics:
- [[detection]]
- [[classification]]
