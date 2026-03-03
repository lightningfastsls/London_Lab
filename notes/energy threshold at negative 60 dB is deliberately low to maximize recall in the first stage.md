---
description: "Energy detector uses -60 dB threshold to catch even faint USVs, accepting high false positive rate for downstream CNN filtering"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
---

# Energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage

The energy detector threshold is set to -60.0 dB, deliberately low to maximize recall in the first detection stage. This permissive threshold means many non-USV candidates will be generated, but since [[two-stage detection uses permissive energy detector followed by CNN precision filter]], the CNN classifier will reject most false positives. The energy mode is set to "peak" -- using the maximum energy value within the frequency band per frame rather than the mean, since [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]]. A maximum bandwidth filter of 20 kHz additionally rejects broadband noise artifacts.

The -60 dB threshold has implications beyond the CNN stage. Since [[transformer-first then VQ-VAE avoids forcing premature discretization]], the transformer trains on bout-level spectrograms assembled from detection output — meaning this threshold ultimately determines what enters the representation learning pipeline. USVs below the threshold are absent from VQ-VAE codebook learning, which matters because [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]].

---

Source:
- DECISIONS.md (ADR-003) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the architectural context
- [[peak energy mode detects narrow-band USVs better than mean energy across the frequency band]] -- why peak mode
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- companion noise filter
- [[recall versus precision tradeoff in two-stage USV detection]] -- the -60 dB choice is a specific instantiation of maximizing recall in the first stage
- [[class weight boosting biases toward recall at the cost of precision]] -- both stages are biased toward recall, compounding the recall emphasis
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the -60 dB threshold determines how much of the USV continuum enters the pipeline for VQ-VAE codebook learning
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- the threshold gates what training data enters the transformer, cascading to VQ-VAE codebook composition
- [[low-amplitude and short-duration USVs are the primary source of false negatives and training bias]] -- calls below threshold are absent from representation learning

Topics:
- [[detection]]
