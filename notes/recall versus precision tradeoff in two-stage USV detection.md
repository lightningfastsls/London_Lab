---
description: "Designed tradeoff -- energy detector maximizes recall while CNN maximizes precision, with system-level balance"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[experimental-methods]]"
---

# recall versus precision tradeoff in two-stage USV detection

The two-stage detection pipeline embodies a designed tradeoff between recall and precision. The energy detector is tuned for maximum recall (catching all USVs, accepting false positives), while the CNN classifier is tuned for precision (rejecting non-USVs). This is not an accidental compromise but a deliberate architectural choice: since [[two-stage detection uses permissive energy detector followed by CNN precision filter]], each stage can be independently optimized for its role. The system-level tradeoff is controlled by the CNN threshold, which users can adjust post-hoc: lower thresholds favor recall (catch everything), higher thresholds favor precision (only high-confidence detections). This tradeoff interacts with [[3x class weight boost compensates for USV class imbalance in CNN training]], which biases the CNN itself toward recall. The recall/precision balance has downstream consequences beyond detection accuracy: since [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]], low recall means the VQ-VAE codebook only learns from the louder portion of the continuum, potentially missing codes that represent quiet or transitional USV types.

---

Source:
- DECISIONS.md (ADR-003, ADR-005) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the architecture creating this tradeoff
- [[energy threshold at negative 60 dB is deliberately low to maximize recall in the first stage]] -- first stage recall bias
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- second stage recall bias
- [[class weight boosting biases toward recall at the cost of precision]] -- related tradeoff
- [[Goffinet et al 2021 showed USVs form a continuum rather than discrete clusters motivating VQ-VAE discretization]] -- the recall/precision balance determines how completely the USV continuum is sampled for VQ-VAE codebook learning; low recall truncates the quiet tail of the distribution

Topics:
- [[detection]]
- [[experimental-methods]]
