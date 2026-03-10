---
description: "The CNN sometimes flags noisy patches because energy patterns within noise can superficially mimic USV spectral structure"
type: finding
confidence: likely
conditions: []
meta_state: current
topics:
  - "[[detection]]"
  - "[[classification]]"
---

# CNN false positives cluster in noisy regions where energy patterns superficially resemble USV structure

While the CNN generally ignores noise well, its most common false positives occur in noisy regions — particularly areas within noise where energy patterns superficially resemble USV structure. The CNN may be picking up on narrowband-like energy concentrations within broadband noise that happen to fall in the USV frequency range. The [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] catches overtly broadband noise at the energy detector stage, but these CNN-level false positives involve more subtle patterns that pass the bandwidth filter and present USV-like features to the CNN. This failure mode suggests the CNN's learned features include spectral shape patterns that can be triggered by noise with the right energy distribution — a structural mimicry effect. These false positives have implications beyond detection accuracy: since detection output feeds transformer training via bout assembly (see [[transformer-first then VQ-VAE avoids forcing premature discretization]]), surviving CNN false positives introduce noise artifacts into the training data, potentially creating spurious codebook entries that represent noise structure rather than genuine USV types.

---

Source:
- Researcher brain-dump on labeling expertise (2026-02-19)

Relevant Notes:
- [[maximum bandwidth filter of 20 kHz rejects broadband noise in energy detection]] -- catches broadband noise upstream, but narrowband-like patterns within noise pass through
- [[noisy USVs are valid positive training samples because the model must learn detection in degraded conditions]] -- training on noisy positives may help reduce this failure mode
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the CNN is the precision stage, so these FPs directly affect system precision
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- the recording environment that generates the noise regions where FPs cluster
- [[electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs]] -- one specific noise pattern; FPs arise from subtler energy patterns that mimic USV structure
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- surviving FPs propagate into transformer training data and may create spurious VQ-VAE codebook entries representing noise structure
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- with K=64 codes, even a small fraction of noise-derived entries reduces the effective vocabulary for real USV variation
- [[including a noise-false-positive class in the USV classifier catches residual detection errors]] -- BootSnap's explicit noise class in the syllable classifier provides a second-pass filter specifically targeting these structural noise mimics
- [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]] -- entropy detection measures spectral complexity directly, catching the narrow-band-like noise patterns that fool energy thresholding with 99.3% precision
- [[HybridMouse CNN plus BiLSTM first combined spatial and temporal features for USV detection outperforming DeepSqueak in low SNR]] -- temporal context from BiLSTM helps disambiguate noisy signals that appear USV-like in individual frames

Topics:
- [[detection]]
- [[classification]]
