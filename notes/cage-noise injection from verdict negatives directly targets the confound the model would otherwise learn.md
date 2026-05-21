---
description: "Augmenting training spectrograms with additive cage noise sampled from verified-negative patches forces the encoder to be invariant to exactly the noise distribution that would otherwise become a discriminative shortcut"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[training-methodology]]"
  - "[[classification-methodology]]"
---

# Cage-noise injection from verdict negatives directly targets the confound the model would otherwise learn

Standard audio augmentation strategies — SpecAugment time/frequency masking, pitch shift, time stretch, random crop — make the model robust to generic perturbations in the input. They do not specifically target *cage acoustics*. If the encoder is learning recording-environment signatures (as the 2026-05-18 VAE comparison memo established), generic augmentation may slow that learning but not prevent it: the cage signature is consistent across many spectrograms of the same cohort, so the encoder can average over augmentation noise to recover it.

A more direct corrective is to inject *actual cage noise* into training spectrograms during augmentation. The lab CNN classifier plan proposes sampling additive noise patches from the 845 hand-curated lab 131204 verdict negatives — patches that the human reviewer confirmed contain no USV signal, only the cage's characteristic ambient noise + tonals. With probability ~0.25 per training sample, a random verdict-negative patch is sampled and blended additively into the spectrogram. The encoder now sees the same syllable surrounded by varying cage noise from sample to sample, breaking the spurious correlation between cage signature and class label.

This is an asymmetric attack on the confound. Generic augmentations attack a confound by making it noisy; cage-noise injection attacks the confound by making the confound's signature *itself* a noise dimension that the encoder must ignore to discriminate classes. The two mechanisms compose — generic augmentation gives broad invariance, cage-noise injection gives targeted invariance. Combined with DANN domain-adversarial training in Phase 1.3, the cage-confound has three independent suppression mechanisms applied to it (cleaning stack at preprocessing time, cage-noise injection at augmentation time, gradient-reversal at optimization time). Defense in depth.

The novelty noted in the plan: cage-noise injection from verdict negatives is not in standard bioacoustic augmentation literature. It's a domain-specific augmentation that leverages our specific dataset asset — the 845 dual-rater negatives. Datasets without high-quality verdict negatives can't apply this technique cheaply.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — the confound this augmentation targets
- [[DANN gradient-reversal enforces invariance without per-batch domain matching]] — the second invariance mechanism that compounds with this
- [[spectrogram SpecAugment-style augmentation with frequency and time masking improves transformer generalization]] — the generic augmentation this complements

Topics:
- [[training-methodology]]
- [[classification-methodology]]
