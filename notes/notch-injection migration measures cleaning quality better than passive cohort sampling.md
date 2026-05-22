---
description: "Painting one cohort's cage tone onto another cohort's spectrograms and measuring whether the modified samples migrate toward the donor cohort in embedding space is an active-perturbation test — much more sensitive than passive observation"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[representation-learning]]"
  - "[[signal-processing]]"
---

# Notch-injection migration measures cleaning quality better than passive cohort sampling

Passive cohort-separation tests like per-band Cohen's d or K-NN same-cohort rate ask whether two cohorts already look different in embedding space. The problem is that they always do, somewhat — two collections of recordings from two animals will differ on dozens of axes, only some of which are cage acoustics. Asking "are the cohorts separable?" sets too low a bar: yes is the default answer regardless of cleaning quality.

The notch-injection test inverts the question. Take cohort B (which lacks cohort A's cage tone). Synthesize cohort A's tone (e.g., 51 kHz lab 131204 cage mask) at +20 dB above local noise floor. Paint that tone onto a copy of cohort B's spectrograms. Run the modified samples through the same encoder. **Migration rate** is the fraction of injected-B samples whose nearest neighbors in embedding space belong to cohort A rather than cohort B.

A clean pipeline produces low migration because the cage tone is suppressed before encoding — the painted samples look like cohort B (which is where they came from) regardless of the tone. A confounded pipeline produces high migration because the tone is intact in the encoding, and the tone is what differentiates the cohorts. The 91.7% baseline on uncleaned wild-vs-lab data quantifies "the encoder is mostly reading the tone." A pipeline that drops migration below 30% has explicitly demonstrated cage tones are NOT the dominant signal it learns.

The pattern generalizes: when a passive test always says yes, look for an active perturbation that can plausibly say no. Migration is exactly the kind of asymmetric test that can — a sufficiently clean pipeline must produce low migration; a confounded one will trivially produce high migration; there's no third option. This makes it the most discriminating of the four Phase 1.0 cleaning gate diagnostics.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[falsifiable cleaning gates with numeric thresholds beat vibes-based judgment]] — the broader pattern this fits into
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — the confound the test targets
- [[adversarial builder-critic separation catches silent performance risks that pass all tests]] — same "active probe beats passive test" insight in a different domain
- [[diagnostic VAE epoch budget must scale with input feature count or migration measurements are spurious]] — the encoder this test depends on must be converged or the migration number is noise, not signal

Topics:
- [[classification-methodology]]
- [[representation-learning]]
- [[signal-processing]]
