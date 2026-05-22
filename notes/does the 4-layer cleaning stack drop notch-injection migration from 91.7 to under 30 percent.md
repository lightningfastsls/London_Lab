---
description: "Open empirical question — Phase 1.0 of the lab CNN classifier plan stakes its entire downstream pipeline on this gate; we have not yet measured the answer on real cohort data"
type: claim
confidence: open
conditions:
  - "Awaiting Phase 1.0 diagnostic runner on real VocalMat + lab 131204 + 5970 spectrograms"
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[signal-processing]]"
---

# Does the 4-layer cleaning stack drop notch-injection migration from 91.7 to under 30 percent

This is the single load-bearing empirical question for the entire lab CNN classifier project. The 2026-05-18 VAE comparison memo measured 91.7% migration on uncleaned wild-vs-lab spectrograms — meaning the encoder treats injected cage tones as cohort identity nearly all the time. The Phase 1.0 cleaning gate sets a pass threshold of <30% on cleaned spectrograms. The 4-layer cleaning stack — soft-notch (lab 131204 only), Boll 1979 baseline subtraction, global MAD normalization, per-recording Z-score — is designed to bridge the 91.7→30 gap. **Whether it actually does is unknown until Phase 1.0 runs on real cohort data.**

Three possible outcomes:
- **Migration ≤ 30%**: the cleaning is sufficient; modules 18.2–18.5 unlock. The plan continues as designed.
- **Migration 30–70%**: cleaning helps but not enough. Iteration required. Likely fixes: broadband whitening (suppresses cage-broad spectral signatures, not just tonals), per-band Z-score (catches sub-band differences global Z-score misses), or noise floor matching (subtracts cohort-specific noise floors uniformly).
- **Migration > 70%**: cleaning barely moves the needle. The cleaning architecture itself is wrong, not the parameters. Likely cause: cage acoustics aren't dominated by the spectral features the stack targets — they're dominated by reverberation, temporal patterns, or wide-band features that survive all four layers intact. Phase 2 starts.

The question is open because the 18.1 module code exists (per the implementation agent's 2026-05-21 report — 31/31 tests pass on synthetic data) but has not been executed on the real 12 GB VocalMat dataset (deferred to Module 18.2) plus our own lab + wild cohorts. The first real-data run is the answer.

Notable: the question is binary-with-thresholds, not a continuum. The plan does not accept "migration is somewhat lower than before" as a pass — the gate either crosses 30% or it doesn't. This is the same falsifiable structure documented in [[falsifiable cleaning gates with numeric thresholds beat vibes-based judgment]].

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[notch-injection migration measures cleaning quality better than passive cohort sampling]] — the diagnostic that answers this question
- [[falsifiable cleaning gates with numeric thresholds beat vibes-based judgment]] — why the question is yes/no
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — the underlying problem this test is calibrated against

Topics:
- [[classification-methodology]]
- [[signal-processing]]
