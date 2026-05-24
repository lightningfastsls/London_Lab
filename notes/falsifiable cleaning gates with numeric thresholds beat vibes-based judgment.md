---
description: "Concrete pass criteria like notch-injection migration under 30 percent convert 'is this clean enough?' from subjective judgment into testable empirical claim — gate either passes or you iterate"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[experimental-methods]]"
  - "[[signal-processing]]"
---

# Falsifiable cleaning gates with numeric thresholds beat vibes-based judgment

Phase 1.0 of the lab CNN classifier plan defines four numeric pass criteria that the cleaned spectrogram pipeline must meet before any classifier training begins: notch-injection migration <30%, max per-10kHz-sub-band Cohen's d <0.3, K-NN same-cohort rate <0.85, raw-pixel PCA PC1 Cohen's d <1.5. Each threshold is anchored to a measured baseline on uncleaned data (91.7%, 0.4–2.0, 0.98–1.0, +5.83 respectively) so the gate is calibrated against the actual problem rather than against arbitrary numbers.

The methodological move is converting the question "is this cleaning enough?" — which has no answer because everyone has different intuitions about "enough" — into "does this cleaning configuration cross threshold X on metric Y?" — which has exactly one answer per run. This eliminates the bargaining failure mode where each subsequent reviewer agrees the cleaning *looks* better than before but no one can decide whether it's good *enough* to proceed. The plan explicitly states it will "fail honestly (not silently) if Phase 1.0 cleaning gate doesn't pass," which is the entire point: a falsifiable gate has the property that you can lose.

This is the same pattern Popper named for science generally — claims become useful when they can be wrong in a specific, observable way. Applied to system engineering, it means specifying exit criteria before starting work, not as you go. The time to write down "what does success look like numerically?" is before the work starts, when you can still reason about whether the criteria are calibrated correctly. Writing them after the fact invites motivated reasoning toward whatever the result happens to be.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — the problem the gate is calibrated against
- [[notch-injection migration measures cleaning quality better than passive cohort sampling]] — the gate's primary diagnostic
- [[adversarial builder-critic separation catches silent performance risks that pass all tests]] — related pattern: building escape hatches before they're needed
- [[diagnostic VAE epoch budget must scale with input feature count or migration measurements are spurious]] — the threshold stays falsifiable only when the measurement underneath it is trustworthy, which requires latent convergence

Topics:
- [[classification-methodology]]
- [[experimental-methods]]
- [[signal-processing]]
