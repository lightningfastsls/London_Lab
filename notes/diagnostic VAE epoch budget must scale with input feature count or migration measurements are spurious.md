---
description: "Smoke-test VAE epoch defaults calibrated on 32x32 synthetic data silently produce degenerate K-NN diagnostics at 227x227 real spectrograms — under-trained latents give random geometry that any +2 sigma cohort injection saturates to migration=1.0."
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification-methodology]]"
  - "[[representation-learning]]"
  - "[[signal-processing]]"
---

# Diagnostic VAE epoch budget must scale with input feature count or migration measurements are spurious

When a cleaning-validation diagnostic uses a learned representation (a VAE encoder) to measure cohort migration, the training schedule of that representation is part of the diagnostic specification — not a tuning knob. Module 18.2a's cleaning gate produced contradictory verdicts on the same cleaned spectrograms (all 4 layers active, no upstream changes) purely as a function of `--n-epochs`: with 4 epochs (the smoke-test default calibrated on 32x32 synthetic data, 1,024 features) `notch_injection_migration = 1.00` and the gate said NO-GO; with 32 epochs on the same 227x227 real spectrograms (~51,529 features, ~50x the smoke regime) migration dropped to 0.00 and the gate said GO. The non-VAE raw-pixel diagnostics in the same runs (`per_band_cohens_d`, `raw_pixel_pca_d`) were *identical* across both, isolating the cause to VAE training rather than the data.

The mechanism is geometric. With the cleaning stack working as designed, cohorts compress to near-overlap (`per_band_d = 0.07`, `PCA_d = 0`). The residual cohort signal is small. An under-trained VAE has not yet found reconstruction-loss minima, so the encoder's latent geometry is essentially random with respect to that small signal. A coherent +2 sigma notch injection on cohort B then pushes *all* injected samples to one side of whatever weak decision boundary the random latent happens to expose, saturating migration at 1.0. At sufficient epochs the latent stabilizes around real reconstruction structure, the cleaned cohorts sit in the same manifold region, and the same +2 sigma injection produces a small two-sided migration — the correct "cleaning worked" signal. So a high migration number from an under-trained encoder is not evidence of dirty data; it's evidence that the diagnostic is not yet measuring anything.

Two operational rules follow. **(1) Heuristic:** training compute should scale with feature dimensionality; `epochs ∝ √(feature_count)` is a defensible rule of thumb for fixed batch size and architecture, so 32x32 → 227x227 (~7x linear, ~50x pixels) needs at least ~7x more epochs to reach comparable latent stability. **(2) Safety net:** add a convergence check (loss-plateau detection) that re-trains when reconstruction loss is still falling at the configured epoch budget — the convergence check is the principled fix; the heuristic multiplier is the cheap-and-effective fallback. The deeper lesson is that smoke-test defaults are not pre-registered diagnostic schedules: calibrating thresholds on small synthetic data and applying them at production feature counts without epoch adjustment is a silent specification violation, exactly the failure mode [[falsifiable cleaning gates with numeric thresholds beat vibes-based judgment]] is designed to prevent — except the falsifiable gate only stays falsifiable when the *measurement* underneath it remains trustworthy. This is also why [[notch-injection migration measures cleaning quality better than passive cohort sampling]] requires a converged encoder to discriminate: a random latent answers "yes, migration is high" to any injection, which is the same failure mode as a passive test always answering "yes, cohorts differ." And it answers [[does the 4-layer cleaning stack drop notch-injection migration from 91.7 to under 30 percent]] in a qualified way: yes, but only when the VAE encoder used to measure migration is trained long enough to have a non-random latent geometry.

---

Source: Module 18.2a cleaning-validation gate runs on `worktree-lab-cnn-classifier-plan` — commits `9e4540d9`, `0a47015e`, `a515e2f6`; audit trail `docs/handoffs/cleaning-validation-report.n4-NOGO.md` and `docs/handoffs/cleaning-validation-report.md` Interpretation section; documentation hooks in `train_diagnostic_vae` docstring and `--n-epochs` CLI help text shipped in `a515e2f6`.

Relevant Notes:
- [[notch-injection migration measures cleaning quality better than passive cohort sampling]] — the diagnostic that goes spurious without convergence
- [[falsifiable cleaning gates with numeric thresholds beat vibes-based judgment]] — the threshold (0.30) is only falsifiable when the measurement underneath it is trustworthy
- [[does the 4-layer cleaning stack drop notch-injection migration from 91.7 to under 30 percent]] — the load-bearing question that this lesson qualifies
- [[AMVOC trains for only 2 epochs deliberately because the undercomplete bottleneck acts as implicit regularizer]] — parallel case where short training is correct; the distinction is that AMVOC measures reconstruction-driven clustering on 64x160 input, not cohort migration on 227x227 input with a +2 sigma probe

Topics:
- [[classification-methodology]]
- [[representation-learning]]
- [[signal-processing]]
