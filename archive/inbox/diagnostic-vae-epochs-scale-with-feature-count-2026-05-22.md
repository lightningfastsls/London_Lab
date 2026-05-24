---
title: Diagnostic VAE epoch budget must scale with input feature count
date_captured: 2026-05-22
source_type: methodology
status: applied
---

## Claim

Diagnostic VAE epoch budgets must scale with input feature count. Smoke-test
defaults calibrated on 32×32 synthetic data (1,024 features) silently produce
degenerate K-NN diagnostics when applied to real spectrograms (227×227 =
51,529 features, ~50× the smoke regime).

## Evidence

Module 18.2a cleaning-validation gate (commits `9e4540d9`, `0a47015e`,
`a515e2f6` on `worktree-lab-cnn-classifier-plan`):

| Run | `--n-epochs` | `notch_injection_migration` on `all_layers` | Verdict |
|-----|--------------|---------------------------------------------|---------|
| First | 4 (smoke-test default) | 1.00 | NO-GO (false) |
| Second | 32 | 0.00 | GO (correct) |

Raw-pixel diagnostics (`per_band_cohens_d`, `raw_pixel_pca_d`), which do not
use the VAE, were **identical** between the two runs — isolating the cause to
VAE training.

Audit trail: `docs/handoffs/cleaning-validation-report.n4-NOGO.md` and
`docs/handoffs/cleaning-validation-report.md` Interpretation section.

## Mechanism

With cohorts compressed to near-overlap by the full cleaning stack
(`per_band_d = 0.07`, `PCA_d = 0`), the residual cohort signal is small. An
under-trained VAE maps that small signal into a noisy latent geometry — the
decoder hasn't yet learned to reconstruct, so the encoder's geometry is
essentially random. A coherent +2σ notch injection on cohort B then pushes
**all** cohort-B encodings to one side of whatever weak decision boundary the
random latent happens to find, giving migration = 1.0.

At sufficient epochs the latent geometry stabilizes around real reconstruction
loss minima. The cleaned cohorts then sit near the same manifold region and
the +2σ injection produces a small, two-sided migration — the correct signal
that cleaning worked.

## Generalization

Applies to any diagnostic that uses learned representations to measure cohort
properties.

- **Rule of thumb**: training compute should scale with feature
  dimensionality. A reasonable heuristic is `epochs ∝ √(feature_count)` for
  fixed batch size and architecture, so 32×32 → 227×227 (~7× linear, ~50×
  pixels) needs at least ~7× more epochs to match latent geometry stability.

- **Safety net**: add a convergence check (loss-plateau detection) that
  re-trains when reconstruction loss is still falling at the configured epoch
  budget. The convergence check is the principled fix; the heuristic
  multiplier is the cheap-and-effective fix.

- **Pre-registration discipline**: when a diagnostic uses a learned model,
  the model's training schedule is part of the diagnostic's specification.
  Calibrating thresholds on 32×32 smoke data and then applying them at
  227×227 without epoch adjustment is a silent specification violation.

## Proposed wiki-links

- [[notch-injection migration measures cleaning quality better than passive cohort sampling]]
  — measures it, but only when the latent is converged.
- [[falsifiable cleaning gates with numeric thresholds beat vibes-based judgment]]
  — the threshold (0.30) is only falsifiable when the measurement is
  trustworthy, which requires convergent latent geometry.
- [[does the 4-layer cleaning stack drop notch-injection migration from 91.7 to under 30 percent]]
  — the answer is "yes, but only when the VAE is trained long enough."

## Documentation hooks

- `train_diagnostic_vae` docstring (in `scripts/cnn_cleaning_validation.py`)
  now documents the epoch-scaling rule.
- `--n-epochs` CLI help text describes the smoke-vs-real trade-off.
- Both changes shipped in commit `a515e2f6` (18.1.x carve-out).

## Source

- `docs/handoffs/cleaning-validation-report.md` Interpretation section
- Module 18.2a IMPLEMENTATION_PROGRESS entry (2026-05-22)
- Commits `9e4540d9`, `0a47015e`, `a515e2f6` on `worktree-lab-cnn-classifier-plan`
- Master-reviewer report: `docs/reviews/cnn-download-vocalmat-sample-review.md`
  (this discovery was originally flagged as Finding #1 WARNING and addressed
  in the carve-out commit)
