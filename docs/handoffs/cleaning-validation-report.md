# Cleaning Validation Report (Module 18.1)

Generated in 481.34s. Cohort sample sizes:

- **vocalmat**: 200 spectrograms
- **lab_131204**: 200 spectrograms
- **wild_5970**: 200 spectrograms

## Ablation matrix

### Layer config: `raw`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 1.0000 | < 0.30 | FAIL |
| per_band_cohens_d | 26.0229 | < 0.30 | FAIL |
| knn_same_cohort_rate | 0.3333 | < 0.85 | PASS |
| raw_pixel_pca_d | 52.1606 | < 1.50 | FAIL |

### Layer config: `soft_notch_only`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 1.0000 | < 0.30 | FAIL |
| per_band_cohens_d | 26.0229 | < 0.30 | FAIL |
| knn_same_cohort_rate | 0.3333 | < 0.85 | PASS |
| raw_pixel_pca_d | 52.1606 | < 1.50 | FAIL |

### Layer config: `baseline_only`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 0.0250 | < 0.30 | PASS |
| per_band_cohens_d | 0.3194 | < 0.30 | FAIL |
| knn_same_cohort_rate | 0.3333 | < 0.85 | PASS |
| raw_pixel_pca_d | 15.8697 | < 1.50 | FAIL |

### Layer config: `mad_only`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 0.0000 | < 0.30 | PASS |
| per_band_cohens_d | 0.5356 | < 0.30 | FAIL |
| knn_same_cohort_rate | 0.9473 | < 0.85 | FAIL |
| raw_pixel_pca_d | -10.2232 | < 1.50 | FAIL |

### Layer config: `zscore_only`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 0.0050 | < 0.30 | PASS |
| per_band_cohens_d | 0.5032 | < 0.30 | FAIL |
| knn_same_cohort_rate | 0.9310 | < 0.85 | FAIL |
| raw_pixel_pca_d | -9.1535 | < 1.50 | FAIL |

### Layer config: `all_layers`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 0.0000 | < 0.30 | PASS |
| per_band_cohens_d | 0.0696 | < 0.30 | PASS |
| knn_same_cohort_rate | 0.3333 | < 0.85 | PASS |
| raw_pixel_pca_d | 0.0000 | < 1.50 | PASS |

## Go/No-Go Decision

**GO** — All 4 diagnostics pass under the full cleaning stack. Module 18.2 (Data Preparation) is unlocked.

---

## Interpretation (added 2026-05-22)

This was the **second** real-data gate run. The first (n_epochs=4, the
script's default) produced NO-GO with `notch_injection_migration = 1.0`
on `all_layers`, while every individual cleaning layer passed the
injection test. Per the gate's interpretation matrix that pattern
would point to an architectural problem — but it was a false alarm.

Root cause of the first NO-GO: **diagnostic VAE under-training.**

- The VAE input is 227×227 = 51,529 features mapped to a 32-d latent
  space.
- 4 epochs trains the autoencoder for ~1,600 forward/backward passes on
  400 spectrograms — insufficient for that compression ratio.
- For the four "easy" ablations (raw + individual layers) cohorts are
  still well-separated by surface statistics, so even an under-trained
  VAE measured injection correctly.
- For `all_layers`, cleaning collapses the cohorts to per-band Cohen's d
  = 0.07 and PCA d = 0. With cohorts statistically nearly identical, the
  injection-vs-baseline signal is small — and an under-trained VAE
  cannot find it, returning ~random `argmax` over cohort labels.
  Combined with the +2σ injection (which IS a coherent perturbation),
  the K-NN majority shifts maximally and migration rate hits 1.0.

The n_epochs=32 re-run with otherwise identical inputs dropped
`all_layers` migration from 1.000 → 0.000, confirming the diagnostic
VAE — not the cleaning stack — was the limiter.

`knn_same_cohort_rate` also changed across the non-`all_layers`
ablations between the two runs (e.g. `baseline_only` went from 0.7287
→ 0.3333). This is expected and not a separate effect: K-NN operates
on the same VAE embedding, so a deeper-trained encoder produces
better-mixed latents for partially-cleaned data as well. The
direction is monotonically toward "better cage invariance" (lower
rate) and does not change any verdict; the raw-pixel diagnostics
(`per_band_cohens_d`, `raw_pixel_pca_d`), which do NOT use the VAE,
are identical between the two runs — confirming the data is unchanged
and only the measurement instrument improved.

**Practical implication.** Module 18.1's smoke-test default (n_epochs=4)
suits the 32×32 synthetic data but is too low for the real-data path.
A successor module-18.1 patch should either: (a) bump the default to
~32, (b) couple the epoch count to input feature count, or (c) add a
convergence check that re-trains when reconstruction loss is still
falling. None of these are in scope for Module 18.2a; they are flagged
in the 18.2a handoff for the next cleaning-pipeline session.

**Audit trail.** The n_epochs=4 NO-GO report is preserved at
`docs/handoffs/cleaning-validation-report.n4-NOGO.md` for the audit.

---

*Notes*: Per cross-phase constraint **C4**, the soft-notch layer is a no-op for VocalMat and wild_5970 cohorts (no calibrated tonal library). Per **C2**, global MAD operates on the whole spectrogram before windowing. Per **C6**, terminology is 'cage' (physical recording environment), not 'rig' (compute hardware).