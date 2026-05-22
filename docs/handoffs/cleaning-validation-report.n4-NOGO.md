# Cleaning Validation Report (Module 18.1)

Generated in 118.87s. Cohort sample sizes:

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
| knn_same_cohort_rate | 0.7287 | < 0.85 | PASS |
| raw_pixel_pca_d | 15.8697 | < 1.50 | FAIL |

### Layer config: `mad_only`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 0.0050 | < 0.30 | PASS |
| per_band_cohens_d | 0.5356 | < 0.30 | FAIL |
| knn_same_cohort_rate | 0.8947 | < 0.85 | FAIL |
| raw_pixel_pca_d | -10.2232 | < 1.50 | FAIL |

### Layer config: `zscore_only`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 0.0050 | < 0.30 | PASS |
| per_band_cohens_d | 0.5032 | < 0.30 | FAIL |
| knn_same_cohort_rate | 0.9480 | < 0.85 | FAIL |
| raw_pixel_pca_d | -9.1535 | < 1.50 | FAIL |

### Layer config: `all_layers`

| Diagnostic | Value | Threshold | Verdict |
|---|---|---|---|
| notch_injection_migration | 1.0000 | < 0.30 | FAIL |
| per_band_cohens_d | 0.0696 | < 0.30 | PASS |
| knn_same_cohort_rate | 0.3333 | < 0.85 | PASS |
| raw_pixel_pca_d | 0.0000 | < 1.50 | PASS |

## Go/No-Go Decision

**NO-GO** — One or more diagnostics fail under the full cleaning stack. Module 18.2 is blocked until cleaning is iterated.

Failed criteria:
- `notch_injection_migration`

---

*Notes*: Per cross-phase constraint **C4**, the soft-notch layer is a no-op for VocalMat and wild_5970 cohorts (no calibrated tonal library). Per **C2**, global MAD operates on the whole spectrogram before windowing. Per **C6**, terminology is 'cage' (physical recording environment), not 'rig' (compute hardware).