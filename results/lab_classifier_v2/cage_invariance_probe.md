# Module 18.4 — VAE falsifiable cage test on v2 encoder features

- Encoder checkpoint: `results/lab_classifier_v2/best.pt`
- Source (vocalmat) patches: 1000 · Target (lab) patches: 1000
- k (k-NN): 5 · notch fraction: 0.15

| Test | Value | Threshold | Direction | Pass |
|---|---|---|---|---|
| knn_same_cohort_rate | 0.9817 | 0.85 | less_than | ❌ |
| pca_pc1_cohens_d | 1.8710 | 1.5 | less_than | ❌ |
| per_dimension_max_cohens_d | 3.3068 | 0.3 | less_than | ❌ |
| notch_injection_migration_features | 0.0190 | 0.3 | less_than | ✅ |

## Overall: ❌ FAIL — at least one criterion failed

This is the 18.4 VAE gate (stronger than 18.1's raw-spectrogram pass: it tests invariance of the *learned* features after adversarial training). Combined with the linear cage probe (<0.65) and the collapse tripwire (syllable F1 ≥ v1−0.05), all three must pass to ship v2.