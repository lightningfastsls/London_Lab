---
title: Lab-Cleaned USV Syllable Classifier (CNN, VocalMat-anchored) — Plan
date_captured: 2026-05-21
source_type: project-plan
source_path: .claude/worktrees/lab-cnn-classifier-plan/PLAN_lab_cnn_classifier.md
status: unprocessed
---

## Summary

Plan for training a 12-class USV syllable-type CNN on lab-recorded mouse USVs. Uses VocalMat's 12,954-spectrogram labeled dataset (Grimsley 2011 taxonomy, Apache 2.0) as the training base, our existing cleaning pipeline (soft-notch + Boll 1979 baseline subtraction + global MAD + per-recording Z-score), and optionally a small set of user-curated wild-mouse labels (Phase 1.4) for OOD transfer evaluation.

## Motivation

1. VAE comparison (2026-05-18 memo) showed cohort separation was cage-driven, not biological. Any classifier trained on raw spectrograms inherits this confound.
2. VocalMat transfer test failed (κ=0.13) mechanistically because of wild-mouse call-duration mismatch with VocalMat's 0.22s lab-optimized patch. Lab-to-lab is the natural first cut.
3. VocalMat data is Apache 2.0. Our 845 hand-curated lab 131204 verdicts are also lab data. Soft-notch calibrated only for lab 131204. Three constraints point lab-first.

## Key Decisions (Locked)

| # | Decision |
|---|---|
| 1 | License: Apache 2.0 — proceed (verified at github.com/ahof1704/VocalMat/LICENSE) |
| 2 | Taxonomy: VocalMat's 12 classes (Grimsley 2011) — 12,954 examples vs 0 of our own |
| 3 | Architecture: ResNet-18 ImageNet-pretrained (primary); EfficientNet-B0 fallback |
| 4 | Domain adaptation: DANN head (gradient-reversal cage discriminator, Ganin 2015) |
| 5 | Augmentation: SpecAugment + cage-noise injection (from 845 verdict negatives) + pitch/time |
| 6 | Wild data: opt-in Phase 1.4 — user labels ~200 wild calls for OOD eval, not training |

## VocalMat Dataset Facts

- 12,954 training images, single-rater (experienced experimenter). Expect 5–15% label noise in minority classes.
- Test set 4,441 USVs: ≥2 investigators + consensus arbitration (gold-standard; not part of training).
- Source: 5 strains × 3 neonatal ages (P5, P10, P15) × both sexes. ~30 sub-cohorts — good for cage-invariance.
- OSF data at: https://osf.io/bk2uj/
- VocalMat paper: Fonseca et al. 2021, eLife 10:e59161

## Class Distribution (stratification required)

| Class | Count | % |
|---|---|---|
| Noise | 2,083 | 16.1% |
| Step up | 1,814 | 14.0% |
| Down-FM | 1,775 | 13.7% |
| Short | 1,713 | 13.2% |
| Chevron | 1,594 | 12.3% |
| Up-FM | 1,191 | 9.2% |
| Flat | 1,134 | 8.8% |
| Two steps | 701 | 5.4% |
| Step down | 389 | 3.0% |
| Complex | 350 | 2.7% |
| Reverse Chevron | 136 | 1.1% — borderline |
| Multi-steps | 74 | 0.6% — borderline |

Imbalance ratio: 24.5× (Step-up vs Multi-steps). Multi-steps gets ~59 training examples at 80/10/10 split.

## Existing Cleaning Infrastructure (DO NOT REBUILD)

| Component | Location | Status |
|---|---|---|
| Soft-notch tonal removal | src/usv_spectrogram/app/core/notch.py | Production, 15/15 tests pass |
| Boll 1979 baseline subtraction | src/usv_spectrogram/app/core/denoise.py | Production, opt-in via --subtract-baseline |
| Global MAD normalization | src/usv_spectrogram/app/core/sliding_inference.py:389–424 | Always-on in CNN inference |
| Per-recording Z-score | src/usv_spectrogram/postprocessing/normalization.py | Implemented but dormant — wire in |
| Tonal libraries | data/lab_tonal_lines/lab_131204.json | Calibrated for lab 131204 only |

Net new cleaning code: probably zero. Need to enable + wire existing components.

## Phase 1.0 — Cleaning Validation Gate (BLOCKING)

Falsifiable test: re-run VAE diagnostic on cleaned spectrograms across VocalMat + lab 131204 + wild 5970.

Pass criteria (all four):
| Test | Current (raw) | Pass threshold |
|---|---|---|
| Notch-injection migration | 91.7% (ours) / 58.5% (DS) | < 30% |
| Per-10kHz sub-band max Cohen's d | 0.4–2.0 | < 0.3 |
| K-NN same-cohort rate | 0.98–1.0 | < 0.85 |
| Raw-pixel PCA PC1 Cohen's d | +5.83 | < 1.5 |

Deliverables: scripts/cnn_cleaning_validation.py + docs/handoffs/cleaning-validation-report.md
Estimated effort: 3–5 days.

## Phase 1.1 — Data Preparation

Steps: download VocalMat (OSF), resample 300→250 kHz via resample_poly(5,6), apply cleaning stack, generate 227×227 RGB spectrograms at VocalMat STFT params (Hamming 256, hop 128, NFFT 1024 at 250 kHz, 0.22s duration), build stratified 80/10/10 splits with recording-level grouping (no leakage).

Deliverables: data/vocalmat/ manifest CSV, scripts/cnn_prepare_training_data.py, data/lab_cnn_training/{train,val,test}/

## Phase 1.2 — Baseline ResNet-18 Training

Training loop: timm.create_model('resnet18', pretrained=True, num_classes=12), AdamW + cosine LR, ~50 epochs, class-weighted CE, SpecAugment, pitch shift ±10%/time stretch ±20%/random crop ±5%.

Validation pass thresholds:
- Macro F1 on VocalMat val > 0.65
- Per-class precision all ≥ 0.40
- Held-out 845 USV/noise accuracy > 0.80
- Held-out 845 syllable-type entropy mean ≤ log(6)

Deliverables: models/lab_classifier_v1/best.pt, results/lab_classifier_v1/confusion_matrix.png
Estimated effort: 1 week.

## Phase 1.3 — DANN Cage-Confound-Aware Training

Extend 1.2 with: shared ResNet-18 trunk, 12-way class head, N-way cage discriminator head, gradient-reversal with λ schedule 0→1 (Ganin et al. 2015).

Additional pass thresholds:
- Linear cage probe accuracy on encoder features < 65% (vs random ~50% for 2 cages)
- Syllable F1 vs v1 baseline no worse than -0.05

Deliverables: models/lab_classifier_v2/best.pt, results/lab_classifier_v2/cage_invariance_probe.md
Estimated effort: 3–5 days (~50 lines new code over v1).

## Phase 1.4 — Wild Transfer Evaluation (User-Enabled)

Pre-requisite: user labels ~200 wild calls from 5970 using feedback_labeling_queue_folder.md workflow. Stored at data/wild_labels/5970_human_verified.csv.

Domain gap candidates (NOT strain monoculture — VocalMat already has 5 strains):
1. Cage acoustics — mitigated by DANN + cleaning
2. Call-duration distribution — wild 30–50ms median vs lab 30–200ms range (same mechanism as κ=0.13 failure)

Decision point: v2 generalizes meaningfully better than v1 → ship; no → document + defer Phase 2.

## Risk Register (High-priority items)

- Cleaning fails Phase 1.0 gate (High): iterate cleaning, possibly broadband whitening
- Minority classes borderline trainable: Multi-steps n=74, Reverse-Chevron n=136 (High): open question 5 binding before Phase 1.2
- Single-rater training labels, 5–15% noise expected in minority classes (Medium)
- Wild transfer remains poor despite DANN (Medium): NOT strain monoculture; cage acoustics + call-duration are real gaps
- --subtract-baseline interaction with VocalMat spectrogram conventions (Medium): validate PNGs visually in Phase 1.1 ablation
- VocalMat includes neonatal mice (P5–P15); adult wild + lab may differ acoustically (Medium)

## Open Questions (Must resolve before /implement)

1. Patch size: add shorter 0.08s patch variant for Phase 1.4 wild eval? (+1 day cost)
2. Wild labels timing: parallel with Phase 1.1–1.3 or wait for v2 confidence scores?
3. Perch 2.0 sidequest: test bioacoustic-pretrained embeddings + linear probe in Phase 1.2? (~1 day, recommended)
4. DANN domain granularity: 2-cage (lab_131204 vs vocalmat) or per-recording-session (~50–100 cages)? Recommendation: start with 2.
5. Minority-class strategy: (a) keep all 12 + oversample/focal, (b) collapse step-family → 9 classes, (c) drop Multi-steps + Reverse-Chevron → 10 classes. Recommendation: (a) for v1 baseline; revisit if minority precision < 0.20.

## Constraints Captured from Plan

- DO NOT modify corpus.py — this pipeline uses VocalMat's 250 kHz convention internally; canonical 300 kHz invariant stays intact
- DO NOT modify detection CNN (hard_neg_retrain/best_model.pt) — this is downstream classifier only
- New code lives in src/usv_spectrogram/classifier/ — isolated from rest of pipeline
- Soft-notch calibration exists ONLY for lab 131204 — new lab data requires calibrate_lab_tonal_lines.py run
- Production model is models/hard_neg_retrain/best_model.pt — DEPRECATED: models/matched_windows/ and models/production/

## Success Definition

After Phase 1.4:
- 12-class lab classifier with macro F1 > 0.65 on VocalMat val
- Encoder provably cage-invariant (linear probe < 65%)
- Wild transfer eval (~200 calls): meaningful generalization → ship; honest failure → document + scope Phase 2
- Plan fails honestly (not silently) if Phase 1.0 cleaning gate doesn't pass

## References

- Plan file: .claude/worktrees/lab-cnn-classifier-plan/PLAN_lab_cnn_classifier.md
- VAE comparison memo: docs/handoffs/2026-05-18_vae_comparison_memo.md
- Soft-notch design: docs/handoffs/2026-05-11_adaptive-soft-notch.md
- Pre-CNN baseline subtraction: docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md
- VocalMat code/data: https://github.com/ahof1704/VocalMat (Apache 2.0), OSF: https://osf.io/bk2uj/
- Grimsley taxonomy: Grimsley, Hazlett, Wenstrup 2013, PLOS One 8(3):e59428
- DANN: Ganin & Lempitsky 2015, arXiv:1409.7495
- SpecAugment: Park et al. 2019, arXiv:1904.08779
- Perch 2.0: 2025, arXiv:2512.03219
- Memory refs: project_vae_comparison_complete, project_vocalmat_transfer_test, project_clustering_analysis_lab, feedback_cnn_inference_global_mad, feedback_cage_not_rig_terminology
