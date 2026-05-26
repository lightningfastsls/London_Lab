# Plan: Lab-Cleaned USV Syllable Classifier (CNN, VocalMat-anchored)

> **⚠ ERRATA (2026-05-25):** the references below to "845 hand-curated lab
> 131204 verdicts" in `classified_detections_lab_131204_clean.csv` are wrong.
> That file is the 40,787-row clustering working set. The real verdicts are
> 844 rows in `results/lab_*_review/review_index_annotated.csv` (usv/noise
> only — NO Grimsley labels). The held-out validation criteria (USV/noise
> accuracy + entropy) still stand but are DEFERRED; there is no
> Grimsley-macro-F1 held-out gate. See
> `docs/handoffs/2026-05-25_ERRATA_held-out-845.md`.

## Goal

Train a 12-class syllable-type CNN on **lab-recorded** mouse USVs using:
- VocalMat's published 12,954-spectrogram labeled dataset (Grimsley 2011 taxonomy) as the training base,
- Our existing cleaning pipeline (soft-notch + Boll 1979 baseline subtraction + global MAD normalization + per-recording Z-score) to suppress recording-environment confound,
- A small amount of user-curated wild-mouse labels (Phase 1.4) to test domain transfer.

Output: a published model artifact + evaluation report demonstrating that the classifier learns syllable structure rather than recording-environment artifacts.

## Why

1. **The VAE comparison showed cohort separation was cage-driven, not biological** (memo: `docs/handoffs/2026-05-18_vae_comparison_memo.md`). Any classifier trained on raw spectrograms inherits this risk.
2. **The κ=0.13 VocalMat transfer test failed mechanistically because of wild-mouse call durations in a lab-trained model** (`PLAN_test_vocalmat_classifier.md`). The natural first cut is lab-to-lab: train and deploy on lab data where the call-duration distribution matches the model's design assumption.
3. **VocalMat data is Apache 2.0** — usable for derived models with attribution. Our 845 hand-curated lab 131204 verdicts are also lab data. Soft-notch is calibrated only for lab 131204. Three independent constraints point lab-first.

## Decisions Already Made

| # | Decision | Rationale |
|---|---|---|
| 1 | **License: Apache 2.0** — go ahead | Verified at `github.com/ahof1704/VocalMat/LICENSE`; permits derived works incl. commercial |
| 2 | **Taxonomy: VocalMat's 12 classes** (Grimsley 2011) | 12,954 labeled examples vs 0 human-labeled syllable types on our side. Map to Holy & Guo 7 at inference time if needed. |
| 3 | **Architecture: ResNet-18 ImageNet-pretrained** (primary) | Right capacity for ~13k examples; SOTA evidence for bioacoustics at this scale; abundant transfer-learning support via `timm`. Fallback: EfficientNet-B0 (5.3M params, lower overfit risk). |
| 4 | **Domain adaptation: DANN head** (gradient-reversal cage discriminator) | Cheapest cage-invariance method; near-zero engineering cost; well-documented since 2015. |
| 5 | **Augmentation: SpecAugment + cage-noise injection + pitch/time** | Industry-standard for audio CNN training; cage-noise injection from our 845 verdict negatives is novel and directly targets confound. |
| 6 | **Wild data: opt-in Phase 1.4** | User will hand-label a small wild set; used for OOD evaluation, not training (Phase 1). |

## Existing Infrastructure (DO NOT REBUILD)

Audit found we already have most of the cleaning pipeline needed:

| Component | Location | Status | What it does |
|---|---|---|---|
| Soft-notch tonal removal | `src/usv_spectrogram/app/core/notch.py` | Production, 15/15 tests pass | Suppresses cage equipment tonals (51 kHz, 46.58 kHz on lab 131204) |
| Boll 1979 baseline subtraction | `src/usv_spectrogram/app/core/denoise.py` | Production, opt-in via `--subtract-baseline` | Per-bin temporal noise-floor subtraction in linear magnitude. Two modes: `percentile`, `median_envelope` |
| Global MAD normalization | `src/usv_spectrogram/app/core/sliding_inference.py:389–424` | Always-on in CNN inference | Normalizes whole spectrogram once before window cropping |
| Per-recording Z-score | `src/usv_spectrogram/postprocessing/normalization.py` | **Implemented but dormant** | Wire into batch detection for additional cage-invariance |
| Tonal libraries | `data/lab_tonal_lines/lab_131204.json` | Calibrated for lab 131204 only | Empirical facts; new cages require `scripts/calibrate_lab_tonal_lines.py` |

**Net new code required for cleaning: probably zero.** We need to *enable* and *wire* existing components and verify against the falsifiable cage-confound test.

---

## VocalMat Dataset Characteristics (verified 2026-05-20)

**Labeling protocol** (Fonseca et al. 2021, eLife Methods):
- Training set (12,954 images): **labeled by one "experienced experimenter"** — single-rater. No inter-rater agreement reported. No quantified label-noise rate.
- Test set (4,441 USVs across 7 recordings): **≥ 2 investigators with consensus arbitration** by a third investigator — gold-standard. Not part of the 12,954 training images.
- Implication: Expect 5–15% label noise in minority classes. Training-set labels are *better than auto-clustering*, but *not equivalent to* multi-rater consensus.

**Source recordings:** 5 mouse strains (C57Bl6/J, NZO/HlLtJ, 129S1/SvImJ, NOD/ShiLtJ, PWK/PhJ) × 3 ages (P5, P10, P15) × both sexes. ~30 sub-cohorts represented — **good for cage-invariance**: the model sees diverse recording conditions during training, not a monoculture.

**Class counts (stratification must respect this):**

| Class | Count | % | Risk tier |
|---|---|---|---|
| Noise | 2,083 | 16.1% | |
| Step up | 1,814 | 14.0% | |
| Down-FM | 1,775 | 13.7% | |
| Short | 1,713 | 13.2% | |
| Chevron | 1,594 | 12.3% | |
| Up-FM | 1,191 | 9.2% | |
| Flat | 1,134 | 8.8% | |
| Two steps | 701 | 5.4% | |
| Step down | 389 | 3.0% | |
| Complex | 350 | 2.7% | |
| Reverse Chevron | 136 | 1.1% | ⚠ Borderline trainable |
| Multi-steps | 74 | 0.6% | ⚠ Borderline trainable |

**Imbalance ratio: Step-up vs Multi-steps = 24.5×.** With 80/10/10 split, Multi-steps gets ~59 training examples — close to ResNet-18's reasonable lower bound. Open question 5 (below) asks how to handle.

---

## Phase 1.0 — Cleaning Validation Gate (BLOCKING)

### Goal
Verify that enabling our existing cleaning stack (soft-notch + baseline subtraction + global MAD + per-recording Z-score) suppresses cage-confound enough to make a syllable classifier honest.

### The Falsifiable Test

Re-run the VAE diagnostic from `vae-pytorch-pivot` worktree on **cleaned** spectrograms across (a) VocalMat's lab data, (b) our lab 131204, (c) our wild 5970.

Pass criteria (all four must hold):

| Test | Current (raw) | Pass threshold (cleaned) |
|---|---|---|
| Notch-injection migration | 91.7% (ours) / 58.5% (DS) | **< 30%** |
| Per-10kHz sub-band max Cohen's d | 0.4–2.0 | **< 0.3** |
| K-NN same-cohort rate | 0.98–1.0 | **< 0.85** |
| Raw-pixel PCA PC1 Cohen's d | +5.83 (ours) | **< 1.5** |

If any criterion fails → stop, iterate on cleaning before proceeding to Phase 1.1.

### Steps
1. Build a small script `scripts/cnn_cleaning_validation.py` that:
   - Loads a sample of VocalMat + lab 131204 + 5970 spectrograms
   - Applies the full cleaning stack with each component toggleable (ablation matrix)
   - Re-runs the synthetic-notch injection test (from VAE adversarial review)
   - Re-runs per-band Cohen's d on cleaned spectrograms
   - Trains a quick 32-dim VAE (4–8 epochs is enough for diagnostic) and runs K-NN + PC1 tests
2. Output: `docs/handoffs/cleaning-validation-report.md` with pass/fail per criterion and ablation table.

### Deliverables
- `scripts/cnn_cleaning_validation.py`
- `docs/handoffs/cleaning-validation-report.md`
- Go/no-go decision: proceed to Phase 1.1 or iterate

### Estimated effort
3–5 focused days. No new cleaning code expected; this is plumbing existing components and running diagnostics.

---

## Phase 1.1 — Data Preparation

### Steps

1. **Download VocalMat dataset** from OSF (`https://osf.io/bk2uj/`) into `data/vocalmat/` (gitignored).
2. **Verify class counts and balance.** Save manifest to `data/vocalmat/manifest.csv` with columns: `path, class, source_recording, duration_ms`.
3. **Resample our lab 131204 + 5970 from 300 kHz → 250 kHz** via `scipy.signal.resample_poly(up=5, down=6)` to match VocalMat's pipeline. Output: parallel chunked WAVs.
4. **Apply cleaning stack uniformly** to all training/validation data:
   - Soft-notch (where library exists — lab 131204 only at first)
   - `--subtract-baseline median_envelope` (works on any recording)
   - Global MAD normalization (always-on)
   - Per-recording Z-score (newly wired)
5. **Generate spectrograms** using VocalMat's exact STFT params (Hamming 256, hop 128, NFFT 1024 at 250 kHz, 227×227 RGB patches, 0.22s duration). Store as PNGs alongside class label.
6. **Build train/val/test splits** stratified by class (80/10/10), with **recording-level grouping** to prevent leakage (no call from the same source recording appears in both train and val).

### Deliverables
- `data/vocalmat/` populated, manifest CSV
- `scripts/cnn_prepare_training_data.py`
- `data/lab_cnn_training/{train,val,test}/` with PNG patches + CSV manifests

### Risk
**Class imbalance is more severe than first assumed.** Multi-steps (74 total) and Reverse-Chevron (136 total) sit 13–25× smaller than the top classes. Phase 1.1 must use stratified 80/10/10 split AND class-weighted CE / focal loss / oversampling. The minority-class strategy decision (open question 5) is binding before Phase 1.2 starts.

---

## Phase 1.2 — Baseline ResNet-18 Training

### Steps

1. **Implement training loop** in `scripts/train_lab_classifier.py`:
   - `timm.create_model('resnet18', pretrained=True, num_classes=12)`
   - AdamW, cosine LR schedule, ~50 epochs to convergence
   - Class-weighted cross-entropy
   - SpecAugment augmentation (time + frequency masking)
   - Standard image augmentation: pitch shift ±10%, time stretch ±20%, random crop ±5%
   - Track per-class precision/recall + confusion matrix every 5 epochs
2. **Train on cleaned VocalMat data only** (Phase 1.1 output, excluding our 845 lab 131204 verdicts which are held out).
3. **Log to `results/lab_classifier_v1/`** — training curves, best checkpoint, val confusion matrix.

### Validation criteria

| Metric | Pass threshold |
|---|---|
| Macro F1 on VocalMat val split | > 0.65 (VocalMat reports 0.92+ on their original) |
| Per-class precision | All ≥ 0.40 (no class completely fails) |
| Held-out 845 USV/noise accuracy | > 0.80 |
| Held-out 845 syllable-type entropy | Mean ≤ log(6) (≈ peaky predictions, not "everything is Flat") |

### Deliverables
- `models/lab_classifier_v1/best.pt`, training config, metric tables
- `results/lab_classifier_v1/confusion_matrix.png` + `eval_report.md`

### Estimated effort
1 week including dataset bring-up, training, and analysis.

---

## Phase 1.3 — Cage-Confound-Aware Training (DANN)

### Steps

1. **Extend training loop** with a domain discriminator head:
   - Encoder: shared ResNet-18 trunk
   - Class head: 12-way syllable classifier (cross-entropy)
   - Domain head: N-way cage discriminator (VocalMat / lab 131204; eventually 5970 if Phase 1.4 lands)
   - Gradient-reversal layer between trunk and domain head, with λ schedule from 0 → 1 over training
2. **Train** with same hyperparameters as 1.2, plus λ schedule per Ganin et al. 2015.
3. **Re-run all Phase 1.2 evals** + the cage-invariance probe:
   - Freeze encoder, train a linear classifier on encoder features to predict cage. Accuracy near chance = success.
   - Run VAE falsifiable test on encoder features (not raw spectrograms).

### Validation criteria (in addition to Phase 1.2)

| Metric | Pass threshold |
|---|---|
| Linear cage probe accuracy on encoder features | < 65% (vs random ~50% for 2 cages) |
| Syllable F1 vs 1.2 baseline | No worse than -0.05 (small drop OK; large drop = encoder collapsed under adversarial loss) |

### Deliverables
- `models/lab_classifier_v2/best.pt` + comparison report vs v1
- `results/lab_classifier_v2/cage_invariance_probe.md`

### Estimated effort
3–5 days. The DANN head is ~50 lines of new code on top of v1.

---

## Phase 1.4 — Wild Transfer Evaluation (USER-ENABLED)

### Domain-gap reality check
VocalMat's training data spans 5 mouse strains × 3 neonatal ages — already heterogeneous. So if wild transfer fails, the cause is **NOT strain monoculture**. The two real candidates are:
1. **Cage acoustics** (mitigated by DANN in Phase 1.3 + our cleaning stack)
2. **Call-duration distribution** (wild 30–50 ms median vs lab 30–200 ms range — same mechanism that killed κ=0.13 transfer)

Optional probe (1 day cost): re-train with **short-patch (0.08s) variant** to address (2). Open question 1 below.

### Pre-requisite
**User labels a small wild syllable-type set.** Recommended: ~200 calls from 5970, balanced across the 12 Grimsley classes as best the data allows. Workflow follows `feedback_labeling_queue_folder.md`: PNGs copied to a flat folder with stable-ID prefixes (`typ01_*.png`, `typ02_*.png`, ...).

**Labeling protocol:** Single-rater is acceptable for OOD eval (matches VocalMat's training-set protocol). If you want stronger evidence, dual-label a subset (~50 calls) yourself or have a second labeler — flag disagreements for special discussion.

### Steps

1. **User-provided wild labels** stored at `data/wild_labels/5970_human_verified.csv` (columns: `wav_stem, det_start_s, det_end_s, syllable_type_grimsley`).
2. **Run inference** on these calls with v1 (baseline) and v2 (DANN) classifiers.
3. **Report:**
   - Per-class precision/recall on wild data (lab → wild generalization)
   - Confusion matrix comparing wild vs lab patterns
   - Confidence distribution: are wild predictions less confident?
4. **Decision point:** Does v2 generalize meaningfully better than v1?
   - **Yes** → ship v2 as the production lab+wild classifier
   - **No** → stop and document. Phase 2 (wild-specific training or stronger DA) needed before wild deployment.

### Deliverables
- `data/wild_labels/5970_human_verified.csv` (user-produced)
- `results/wild_transfer_eval/{v1,v2}_eval.md` with confusion matrices
- Decision memo: ready-for-production or Phase 2-required

### Estimated effort
1–2 days assuming user labeling is done. Labeling itself: depends on user pace; 200 calls at ~30s/call ≈ 100 minutes.

---

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Cleaning stack fails Phase 1.0 gate | High | Iterate cleaning before training. May require new component (broadband whitening, per-band z-score). |
| Minority classes borderline trainable (Multi-steps n=74, Reverse-Chevron n=136) | **High** | Resolve open question 5 before Phase 1.2 (keep-all / collapse / drop). Class-weighted CE + oversampling + focal loss regardless. Monitor per-class precision/recall, not just macro F1. |
| Single-rater training labels (5–15% noise expected in minority classes) | Medium | Don't treat as gold-standard. When our 845 dual-rater-quality lab 131204 verdicts disagree with model predictions on minority classes, trust the verdicts. Consider active-learning relabeling in Phase 2 if minority-class precision is too poor. |
| Wild transfer remains poor despite DANN | Medium | This is Phase 2 — document and stop, don't force production. NOTE: strain monoculture is NOT the issue (VocalMat already spans 5 strains); cage acoustics + call-duration distribution are the real domain gaps. |
| `--subtract-baseline` interacts badly with VocalMat's spectrogram conventions | Medium | Validate output PNGs visually in Phase 1.1 ablation. |
| User-labeled wild set is too small for statistical inference | Low | 200 calls gives ~17/class on average — descriptive only, not inferential. Be honest in reporting. |
| `corpus.py` constants override conflict | Low | Keep Phase 1 pipeline in its own module (`src/usv_spectrogram/classifier/`); document the 250 kHz override; do NOT modify `corpus.py`. |
| Lab tonal library only exists for one rig | Medium (limits Phase 2) | Lab 131204 is the only lab data we have. New lab data requires `calibrate_lab_tonal_lines.py` run. |
| VocalMat training data includes neonatal mice (P5–P15); our adult wild + lab mice may have different acoustic features | Medium | Test-time domain gap. If wild transfer fails in Phase 1.4, this is the second most likely cause after cage acoustics. Mitigation: filter VocalMat data to older ages (P15) if available in metadata, or accept the gap and document. |

---

## What's NOT in this plan

- **A new cleaning component.** Audit found `--subtract-baseline` + per-recording Z-score already exist; we need to wire and validate, not build.
- **Modifications to the detection CNN.** This is a downstream classifier. Detection model (`hard_neg_retrain/best_model.pt`) is unchanged.
- **DeepSqueak replacement.** This complements DeepSqueak, doesn't replace it. We may compare downstream.
- **Wild-mouse training (Phase 2).** Explicitly deferred until Phase 1.4 reveals whether the lab classifier transfers.
- **Modifications to `corpus.py`.** This pipeline uses VocalMat's 250 kHz convention internally; the canonical 300 kHz invariant remains intact for the rest of the project.

---

## Open Questions (for user before /implement)

1. **Patch size in seconds:** VocalMat's 0.22s patch was the wild-mouse killer (82% silence around 30–50ms calls). For *lab* data (30–200ms calls), 0.22s is appropriate. But: do we want to extend Phase 1.4 with a **shorter patch (e.g., 0.08s)** variant trained on the same data, in case wild generalization improves? +1 day cost.
2. **Wild labels priority:** Once Phase 1.0 passes, do you want to start labeling in parallel with Phase 1.1–1.3 (the model is the bottleneck) or wait for Phase 1.3 to complete (gives you v2 confidence scores to bootstrap your labeling)?
3. **Perch 2.0 sidequest:** Should we burn ~1 day in Phase 1.2 to also test Perch 2.0 audio-pretrained embeddings + linear probe? Recent (2025) evidence says bioacoustic-pretrained embeddings often beat ImageNet-pretrained CNNs on cross-domain bioacoustic tasks. Cheap to test. Recommend yes.
4. **Domain head granularity:** DANN can discriminate (a) just `lab_131204` vs `vocalmat` (2 cages — minimum), or (b) per-recording-session (~50–100 cages — much finer, more aggressive invariance). I'd start with (a). Confirm or override.
5. **Minority-class strategy:** Multi-steps (74 examples) and Reverse-Chevron (136 examples) sit 13–25× smaller than top classes. Three options:
   - **(a) Keep all 12 classes** with oversampling + focal loss — most faithful to VocalMat's taxonomy. Risk: poor minority-class precision; may produce "everything-becomes-Step-up" failure mode on rare types.
   - **(b) Collapse step-family** — merge Multi-steps + Two-steps + Step-up + Step-down → single "Multi-step family" class → 9-class model. Risk: loses biological granularity Grimsley taxonomy was designed for.
   - **(c) Drop Multi-steps + Reverse-Chevron entirely** → 10-class model. Risk: ignores rare-but-real syllable types; downstream consumers expecting 12-class won't match.
   Recommendation: start with **(a)** for Phase 1.2 baseline. Inspect v1 confusion matrix. If minority classes have precision < 0.20, revisit (b) or (c) before Phase 1.3.

---

## Success Definition

This plan succeeds if, after Phase 1.4:
- We have a 12-class lab USV classifier with macro F1 > 0.65 on held-out VocalMat val data,
- The encoder is provably cage-invariant (linear probe < 65% accuracy),
- Wild transfer eval (~200 calls) shows either meaningful generalization (then we ship) or honest failure (then we scope Phase 2),
- All decisions are documented and reproducible from this plan + the Phase deliverable handoffs.

The plan **fails honestly** (not silently) if Phase 1.0 cleaning gate doesn't pass — that's the explicit purpose of the gate.

---

## References

- Memory: `project_vae_comparison_complete`, `project_vocalmat_transfer_test`, `project_clustering_analysis_lab`, `feedback_cnn_inference_global_mad`, `feedback_cage_not_rig_terminology`
- VAE adversarial review: `docs/handoffs/2026-05-18_vae_comparison_memo.md`
- Soft-notch design: `docs/handoffs/2026-05-11_adaptive-soft-notch.md`
- Pre-CNN baseline subtraction: `docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md`
- VocalMat paper: Fonseca et al. 2021, eLife 10:e59161 — `https://elifesciences.org/articles/59161`
- VocalMat code/data: `https://github.com/ahof1704/VocalMat` (Apache 2.0), data at `https://osf.io/bk2uj/`
- Grimsley taxonomy: Grimsley, Hazlett, Wenstrup 2013, *PLOS One* 8(3):e59428
- DANN: Ganin & Lempitsky 2015, ICML — `arXiv:1409.7495`
- SpecAugment: Park et al. 2019 — `arXiv:1904.08779`
- Perch 2.0: 2025 — `arXiv:2512.03219`
