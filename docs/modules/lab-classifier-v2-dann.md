# Module 18.4 — Cage-invariant lab classifier (DANN)

**Status:** code implemented + locally verified (CPU). GPU training + full
evaluation run on the rig (`cloudyclaude`). ROADMAP: `ROADMAP_lab_cnn_classifier.md §18.4`.
**Predecessor:** Module 18.3 (12-class ResNet-18, `results/lab_classifier_v1/`).

---

## Why DANN

The v1 classifier learns features that are good at the 12-class Grimsley
syllable task — but those features can also silently encode *which recording
cage* a patch came from (intensity, noise floor, tonal-notch artifacts). Because
v1 is trained only on VocalMat and applied to lab/wild data from different cages,
that cage signal is a confound: the model can "cheat" by keying on the cage
instead of the call.

DANN (Ganin & Lempitsky 2015, arXiv:1409.7495) removes the confound during
training. A second head — a **2-way cage discriminator** (D4: `lab_131204` vs
`vocalmat`) — is attached to the shared encoder through a **gradient-reversal
layer (GRL)**:

- **Forward**, the GRL is the identity (downstream sees unmodified features).
- **Backward**, it multiplies the gradient by **−λ**.

So the domain head *minimises* its own cage-classification loss (gets good at
cage discrimination), but the reversed gradient pushes the encoder the opposite
way (gets bad at it). The equilibrium is a representation from which cage is no
longer linearly recoverable — yet syllable classification still works because the
class head keeps its normal (un-reversed) gradient.

```
            ┌───────────────┐   class head   →  12-way syllable logits  (focal loss)
 patch →    │  ResNet-18    │──┤
            │  encoder (512)│  └─ GRL(−λ) → domain head → 2-way cage logits (CE loss)
            └───────────────┘
```

## Unsupervised domain adaptation (important framing)

Our lab and wild data was **never syllable-labelled** — VocalMat is the only
labelled dataset. So:

- **Source domain** = VocalMat (labelled): trains both heads.
- **Target domain** = lab `131204` (unlabelled): trains the domain head only.
- 12-way macro-F1 is measurable **only on the VocalMat test split**.
- On lab, the only ground truth is **binary usv/noise** (the 844 held-out set);
  collapse the 12-class output to usv/noise (`argmax == "Noise"` → noise) to
  score it. The 844 set is **83.5 % usv**, so report **per-class** (noise recall),
  never pooled accuracy (a trivial "always-usv" baseline scores 0.835).

## The λ schedule and encoder collapse

λ ramps on the Ganin schedule `λ(p) = 2/(1+e^{−γp}) − 1`, `p = epoch/total_epochs`,
γ = 10. λ ≈ 0 at epoch 0 (pure warm-up, no adversarial pressure) → ≈ 1 at the
final epoch.

The schedule is gentle for a reason. The central failure mode is **encoder
collapse**: if λ is too aggressive the encoder finds a *trivially* cage-invariant
representation that also throws away the syllable signal — cage-invariant but
syllable-blind. We guard against it three ways:

1. **Warm-start** from the v1 checkpoint (`--v1-checkpoint`) so v2 begins as a
   competent classifier and the adversary only removes the *residual* cage signal.
2. **Slow λ ramp** (sigmoid from 0).
3. **Collapse tripwire**: `syllable macro-F1 ≥ v1 − 0.05`. If v2's F1 drops by
   more than 0.05, training collapsed — **STOP** and revisit λ (per ROADMAP exit
   criteria), do not ship.

## The three ship gates

| Gate | Metric | Pass | Where computed |
|---|---|---|---|
| Cage probe | linear cage probe acc on v2 encoder features | **< 0.65** | `train_lab_classifier_v2.py` (end of run) |
| Collapse | syllable macro-F1 (VocalMat test) vs v1 | **≥ v1 − 0.05** | `comparison_v1_vs_v2.md` |
| VAE re-run | 4 falsifiable cage tests on encoder features | **all 4 pass** | `run_vae_diagnostic_on_encoder.py` → `cage_invariance_probe.md` |

The **linear** cage probe (Alain & Bengio 2016) is deliberate: a powerful
non-linear probe could recover cage from almost any representation, so it would
not distinguish a cage-invariant encoder from a cage-entangled one. Linear
decodability is the right operationalisation of "the encoder forgot the cage."

The VAE re-run applies the same four 18.1 criteria — but to the **learned
features** rather than raw pixels (stronger evidence):

| Criterion | Threshold | Notes |
|---|---|---|
| k-NN same-cohort rate | < 0.85 | reused from 18.1 (accepts 2-D embeddings) |
| PC1 Cohen's d | < 1.50 | computed inline (18.1's `raw_pixel_pca_d` needs 3-D) |
| per-dimension max Cohen's d | < 0.30 | feature-space analogue of 18.1 per-band d |
| notch-injection migration | < 0.30 | encode notched target patches; measure k-NN migration toward source |

> **Sample-size caveat (per-dimension max Cohen's d).** This criterion takes the
> maximum d over all 512 feature dimensions, so its 0.30 threshold — calibrated
> for 18.1's ~10 frequency bands — suffers multiple-comparisons inflation at small
> n: expected max|d| ≈ 0.34 at n=200 (false FAIL on a clean encoder) vs ≈ 0.15 at
> n=1000 (safe). **Run with ≥ 500 patches per cohort** (`--max-per-cohort`, default
> 1000); the script warns below 500.

## Known transfer risk (eyes open)

Prior `project_vocalmat_transfer_test` measured VocalMat→our-data transfer at
**κ = 0.13** (poor), driven substantially by the **call-duration / repertoire
distribution gap — NOT just cage**. Cleaning + DANN address cage; they do *not*
fix a duration/repertoire mismatch. If the 844 held-out eval is poor, that is an
**informative** result (pivot to CDAN/DSAN, or accept that labelled lab data is
required), not a bug.

## Pixel-level cage is already handled

Phase 1.0 cleaning is formally NO-GO but **accepted as a soft-notch test
artifact** (`cleaning-validation-report.n4-NOGO.md`): under the full `all_layers`
stack the substantive cage metrics pass (per-band Cohen's d 26→0.07; raw-pixel
PCA d 52→0.00; knn 0.33). The lone FAIL (`notch_injection_migration = 1.0`) is a
no-op — soft-notch needs a calibrated tonal library (C4) that was not supplied;
the patches were actually cleaned with **soft-notch OFF + `baseline_mode=percentile`**.
So the pixel-level cage is handled; DANN's job is the **feature-level residual**.

## The 5-file set

| File | Role |
|---|---|
| `src/usv_spectrogram/classifier/dann.py` | GRL, `grad_reverse`, `DomainHead`, `LambdaSchedule`, `ResNet18DANN` |
| `src/usv_spectrogram/classifier/cage_probe.py` | `linear_cage_probe` on frozen encoder features |
| `scripts/train_lab_classifier_v2.py` | DANN training CLI (warm-start, source/target loaders, cage probe, comparison) |
| `scripts/run_vae_diagnostic_on_encoder.py` | VAE 4-criteria re-run on encoder features → `cage_invariance_probe.md` |
| `docs/modules/lab-classifier-v2-dann.md` | this doc |

**Additivity:** the Module 18.3 files (`model.py`, `training.py`, `losses.py`,
`augmentation.py`, `scripts/train_lab_classifier.py`) are *imported, never
modified* — v1 remains the non-DANN reference.

## How to run

Training (rig, GPU). The source manifests + the 18 GB patch pool live in the
`lab-cnn-classifier-plan` worktree / on the rig; the warm-start checkpoint is in
main:

```bash
python scripts/train_lab_classifier_v2.py \
    --train-csv data/lab_cnn_training/train/manifest.csv \
    --val-csv   data/lab_cnn_training/val/manifest.csv \
    --test-csv  data/lab_cnn_training/test/manifest.csv \
    --domain-unlabeled-csv data/lab_cnn_training/domain_unlabeled.csv \
    --v1-checkpoint results/lab_classifier_v1/best.pt \
    --output-dir results/lab_classifier_v2/ \
    --domain-granularity 2cage --epochs 50 --batch-size 64 --device cuda
```

CPU smoke (wiring check): add `--no-pretrained --epochs 2 --max-source-samples 200
--max-target-samples 200 --device cpu`.

VAE gate (after training):

```bash
python scripts/run_vae_diagnostic_on_encoder.py \
    --v2-checkpoint results/lab_classifier_v2/best.pt \
    --source-val-csv data/lab_cnn_training/val/manifest.csv \
    --domain-unlabeled-csv data/lab_cnn_training/domain_unlabeled.csv \
    --output-dir results/lab_classifier_v2/ --device cuda
```

## Outputs (`results/lab_classifier_v2/`)

- `best.pt` — v2 weights (`state_dict` + history + warm-start report + λ schedule).
- `metrics.json` — val/test macro-F1, per-class precision/recall, confusion matrix, `cage_probe_acc`.
- `confusion_matrix.png` — test confusion matrix.
- `comparison_v1_vs_v2.md` — side-by-side metric table + collapse/cage gate verdict.
- `cage_invariance_probe.md` — the 4-criteria VAE gate report.

## Tests

- `tests/classifier/test_dann.py` (14) — GRL forward/backward (grad×−λ), DomainHead
  shapes, λ-schedule endpoints/monotonicity, `ResNet18DANN` 3-tuple shapes &
  eval determinism, adversarial training smoke (no-NaN), collapse-detection F1 band.
- `tests/classifier/test_cage_probe.py` (7) — positive control (>0.90 when cage
  signal exists), negative control (≈0.50 on signal-free data), frozen-encoder
  invariant, single-cage / CPU edge cases.

Both written by `test-architect` **before** implementation (TDD); their
expectations are spec and were not modified during implementation.
