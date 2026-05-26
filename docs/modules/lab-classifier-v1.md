# Lab USV Syllable Classifier v1 (Module 18.3)

> Train a 12-class syllable classifier (Grimsley 2011 taxonomy) on the
> VocalMat-curated manifest from Module 18.2b, using a timm ResNet-18
> ImageNet-pretrained backbone with AdamW + cosine LR + class-weighted
> focal loss + SpecAugment + cage-noise injection. Optional D3 sidequest:
> Perch 2.0 frozen-embedding linear probe for cross-domain comparison.

## What it is

Module 18.3 is the *baseline* classifier shipment. It assumes the prep
work in Module 18.2b is complete and the four pre-existing modules
(`resample.py`, `dataset.py`, `cleaning_pipeline.py`, `diagnostics.py`)
have produced a deterministic train/val/test split + sanity patches. The
shipment closes the gap from "data ready" → "model architecture, loss,
training loop, and CLI exist". The real GPU training run that produces
shippable numbers is gated by Stream V, not by this module.

## Architecture

```
src/usv_spectrogram/classifier/
  model.py              # build_resnet18_classifier() — timm factory
  augmentation.py       # SpecAugment + cage-noise injection
  losses.py             # class-weighted focal loss (plain-mean reduction)
  training.py           # TrainingConfig + train_classifier loop
scripts/
  train_lab_classifier.py     # CLI entry point (argparse)
  train_perch2_probe.py       # D3 sidequest (DEFERRED — Stream V)
tests/classifier/
  test_model.py               #  8 tests
  test_augmentation.py        # 13 tests
  test_losses.py              #  7 tests
  test_training.py            # 10 tests (incl. <90 s CPU smoke)
  test_train_lab_classifier.py #  5 tests (CLI argparse + validation)
  test_train_perch2_probe.py  #  4 tests (skipped — Perch not installed)
docs/modules/
  lab-classifier-v1.md        # this file
```

## Model factory (`model.py`)

```python
from usv_spectrogram.classifier.model import (
    NUM_CLASSES,                 # 12 — Grimsley 2011 taxonomy
    build_resnet18_classifier,   # (num_classes, pretrained) -> nn.Module
)
```

Wraps `timm.create_model("resnet18", pretrained, num_classes=12)`. Input
is `(B, 3, 227, 227)`, output is `(B, 12)` logits. The 227×227 spatial
size matches VocalMat's patch convention (see Module 18.2b).

### Why ResNet-18 and not a deeper backbone

PLAN §"Phase 1.2" picks ResNet-18 because the supervised dataset is small
(~12 k examples after Module 18.2b's split). Deeper backbones overfit
without proportionally more data. EfficientNet-B0 is the documented
fallback if ResNet-18 underfits — same input shape, similar param count,
slightly better ImageNet transfer characteristics.

## Augmentation (`augmentation.py`)

Two transforms compose to fight the cage-confound documented in the VAE
comparison memo (`docs/handoffs/2026-05-18_vae_comparison_memo.md`):

```python
from usv_spectrogram.classifier.augmentation import (
    AugmentationConfig,   # frozen hyperparam bundle
    specaugment,          # Park et al. 2019 time+freq masking
    inject_cage_noise,    # additive blend of verdict-negative patches
)
```

### SpecAugment

Implements Park et al. 2019 (arXiv:1904.08779) time + frequency masking.
With `m_T = m_F = 2`, two time masks of width up to 20 frames and two
frequency masks of width up to 16 bins are applied per patch.

Uses the **legacy `np.random.*` global state** for determinism — callers
seed via `np.random.seed(...)`. This is deliberate so the test contract
can reproduce augmentations by seeding once at the top of a test.

### Cage-noise injection

A new-to-this-codebase trick. With probability
`cage_noise_inject_prob` (default 0.25), a verdict-negative patch is
loaded from `cage_noise_paths`, resized to the input shape, and added
with a uniformly-drawn weight in [0.05, 0.30]. This directly trains the
classifier to remain class-correct under cage-noise contamination —
attacking the cage-confound at training time rather than relying on
post-hoc DANN (Module 18.4).

Degrades gracefully: empty `cage_noise_paths` or `prob=0` → identity.

## Loss (`losses.py`)

```python
focal_loss(logits, targets, class_weights, gamma=2.0) -> scalar
```

Implements Lin et al. 2017 focal loss:

```
focal_loss = mean_b( w[t_b] · (1 - p_{t_b})^γ · (-log p_{t_b}) )
```

### Reduction choice: plain-mean, NOT PyTorch's weighted-mean

The reduction is `sum / B` (plain mean over the batch), **not** the
PyTorch default `sum / sum(weights[targets])` (weighted mean). This is
deliberate. Under weighted-mean reduction, the per-sample weight `w[t]`
appears in *both* numerator and denominator in any single-sample limit,
cancelling exactly. That defeats the D5 strategy ("class weights amplify
minority gradients"). Plain-mean preserves the intended scaling.

The price is that `focal_loss(..., gamma=0)` matches plain-mean weighted
CE, **not** PyTorch's default weighted CE. The test at
`tests/classifier/test_losses.py::test_focal_loss_gamma0_equals_weighted_ce`
documents this with the matching reference expression.

### Why focal loss at all

PLAN §"D5 — Minority class strategy" inventories the 12-class imbalance:
Multi-steps and Reverse Chevron each have fewer than 200 training
patches, while Step up and Flat each have over 2000. Vanilla CE
under-trains the minority classes. Focal loss + per-class weights
attacks the imbalance from two directions: weights up-weight minority
gradients globally, the focal term `(1-p_t)^γ` down-weights easy
majority examples that already classify well.

## Training loop (`training.py`)

```python
from usv_spectrogram.classifier.training import (
    TrainingConfig,
    train_classifier,
)
```

`train_classifier` does:

- AdamW optimizer with `weight_decay=1e-4`.
- Cosine LR schedule with linear warmup (default `warmup_epochs=0`; set
  to 3 for real 50-epoch runs; smoke tests use 0 because epochs=1 or 2).
- Class weights computed from train-loader label distribution
  (inverse-frequency, normalised so `sum(weights) == n_classes`).
- Per-epoch validation with macro F1; best checkpoint saved to
  `output_dir / "best.pt"`.
- Early stopping after `early_stop_patience` epochs without macro-F1
  improvement.
- Final eval on val + test loaders + held-out 845 CSV.

### Returned metrics dict

```python
{
    "macro_f1_val":             float,
    "macro_f1_test":            float,
    "per_class_precision":      list[float],   # length 12
    "per_class_recall":         list[float],   # length 12
    "confusion_matrix":         list[list[int]],
    "usv_noise_acc":            float,         # from held-out 845
    "syllable_entropy_mean":    float,         # nats, in [0, log(12)]
    "history":                  list[dict],    # per-epoch (epoch, train_loss, macro_f1_val)
}
```

### Held-out 845 evaluation — DEFERRED (see ERRATA)

> **CORRECTION (2026-05-25):** see
> `docs/handoffs/2026-05-25_ERRATA_held-out-845.md`. Two errors were found
> when the rig tried to run this: (1) the verdict set is NOT
> `classified_detections_lab_131204_clean.csv` (that's the 40,787-row
> clustering working set) — it's 844 rows across
> `results/lab_{cluster0,cluster1,cluster2,noise}_review/review_index_annotated.csv`,
> with only a `usv`/`noise` `verdict`, no Grimsley labels; (2) there is **no
> Grimsley macro-F1 held-out gate** — lab data was never Grimsley-labeled, so
> that gate is unbuildable and is removed.

The smoke / unit-test path computes label-distribution statistics from a
stub CSV and is fine for the test contract. The **real** held-out lab eval
(the two PLAN gates below) is **DEFERRED** — the 844 verdict calls have only
annotated review figures on disk, not clean 227×227 model patches, so they
must be re-extracted from `USV_lab_131204_chunked_2s_full/` first (prep task
in the ERRATA). The two evaluable gates are **USV/noise accuracy > 0.80** and
**syllable-type entropy mean ≤ log(6)** — NOT a Grimsley macro F1.

### `device='auto'` resolution

`TrainingConfig(device='auto')` resolves to `'cuda'` if
`torch.cuda.is_available()`, else `'cpu'`. A `device='cuda'` request on a
host without CUDA silently falls back to CPU rather than crashing — useful
for the CI smoke path. Real training scripts should pass `device='cuda'`
explicitly to fail fast if the GPU isn't visible.

## CLI (`scripts/train_lab_classifier.py`)

Required flags:

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--train-csv` | path | required | Module 18.2b train manifest |
| `--val-csv` | path | required | Module 18.2b val manifest |
| `--test-csv` | path | required | Module 18.2b test manifest |
| `--held-out-845` | path | required | `classified_detections_lab_131204_clean.csv` |
| `--output-dir` | path | required | best.pt + metrics.json land here |
| `--epochs` | int | 50 | must be > 0 |
| `--batch-size` | int | 64 | must be > 0 |
| `--lr` | float | 1e-3 | must be > 0 |
| `--warmup-epochs` | int | 3 | must be ≤ epochs |
| `--focal-gamma` | float | 2.0 | Lin et al. 2017 default |
| `--device` | str | "auto" | one of {auto, cpu, cuda} |
| `--num-workers` | int | 2 | DataLoader worker count |
| `--no-pretrained` | flag | False | smoke / offline runs |

Validation happens before any heavy work: missing paths → exit 1 with a
clear message; bad config (epochs/batch_size/lr ≤ 0, warmup > epochs)
→ exit 2 with the offending field named.

### Production training command (Stream V)

```bash
.venv/bin/python scripts/train_lab_classifier.py \
    --train-csv data/lab_cnn_training/train/manifest.csv \
    --val-csv   data/lab_cnn_training/val/manifest.csv \
    --test-csv  data/lab_cnn_training/test/manifest.csv \
    --held-out-845 classified_detections_lab_131204_clean.csv \
    --output-dir results/lab_classifier_v1/ \
    --epochs 50 --batch-size 64 --warmup-epochs 3 \
    --device cuda --num-workers 4
```

## Perch 2.0 linear probe (D3 sidequest — DEFERRED)

`scripts/train_perch2_probe.py` is intentionally **not shipped** in this
module. The test contract at `tests/classifier/test_train_perch2_probe.py`
specifies macro F1 > 0.30 on a synthetic dataset of IID Gaussian noise
with **permuted labels** — a structurally unfulfillable bar (mutual
information `I(features; labels) = 0` by construction, so no encoder can
exceed the 1/12 random baseline). The test architect's reasoning
confused class balance with class separability. Until the test is
amended (e.g., to use a real subset of the 12,178-row supervised
manifest, or to lower the threshold to ≤ random baseline), there's no
honest way for an embedding pipeline to pass.

The 4 Perch tests skip cleanly via an `ImportError`-driven `skipif`
guard. Stream V's successor handoff documents the deferral and proposes
fix options for the test.

## Validation criteria (PLAN §"Validation criteria")

The training run is SHIP-eligible when (corrected per
`docs/handoffs/2026-05-25_ERRATA_held-out-845.md`):

- Macro F1 > 0.65 on VocalMat test split
- Per-class precision ≥ 0.40 on every class
- Confusion matrix shows no class collapsing into a different class with
  > 0.40 mass
- Held-out lab gates — **DEFERRED** (not a 18.3 blocker): USV/noise
  accuracy > 0.80 AND syllable-type entropy ≤ log(6), on the 844-row
  `results/lab_*_review/` verdict set, once clean patches are re-extracted.
  (There is **no** Grimsley-macro-F1 held-out gate — see ERRATA.)
- Perch 2.0 linear probe macro F1 reported (D3 deliverable; comparison
  only, no gate) — **deferred pending test repair**

If any criterion fails, do NOT collapse the taxonomy or modify the
validation thresholds — re-investigate per D5's "revisit only if
precision < 0.20" guidance.

## Test counts (post-shipment)

- 43 new tests added (8 model + 13 augmentation + 7 losses + 10 training + 5 CLI)
- 4 new tests skipped (Perch — deferred)
- 100 baseline 18.2a/b tests still pass (0 regressions)
- **143 passed, 4 skipped, 0 failed** in the full classifier suite

## Known limitations & follow-up tickets

1. **Stream V (real-data GPU training)** is deferred — no GPU on this
   WSL host. All code is CPU-runnable but the 50-epoch run on 9,741
   training patches needs a GPU to finish in reasonable wall time. See
   the Stream V successor handoff.
2. **Perch 2.0 probe** is deferred (see above section). Successor
   handoff has the proposed test repair.
3. **Held-out 845 real-inference path** is stubbed. Stream V must
   extend `_evaluate_held_out_845` to load patches and run model
   inference.
4. **Test 6 reference reduction** was amended in this shipment (one-line
   change, fully documented in the test's docstring) to use plain-mean
   weighted CE instead of PyTorch's default weighted-mean. See `losses.py`
   docstring for the rationale.
5. **TrainingConfig.warmup_epochs default** is 0 (not 3 as in the
   ROADMAP code stub) so that small-epoch smoke runs work without
   explicit override. Real training scripts should pass `--warmup-epochs 3`.

## Inherited Tier-2 tickets (NOT addressed in 18.3)

From Module 18.2b (still open):

1. `cleaning_pipeline.py` produces all-zero output on long lab WAVs when
   `baseline_mode='median_envelope'` chains with `_apply_global_mad`.
   Workaround at `scripts/cnn_prepare_training_data.py:376`
   (`baseline_mode='percentile'`).
2. `scripts/cnn_prepare_training_data.py:_collect_wav_rows` uses
   non-recursive `glob`. Worked around by `scripts/cnn_wild_topup.py`.

Both must be resolved before Module 18.4 (DANN).
