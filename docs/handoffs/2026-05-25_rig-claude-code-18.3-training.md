# Module 18.3 Stream V — GPU rig execution (Claude Code entry point)

**Date:** 2026-05-25
**You are:** Claude Code running on the GPU rig. This is your entry point.
**Repo:** `git@github.com:lightningfastsls/London_Lab.git`
**Branch:** `worktree-lab-cnn-classifier-plan`
**Predecessor (CPU box):** Module 18.3 code shipment — COMPLETE. Code +
tests are pushed; this handoff is the operational guide for running the
real GPU training the CPU box couldn't.
**Deeper spec:** `docs/handoffs/2026-05-24_module-18.3-stream-v-gpu-training.md`
(full exit criteria + the held-out-845 code addition) and
`docs/modules/lab-classifier-v1.md` (architecture). Read both before training.

---

## TL;DR of what you're doing

The classifier code is written, reviewed, hardened, and green on CPU
(180 passed, 4 skipped). It has never been trained on a GPU. Your job:
get the code, place the data, set up a CUDA env, run the 50-epoch
training, and report whether it clears the ROADMAP §18.3 SHIP gate.
**You should NOT need to write much new code** — only the held-out-845
real-inference path (see Step 6).

---

## Step 1 — Get the code

```bash
git clone git@github.com:lightningfastsls/London_Lab.git
cd London_Lab
git checkout worktree-lab-cnn-classifier-plan
git log --oneline -1   # should show the "Module 18.3 ResNet-18 baseline" commit
```

If the repo is already cloned: `git fetch origin && git checkout worktree-lab-cnn-classifier-plan && git pull`.

> NOTE: the project's working code lives under the worktree layout on the
> CPU box, but on the rig you can just check out the branch directly into
> a normal clone. All paths below are **relative to the repo root**, so
> they resolve identically in a plain checkout.

## Step 2 — Set up the CUDA environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install timm==1.0.27          # required by classifier/model.py; not in requirements.txt
# Ensure a CUDA build of torch is installed (the CPU box used torch 2.10.0+cu128).
.venv/bin/python -c "import torch, timm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('timm', timm.__version__)"
```

**Gate:** `torch.cuda.is_available()` MUST print `True`. If it prints
`False`, stop and fix the torch/CUDA install before training — the whole
point of the rig is the GPU.

## Step 3 — Place + verify the training data

The data is **gitignored** (too large for git) and arrives via a separate
transfer from the CPU box (rsync/scp). Stream V (this training) needs only
**~650 MB**, NOT the full 18 GB — the 18 GB lab/wild patch pool is Module
18.4 (DANN) input and is not consumed here.

Expected on disk after transfer (paths relative to repo root):

| Path | Size | What |
|---|---|---|
| `data/vocalmat_full/` | 627 MB | 12,178 VocalMat PNGs (the supervised training images) |
| `data/lab_cnn_training/train/manifest.csv` | ~1 MB | 9,741 rows |
| `data/lab_cnn_training/val/manifest.csv` | ~125 KB | 1,220 rows |
| `data/lab_cnn_training/test/manifest.csv` | ~125 KB | 1,217 rows |
| `classified_detections_lab_131204_clean.csv` | 22 MB | held-out 845 verdict set (repo root) |

Verify:

```bash
du -sh data/vocalmat_full/                                  # ~627M
wc -l data/lab_cnn_training/{train,val,test}/manifest.csv   # 9742 / 1221 / 1218 (incl. headers)
ls classified_detections_lab_131204_clean.csv               # exists at repo root
# Confirm every referenced PNG exists:
.venv/bin/python -c "
import pandas as pd; from pathlib import Path
miss=0; tot=0
for s in ['train','val','test']:
    for p in pd.read_csv(f'data/lab_cnn_training/{s}/manifest.csv')['path']:
        tot+=1; miss+= 0 if Path(p).exists() else 1
print(f'referenced={tot} on_disk={tot-miss} missing={miss}')   # missing MUST be 0
"
```

If `missing` > 0, the transfer is incomplete — do NOT train. Re-run the
data transfer for the missing class folders.

## Step 4 — Run the test suite (sanity gate)

```bash
.venv/bin/python -m pytest tests/classifier/ -q
```

**EXPECTED ON A CUDA HOST: 179 passed, 1 failed, 4 skipped.** The single
failure is **expected and is NOT a regression**:

> `tests/classifier/test_training.py::test_trainingconfig_auto_device_resolves_cpu`
> contains `assert not torch.cuda.is_available()` at line 331 — a
> *precondition* written for the CPU-only build box. On a GPU host this
> precondition is false, so the test errors. This is a host-specific test,
> not a code defect.

To get a clean run on the rig, exclude that one test:

```bash
.venv/bin/python -m pytest tests/classifier/ -q -k "not auto_device_resolves_cpu"
# expect: 179 passed, 4 skipped
```

Do NOT "fix" the failing test by editing its expectation — it's a
test-architect spec file. If you want it to skip cleanly on CUDA hosts,
that's a `@pytest.mark.skipif(torch.cuda.is_available(), ...)` change that
must be discussed with the user first (it's a tests-as-spec modification).
Note it in your report instead.

The 4 skips are the Perch 2.0 probe tests (deferred — see Step 7).

## Step 5 — First training run (val/test gates)

```bash
.venv/bin/python scripts/train_lab_classifier.py \
    --train-csv  data/lab_cnn_training/train/manifest.csv \
    --val-csv    data/lab_cnn_training/val/manifest.csv \
    --test-csv   data/lab_cnn_training/test/manifest.csv \
    --held-out-845 classified_detections_lab_131204_clean.csv \
    --output-dir results/lab_classifier_v1/ \
    --epochs 50 --batch-size 64 --lr 1e-3 \
    --warmup-epochs 3 --focal-gamma 2.0 \
    --device cuda --num-workers 4
```

**`--warmup-epochs 3` is mandatory for the real run** — the argparse
default is 0 (a deliberate choice so short smoke runs don't trip the
`warmup <= epochs` validator). The cosine LR schedule needs the 3-epoch ramp.

Expected wall-clock: ~30–90 min on a single consumer GPU (RTX 3060+),
~11 M params, 9,741 patches at batch 64. Bump `--batch-size` if VRAM allows.

This run produces:
- `results/lab_classifier_v1/best.pt` (best checkpoint by val macro F1)
- `results/lab_classifier_v1/metrics.json` (val/test macro F1, per-class
  precision/recall, confusion matrix)

It gives you the **val/test gates**: macro F1 > 0.65 on test, per-class
precision ≥ 0.40. The held-out-845 macro-F1 > 0.80 gate needs Step 6.

## Step 6 — The ONE code addition: held-out 845 real inference

The shipped `_evaluate_held_out_845` in
`src/usv_spectrogram/classifier/training.py` currently computes only
**label-distribution statistics** from the CSV (no model inference) — this
satisfies the smoke tests but does NOT evaluate the trained model on the
845 held-out patches. To clear the **held-out 845 macro F1 > 0.80** gate
you must extend it to load patches and run inference.

See `docs/handoffs/2026-05-24_module-18.3-stream-v-gpu-training.md`
§"Required CODE additions for Stream V" for the exact change. Summary:

- Add optional `model`, `device`, `patches_dir` params to
  `_evaluate_held_out_845`. When present, load each held-out patch, batch-
  predict, and emit `held_out_845_macro_f1`, real `held_out_845_usv_noise_acc`,
  and a 12×12 `held_out_845_confusion_matrix`.
- Keep the label-distribution path as a fallback so the existing smoke +
  adversarial tests still pass unchanged.
- Add NEW tests for the new path (synthetic patches with known labels). Do
  NOT modify the existing `test_training.py` expectations.

**Open question to resolve first:** the 845 held-out rows come from
`classified_detections_lab_131204_clean.csv` (lab 131204 detections). Their
patch images are NOT in `data/vocalmat_full/` — they're lab patches. Two
options:
1. Transfer the subset of `data/lab_cnn_training/patches/lab/` that
   corresponds to the 845 rows (a few hundred MB at most), OR
2. Render the 845 patches on the rig from the lab 131204 WAVs using the
   existing cleaning + extraction pipeline (heavier; needs the WAVs).

Pick (1) if the patch-key mapping from CSV rows to `patches/lab/` filenames
is recoverable; otherwise (2). Flag to the user which you chose.

## Step 7 — Perch 2.0 probe (D3 sidequest)

DEFERRED on the CPU box because the test contract
(`tests/classifier/test_train_perch2_probe.py`) demands macro F1 > 0.30 on
IID Gaussian noise with permuted labels — structurally impossible
(`I(features; labels) = 0`). Two options in the Stream V handoff
§"Perch 2.0 linear probe": (A) repair the test to use a real-patch subset
then install Perch + implement, or (B) defer permanently (allowed — it's a
"no gate" comparison). **Recommend Option A** — it informs whether 18.4's
DANN is needed. Confirm Perch 2.0's distribution channel (TF Hub / HF) at
install time.

## Step 8 — Confusion-matrix PNG

`results/lab_classifier_v1/confusion_matrix.png` is a ROADMAP exit
deliverable. Render it from the `confusion_matrix` in `metrics.json` using
`matplotlib.imshow` + class labels from
`usv_spectrogram.classifier.dataset.GRIMSLEY_12_CLASSES`. ~15 LOC, inline
in `scripts/train_lab_classifier.py` after `train_classifier` returns
(don't sprout a viz module for one chart).

## SHIP gate (ROADMAP §18.3 lines 564–577)

Stream V is SHIP-eligible when ALL hold:

- [ ] `pytest tests/classifier/ -q -k "not auto_device_resolves_cpu"` → 179 passed, 4 skipped (+ any new held-out tests), 0 failed
- [ ] Macro F1 > 0.65 on VocalMat test split
- [ ] Per-class precision ≥ 0.40 on every class
- [ ] Held-out 845 lab verdict: macro F1 > 0.80
- [ ] Held-out 845 USV/noise accuracy > 0.80
- [ ] Confusion matrix: no class collapses into another with > 0.40 mass
- [ ] `models/lab_classifier_v1/best.pt` (or under `--output-dir`) loads via `torch.load(...)["state_dict"]`
- [ ] `results/lab_classifier_v1/eval_report.md` + `confusion_matrix.png` exist
- [ ] (Conditional) Perch probe macro F1 reported (no gate)

**If per-class precision < 0.20 on Multi-steps or Reverse-Chevron:** STOP.
Do NOT collapse the taxonomy or move the thresholds. Revisit D5 first
(raise the per-class weight in `_compute_class_weights`, or add a
`WeightedRandomSampler`).

## Files NOT to touch (carry-over)

- `src/usv_spectrogram/corpus.py`
- `src/usv_spectrogram/app/core/{sliding_inference,notch,denoise}.py`
- `src/usv_spectrogram/postprocessing/normalization.py`
- `scripts/run_batch_detection.py`
- `src/usv_spectrogram/classifier/{cleaning_pipeline,diagnostics,resample,dataset}.py`
- `scripts/cnn_prepare_training_data.py`
- All pre-existing 18.2a/b tests + the 18.3 test-architect spec files
  (`test_{model,augmentation,losses,training,train_lab_classifier}.py`).
  You MAY add new test files; do NOT edit existing expectations.
- The shipped 18.3 source (`model.py`, `augmentation.py`, `losses.py`,
  `scripts/train_lab_classifier.py`). The ONLY file expected to gain code
  is `training.py` (Step 6 held-out real inference) + a confusion-matrix
  renderer (Step 8).

## When done

- Append a dated entry to `IMPLEMENTATION_PROGRESS.md`: per-class metrics,
  held-out 845 results, Perch comparison (or deferral verdict).
- Update `ops/goals.md` "Lab CNN Classifier" thread: Module 18.3 fully
  CLOSED, Module 18.4 (DANN) UNLOCKED.
- Write the Module 18.4 orchestrator handoff (DANN cage-invariance using
  the 235,726-row `domain_unlabeled.csv` — which needs the full 18 GB
  patch pool transferred, unlike 18.3).
- Run `/wrap-session` for an HTML results report, then commit + push.
- Mark this handoff complete.

## Tier-2 tickets inherited from 18.2b (BLOCKING for 18.4, not for this run)

1. `cleaning_pipeline.py` all-zero output on long lab WAVs when
   `baseline_mode='median_envelope'` chains with `_apply_global_mad`.
   Workaround at `cnn_prepare_training_data.py:376`.
2. `cnn_prepare_training_data.py:_collect_wav_rows` non-recursive `glob`
   skipped 853 wild WAVs; worked around by `cnn_wild_topup.py`.

Resolve both before Module 18.4.
