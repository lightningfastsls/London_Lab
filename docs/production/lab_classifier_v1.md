# Lab Syllable-Type Classifier — v1 (PRODUCTION)

> **What this is:** A 12-class mouse-USV syllable-type classifier — a timm ImageNet-pretrained
> ResNet-18 fine-tuned on the VocalMat-curated (Grimsley 2011 taxonomy) image set. It takes one
> spectrogram patch per call and returns a syllable class + softmax probabilities. Index 0 is
> `Noise`, so it doubles as a usv/noise gate.
> **Status:** PRODUCTION (Module 18.3, trained 2026-05-25). This is the current shipping classifier.
> **Production artifact:** `results/lab_classifier_v1/best.pt` (44.8 MB; ResNet-18 + 12-class head).
> **Do NOT use** `results/lab_classifier_v2/best.pt` — the DANN successor (Module 18.4) is a SHELVED
> dead end; see [§ v2/DANN is a shelved dead end](#v2dann-is-a-shelved-dead-end).
> **Sibling docs:** module spec `../modules/lab-classifier-v1.md`; cleaning-domain notes
> `../modules/cleaning-subsystems.md`; CNN *detection* (a different model) `../modules/cnn-classifier.md`.

The 12 classes (`Noise`, `Step up`, `Down-FM`, …) are syllable *types*, not the upstream USV
*detector*. This classifier assumes you already have a detected call and want to know what kind of
call it is. The detection pipeline (CNN + hysteresis) is separate — see
[CNN detection/classifier](../modules/cnn-classifier.md).

---

## 1. Operate

### 1.1 The 12 classes

The class list is `GRIMSLEY_12_CLASSES`, defined verbatim at
`src/usv_spectrogram/classifier/dataset.py:46-59`. **Order is load-bearing** — the model's output
column `i` maps to `GRIMSLEY_12_CLASSES[i]`, and index 0 (`Noise`) is the usv/noise collapse pivot.

| Idx | Class | Notes |
|---|---|---|
| 0 | `Noise` | Non-USV. `argmax == 0` → "noise" verdict when collapsing to binary. |
| 1 | `Step up` | |
| 2 | `Down-FM` | Downward frequency-modulated sweep. |
| 3 | `Short` | Short-duration call. |
| 4 | `Chevron` | |
| 5 | `Up-FM` | Upward frequency-modulated sweep. |
| 6 | `Flat` | |
| 7 | `Two steps` | |
| 8 | `Step down` | |
| 9 | `Complex` | Natural absorber of borderline shapes (lowest precision class). |
| 10 | `Reverse Chevron` | Minority class (~few hundred patches). |
| 11 | `Multi-steps` | Minority class; lowest recall. |

Taxonomy source: Grimsley et al. 2011. The names match VocalMat's published categories so the
VocalMat-curated training images transfer.

### 1.2 Running inference on new patches

The canonical, in-repo inference harness is
**`scripts/experiments/label_patches_v1.py`**. It is the live equivalent of the original
"v1 reproduction harness" (`archive/cleaning_legacy/stack1/scripts/experiments/patch_duration_sweep.py`)
and uses the exact same preprocessing. The published *faithful VocalMat port*
(`run_inference.py`) lives only inside the `vocalmat-classifier-test` worktree
(`docs/modules/cleaning-subsystems.md:112`); `label_patches_v1.py` reproduces its convention and is
the one to use day-to-day.

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/label_patches_v1.py \
    --manifest <MANIFEST>.csv \
    --output   <OUTPUT_LABELS>.csv \
    --checkpoint results/lab_classifier_v1/best.pt \
    --device cpu \
    --batch-size 64
```

Smoke command (reproduces the known held-out numbers — run this first to prove the model loads and
infers correctly before spending time on a large render):

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/label_patches_v1.py \
    --manifest data/lab_cnn_training/held_out_844/manifest.csv \
    --output   /tmp/labels_v1_held_out_844.csv
```

Because the held-out manifest carries a `usv_verdict` column, this also prints an oracle smoke
block whose **balanced accuracy should land at ~0.81** (see [§1.5](#15-headline-metrics)).

#### Arguments (`label_patches_v1.py:122-136`)

| Flag | Type | Default | What it does / when to change |
|---|---|---|---|
| `--manifest` | path | **required** | CSV with a column of PNG patch paths (default column name `path`). Paths may be absolute or relative to repo root. |
| `--output` | path | **required** | Where the per-patch labels CSV is written. Parent dirs are created. |
| `--checkpoint` | path | `results/lab_classifier_v1/best.pt` | The model. Leave at default for production. |
| `--path-column` | str | `path` | Manifest column holding the PNG path. |
| `--id-column` | str | `None` | Optional manifest column carried through as a `call_id` column (e.g. `wav_stem`). |
| `--device` | str | `cpu` | `cpu` or `cuda`. CPU is fine for a few thousand patches; use `cuda` on the rig for ~40k. |
| `--batch-size` | int | `64` | DataLoader-free manual batching; raise on GPU. |
| `--high-conf-threshold` | float | `0.85` | Rows with `top1_prob >=` this are flagged `high_confidence=True`. Used downstream as an "eval-gold" filter. |

The script **exits with an error** if the path column is missing, or if any referenced PNG does not
exist on disk (`label_patches_v1.py:149-159`).

### 1.3 Required input patch format (CRITICAL — this is a faithful VocalMat port)

The model was trained on VocalMat-convention patches and the inference transform is transcribed
verbatim from the reproduction harness. Patches MUST be produced the same way or the numbers are
meaningless.

The inference transform (`label_patches_v1.py:56-62`):

```python
transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
```

applied to `Image.open(p).convert("L")`. **There is NO ImageNet `Normalize`** — input pixels are
raw `[0,1]` after `ToTensor`. The `227×227` size is VocalMat's patch convention
(`docs/modules/lab-classifier-v1.md:52`); `Grayscale(num_output_channels=3)` replicates the single
luminance channel three times to fit the 3-channel ImageNet backbone.

The **patch rendering chain** that produced the training/eval PNGs (from
`patch_duration_sweep.py:13-16, 96-119`):

1. Resample source WAV **300 kHz → 250 kHz** (`scipy.signal.resample_poly`, factors `5/6`;
   constants at `src/usv_spectrogram/classifier/__init__.py:32-34`:
   `TARGET_SAMPLE_RATE_HZ=250_000`, `RESAMPLE_UP=5`, `RESAMPLE_DOWN=6`).
   Note: this 250 kHz target is **specific to this classifier**; it is NOT the corpus 300 kHz used
   everywhere else (`src/usv_spectrogram/corpus.py`). Do not propagate 250 kHz into other modules.
2. STFT (Hamming window, n_fft 1024, win 256, hop 128) → dB spectrogram (`_spectrogram_db`).
3. `clean_spectrogram(spec, CleaningConfig(baseline_mode="percentile"), recording_id=...)`
   (`patch_duration_sweep.py:108-109`). `baseline_mode="percentile"` is mandatory: the
   `median_envelope` mode chains badly with `_apply_global_mad` and produces all-zero patches on
   long lab WAVs (`docs/modules/lab-classifier-v1.md:301-307`).
4. Call-centered crop to a fixed window (~0.22 s in the original extraction), min-max → uint8,
   stack to 3-channel, save PNG.

If you only have detected calls and need patches, re-extract from the chunked WAVs in
`USV_lab_131204_chunked_2s_full/` using the chain above (this directory is **not present on the CPU
box** — see §2.5; the un-chunked WAVs live in `USV_lab_131204/`); **do not** feed contour-masked VAE patches
(`alpha3_patches`) — those are illegible vertical bands for this model. The human-readable substrate
is the normal spectrogram render, per the project memory note `feedback_shape_labeling_substrate`.

### 1.4 Output format

`label_patches_v1.py` writes one row per input patch (`label_patches_v1.py:167-178`):

| Column | Meaning |
|---|---|
| `call_id` | (only if `--id-column` given) value carried through from that manifest column. |
| `path` | the input PNG path (echoed from the manifest). |
| `top1_class` | predicted class name, `GRIMSLEY_12_CLASSES[argmax]`. |
| `top1_idx` | predicted class index 0–11. |
| `top1_prob` | softmax probability of the top-1 class (max over the 12). |
| `high_confidence` | bool, `top1_prob >= --high-conf-threshold` (default 0.85). |
| `softmax_12` | JSON list of all 12 softmax probabilities (5-dp rounded), in `GRIMSLEY_12_CLASSES` order. |

The script also prints, to stdout, the top-1 class distribution and (if `usv_verdict` is present)
the usv/noise smoke metrics.

For the held-out smoke set, the binary collapse rule is:
**`argmax == 0` (`Noise`) → noise verdict; any other class → usv verdict**
(`label_patches_v1.py:97-98`). This is how the 12-way head becomes a usv/noise gate.

A real example output already on disk: `results/alpha3/labels_v1_held_out_844.csv` (the 844 held-out
patches scored by this exact script).

### 1.5 Headline metrics

**VocalMat test/val split (the SHIP gate; `results/lab_classifier_v1/metrics.json` &
`eval_report.md`):**

| Metric | Value | Gate | Status |
|---|---|---|---|
| Macro-F1, VocalMat **test** split | **0.7669** (`metrics.json:3`) | > 0.65 | PASS |
| Macro-F1, VocalMat **val** split | 0.7693 (`metrics.json:2`) | informational | — |
| Min per-class precision | 0.5957 (`Complex`, idx 9) | ≥ 0.40 | PASS |
| Max off-diagonal confusion fraction | 0.083 (no class collapse) | < 0.40 | PASS |

Lowest recall is `Multi-steps` (idx 11) at 0.4286 (`metrics.json:30`) — not a gated metric but worth
noting on the smallest class. Full per-class precision/recall: `eval_report.md:39-52`.

> **WARNING — stale fields in `metrics.json`:** the keys `usv_noise_acc` (0.0) and
> `syllable_entropy_mean` (2.4849) at `metrics.json:202-203` are **garbage** from an accidental
> label-distribution fallback that keyed on the wrong CSV. **Ignore them.** The real held-out
> numbers come from the patch-loading evaluator below, not from `metrics.json`. See
> `eval_report.md:71-94`.

**Held-out lab verdict set (844 calls, 705 usv / 139 noise = 83.5% usv), v1 scored via the patch
loader:**

| Metric | Value | Source |
|---|---|---|
| Balanced accuracy (usv/noise) | **0.812** | `docs/plans/ROADMAP_ALPHA3_VOCALMAT_LABELED.md:55,63`; reproduced by the smoke command above |
| Noise recall | **0.640** | same |
| USV recall | **0.984** | same |

Interpretation: v1 almost never misses a real USV (98.4%) but catches only 64% of noise. **The 0.64
noise-recall is a lab→VocalMat DOMAIN GAP, not a cleaning bug** — `run_inference.py` is a faithful
VocalMat port (`docs/modules/cleaning-subsystems.md:112`,
`docs/handoffs/2026-05-27_repo-housekeeping-followup.md:13`). Closing it needs matched-domain data
(human-in-the-loop fine-tune), not a preprocessing tweak.

> **Pooled accuracy is misleading here.** On an 83.5%-usv set a trivial "always usv" classifier
> scores 0.835. Always read the per-class (noise-recall / usv-recall) rows, never pooled accuracy
> (`scripts/evaluate_held_out_844.py:25-30`).

### 1.6 Patch duration is an operating-point knob, not a fix

The vault hypothesis was that shrinking the analysis window so a short call *fills* more of the
227×227 patch would improve transfer. The sweep
(`archive/cleaning_legacy/stack1/scripts/experiments/patch_duration_sweep.py`, durations
`[0.22, 0.14, 0.08, 0.05]` s at line 47) **falsified that**: balanced accuracy stays flat at ~0.81
across all four durations. Shrinking the window only **trades usv-recall ↔ noise-recall** — it moves
the operating point along the same curve; it does not improve discrimination. Per the memory note
`project_lab_transfer_v1_vs_dann_patchsweep`. So: if you need more noise rejection at the cost of
some USV recall, a shorter window is a lever — but do not expect a better classifier from it.

### 1.7 v2/DANN is a SHELVED dead end

**Do NOT use `results/lab_classifier_v2/best.pt`.** Module 18.4 trained a domain-adversarial (DANN)
ResNet-18 to remove the cage confound. It **destroyed base noise-rejection**:

| Metric | v1 (production) | v2 (DANN) | Source |
|---|---|---|---|
| Syllable macro-F1 (test) | 0.7669 | 0.6359 | `results/lab_classifier_v2/comparison_v1_vs_v2.md:5` |
| Held-out balanced accuracy | 0.812 | **0.5756** | `results/lab_classifier_v2/held_out_844_eval.md:16` |
| Held-out noise recall | 0.640 | **0.1583** | `held_out_844_eval.md:13` |
| Held-out usv recall | 0.984 | 0.9929 | `held_out_844_eval.md:14` |
| Cage linear-probe acc | n/a | 1.000 (still cage-decodable) | `comparison_v1_vs_v2.md:7` |

v2 tripped the collapse tripwire (F1 drop > 0.05) **and** the cage gate (probe ≥ 0.65), verdict
**DO NOT SHIP** (`comparison_v1_vs_v2.md:13-14`). The v2 checkpoint is a phantom — it was evaluated and
rejected and `results/lab_classifier_v2/best.pt` **is not on disk** (verified 2026-06-21; only the v2
eval artifacts — `comparison_v1_vs_v2.md`, `held_out_844_eval.{md,json}`, `metrics.json` — were kept).
The DANN approach was SHELVED 2026-05-27 (`project_lab_cnn_classifier_scope`).

### 1.8 Troubleshooting / Gotchas

- **`ModuleNotFoundError: usv_spectrogram`** → you forgot `PYTHONPATH=src`. All commands above set it.
- **Wrong/garbage held-out numbers from `metrics.json`** → you read `usv_noise_acc` /
  `syllable_entropy_mean`. Those are stale. Use the patch-loading smoke run instead (§1.2).
- **All-zero / black patches on long lab WAVs** → you used `CleaningConfig(baseline_mode="median_envelope")`.
  Use `baseline_mode="percentile"` (`patch_duration_sweep.py:109`).
- **Patches look like vertical bands and predictions are nonsense** → you fed contour-masked VAE
  patches (`alpha3_patches`). This model wants normal spectrogram renders.
- **Noise recall only ~0.64, seems low** → expected. It is a lab→VocalMat domain gap, not a bug;
  do not "fix" it by changing cleaning. v1 trades almost no USV recall for it (0.984).
- **You ran inference at 300 kHz** → wrong. This classifier's chain resamples to **250 kHz**
  (`classifier/__init__.py:32-34`). 250 kHz is local to this module only.
- **`best.pt` loads with "missing/unexpected keys"** → the loader uses `strict=False` and prints a
  count (`label_patches_v1.py:70-74`); a small count from buffer naming differences is benign, but a
  large missing-key count means you built the wrong architecture (it must be a plain
  `build_resnet18_classifier(num_classes=12)`, NOT `ResNet18DANN`).
- **Checkpoint structure**: `best.pt` is a dict; the weights are under `["state_dict"]`
  (`label_patches_v1.py:67-68`, `eval_report.md:17`).

---

## 2. Internals

### 2.1 Architecture

`build_resnet18_classifier(num_classes=12, pretrained=True)` at
`src/usv_spectrogram/classifier/model.py:20-45` wraps
`timm.create_model("resnet18", pretrained=pretrained, num_classes=12)`. Input `(B, 3, 227, 227)`,
output `(B, 12)` logits. `NUM_CLASSES = 12` at `model.py:17`.

ResNet-18 (not a deeper net) was chosen because the supervised set is small (~12k examples); deeper
backbones overfit (`docs/modules/lab-classifier-v1.md:54-60`). EfficientNet-B0 is the documented
fallback if it underfits — it didn't (0.77 macro-F1 cleared the 0.65 gate).

At inference time `pretrained=False` is passed (`label_patches_v1.py:69`) — all weights come from
`best.pt`, not from ImageNet.

### 2.2 Class mapping

`GRIMSLEY_12_CLASSES` (`src/usv_spectrogram/classifier/dataset.py:46-59`) is the single source of
truth for index↔name and is imported everywhere (training, inference, evaluators). `Noise` is
**index 0** and is the usv/noise collapse pivot (`evaluate_held_out_844.py:62`,
`label_patches_v1.py:52`).

### 2.3 Inference data flow

`scripts/experiments/label_patches_v1.py`:
- `load_oracle(ckpt, device)` (`:65-75`) — `torch.load(..., map_location="cpu")`, unwrap
  `["state_dict"]`, `build_resnet18_classifier(num_classes=12, pretrained=False)`,
  `load_state_dict(strict=False)`, `.eval().to(device)`.
- `score_paths(model, paths, device, batch_size)` (`:78-87`) — manual batching, `_TF` transform per
  PNG, `torch.softmax(logits, dim=-1)` → `(N, 12)` probability matrix. Decorated `@torch.no_grad()`.
- `usv_noise_smoke(argmax12, verdict)` (`:90-119`) — binary collapse + balanced-accuracy smoke.
- `main()` (`:122-203`) — arg parsing, path resolution/existence check, scoring, CSV write,
  distribution print, optional smoke.

The standalone binary evaluator `scripts/evaluate_held_out_844.py` is **v2-targeted by default**
(`--checkpoint` defaults to `results/lab_classifier_v2/best.pt`, line 264) and builds a
`ResNet18DANN`, so it is the wrong tool for v1. For v1, use `label_patches_v1.py`. (The evaluator's
`_binary_metrics` / `_twelve_class_breakdown` at `:148-208` are the reference implementations for
the per-class metric definitions, including the "pooled accuracy is misleading" guard at `:197`.)

### 2.4 Training (for the record — already done, do not re-run casually)

Training code: `src/usv_spectrogram/classifier/{model,augmentation,losses,training}.py`. CLI:
`scripts/train_lab_classifier.py` (the live copy now sits in
`archive/cleaning_legacy/stack1/scripts/train_lab_classifier.py`; the canonical CLI flag table is
`docs/modules/lab-classifier-v1.md:202-216`).

Key training facts (`docs/modules/lab-classifier-v1.md`, `eval_report.md`):
- Backbone: timm ResNet-18, ImageNet-pretrained.
- Loss: class-weighted focal loss, `gamma=2.0`, **plain-mean reduction** (`sum/B`, NOT PyTorch's
  weighted-mean) — see `losses.py` and `docs/modules/lab-classifier-v1.md:109-122`. The plain-mean
  choice preserves the minority-class up-weighting that weighted-mean would cancel.
- Class weights: inverse-frequency from the train split, normalized so the mean weight is 1.0
  (sum = 12) — `dataset.py` `build_stratified_split`.
- Optimizer: AdamW, `weight_decay=1e-4`; cosine LR with linear warmup (3 epochs in the real run).
- Augmentation: SpecAugment (Park 2019; 2 time masks ≤20 frames, 2 freq masks ≤16 bins) +
  cage-noise injection (prob 0.25, blend weight U[0.05, 0.30]) — `augmentation.py`,
  `docs/modules/lab-classifier-v1.md:62-95`.
- Run: 50 epochs, batch 64, 9,741 training patches, ~16 min on one RTX 3060 Ti (rig `cloudyclaude`),
  2026-05-25 (`eval_report.md:5-8`).

**Split invariant (do not violate when re-splitting):** every `source_recording` lands in exactly
one of train/val/test (recording-level grouping), because raw spectrograms cluster by recording
environment ("cage"), not biology — splitting a recording across folds leaks the cage feature and
inflates the score (`dataset.py:5-16`). Per-class stratification is kept within ±5%.

### 2.5 Training data location

- Training/val/test manifests: `data/lab_cnn_training/{train,val,test}/manifest.csv` (per the
  production training command, `docs/modules/lab-classifier-v1.md:225-232`). On this CPU box only the
  held-out set is currently materialized under `data/lab_cnn_training/` — the train/val/test patch
  folders were rig-side during training.
- Held-out lab verdict set (844 rows): `data/lab_cnn_training/held_out_844/manifest.csv` +
  `data/lab_cnn_training/held_out_844/patches/*.png` (844 PNGs). Manifest columns:
  `path, usv_verdict, wav_stem, det_start_s, det_end_s, source_review, traditional_label, couple`.
  `usv_verdict ∈ {usv, noise}`. **This set has only usv/noise verdicts, NOT Grimsley labels** — lab
  131204 was never 12-class labeled, which is why there is no Grimsley held-out macro-F1 gate
  (`eval_report.md:71-83`).
- Source WAVs for re-extraction: `USV_lab_131204_chunked_2s_full/` (`patch_duration_sweep.py:46`).
  **Not materialized on the CPU box** (verified 2026-06-21) — like the train/val/test patch folders
  it was rig-side. The un-chunked source WAVs do exist locally under `USV_lab_131204/` (83 files);
  the 2 s chunked-render directory must be regenerated (or rsynced from the rig) before any
  re-extraction.

### 2.6 Result artifacts (`results/lab_classifier_v1/`)

| File | What it is |
|---|---|
| `best.pt` (44.8 MB) | the model; weights under `["state_dict"]`. |
| `metrics.json` | val/test macro-F1, per-class precision/recall, 12×12 confusion. **`usv_noise_acc` / `syllable_entropy_mean` are stale — ignore.** |
| `eval_report.md` | the SHIP-gate report (authoritative for VocalMat metrics). |
| `confusion_matrix.png` | 12×12 confusion render. |
| `per_class_pr.png`, `cross_domain_usv_noise.png` | diagnostic plots. |
| `test_exemplars_correct.png`, `test_mistakes.png` | qualitative panels. |

### 2.7 Where to change things

- **New patches to label** → write a manifest with a `path` column, run `label_patches_v1.py`. No
  code change.
- **Different usv/noise operating point** → re-extract at a shorter window (§1.6). The collapse
  threshold itself (`argmax==0`) is structural; don't change it.
- **Retrain / fine-tune** → `src/usv_spectrogram/classifier/` + `scripts/train_lab_classifier.py`.
  Honor the recording-level split invariant (§2.4) and the plain-mean focal reduction. The next
  real improvement is matched-domain fine-tuning to close the noise-recall gap, NOT another DANN
  attempt (which failed — §1.7).
- **Do NOT** edit `corpus.py` or assume 300 kHz here; this module is 250 kHz internally.
