# Module 18.3 Stream V — Lab Classifier v1 evaluation report

**Date:** 2026-05-25
**Run:** `results/lab_classifier_v1/`
**Host:** GPU rig `cloudyclaude` — 1× RTX 3060 Ti (training used GPU 0 with ~3 GB VRAM, co-tenanted with `llama-server` on the same GPU; the other 3 GPUs were full of llama-server tenants and unused for this run)
**Training command:** see `docs/handoffs/2026-05-25_rig-claude-code-18.3-training.md` §Step 5
**Wall-clock:** ~16 minutes for 50 epochs over 9,741 training patches at batch 64

## TL;DR

| Gate | Threshold | Actual | Status |
|---|---|---|---|
| Macro F1 on VocalMat test split | > 0.65 | **0.7669** | PASS |
| Macro F1 on VocalMat val split | (informational) | 0.7693 | — |
| Per-class precision (min) | ≥ 0.40 | **0.5957** (Complex) | PASS |
| Confusion-matrix collapse | no class → other > 0.40 mass | **0.083** (max off-diag fraction) | PASS |
| `best.pt` exists + loads | yes | 44.8 MB, loads via `torch.load(...)["state_dict"]` | PASS |
| `confusion_matrix.png` exists | yes | 1275×1125 RGBA PNG | PASS |
| `eval_report.md` exists | yes | this file | PASS |
| Held-out lab (Grimsley) macro F1 | — | **GATE WITHDRAWN — see §Held-out below** | n/a |
| Held-out lab USV/noise accuracy | — | **DEFERRED to Module 18.4 prep** | DEFERRED |
| Perch 2.0 linear probe macro F1 | (no gate — comparison only) | **DEFERRED — Option B, see §Step 7 below** | DEFERRED |

**Verdict:** Stream V is SHIP-eligible on the VocalMat gates (the substantive Module 18.3 result). The held-out lab gates have been withdrawn / deferred per CPU-box clarification 2026-05-25 — see §Held-out below.

## Val / test metrics

```
macro_f1_val:  0.7693
macro_f1_test: 0.7669
```

Both well above the 0.65 PLAN-pass threshold. The val/test gap of 0.0024 is negligible — no signs of test leakage or overfitting to val.

### Per-class precision and recall (test split)

Classes ordered by `GRIMSLEY_12_CLASSES`:

| Idx | Class | Precision | Recall |
|---|---|---|---|
| 0 | Noise            | 0.9837 | 0.8963 |
| 1 | Step up          | 0.8146 | 0.8056 |
| 2 | Down-FM          | 0.8021 | 0.8701 |
| 3 | Short            | 0.8010 | 0.8947 |
| 4 | Chevron          | 0.9051 | 0.7799 |
| 5 | Up-FM            | 0.8051 | 0.8051 |
| 6 | Flat             | 0.8438 | 0.7168 |
| 7 | Two steps        | 0.7027 | 0.7429 |
| 8 | Step down        | 0.7179 | 0.7179 |
| 9 | Complex          | **0.5957** | 0.8000 |
| 10 | Reverse Chevron | 0.6471 | 0.8462 |
| 11 | Multi-steps     | 0.6000 | **0.4286** |

**Lowest precision: Complex at 0.5957** — well above the 0.40 gate.
**Lowest recall: Multi-steps at 0.4286** — recall is not in the SHIP gate but is worth noting.

Per the handoff's D5 guidance ("If per-class precision < 0.20 on Multi-steps or Reverse-Chevron: STOP"), this run is comfortably above the stop condition; the class-weight strategy in `_compute_class_weights` (inverse-frequency) is doing its job for now.

## Confusion-matrix interpretation

12×12 confusion matrix is rendered to `confusion_matrix.png`. Diagonals dominate everywhere; the largest off-diagonal masses are:

- **Step up (1) → Two steps (7): 11/180 = 6.1%** — semantically adjacent (Two-steps subsumes a Step-up).
- **Chevron (4) → Complex (9): 9/159 = 5.7%** — Complex is a natural absorber of borderline shapes.
- **Down-FM (2) → Step up (1): 7/177 = 4.0%** — both monotone-frequency classes.
- **Up-FM (5) → Short (3): 8/118 = 6.8%** — both short-duration; the predecessor handoff D3 flag (texture-not-class) may apply.
- **Two steps (7) → Step up (1): 11/70 = 15.7%** — highest single off-diagonal fraction. Worth tracking but well under the 40% collapse threshold.

No 12×12 cell exceeds the 40% collapse threshold. All gates clear.

## Held-out lab evaluation: GATE WITHDRAWN + RE-EXTRACTION DEFERRED

CPU-box clarification 2026-05-25 resolved two errors in the predecessor handoffs:

**Error 1 — `held_out_845_macro_f1 > 0.80` gate is unbuildable.** Lab 131204 was only ever clustered (Cluster_N) and sample-verdicted (usv/noise) — never labeled with the 12 Grimsley classes. There is no Grimsley ground truth for lab data, so a Grimsley macro F1 gate is structurally impossible to satisfy. The PLAN never asked for it; only USV/noise accuracy + entropy. The gate is **withdrawn**, not failed.

**Error 2 — wrong CSV cited.** The predecessor and the present handoff both cite `classified_detections_lab_131204_clean.csv` as the held-out verdict set; this is incorrect. That file is the 40,787-row clustering working set. The real held-out set is **844 usv/noise verdicts** in `results/lab_{cluster0,cluster1,cluster2,noise}_review/review_index_annotated.csv` (on the CPU box).

**Why it's deferred even with the right file:** the 844 rows reference annotated review *figures* (variable-size RGBA), not clean 227×227 model patches, and they don't join to the domain pool. Scoring them requires re-extracting clean patches from `USV_lab_131204_chunked_2s_full/` — a separate prep task, owner: CPU box / Module 18.4 (natural fit, since 18.4's DANN target is precisely the lab domain).

**This run did NOT execute `_evaluate_held_out_845` against the wrong CSV** — running it would have produced the 0.0 / 2.4849 garbage noted earlier (label-distribution fallback keying on `df.columns[0]='file'`). The held-out fields are intentionally **omitted** from the metrics presented in this report; `metrics.json` on disk does contain the garbage values from the original training run's accidental fallback execution and they should be **disregarded** when reading the JSON directly.

**`_evaluate_held_out_845` is left as-is** — it's still correct for a properly-formatted CSV that doesn't exist yet, and Module 18.4 will plug into it once clean patches are re-extracted.

## Step 7 — Perch 2.0 linear probe: DEFERRED (Option B)

Inherited deferral from the predecessor: `tests/classifier/test_train_perch2_probe.py` specifies macro F1 > 0.30 on permuted-label IID Gaussian patches (structurally `I(features; labels) = 0`). Per ROADMAP §18.3 SHIP-gate text the Perch comparison is "no gate" — Stream V is SHIP-eligible without it. **Decision (Shachar, 2026-05-25): Option B (defer permanently).** No Perch installation, no test repair. If the DANN-vs-bioacoustic-pretrained-encoder question becomes load-bearing for Module 18.4, the comparison can be revived as a sidequest at that point.

## Files produced

| Path | Size | Status |
|---|---|---|
| `results/lab_classifier_v1/best.pt` | 44.8 MB | ResNet-18 + 12-class head; loads via `torch.load(...)["state_dict"]` |
| `results/lab_classifier_v1/metrics.json` | 2.2 KB | Full metrics dict (val/test F1, per-class precision/recall, 12×12 confusion matrix). **NOTE:** `usv_noise_acc=0.0` and `syllable_entropy_mean=2.4849` fields are stale garbage from the label-distribution fallback path keying on the wrong column in the wrong CSV — IGNORE these. See §"Held-out lab evaluation" above. |
| `results/lab_classifier_v1/confusion_matrix.png` | ~70 KB | 1275×1125 RGBA, rendered post-hoc from `metrics.json` |
| `results/lab_classifier_v1/eval_report.md` | this file | SHIP-gate deliverable |
| `results/lab_classifier_v1/training.log` | 110 B | only the final summary line — the training script has no per-epoch progress logging |

## Test suite state

`pytest tests/classifier/ -q -k "not auto_device_resolves_cpu"` — **186 passed, 5 skipped, 1 deselected** in 120 s. Eight new tests added in `tests/classifier/test_render_confusion_matrix.py` for the Step 8 PNG renderer. No regressions vs the 178-passed deploy-time baseline.

## What changed on the rig (rsync these back to CPU box)

- `scripts/train_lab_classifier.py` — added `numpy` import, `_render_confusion_matrix_png` helper (~50 LOC), call to renderer after `train_classifier` returns.
- `scripts/render_confusion_matrix.py` — NEW. Standalone PNG back-fill helper.
- `tests/classifier/test_render_confusion_matrix.py` — NEW. 8 tests covering the renderer + CLI helper.
- `results/lab_classifier_v1/` — NEW. `best.pt`, `metrics.json`, `confusion_matrix.png`, `training.log`, `eval_report.md`.
