# Module 18.3 Stream V — rig session closeout (2026-05-25)

**Origin:** GPU rig (`cloudyclaude`, `/data/mickey_london_lab/` non-git rsync deployment).
**Predecessor handoff:** `docs/handoffs/2026-05-25_rig-claude-code-18.3-training.md` (this session executed it).
**Status:** **18.3 SHIP** on the VocalMat gates. Held-out lab evaluation deferred to Module 18.4 prep per CPU-box clarification 2026-05-25. Perch 2.0 probe deferred (Option B).

---

## TL;DR

- Step 5 (50-epoch training) — **DONE** in 16 min on GPU 0. All VocalMat SHIP gates PASS: macro F1 val=0.7693, test=0.7669, min per-class precision=0.5957 (Complex), confusion-matrix collapse max=8% off-diagonal (gate < 40%).
- Step 6 (held-out 845 / 844 lab evaluation) — **DEFERRED** to Module 18.4 prep. Two upstream errors corrected (see §Corrections below); `_evaluate_held_out_845` left as-is.
- Step 7 (Perch 2.0 linear probe) — **DEFERRED** (Option B). No new code; no SHIP-gate consequence.
- Step 8 (confusion-matrix PNG renderer) — **DONE** + verified. Inline render in `train_lab_classifier.py` + standalone back-fill helper + 8 new tests, all green.
- Test suite — **186 passed, 5 skipped, 0 failed** (vs 178/4 baseline; +8 new from Step 8).

---

## Corrections to upstream handoffs

Two errors in the predecessor chain (`2026-05-22_…`, `2026-05-24_…`, `2026-05-25_rig-…`) were resolved by Shachar on 2026-05-25:

1. **`held_out_845_macro_f1 > 0.80` gate is unbuildable and is hereby WITHDRAWN, not failed.** Lab 131204 was only ever clustered (Cluster_N) and sample-verdicted (usv/noise) — never labeled with the 12 Grimsley classes. There is no Grimsley ground truth for lab data. The PLAN never asked for it; only USV/noise accuracy + entropy. The gate was over-specified in `docs/handoffs/2026-05-24_module-18.3-stream-v-gpu-training.md` and propagated through 2026-05-25.

2. **The wrong CSV was cited as the "held-out 845 verdict set" throughout.** `classified_detections_lab_131204_clean.csv` (40,787 rows of `Cluster_*` upstream-clustering output) is NOT the verdict set; it's the clustering working set. The real held-out set is **844 usv/noise verdicts** in `results/lab_{cluster0,cluster1,cluster2,noise}_review/review_index_annotated.csv` (CPU-box-only). But even with the right file, scoring requires re-extracting clean 227×227 patches from `USV_lab_131204_chunked_2s_full/` — the annotated-figure RGBAs in the review directories are variable-size and don't join to the domain pool. **Re-extraction is a separate prep task, inherited by Module 18.4** (natural fit — DANN's target IS the lab domain).

CPU box should fix the cite in `ROADMAP_lab_cnn_classifier.md` §18.3, `docs/modules/lab-classifier-v1.md` §"Held-out 845 evaluation", `docs/handoffs/2026-05-22_…`, `2026-05-24_…`. The rig has already corrected `results/lab_classifier_v1/eval_report.md`.

---

## SHIP-gate verification (VocalMat gates only)

| Gate | Threshold | Actual | Status |
|---|---|---|---|
| `pytest tests/classifier/ -k "not auto_device_resolves_cpu"` | 0 failed | 186 passed, 5 skipped | PASS |
| Macro F1 on VocalMat test split | > 0.65 | 0.7669 | PASS |
| Per-class precision (min) | ≥ 0.40 | 0.5957 (Complex) | PASS |
| Confusion-matrix max off-diagonal mass | < 0.40 | 0.083 | PASS |
| `best.pt` exists + loads via `torch.load(...)["state_dict"]` | yes | 44.8 MB | PASS |
| `confusion_matrix.png` exists | yes | 1275×1125 RGBA | PASS |
| `results/lab_classifier_v1/eval_report.md` exists | yes | written | PASS |
| Held-out lab Grimsley macro F1 | (withdrawn) | — | n/a |
| Held-out lab USV/noise accuracy | (deferred) | — | DEFERRED |
| Perch 2.0 probe | (no gate) | — | DEFERRED (Option B) |

All non-deferred gates: **PASS**. Module 18.3 is SHIP-eligible on the VocalMat result.

---

## Entry to append to `IMPLEMENTATION_PROGRESS.md` (CPU box)

```
### 2026-05-25 — Module 18.3 Stream V (GPU real-data results)

SHIP on VocalMat gates. Held-out lab eval deferred to Module 18.4 prep.

Training: 50 epochs, batch 64, AdamW lr=1e-3, warmup=3, focal γ=2.0, ResNet-18
+ ImageNet pretrained. Single RTX 3060 Ti (co-tenanted), ~16 min wall-clock.

VocalMat val/test (PASS):
  - macro_f1_val  = 0.7693
  - macro_f1_test = 0.7669

Per-class precision (test split; gate ≥ 0.40 on every class — PASS):
  Noise 0.9837 | Step up 0.8146 | Down-FM 0.8021 | Short 0.8010
  Chevron 0.9051 | Up-FM 0.8051 | Flat 0.8438 | Two steps 0.7027
  Step down 0.7179 | Complex 0.5957 | Reverse Chevron 0.6471 | Multi-steps 0.6000

Per-class recall (test split):
  Noise 0.8963 | Step up 0.8056 | Down-FM 0.8701 | Short 0.8947
  Chevron 0.7799 | Up-FM 0.8051 | Flat 0.7168 | Two steps 0.7429
  Step down 0.7179 | Complex 0.8000 | Reverse Chevron 0.8462 | Multi-steps 0.4286

Confusion-matrix interpretation: largest off-diagonal mass is Two-steps →
Step-up at 15.7% (well under the 40% collapse threshold). Step-up → Two-steps
6.1%, Chevron → Complex 5.7% — all taxonomically adjacent, not class collapse.

Held-out lab eval: DEFERRED to Module 18.4 prep. Two errors corrected:
  1. held_out_845_macro_f1 > 0.80 gate withdrawn — lab data never Grimsley-
     labeled, so Grimsley macro F1 on lab is unbuildable.
  2. Real verdict file is results/lab_{cluster0,cluster1,cluster2,noise}_
     review/review_index_annotated.csv (844 usv/noise verdicts), NOT
     classified_detections_lab_131204_clean.csv. Even the right file needs
     clean 227×227 patches re-extracted from USV_lab_131204_chunked_2s_full/
     — owner: CPU box / Module 18.4.

Perch 2.0 probe: DEFERRED (Option B). No new code, no gate.

Code shipment (rsync rig → CPU box):
  - scripts/train_lab_classifier.py — added numpy import, inline confusion-
    matrix render call, _render_confusion_matrix_png helper.
  - scripts/render_confusion_matrix.py — NEW standalone back-fill helper.
  - tests/classifier/test_render_confusion_matrix.py — NEW. 8 tests.

Test suite: 186 passed, 5 skipped, 0 failed (vs 178/4 deploy-time baseline).

Artifacts on rig (rsync these back):
  - results/lab_classifier_v1/best.pt (44.8 MB)
  - results/lab_classifier_v1/metrics.json — held-out fields contain stale
    garbage (0.0 / 2.4849) from the label-distribution fallback firing on
    the wrong CSV; IGNORE those fields, the val/test/per-class data is fine.
  - results/lab_classifier_v1/confusion_matrix.png
  - results/lab_classifier_v1/eval_report.md
```

---

## Update to append to `ops/goals.md` "Lab CNN Classifier" thread

```
Module 18.3 — Lab USV syllable classifier (12-class ResNet-18):
  STATUS: SHIP on VocalMat gates (2026-05-25).
  Held-out lab eval: deferred to Module 18.4 prep (re-extraction task).
  Perch 2.0 probe: deferred (Option B, no gate).
  Result: val 0.7693, test 0.7669 macro F1; min per-class precision 0.5957.

Module 18.4 — DANN cage-invariance training: UNLOCKED.
  Prerequisite (Tier-2, blocking 18.4 evaluation, not training):
    - Re-extract clean 227×227 patches from USV_lab_131204_chunked_2s_full/
      for the 844 review_index_annotated.csv usv/noise verdicts, so the
      held-out lab eval can finally be scored.
```

---

## Rsync recipe (rig → CPU box)

Run from the CPU box:

```bash
RIG=cloudyclaude  # or 100.113.224.57
DEST=/home/shachar/projects/mickey_london_lab/.claude/worktrees/lab-cnn-classifier-plan

# Code changes
rsync -aR \
    $RIG:/data/mickey_london_lab/./scripts/train_lab_classifier.py \
    $RIG:/data/mickey_london_lab/./scripts/render_confusion_matrix.py \
    $RIG:/data/mickey_london_lab/./tests/classifier/test_render_confusion_matrix.py \
    $RIG:/data/mickey_london_lab/./docs/handoffs/2026-05-25_rig-stream-v-closeout.md \
    $RIG:/data/mickey_london_lab/./docs/handoffs/2026-05-25_module-18.4-dann-orchestrator.md \
    $DEST/

# Results (best.pt is 44.8 MB; --partial-dir guards interrupted transfers)
rsync -aR --partial-dir=.partial \
    $RIG:/data/mickey_london_lab/./results/lab_classifier_v1/ \
    $DEST/

# Then on CPU box: git status, commit, push.
```

---

## What happens next

1. CPU-box session reads this closeout, runs the rsync recipe, commits the code changes + results to the worktree branch `worktree-lab-cnn-classifier-plan`, opens a PR or merges as appropriate.
2. CPU-box session appends the IMPLEMENTATION_PROGRESS.md entry above, updates `ops/goals.md`, fixes the wrong-CSV citations in `ROADMAP_lab_cnn_classifier.md` + `docs/modules/lab-classifier-v1.md` + the predecessor handoffs.
3. Module 18.4 starts via `docs/handoffs/2026-05-25_module-18.4-dann-orchestrator.md`.

This handoff: marked complete by appending a "Closed by 2026-05-25_rig-stream-v-closeout.md" line to the bottom of `docs/handoffs/2026-05-25_rig-claude-code-18.3-training.md` on the CPU box.
