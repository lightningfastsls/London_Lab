# ERRATA — the "held-out 845" lab verdict set (Module 18.3)

**Date:** 2026-05-25
**Status:** Authoritative correction. Supersedes the held-out-845 claims in
`PLAN_lab_cnn_classifier.md`, `ROADMAP_lab_cnn_classifier.md` §18.3, the three
Module 18.3 handoffs, `docs/modules/lab-classifier-v1.md`, and the vault note
`notes/held-out dual-rater verdicts serve as independent acceptance test...md`.

Discovered when the rig session tried to run Step 6 (held-out evaluation) and
hit a data-shape mismatch. Two distinct errors compounded.

## Error 1 — the file identity is wrong

Every doc that names a file for the "845 dual-rater verdicts" points to
`classified_detections_lab_131204_clean.csv`. **That is the wrong file.** It is
the **40,787-row lab clustering working set** (columns `label`=`Cluster_1..26`,
`tier`, `det_user_action`=all-NaN). It has **no per-row USV/noise verdict and no
Grimsley class labels.**

The **real** verdict set is **844 rows** across four cluster-review CSVs:

```
results/lab_cluster0_review/review_index_annotated.csv   (200)
results/lab_cluster1_review/review_index_annotated.csv   (244)
results/lab_cluster2_review/review_index_annotated.csv   (200)
results/lab_noise_review/review_index_annotated.csv      (200)   = 844
```

Relevant columns: `verdict` (705 `usv` / 139 `noise`), `wav_stem`,
`det_start_s`, `det_end_s`, `png`. **No Grimsley class column** — only
`traditional_label` = `Cluster_N`.

## Error 2 — the "Grimsley macro F1 > 0.80" held-out gate is unbuildable

The predecessor handoff (`2026-05-24_module-18.3-resnet-supervised-baseline.md`
line 80), the Stream V handoff, the rig handoff, the module doc, and ROADMAP
line 818 all list **"Held-out 845 lab verdict: macro F1 > 0.80"**. This gate
**cannot exist** and is **removed**.

Reason: lab 131204 was only ever (a) clustered into `Cluster_N` (HDBSCAN, a
different taxonomy than Grimsley) and (b) sample-verdicted `usv`/`noise`. It was
**never hand-labeled into the 12 Grimsley classes**, so there is no Grimsley
ground truth for lab data anywhere. No re-extraction or join can produce it
(short of a new 12-class hand-labeling effort). The model is trained on
VocalMat (which has Grimsley labels); our lab data has only cluster IDs +
binary verdicts.

The PLAN's actual validation criteria (PLAN §"Validation criteria", and ROADMAP
§18.3 exit criteria) were always only:

- **Held-out USV/noise accuracy > 0.80**
- **Held-out syllable-type entropy mean ≤ log(6) ≈ 1.79**

The Grimsley-macro-F1 framing was introduced in the handoffs, inconsistent with
PLAN/ROADMAP. Drop it.

## Error 3 (blocker for even the two real gates) — no clean patches on disk

Even the two real gates can't run as-is. The only images for the 844 calls are
**annotated matplotlib review figures** (variable-size RGBA, ~424×306, with axes
and chrome) — unusable as model input (the model expects clean 227×227
VocalMat-style patches). They do **not** join to the 18 GB domain pool
(`data/lab_cnn_training/domain_unlabeled.csv` keys patches by sequential
`pNNNN`, with no `det_start_s` timing key).

## What Module 18.3 actually shipped

The model passed every gate the PLAN set for it, on the **VocalMat held-out
test split** (the proper labeled accuracy evaluation):

- macro F1 val 0.7693, test 0.7669
- min per-class precision 0.5957 (Complex)
- no class collapse

The **held-out lab eval is DEFERRED** (it was always a coarse domain-transfer
sanity probe, not the accuracy result). It is not a Module 18.3 blocker.

## Prep task to actually run the held-out USV/noise + entropy gates (future)

Grounded and bounded; owner = CPU box or Module 18.4:

1. Concat the 4 review CSVs → 844 rows (`verdict`, `wav_stem`, `det_start_s`, `det_end_s`).
2. Lab WAVs exist at `USV_lab_131204_chunked_2s_full/{wav_stem}.wav` (verified).
   Re-extract each call at `[det_start_s, det_end_s]` through the **same**
   cleaning+extraction as training (import `_wav_to_patches` from the frozen
   `scripts/cnn_prepare_training_data.py`, `baseline_mode='percentile'`,
   227×227). Do NOT modify that frozen script.
3. Write a held-out manifest (`path`, `verdict`) + the 844 clean patches.
4. Eval: `usv_noise_acc = mean((argmax != Noise_index) == (verdict=='usv'))`;
   `entropy_mean = mean(-Σ softmax·log softmax)`; check ≤ log(6).

This unblocks **only** USV/noise accuracy + entropy — never a Grimsley gate.

## Read-only note flagged for KG pipeline correction

`notes/held-out dual-rater verdicts serve as independent acceptance test for
single-rater-trained classifiers.md` repeats the file-identity error (cites
`classified_detections_lab_131204_clean.csv` as the 845 dual-rater set) and
implies a Grimsley-quality verdict set. `notes/` is READ-ONLY; this needs
correction via the /reduce pipeline (or a /reweave pass), not a direct edit.
The *methodological principle* in the note (use the highest-quality label
subset as the acceptance test) is sound; only the concrete file + the
implication that lab calls carry Grimsley labels are wrong.
