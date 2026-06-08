# Handoff — Redo the CNN-iteration evaluation correctly (2026-05-27)

## Why this handoff exists
We are building an **illustrative** "CNN got better across iterations" comparison
(no scientific weight — just to show progress). A prior session (this one) produced
a result the user correctly rejected as **wrong**:

> "production's FP filter drops ~70% of real USVs (recall 69% raw → 18.8% after pipeline)"

The user's instinct: *"No way the FP filter drops that many USVs — you're doing
something wrong."* They are right. Two things to fix:
1. **The eval is methodologically broken** (provenance mismatch — see below).
2. **We used the wrong, tiny label set** (34 files / 178 intervals). The user says
   we have **~15,000 labels**. We do — they were never loaded.

Your job: (A) settle **which FP filter** the user means, (B) **find & load all
~15k labels**, (C) **re-run the comparison correctly**.

## THE BUG (do not repeat it)
The previous eval scored each model's **final, post-FP-filter events** against a
ground truth built from a **different pipeline stage**:
- `results/batch_5970/manual_review_all_detections.csv` = **raw/pre-filter** matched
  detections (the 243 events humans reviewed). Stem `0002096` shows **25** detections.
- `results/batch_5970*/detections/*.json` = **post-FP-filter final** events. Same stem
  `0002096`: matched JSON has **1** event, production JSON has **0**.

Scoring post-filter events against a pre-filter GT makes recall collapse as an
artifact. **These two sources do not correspond.** Whatever eval you build, the
predictions and the labels must come from the **same** detection run / same stage.

## TASK A — Which FP filter? (ASK THE USER)
Multiple FP-filter artifacts exist; the prior session assumed
`models/hard_neg_retrain/fp_filter.pkl`. Candidates:
- `models/hard_neg_retrain/fp_filter.pkl`  (+ `.json`)
- `models/hard_neg_retrain/fp_filter_no_duration.pkl`  (+ `.json`)  ← a variant w/o the duration feature
- `models/matched_windows/fp_filter.pkl`  (+ `.json`)
- Trainers: `scripts/train_fp_filter.py`, `scripts/train_lab_fp_filter.py`
- Loader/feature API: `src/usv_spectrogram/postprocessing/fp_filter.py`,
  `event_features.py` (11 features). Applied in `run_batch_detection.py:145-154`.
Confirm with the user which filter they mean and on which model. Note the FP filter
operates on **hysteresis EVENTS** (event features), not on individual patches.

## TASK B — Find ALL labeled data (~15k). Strong candidates (verified row counts):
| Path | rows | label col | granularity |
|---|---|---|---|
| `data/training/matched_windows/{train,val,test}.csv` | 10,712 + 2,140 + 1,830 = **14,682** | `label` ∈ {USV, Not USV} | per-PATCH (has `spectrogram_path`, `candidate_id`, `source_file`) |
| `data/training/matched_windows_v2/{train,val,test}.csv` | 11,477 + 2,140 + 1,830 | `label` | per-patch |
| `data/training/lab_finetune_v1/csv/{train,val,test}.csv` | 11,949 + 2,239 + 1,830 | (check) | per-patch (LAB) |
| `data/labels_combined.csv` | 1,380 | (check) | labeling-tool output |
| `results/batch_5970/detections_for_training.csv` | 8,037 | (check) | per-detection |
| `models/hard_neg_retrain/evaluation/predictions.csv` | 1,829 | `true_label`/`predicted_label`/`confidence` | per-patch TEST set, ALREADY scored |
**Most likely "the 15k" = `data/training/matched_windows/` (≈14.7k).** These are
per-patch USV/Not-USV labels — clean, no time-alignment, ideal for comparing the
CNN classifiers directly. Confirm with user which set they consider canonical.
Also search `USV_Detections/`, `labeling_archives/`, `noise_samples/` for more.

## TASK C — Re-run the comparison correctly
Distinguish TWO eval levels — they answer different questions:
1. **CNN discrimination (per-patch)** — load the ~15k labeled patches
   (`spectrogram_path` images, or re-extract), run each model, compute
   recall/precision/FP at the patch level. Clean, large, no sliding-window or
   time-base pitfalls. This is the honest "which model classifies better" axis.
   - **First CNN caveat:** `models/production/best_model.pt` (the Mar-25 first CNN,
     101K params; byte-identical to `full_retrained_cnn/`) was trained on a DIFFERENT
     extraction: **inferno colormap, 25–110 kHz** (`scripts/extract_split_spectrograms.py:71-75`).
     Current default is magma/20–120 kHz. Fed the wrong extraction it is SILENT (~0).
     Its patches must be rendered with its native extraction, or its labeled patches
     re-extracted with `ExtractionConfig(freq_min_hz=25_000, freq_max_hz=110_000,
     colormap='inferno')`. The other two models use magma/20–120. (Reproducing the
     first CNN's native extraction is explicitly authorized — handoff
     2026-05-26_cnn-improvement-slide-handoff.md line 124 — but do NOT change
     `ExtractionConfig` DEFAULTS or `corpus.py`; pass legacy values explicitly.)
2. **FP-filter effect (event-level)** — to measure what the FP filter actually does,
   you need per-EVENT human labels that match the SAME detection run. Either:
   (a) run detection fresh on labeled files, get events, then apply the FP filter and
   compare keep/reject against per-event verdicts; or (b) use the FP filter's own
   training/test split (`scripts/train_fp_filter.py` will reveal where its labeled
   events live). Do NOT reuse the pre-filter manual_review CSV as GT for post-filter
   events.

## What IS solid from this session (reuse, don't redo)
- **Model lineage** (`project_cnn_retrain_matched_windows`): production/ = 1st CNN
  (Mar 25, 101K, DEPRECATED, inferno/25-110); matched_windows (Mar 27, 207K);
  hard_neg_retrain (Mar 31, 207K, CURRENT production). The dir named `production/`
  is the OLDEST, not current — naming trap.
- **First-CNN resurrection recipe** (`project_march_model_calibration_collapse`): the
  Mar-25 model is silent under magma/20-120 but fully alive under native inferno/25-110.
  In `SlidingInference._prepare_batch` the colormap is hardcoded `plt.get_cmap('magma')`
  — override by patching `matplotlib.pyplot.get_cmap` → inferno in-process.
- **Pipeline API** (`run_batch_detection.py:107-156`): `TemperatureScaler.load(...).calibrate(logits)`
  → `hysteresis_detect(probs, result.times, HysteresisConfig(onset_threshold=0.6,
  sustain_threshold=0.4, gap_fill_windows=0, min_duration_windows=3))` [v2 best_params,
  `models/hard_neg_retrain/hysteresis_optimization_v2.json`] → `extract_event_features` →
  `FalsePositiveFilter.predict`. Pass `result.times` (window times, NOT the spectrogram
  column times — that bug cost a re-run).
- **Saved full-pipeline batches**: `results/batch_5970/detections/` (matched) and
  `results/batch_5970_v2_full/detections/` (production) — post temp+hysteresis+FP-filter.
  Useful for the matched-vs-prod NOISE story, NOT for recall vs the pre-filter GT.
- **Prior slide work**: `docs/handoffs/2026-05-26_cnn-improvement-slide-handoff.md`
  (finished quantitative slide `presentation/figures/cnn_improvement/cnn_matched_vs_prod_scaled.png`;
  half-built composite). Its finding #4 ("first CNN not tractable / silent") is
  OUTDATED — we resurrected it via native extraction (floods on lab OOD, under-detects
  on wild — measured).

## Relevant constraints (vault / CLAUDE.md)
- `ExtractionConfig` is **CNN-FREEZE**: never change `freq_{min,max}_hz`/colormap
  DEFAULTS or `corpus.py`. Legacy inferno/25-110 is passed explicitly only to reproduce
  the first CNN.
- CNN inference uses **global MAD normalization once** over the whole spectrogram;
  never per-window (`feedback_cnn_inference_global_mad`). `enable_per_window_norm=False`.
- Anchor noise/USV claims on **human labels**, not eyeballing (`feedback_cannot_eyeball_noise_vs_usv`).
- `det_duration_ms` vs `call_length_s` differ up to 10× — visual filters use
  `det_duration_ms` (`feedback_duration_columns_differ`).
- Print params/thresholds/row counts on every analysis run (`feedback_analysis_print_params`).

## Memory notes to read first
- `project_cnn_iteration_eval_5970` (this eval + its caveats; the FLAWED 34-file result)
- `project_march_model_calibration_collapse` (first-CNN silence/resurrection, native extraction)
- `project_cnn_retrain_matched_windows` (lineage), `project_cnn_auc_comparison_slide`
  (calibration-collapse / "recall traded down" framing)

## Open questions for the user (ask early)
1. Which FP filter exactly — `hard_neg_retrain/fp_filter.pkl`, the `_no_duration`
   variant, or another? And applied to which model?
2. Which label set is "the ~15k" — `data/training/matched_windows/`? v2? lab_finetune?
3. Goal of the eval: compare the **CNNs** (per-patch discrimination) or the **full
   deployed pipelines** (event-level, incl. FP filter)? They tell different stories.
