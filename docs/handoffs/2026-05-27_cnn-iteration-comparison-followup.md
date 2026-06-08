# Handoff — 3-model CNN-iteration "got better" comparison (follow-up)

## Status of the predecessor
The FP-filter half of `docs/handoffs/2026-05-27_cnn-iteration-eval-redo.md` is **DONE** — the FP filter does NOT drop USVs (true cost 7.67% interval-level). Report: `results/fp_filter_correct_eval/`. Memory: `project_fp_filter_true_effect`. This handoff covers the *other* half: the illustrative "CNN improved across iterations" comparison, which was deferred.

## Goal
An **illustrative** (no scientific weight) figure showing the 3 CNN iterations got better. Three models:
- `models/production/best_model.pt` — **first CNN** (Mar-25, 101K, inferno/25–110 kHz native extraction). DEPRECATED but used as the iteration-1 baseline.
- `models/matched_windows/best_model.pt` — iteration 2 (Mar-27, 207K, magma/20–120).
- `models/hard_neg_retrain/best_model.pt` — **current production** (Mar-31, 207K, magma/20–120).

## What is already done (REUSE, do not redo)
- **Whole-file sliding ROC is the validated 'right test'** (`project_cnn_auc_comparison_slide`). Predictions on disk: `presentation/figures/cnn_improvement/_predictions/*.csv`. AUC separates **0.877 → 0.914** (matched → prod); earlier 33-file run gave 0.885 → 0.920. Slide: `cnn_matched_vs_prod_scaled.png`.
- **Do NOT use the per-patch matched_windows/test.csv comparison** — it gives a MISLEADING plateau (AUC 0.964/0.989/0.989) because all models see the same curated negatives. This is the trap the FP-filter redo also flagged.
- **First-CNN resurrection recipe** (`project_march_model_calibration_collapse`): the Mar-25 model is silent under magma/20–120 but alive under native **inferno/25–110**. Reproduce via `ExtractionConfig(freq_min_hz=25_000, freq_max_hz=110_000, colormap='inferno')` and force `plt.get_cmap`→inferno (it is hardcoded 'magma' in `sliding_inference._prepare_batch`). On wild home-turf it UNDER-detects (recall 21.9% vs prod 68.8% at equal noise); the lab "flood" was OOD-only.

## Constraints (do not violate)
- `ExtractionConfig` is **CNN-FREEZE**: never change `freq_{min,max}_hz`/colormap DEFAULTS or `corpus.py`. Pass legacy inferno/25–110 *explicitly* only to reproduce the first CNN (authorized).
- Global MAD norm once over the whole spectrogram, `enable_per_window_norm=False`.
- Print params/thresholds/row counts on every run.

## Decision gate (which story to tell)
| Want to show | Use | Avoid |
|---|---|---|
| "CNN discriminates noise better over iterations" | whole-file sliding ROC (done) | per-patch plateau |
| "first → current detection capability" | raw-window recall 22%→69% (first native-inferno vs prod) | event-level (muddied by tuning) |
| "FP suppression matched → prod" | known-noise file FP rate 4.9%→1.3% | random-file rate (both fire on real USVs) |

## Done means
A single illustrative slide/figure with the 3 models on the whole-file ROC axis (reuse existing predictions if current), first CNN rendered with its native extraction, and a one-line caption stating it is illustrative. No new claim about the FP filter (that's settled).
