# Handoff — CNN-improvement presentation slides (2026-05-26)

## Goal
Build presentation slides showing how the USV-detection CNN **got better and better** across
iterations, with the final being the current production model (`hard_neg_retrain` + noise
filter). Two kinds of slide are in play:
- **Qualitative** (before/after spectrograms with detection boxes) — visual proof of noise rejection.
- **Quantitative** (AUC / ROC + noise-rejection metrics) — the numbers.

The honest narrative is **false-positive / noise suppression**, NOT "catching more USVs"
(recall was traded down on purpose). The user has direct memory that the older `matched_windows`
model was "still not good" at rejecting noise — **this is now validated at scale** (see Findings).

## Current status
- ✅ **Quantitative slide DONE and validated** — `cnn_matched_vs_prod_scaled.png` (248 files, 147k windows).
- ✅ Qualitative composite exists but needs rework (dead panel) — see Open Items.
- ❌ Nothing committed. New scripts are untracked in BOTH the worktree branch
  `worktree-cnn-progression-slide` and (copied) the main checkout `scripts/`.

## Model lineage (READ THIS — there's a naming trap)
| Order | Directory | Date | Arch | Role |
|---|---|---|---|---|
| 1st (oldest) | `models/production/` | Mar 25 | [32,64,128], 1.2MB | first CNN. **DEPRECATED.** I nicknamed it "Feb-2" (nickname is questionable — checkpoint is Mar 25). |
| 2nd | `models/matched_windows/` | Mar 27 | [32,96,192], 2.4MB | fixed the train/inference window-extraction mismatch |
| 3rd (CURRENT production) | `models/hard_neg_retrain/` | Mar 31 | [32,96,192], 2.4MB | what's actually deployed now |

⚠️ **TRAP: the directory named `models/production/` is the OLDEST model, NOT the current
production one.** Current production = `hard_neg_retrain`. (CLAUDE.md flags `models/production/`
as deprecated.)

## Key findings (all verified this session)
1. **Frozen-candidate AUC is MISLEADING — do not use it.** Scoring all 3 models on the fixed
   1,829-window test set (`data/training/matched_windows/test.csv`) gives AUC 0.964 / 0.989 /
   0.989 — looks like a plateau. ARTIFACT: every model sees the same curated negatives, which
   don't contain the diverse real-recording noise the production retrain learned to reject.
2. **Whole-file ROC (real sliding pipeline vs human labels) is the right test, and it separates.**
   SCALED run on **248 files** (34 known-noise + 14 USV-bearing + 200 random), raw sliding scores
   (no FP-filter), production defaults (global MAD norm):

   | Model | whole-file ROC-AUC | PR-AUC | FPR@90%recall | known-noise windows flagged |
   |---|---|---|---|---|
   | matched_windows (old) | 0.877 | 0.165 | 0.471 | **4.90%** (1121 FP windows) |
   | production (hard_neg) | **0.914** | **0.348** | **0.341** | **1.32%** (303 FP windows) |

   **HEADLINE: on 27 known-noise files, matched_windows flags 3.7× more noise than production.**
   On the 200 random files the gap is small (6.6% vs 6.0%) because random recordings contain real
   USVs — known-noise files isolate the noise behavior.
3. **The FP-filter masks the model difference.** The saved batches (`results/batch_5970` =
   matched_windows, `results/batch_5970_v2_full` = production) both ran the full FP-filter +
   hysteresis pipeline → near-equal totals (8036 vs 7575). The raw model difference only shows in
   un-filtered sliding scores. The user's "matched was noisy" memory is about the RAW model.
4. **The first CNN (`models/production/`) can't be put on the same axis.** Through today's
   sliding pipeline it outputs ~0.000 on EVERYTHING (silent) — it was trained on a DIFFERENT
   extraction (`inferno` colormap, 25–110 kHz; current is `magma`, 20–120 kHz; see
   `scripts/extract_split_spectrograms.py`). Even fed its native inferno extraction it stays mostly
   silent on noise. Its historical "noise flood" was a deploy-time extraction/window MISMATCH
   (feeding a model the WRONG extraction makes it flag ~98% of noise — reproduced with the newer
   models on inferno). That old inference pipeline does not survive in the repo (no legacy script,
   no saved batch). **Conclusion: faithfully showing the very-first CNN's flood is not tractable;
   focus the story on matched_windows → production.**

## Deliverables (all in `presentation/figures/cnn_improvement/`, main checkout)
- **`cnn_matched_vs_prod_scaled.png`** ← FINAL quantitative slide (248 files). USE THIS.
- `cnn_noise_vs_usv_composite.png` — qualitative 2×2 (noise file 0005656: old 9 boxes → prod 0;
  USV file 0000481: 9→9). **The USV (right) column is dead weight (no change) and must be reworked.**
- `cnn_wholefile_roc.png` — earlier 33-file 3-model ROC. **Includes the misleading silent-Feb-2
  curve (AUC 0.863 on crushed scores). Candidate for deletion to avoid mis-grabbing.**
- `_proof_rendering_fixed.png` — scratch (underscore = not for deck).
- `usv_contact_sheet.png` — 24 candidate USV files for swapping the composite's USV example.
- `_predictions/scaled_window_scores.csv` (14.5 MB, 147k rows: file,category,time_s,y,p_matched,p_prod)
  — reproduces the scaled slide instantly without re-running inference.
- `_predictions/{production,matched_windows,hard_neg_retrain,production_native}.csv` — frozen
  candidate-set scores (the misleading-plateau data; keep for reference).

## Scripts (now in main checkout `scripts/`; also on branch `worktree-cnn-progression-slide`)
- **`cnn_wholefile_roc_expanded.py`** — the scaled run (matched vs prod, 248 files). ~60–90 min CPU
  (slow because sliding inference does a magma colormap + PIL resize per window). Edit `N_RANDOM` to
  change sample size. Re-plotting from the saved CSV is instant if you factor that out.
- `cnn_wholefile_roc.py` — original 33-file 3-model whole-file ROC (hand-labeled set only).
- `make_cnn_composite_slide.py` — 2×2 before/after spectrogram grid (uses per-bin median-subtracted
  rendering so USVs are visible — see gotcha below). Imports helpers from `make_cnn_progression_slide.py`.
- `make_cnn_progression_slide.py` — N-panel single-spectrogram progression renderer.
- `make_cnn_metrics_slide.py` — builds the frozen-candidate AUC + operating-point slide
  (its output `cnn_metrics_slide.png` was DELETED as misleading; keep script for reference only).

## How to reproduce / continue
Run everything from the **main checkout** (`/home/shachar/projects/mickey_london_lab`), `.venv/bin/python`.
- Re-plot scaled slide from saved scores (fast): read `_predictions/scaled_window_scores.csv` and
  re-draw (or just re-run `cnn_wholefile_roc_expanded.py`, which re-does inference — slow).
- Ground truth: `scripts/build_manual_review_labels.py` `ANNOTATIONS` dict (per-file usv/noise
  labels on `results/batch_5970/manual_review_all_detections.csv`). 19 all_noise + 14 USV-bearing
  files. Plus `KNOWN_NOISE_SUFFIXES` (18) hard-coded in `scripts/generate_high_confidence_pngs.py`
  and in the expanded script. Union → 34 known-noise whole files.
- WAVs: full 6,400-file 5970 set lives under `5970/USV{1..5}/usv_lmt_034/` (the flat `5970 USV/`
  dir has only 12). Resolve labeled stems via `5970_manual_review_reviewed/` then `5970/**/`.
- Sliding inference API: `usv_spectrogram.app.core.sliding_inference.SlidingInference(model_path)`
  + `AudioLoader().load(wav)`; `infer(spec_db, times)` → `InferenceResult(probabilities, times, ...)`.
  Defaults (window 100px, hop 10px, energy 0.1, global MAD norm) replicate production — DO NOT
  enable per-window norm (silently kills USVs — see `feedback_cnn_inference_global_mad` memory).

## Open items / next steps
1. **Rework the composite** (`make_cnn_composite_slide.py`): drop the dead USV column; make it a
   multi-noise-file grid — rows = several human-labeled all_noise files, columns = matched_windows
   (old, many boxes) → production (0 boxes). All 19 all_noise files have old-model boxes and
   production = 0 across all of them (verified). Detection JSONs:
   `results/batch_5970/detections/<stem>.json` (matched) and
   `results/batch_5970_v2_full/detections/<stem>.json` (production). Top noise files by old-box
   count: 0005656 (9), 0003502 (5), 0001700 (2), 0003493 (2). NOTE: can't include the first CNN
   as boxes (silent — see finding #4).
2. **Cleanup decision (ask user):** delete `cnn_wholefile_roc.png` (misleading silent-Feb-2 curve)?
3. **Optional:** commit the scripts + figures. Stage by exact path (never `git add -A` — see
   `feedback_no_bulk_stage_in_parallel_chats`). Figures are in `presentation/figures/cnn_improvement/`;
   `_predictions/scaled_window_scores.csv` is 14.5 MB (decide whether to commit or .gitignore).

## Relevant constraints (memories)
- `project_cnn_auc_comparison_slide` — the full AUC story + corrected numbers (this work).
- `project_cnn_retrain_matched_windows` — model lineage.
- `project_batch_detection_runs` — batch_5970 = matched_windows; batch_5970_v2_full = production;
  Feb-2 returns ~0 through current pipeline.
- `feedback_cannot_eyeball_noise_vs_usv` — anchor noise/USV claims on human labels, not my eye;
  rendering gotcha (use per-bin median subtraction, not ref=np.max, or USVs vanish).
- `feedback_cnn_inference_global_mad` — global MAD norm once, not per-window.
- `ExtractionConfig` is CNN-FREEZE: do not change `freq_{min,max}_hz` / colormap defaults; the
  legacy 25–110 kHz / inferno values are passed explicitly only to reproduce the first CNN.
