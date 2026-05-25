# HANDOFF — Denoised-patch regeneration + contour-VAE retrain

**Date:** 2026-05-25  **Status:** READY (predecessor consolidation DONE)
**Predecessor:** `docs/handoffs/2026-05-25_consolidate-contour-pipeline-into-main.md` — the 5 patch-pipeline scripts are committed on branch `consolidate/contour-pipeline-into-main` (commit `9ccbd996`). **First, fast-forward that branch into main** (`git merge consolidate/contour-pipeline-into-main`) so the scripts run from main.
**Canonical plan:** `PLAN_shape_representation_v2.md` (Track 0 + Track B) — read it first; this handoff is the pointer.

## Root cause being fixed
The production contour VAE was trained on **hard-masked** patches that are >95% zeros even for strong calls (verified). The latent space therefore sorts by pitch/duration, not shape (K=20: shape η²=0.12; chevron/valley NMI=0.04 — see `project_shape_registration_clustering`, `project_c06_empty_cluster`). Fix: retrain on **denoised, un-masked** patches.

## What to do
1. **Drop the contour mask.** Build training patches with `src/usv_spectrogram/features/spectrogram_filter.py::prefilter_spectrogram` for denoising instead of the hard contour mask. Use the now-in-main scripts (`window_calls_to_patches.py`, `deepsqueak_focus_stft.py`, `contour_mask_utils.py`).
2. **Inputs stay on the rig:** `contours.parquet` lives at `/data/shachar/contour_vae/results/contour_extraction/<cohort>_focus/contours.parquet` — read in place, do NOT commit data.
3. **Retrain the contour VAE** on the denoised patches.
4. **Compute on the rig** (`shachar@100.113.224.57`, `/data/shachar/contour_vae`, ~50× the box). The box OOM-crashed WSL under load — never full-scan `patches.npz` locally (11 GiB; OOM'd once, see `project_c06_empty_cluster`).

## Decision gate
| Outcome | Action |
|---|---|
| Denoised patches still >90% zeros | STOP — prefilter params wrong; inspect a sample before mass regen |
| Retrained VAE shape η² materially > 0.12 | proceed to re-cluster + compare to registration baseline (η²=0.75) |
| Retrained VAE shape η² ≈ 0.12 (no gain) | STOP — denoising alone insufficient; reconsider registration approach |

## Corpus / constants
Import sample rate, USV band, STFT params from `src/usv_spectrogram/corpus.py` — never redeclare. Load `data/corpus_facts/{dataset}.json` for empirical facts.

## Files to touch / NOT touch
- TOUCH (rig copy): patch-regen outputs under `/data/shachar/contour_vae`, retrain artifacts.
- DO NOT TOUCH: production detection pipeline (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `postprocessing/`), `ExtractionConfig` values, any committed model under `models/`.
- Verify on the rig, not the box. Move code+data by rsync (the rig can't reach GitHub).
