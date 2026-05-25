# HANDOFF — Safely consolidate the contour-VAE patch pipeline INTO MAIN

**Date:** 2026-05-25  **Status:** READY (no code written yet; this is the first job for a fresh session)
**Goal:** get the patch-generation pipeline into the `main` checkout so the denoised-patch
regeneration + VAE retrain (see `PLAN_shape_representation_v2.md`) can run from main —
**without** the bulk-staging / data-loss mistakes this repo has a history of.

## Why this needs care (audited state, 2026-05-25 — do not skip)
- A plain `git merge` does NOT work here: `worktree-contour-masked-vae-pipeline` is **0 commits ahead of main** (empty diff). The pipeline scripts exist only as **untracked/uncommitted files** in that worktree (49 uncommitted items). `worktree-latent-analysis-b-a-c` is 1 commit ahead (+4,474) but its recent work is mostly uncommitted (17 items).
- **`main` is DIRTY:** 19 modified TRACKED files — incl. PRODUCTION code (`src/usv_spectrogram/app/main_window.py`, `postprocessing/hysteresis.py`, `postprocessing/triage.py`, `tests/test_*.py`) and a staged deletion — plus **177 untracked**. These are someone else's in-flight work. **DO NOT stage, commit, revert, or touch them.**
- This repo lost `USV_Detections/` (656 files) once to a bulk `git add` commit. The worktrees contain large data outputs (`.npz`, `.parquet`, HTML, PNG, `results/`) that **must NOT enter main git.**
- A **parallel chat** is actively working in `latent-analysis-b-a-c` (it wrote the `shape_alphabet` productionization). **Do not move/commit its in-flight files.** This handoff is scoped to the **contour-pipeline scripts only**; the latent-analysis consolidation is deferred to coordinate with that chat.

## What to bring into main (EXACTLY these 5 — confirmed missing from main, confirmed self-contained)
Source: `/home/shachar/projects/mickey_london_lab/.claude/worktrees/contour-masked-vae-pipeline/scripts/`
Dest:   `/home/shachar/projects/mickey_london_lab/scripts/`
- `window_calls_to_patches.py`
- `deepsqueak_focus_stft.py`
- `sweep_contour_mask.py`
- `mass_apply_contour_mask.py`
- `contour_mask_utils.py`
Their only non-stdlib/non-pip imports are `from usv_spectrogram import corpus` (already in main) and each other (siblings in `scripts/`). Extra pip dep `statsmodels` is already installed in main `.venv` (0.14.6).
Optional docs (safe, additive): `PLAN_shape_representation_v2.md`, `docs/handoffs/2026-05-25_productionize-shape-registration.md`, this file.

## What NOT to bring
- No `results/`, `*.npz`, `*.parquet`, `*.html`, `*.png`, or any data. The `contours.parquet` inputs stay as data on the rig (`/data/shachar/contour_vae/results/contour_extraction/<cohort>_focus/contours.parquet`) and in the worktree — the regenerate run reads them in place, they are NOT committed.
- Nothing from main's 19 modified files or 177 untracked.
- Nothing from the parallel chat's `shape_alphabet` work.

## SAFE PROCEDURE (run from main checkout `/home/shachar/projects/mickey_london_lab`)
```bash
cd /home/shachar/projects/mickey_london_lab
git branch --show-current          # confirm: main
# 1. branch first (default-branch safety); untracked files follow harmlessly
git switch -c consolidate/contour-pipeline-into-main
# 2. copy the 5 scripts by exact name (cp, not move — leave worktree intact)
SRC=.claude/worktrees/contour-masked-vae-pipeline/scripts
for f in window_calls_to_patches.py deepsqueak_focus_stft.py sweep_contour_mask.py mass_apply_contour_mask.py contour_mask_utils.py; do
  cp "$SRC/$f" "scripts/$f"
done
# 3. verify they import + compile in MAIN (catches stranded deps)
for f in window_calls_to_patches deepsqueak_focus_stft sweep_contour_mask mass_apply_contour_mask contour_mask_utils; do
  .venv/bin/python -m py_compile scripts/$f.py || echo "COMPILE FAIL $f"
done
PYTHONPATH=src:scripts .venv/bin/python -c "import contour_mask_utils, deepsqueak_focus_stft, window_calls_to_patches, sweep_contour_mask, mass_apply_contour_mask; print('imports OK in main')"
# 4. stage BY EXACT PATH ONLY (never git add -A / .)
git add scripts/window_calls_to_patches.py scripts/deepsqueak_focus_stft.py scripts/sweep_contour_mask.py scripts/mass_apply_contour_mask.py scripts/contour_mask_utils.py
# 5. GATE: inspect what is staged — must be exactly 5 ADDED files, ZERO deletions
git diff --cached --stat
git diff --cached --diff-filter=D --name-only      # MUST be empty; any output = STOP
# 6. commit by name only
git commit -m "feat(scripts): port contour-VAE patch-generation pipeline into main"
```
Then the user fast-forwards/merges `consolidate/contour-pipeline-into-main` into main (or keeps as PR).

## Decision gates
| Outcome | Action |
|---|---|
| `py_compile`/import OK, `--cached --stat` = 5 files / 0 deletions | commit, hand back |
| import fails (a stranded dep beyond the 5 + corpus) | STOP; find the missing module in the worktree; add it by path; re-verify. Do not guess. |
| `--diff-filter=D` shows ANY deletion | STOP — something staged a removal; unstage, investigate |
| user wants it on main directly (not a branch) | same steps, skip step 1; the by-path staging keeps main's dirty files untouched |

## After consolidation (next, per PLAN_shape_representation_v2.md)
Track 0 + B: build the DENOISED training patches (drop the contour mask; use
`src/usv_spectrogram/features/spectrogram_filter.py::prefilter_spectrogram`), then retrain
the contour VAE. ROOT CAUSE recap: the VAE was trained on hard-masked patches that are
>95% zeros even for strong calls (verified) — that's why clustering ignored shape. Compute
on the rig (`/data/shachar/contour_vae`, ~50× the box; box OOM-crashed WSL under load).

## Files to touch / NOT touch
- TOUCH: only `scripts/<the 5>` (additive) + optional docs.
- DO NOT TOUCH: main's 19 modified tracked files (production app/postprocessing/tests), the 177 untracked, any `results/`/data, the parallel chat's `shape_alphabet` files.
