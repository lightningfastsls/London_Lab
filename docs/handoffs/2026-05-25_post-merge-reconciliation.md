# HANDOFF — Post-merge reconciliation (run AFTER the contour-pipeline lands in main)

**Date:** 2026-05-25  **Status:** BLOCKED until `2026-05-25_consolidate-contour-pipeline-into-main.md` is done.
**Goal:** collapse the scattered state — box `main` + 22 worktrees + three rig code roots + a parallel
chat's alphabet work — into ONE canonical source of truth, without losing data or clobbering in-flight work.

## Precondition check (do not start until ALL true)
- [ ] The 5 contour-pipeline scripts are committed in main (`git -C <main> ls-files scripts/ | grep -E "mass_apply_contour_mask|window_calls_to_patches|deepsqueak_focus_stft|sweep_contour_mask|contour_mask_utils"` → 5 hits).
- [ ] `git diff --cached` is clean (no half-staged leftovers from the consolidation).
- [ ] You have checked in with the **parallel chat** working in `latent-analysis-b-a-c` (the `shape_alphabet` productionization) — confirm it has finished or agree on a freeze point. Do NOT reconcile its files while it is mid-write.

## The scattered state to reconcile (audited 2026-05-25)
1. **Code, 3 rig roots:** `/opt/mickey_london_lab` (oldest), `/data/mickey_london_lab` (has ridge_tracker), `/data/shachar/contour_vae` (newest contour scripts + the data). Plus box `main` (now has the pipeline) and 22 worktrees.
2. **Throwaway vs keeper rig scripts** in `/data/shachar/contour_vae/`: `rig_R1_true_ridges.py`, `rig_M8_contour_vae.py`, `rig_M9_contrastive.py`, `rig_M10_image_vae.py` (bake-off experiments), `rig_R2_shape_alphabet.py` (productionization — likely keeper). Box copies in `$CLAUDE_JOB_DIR` (ephemeral — will vanish).
3. **Data artifacts** (NOT git): `patches.npz` (16 GB + per-cohort), `latents.parquet`, `best.pt`, `true_registered_ridges.npz`, `shape_alphabet/{shape_call_letters.parquet,...}`, `models/shape_kmeans/k20.joblib`, `m8/m9/m10` outputs — all on rig `/data/shachar/contour_vae/...`.
4. **Open decision** from the parallel chat: `docs/handoffs/2026-05-25_shape-map-and-alphabet-decision.md` (registered-shape transition MI came out LOWER than latent → build a navigable 2-D shape-map, or keep an alphabet?). Reconciliation must record the resolution, not re-litigate it.

## Reconciliation tasks (each is independent; do in this order)

### R-A. Make `main` the code source of truth, sync the rig from it
- Promote keeper rig scripts into main BY PATH (cp from `/data/shachar/contour_vae/rig_R2_shape_alphabet.py` etc. → main `scripts/` or `scripts/experiments/`; decide with user which are keepers vs discard). Same git guardrails as consolidation: stage by exact name, `git diff --cached --diff-filter=D` must be empty.
- Then push code main→rig by **rsync, DRY-RUN FIRST**: `rsync -avn --include='*.py' --include='*/' --exclude='*' <main>/scripts/ shachar@100.113.224.57:/data/<canonical>/scripts/` — review the `-n` output before dropping `-n`. NEVER rsync data dirs blindly. (Rig can't reach GitHub; code moves by rsync — see memory `reference_gpu_rig_cloudyclaude`.)
- Decide ONE canonical rig root (recommend keep `/data/shachar/contour_vae` for the contour work) and write a one-line README in the other two pointing to it. Do NOT delete the others without `git`/backup confirmation.

### R-B. Reconcile the latent-analysis worktree (coordinate w/ parallel chat)
- Its branch is 1 commit ahead (+4,474, 0 deletions) + uncommitted keepers (plan, handoffs, `shape_registered_clustering.py`). With the parallel chat's sign-off, commit the KEEPER code/docs by path and merge that 1 commit + keepers into main (branch-first, deletion-gate). Leave data outputs out of git.
- Record the shape-map-vs-alphabet decision outcome in `PLAN_shape_representation_v2.md` (Track D) + the memory note.

### R-C. Data inventory + .gitignore
- Confirm `.gitignore` (main) excludes `results/`, `*.npz`, `*.parquet` model/data, `*.joblib`, large HTML/PNG — so the promoted scripts can't drag data into git. Add patterns if missing (by-path edit to `.gitignore`, review diff).
- Write a short `docs/DATA_LOCATIONS.md` (or append to project-structure): canonical paths for patches/latents/ridges/alphabet/models on the rig, so future sessions don't re-hunt.

### R-D. Retire stale worktrees (LAST, after R-A/R-B confirmed in main)
- For `contour-masked-vae-pipeline` and `latent-analysis-b-a-c`: verify everything valuable is in main (`git -C <main> ls-files` the promoted files; spot-check), THEN `git worktree remove <path>` (it refuses if uncommitted changes remain — do NOT use `--force` without confirming those changes are junk/data).
- The other ~20 worktrees: just LIST them with last-commit dates for the user (`git worktree list`); propose retirements but do not remove without per-worktree confirmation.

### R-E. Docs + memory + KG
- Run `/refresh-human-docs` to regenerate `docs/human/{PROJECTS,DECISIONS}.md`.
- Update `ops/goals.md` with the consolidated state. Update memory note `project_shape_registration_clustering.md` (mark consolidation done).
- Run `/sync push` if auto-memory needs to land in-repo.

## Decision gates
| Outcome | Action |
|---|---|
| Parallel chat NOT finished in latent-analysis | do R-A only (contour code/rig sync); STOP before R-B; reschedule |
| `git diff --cached --diff-filter=D` non-empty at any commit | STOP — a deletion is staged; unstage + investigate |
| rsync dry-run shows it would delete/overwrite rig data | STOP — narrow includes; never `--delete` on data |
| `git worktree remove` refuses (uncommitted changes) | inspect those changes; commit keepers by path or confirm they're disposable data before `--force` |

## DO NOT
- `git add -A`/`.`; rsync without `-n` first; `git worktree remove --force` on un-inspected changes; touch main's unrelated dirty production files (`app/main_window.py`, `postprocessing/*`, `tests/*`); delete any `*.npz`/`results/` without `git log --`/backup check; reconcile the parallel chat's files without its sign-off.

## After reconciliation
Single source of truth: main has the code, rig mirrors it, data locations documented, stale worktrees retired, the shape-map/alphabet decision recorded. Then proceed to `PLAN_shape_representation_v2.md` Track 0/B (denoised retrain) from a clean main.
