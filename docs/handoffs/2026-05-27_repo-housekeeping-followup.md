# Handoff — Repo housekeeping follow-up (deferred items)

**Date:** 2026-05-27
**Resume with:** `/execute docs/handoffs/2026-05-27_repo-housekeeping-followup.md`
**Predecessor:** `docs/handoffs/2026-05-27_repo-housekeeping.md` (the original DIAGNOSIS+PLAN)
and `docs/handoffs/2026-05-27_repo-housekeeping-triage.md` (the per-finding triage tables).

## What's already done (do NOT redo)

The 2026-05-27 execution session completed H1, H2 (tests), H6, H7, and the *safe subset*
of H3/H4/H5 in 8 commits (`ce1393ad`..`777131f3` on `main`):
- **H1**: `docs/modules/cleaning-subsystems.md` written; VocalMat equivalence resolved (the
  v1 transfer test's `run_inference.py` IS a faithful VocalMat port → 0.64 noise-recall is
  a domain gap, not a cleaning mismatch).
- **H2**: 6 live shape-VAE tests + targets committed; DANN unit snapshotted (`dcd23c39`)
  then removed (`5ce14c39`) — recover via `git show dcd23c39:<path>`.
- **H5 (partial)**: `.gitignore` safety net (`git add results/` 93,121→179 files); 14 human
  noise-labels + reject/review state tracked.
- **H6**: `.codex/{hooks,agents}` committed; machine-local glue gitignored.
- **H3 (partial)**: 3 SAFE-REMOVE worktrees removed (22→19); branch refs kept.
- **H7**: `project_lab_cnn_classifier_scope` memory + `MEMORY.md` reconciled.

## Deferred — what's left, in priority order

### 1. H5 data backup (do FIRST — highest data-loss risk)
The 4 VAE checkpoints are now gitignored but **irreplaceable, with no rig copy**:
```
results/vae_5970_lab/vae_best.pt       results/vae_5970_lab/vae_final.pt
results/vae_ds_5970_lab/vae_ds_best.pt results/vae_ds_5970_lab/vae_ds_final.pt
```
**Action:** rsync to the rig (`shachar@100.113.224.57:/data/mickey_london_lab/...` per the
established move-by-rsync pattern; see `reference_gpu_rig_cloudyclaude`). Then confirm on
the rig before considering them safe.
Also: `results/syllable_split/cnn_rerun*/*.npz` (169 files, 48M) — **no reproducer script
was found**. Decide: locate/confirm the reproducer (then gitignore), or protect by path.

### 2. H4 heavily-referenced root docs (reference-rewrite refactor)
NOT moved this session because they are load-bearing links. Reference counts (via `rg -l`):
| File | refs | File | refs |
|---|---|---|---|
| ROADMAP.md | 63 | PLAN_geometric_shape_clustering_vae.md | 12 |
| ROADMAP_SIS_BENCHMARK.md | 20 | ROADMAP_POST_PROCESSING.md | 9 |
| ROADMAP_lab_cnn_classifier.md | 13 | PLAN_lab_cnn_classifier.md | 9 |
| | | PLAN_shape_representation_v2.md | 8 |
Plus low-ref: PLAN_test_vocalmat_classifier (4), HANDOFF_contour_masked_vae_orchestrator (4),
HANDOFF_TO_CLAUDE_CODE (2), NOTION_NOTES_ROADMAP (2), ROADMAP_KNOWLEDGE_ACTIVATION (1),
ROADMAP_PARTS_FINDER (1).

**Decision gate:**
| Approach | When | Cost |
|---|---|---|
| `git mv` + `rg`-driven sed rewrite of every reference | want them in `docs/` cleanly | rewrite 60+ files; test nothing breaks |
| Leave a redirect stub at root (`# Moved to docs/plans/X.md`) | want minimal churn | N stubs; refs still resolve to stub |
| Leave in place | refs not worth the churn | 0 — accept root clutter |
Recommended: move the **low-ref (≤4)** PLAN/HANDOFF files with ref-rewrite now; leave
`ROADMAP.md` (63) and `ROADMAP_SIS_BENCHMARK.md` (20) in place or stub-redirect — moving
them is disproportionate to the tidiness gain.

### 3. H3 remaining worktrees (back up unique data BEFORE removing)
13 KEEP + ~5 REVIEW worktrees hold the only copy of real deliverables as **never-committed
files** (branch-merged ≠ work-preserved). See the triage table in
`2026-05-27_repo-housekeeping-triage.md`. **Per worktree:** commit-by-path or rsync the
unique artifacts to the rig, verify, THEN `git worktree remove`. Never bulk-remove.
- **Most dangerous:** `lab-cnn-classifier-plan` = 627M / 12,180-file VocalMat training set
  (gitignored, present nowhere else); `contour-masked-vae-pipeline` = 16G `patches.npz`.
- **REVIEW:** `syllable-splitter` has untracked `postprocessing/syllable_splitter.py` +
  `cnn_prob_splitter.py` — a Red-Flag `postprocessing/` path; confirm intent before removal.

### 4. Loose ends
- **`scripts/evaluate_held_out_844.py`** (untracked) imports `ResNet18DANN`, now removed →
  dangling import. Either repoint it to plain ResNet18 (v1 use) or recover `dann.py` from
  `dcd23c39`. v1's 844 eval currently runs via `scripts/experiments/patch_duration_sweep.py`.
- **17 other untracked `docs/`** (prior-session DANN/shape handoffs + reviews): a follow-up
  commit pass, or gitignore the dead-DANN docs (`docs/modules/lab-classifier-v2-dann.md`,
  `docs/reviews/lab-classifier-v2-dann-review.md`).
- **`results/lab_classifier_v2/`** (DANN eval artifacts) is deliberately left tracked/visible
  as the negative-result record — decide whether to commit those .md/.json/.png.

## Files NOT to touch (carried from the predecessor — corpus/CNN freeze)
`src/usv_spectrogram/corpus.py`, `ExtractionConfig`, `app/core/sliding_inference.py`,
`scripts/run_batch_detection.py`, `postprocessing/`, `classifier/training.py`,
`results/lab_classifier_v1/best.pt`. Housekeeping = move/track/document, never edit these.

## Done means
- 4 VAE `.pt` confirmed on the rig (or explicitly accepted as box-only).
- Root PLAN/HANDOFF either moved-with-ref-rewrite or stub-redirected (your call on ROADMAP.md).
- Each removed worktree had its unique data backed up first; KEEP/REVIEW dispositions recorded.
- `git status` untracked count materially reduced again, no accidental deletions
  (`git diff --cached --stat` clean of surprises).
