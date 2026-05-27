# Handoff — Full repository housekeeping

**Date:** 2026-05-27
**Resume with:** `/execute docs/handoffs/2026-05-27_repo-housekeeping.md`
**Why this exists:** User directive, mid-session — *"I don't get how the repo got so
dirty, we must perform housekeeping … in all the projects there are things that just
don't make sense."* The lab-classifier transfer work was halted to produce this. Every
finding below was **measured this session** (git status, OSF API, source-file headers),
not assumed.

> **This is a DIAGNOSIS + PLAN handoff. Nothing was cleaned/committed/deleted.** The
> executing session must get per-item user approval before any destructive or
> outward-facing action (delete, move, commit, worktree-remove). Data dirs especially.

---

## Mission

Make the repo legible again. Two deliverables:
1. **Resolve the contradictions** — the cleaning-subsystem sprawl and the unverified
   VocalMat-cleaning-equivalence are the ones that triggered this; they are load-bearing,
   not cosmetic.
2. **Tidy the tree** — untracked files, stale worktrees, root clutter, orphaned artifacts —
   **without** repeating the 2026 `USV_Detections/` data-loss incident.

---

## METHOD — read before touching anything (data-safety, binding)

- **NEVER `git add -A` / `git add .`.** Stage by exact path. The `78d1c70` incident deleted
  656 files under `USV_Detections/` via bulk staging. (`feedback_no_bulk_stage_in_parallel_chats`,
  CLAUDE.md "Git Data Safety".)
- **Before any cleanup commit:** `git diff --cached --stat`; any unexpected deletions = STOP.
- **`USV_Detections/`, `results/`, `5970 USV/`, training data, `models/*.pt` are DATA.**
  Treat untracked entries there as "needs a keep/gitignore decision," never auto-delete.
- **Files NOT to touch (corpus/CNN freeze):** `src/usv_spectrogram/corpus.py`,
  `ExtractionConfig`, `app/core/sliding_inference.py`, `scripts/run_batch_detection.py`,
  `postprocessing/`, `src/usv_spectrogram/classifier/training.py`,
  `results/lab_classifier_v1/best.pt`. Housekeeping = moving/tracking/documenting, **not**
  editing these.
- **Worktree removals:** verify the branch is merged or truly abandoned (`git log main..<branch>`)
  before `git worktree remove`. Unmerged work in a worktree is the only copy.

---

## FINDINGS (by severity)

### 🔴 H1 — Cleaning-subsystem sprawl + unverified VocalMat equivalence (the core "doesn't make sense")

Three+ independent cleaning implementations coexist; their relationship is undocumented and
partly unverified.

| Impl | File | Purpose (per its own header) |
|---|---|---|
| Classifier 4-layer | `src/usv_spectrogram/classifier/cleaning_pipeline.py` | Pre-CNN cleaning for 18.x. Docstring C2: *reproduces `sliding_inference.py:_apply_mad_normalization` **byte-for-byte***; C3: soft-notch/baseline/Z-score are **wrappers** around existing impls. |
| Production app | `src/usv_spectrogram/app/core/notch.py`, `app/core/denoise.py` | Live PyQt6 detection cleaning (`subtract_temporal_baseline`, notch). The "existing impls" C3 wraps. |
| SIS prefilter | `src/usv_spectrogram/features/spectrogram_filter.py` | Stateless prefilter for SIS-benchmark modules 17.3/17.5/17.6 (ridge/Oren/AMVOC). Separate lineage. |
| DeepSqueak contour (worktree-only) | per `project_cleaning_stacks_three_distinct` | Focus-STFT contour port feeding the contour-masked VAE. |

**Verified this session:** classifier-cleaning == our production cleaning (the docstring asserts
byte-for-byte + wrapping). **NOT verified anywhere:** that either equals **VocalMat's MATLAB
cleaning**. The 12,178 VocalMat training PNGs were rendered by VocalMat's MATLAB and downloaded
as-is — *our pipeline never re-cleans them* (`2026-05-27_lab-classifier-transfer-solve.md`
Finding 3). So the v1 classifier trains on MATLAB-cleaned images but infers on
our-`clean_spectrogram`-cleaned patches, and **whether those two renderings match is an open,
load-bearing question** (plausibly why v1 noise-recall is only 0.64 while usv-recall is 0.98).

**User's standing claim to adjudicate:** *"matlab and our cleaning pipeline is the same, we
ported the same cleaning code … or the cleaning at inference didn't do the right cleaning."*
The code attributes the lineage to our own `sliding_inference.py` + "Boll baseline subtraction,"
**not** to VocalMat MATLAB. Either the user is recalling a port that the code doesn't document,
or the equivalence was assumed and never checked.

**Actions:**
1. Write a one-page `docs/modules/cleaning-subsystems.md` that names all 3–4 stacks, their
   call sites, and which is canonical for which pipeline. (Doc-only; no code change.)
2. Resolve the VocalMat-equivalence question: locate VocalMat's MATLAB cleaning (the OSF/GitHub
   MATLAB source), compare its denoise/contrast/resize math to `cleaning_pipeline.py`, and either
   (a) document them as equivalent, or (b) flag the mismatch as a real domain-gap source. A cheap
   empirical proxy: take one VocalMat **demo** WAV (`Extras/audio_example.wav` on OSF, the only
   public source audio), render it through our pipeline, and visually/numerically compare to the
   matching VocalMat-rendered crop.
3. Decide whether the 4 stacks can be consolidated or must stay separate (the memory note says
   they are genuinely distinct — confirm, don't force-merge).

### 🟠 H2 — Untracked tests, several for SHELVED work

9 untracked test files in `tests/`:
`tests/classifier/test_cage_probe.py`, `test_dann.py`, `test_dann_adversarial.py`,
`test_vae_diagnostic_encoder.py`, `tests/test_eval_shape_encoder.py`,
`test_eval_shape_vae_v3.py`, `test_shape_encoder_contrastive.py`,
`test_shape_vae_v3_hybrid.py`, `test_train_shape_vae_v3_deriv.py`.

The 3 `*dann*` / `*cage_probe*` tests cover **Module 18.4 DANN**, which was **shelved as a dead
end this session** (`ops/goals.md`, `2026-05-27_lab-classifier-transfer-solve.md`). Untracked
tests are invisible to CI and drift silently.

**Action:** per file, decide commit (still-relevant) vs delete (dead DANN). Run each before
committing — do not commit a red/broken test. Do **not** weaken any pre-existing test's
expectations (anti-greenwashing).

### 🟠 H3 — 22 git worktrees, most stale

`git worktree list` shows 22 worktrees, most pinned to old commits (`737af4b1`, `6b7a03d2`)
for finished or abandoned experiments (e.g. `cnn-progression-slide`, `contour-masked-vae-pipeline`,
`couple-vs-wild-analysis`, `deepsqueak-*`, `feature-viewer-html`, `mickey-*`, `phase2c-mickey-report`,
`reminder-2026-05-19-html-reports`, `ssh-audit-gitignore`, `syllable-splitter`,
`traditional-taxonomy-gallery`, `vae-pytorch-pivot`, `vocalmat-classifier-test`).

⚠️ **Some hold the ONLY copy of real work** — e.g. `lab-cnn-classifier-plan` holds
`data/vocalmat_full/` (the 12,178 labeled images) and the VAE worktrees hold memo'd results.
Do not bulk-remove.

**Action:** for each worktree: `git log main..<branch>` — if empty/merged, `git worktree remove`;
if it holds unmerged commits or unique data, list it for a keep/migrate decision. Produce a
table (worktree → merged? → unique artifacts? → recommend remove/keep) for user sign-off
before removing any.

### 🟡 H4 — Root-level clutter

- **7 loose `PLAN_*` / `HANDOFF_*` at repo root:** `HANDOFF_TO_CLAUDE_CODE.md`,
  `HANDOFF_contour_masked_vae_orchestrator.md`, `PLAN_contour_masked_vae_pipeline.md`,
  `PLAN_geometric_shape_clustering_vae.md`, `PLAN_lab_cnn_classifier.md`,
  `PLAN_shape_representation_v2.md`, `PLAN_test_vocalmat_classifier.md`.
  Convention elsewhere is `docs/handoffs/` and `docs/plans/`. (Note: `feedback_roadmap_not_authoritative`
  — each plan is binding for its scope, so **move, don't delete**.)
- **27 loose `.md` at root total** — audit which belong in `docs/`.
- **Large artifacts at root:** `mickey_meeting_20260427.tar.gz` (25 MB) + extracted
  `mickey_meeting_20260427/` (28 MB); `usv_teaching_20260427.tar.gz` (20 MB) + extracted
  `usv_teaching_20260427/` (22 MB). ~95 MB of presentation bundles loose in the tree.

**Action:** move plans/handoffs into `docs/{plans,handoffs}/` (preserve git history with
`git mv` where tracked); move or gitignore the meeting/teaching bundles (decide: archive dir vs
`.gitignore`).

### 🟡 H5 — Untracked data + results (DATA-SAFETY — handle with care)

Untracked, grouped: **52 `results/`**, **17 `USV_Detections/`**, plus `raven_tables_lab_131204/`,
`reports/`, `archive/`, `presentation/`. These are outputs/data, not code.

**Action:** do NOT auto-stage or auto-delete. Triage: which results are reproducible (gitignore)
vs irreplaceable (track deliberately by path or move to a data store). `USV_Detections/` is the
exact dir lost in `78d1c70` — treat as precious.

### 🟢 H6 — New tooling, intent unclear

Untracked: `.codex/` (config.toml, hooks.json, hooks/, agents/), `.agents/skills/{kcheck,sync}/`,
`.claude/scheduled_tasks.lock`. New Codex/agent infrastructure.

**Action:** confirm with user whether to commit (shared tooling) or gitignore (machine-local).
`.claude/scheduled_tasks.lock` and `*.lock` are almost certainly gitignore.

### 🟢 H7 — Deprecated/phantom artifacts in docs

- `results/lab_classifier_v2/best.pt` (DANN) is **referenced in handoffs but absent locally** —
  only ever a cautionary baseline; now formally shelved.
- Memory `project_lab_cnn_classifier_scope` still says "18.4/DANN handoff READY, only rig auth
  remains" — **stale**; superseded by the 2026-05-27 shelve. Update it.

**Action:** reconcile docs/memory with reality (v1 = production, DANN dead). Partly done in
`ops/goals.md` this session.

---

## RECOMMENDED SEQUENCE (triage order)

1. **H1 cleaning doc + VocalMat-equivalence probe** — highest leverage; unblocks the classifier
   decision the user paused on. Pure investigation + one doc.
2. **H7 doc/memory reconciliation** — cheap, removes active misinformation.
3. **H2 untracked tests** — decide commit/delete; run first.
4. **H4 root clutter** — `git mv` plans/handoffs into `docs/`; relocate the 95 MB bundles.
5. **H6 tooling** — one user decision (commit vs gitignore).
6. **H3 worktrees** — produce the merged/unique table, get sign-off, then remove the dead ones.
7. **H5 data/results** — slowest; per-path keep/gitignore decisions, data-safety throughout.

Each step: present the specific diff/move/remove list, get OK, act, verify (`git diff --cached --stat`).

---

## ALREADY DONE THIS SESSION (so the executor doesn't redo it)

- **DANN/v2 shelved, v1 = production lab classifier** — recorded in `ops/goals.md` (Module 18.4
  flipped UNLOCKED → SHELVED; held-out eval folded in at v1 = balanced-acc **0.812**).
- **Re-rendering VocalMat verified dead** — direct OSF `bk2uj` API inspection: Audios = 7
  detection-GT WAVs `{1303,1304,1773,1774,1794,1795,1798}` + `*_GT.xlsx`; ZERO overlap with the
  64 recordings behind the 12,178 labeled images; classification source audio not distributed.
- **Patch-duration sweep re-confirmed** — `scripts/experiments/patch_duration_sweep.py`,
  BASELINE-A 0.812, durations flat ~0.80 (operating-point knob). Output:
  `$CLAUDE_JOB_DIR/sweep_out/sweep_results.json`.

---

## Done means (exit criteria)

- `docs/modules/cleaning-subsystems.md` exists and names every cleaning stack + canonical owner.
- The VocalMat-MATLAB-vs-our-pipeline equivalence is either documented as equivalent or flagged
  as a mismatch (with the demo-WAV comparison as evidence).
- A worktree triage table exists and the dead worktrees are removed (with user sign-off).
- Root `PLAN_*`/`HANDOFF_*` moved into `docs/`; the 95 MB meeting/teaching bundles relocated or
  gitignored.
- `git status` untracked count materially reduced, with **every** data-dir decision explicit and
  **no** accidental deletions (`git diff --cached --stat` clean of surprises).
- `ops/goals.md` + `project_lab_cnn_classifier_scope` memory reconciled with the v1/DANN reality.
