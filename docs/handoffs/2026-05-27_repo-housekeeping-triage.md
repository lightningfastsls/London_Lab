# Repo Housekeeping — Triage Tables (for sign-off)

**Date:** 2026-05-27
**Parent:** `docs/handoffs/2026-05-27_repo-housekeeping.md` (the DIAGNOSIS+PLAN)
**Status:** Investigation COMPLETE (6 read-only agents). H1 doc + H7 memory reconciliation
DONE. Everything below is **pending per-item user sign-off** — nothing destructive has
been executed.

All findings verified against HEAD `75395217`. No `git add -A`. Stage by exact path only.

---

## H1 — Cleaning sprawl + VocalMat equivalence — ✅ RESOLVED
- `docs/modules/cleaning-subsystems.md` written: names all **4** stacks + canonical owners.
- **Correction landed:** the DeepSqueak contour port is **tracked on `main`** (not
  worktree-only as the handoff/memory said — verified `git ls-files`).
- **VocalMat verdict:** the v1 transfer test (`run_inference.py`) IS a faithful VocalMat
  port → its 0.64 noise-recall is a **genuine domain gap, not a cleaning mismatch**.
  `cleaning_pipeline.py` (our-cohort prep) is NOT a VocalMat port (diverges on 5 axes).

## H7 — Doc/memory reconciliation — ✅ DONE
- `project_lab_cnn_classifier_scope` memory note + `MEMORY.md` index updated to v1=production,
  DANN=dead-end, `results/lab_classifier_v2/best.pt`=phantom. `ops/goals.md` already done.

---

## H2 — Untracked tests (10 found; all GREEN — 215 passed, 5 intentional skips)

| File | Module tested | pytest | Class | Recommendation |
|---|---|---|---|---|
| `tests/classifier/test_cage_probe.py` | `classifier.cage_probe` | 6 ✓ | DEAD-DANN | DELETE (w/ target) |
| `tests/classifier/test_dann.py` | `classifier.dann` | 17 ✓ | DEAD-DANN | DELETE (w/ target) |
| `tests/classifier/test_dann_adversarial.py` | `classifier.dann` | 21 ✓ | DEAD-DANN | DELETE (w/ target) |
| `tests/classifier/test_vae_diagnostic_encoder.py` | `scripts/run_vae_diagnostic_on_encoder.py` | 20 ✓ | DEAD-DANN (18.4 review tool) | REVIEW → lean DELETE |
| `tests/test_eval_shape_encoder.py` | `scripts/eval_shape_encoder.py` | 15 ✓ | LIVE shape | COMMIT (w/ target) |
| `tests/test_eval_shape_vae_v3.py` | `scripts/eval_shape_vae_v3.py` | 9✓ 5 skip | LIVE shape | COMMIT (w/ target) |
| `tests/test_shape_encoder_contrastive.py` | `scripts/experiments/train_shape_encoder_contrastive.py` | 20 ✓ | LIVE shape | COMMIT (w/ target) |
| `tests/test_shape_vae_v3_hybrid.py` | `scripts/experiments/train_shape_vae_v3_hybrid.py` | 50 ✓ | LIVE shape | COMMIT (w/ target) |
| `tests/test_shape_vae_v3_hybrid_hardening.py` *(10th, not in handoff's list of 9)* | same hybrid | 25 ✓ | LIVE shape | COMMIT (w/ target) |
| `tests/test_train_shape_vae_v3_deriv.py` | `scripts/experiments/train_shape_vae_v3_deriv.py` | 32 ✓ | LIVE-ish (Pathway A) | COMMIT (confirm Pathway A not abandoned) |

**⚠ Coupling landmine:** the **tracked** `src/usv_spectrogram/classifier/__init__.py` has an
**uncommitted edit** adding `from .dann import …` + `from .cage_probe import linear_cage_probe`
to the public API. If `dann.py`/`cage_probe.py` are deleted, this edit **must be reverted**
or `import usv_spectrogram.classifier` raises ImportError and the whole classifier suite goes
red. DANN tests + DANN modules + `__init__.py` edit + `scripts/train_lab_classifier_v2.py` are
**one coupled unit** — handle together.

**Note:** the 6 COMMIT tests reference **untracked target scripts**. Committing a test without
its target → collection-time ImportError in CI. Commit each test *with* its `scripts/…` target.

## H3 — Worktrees (21; main at `75395217`)

| Worktree | Branch merged? | Unique untracked artifacts | Recommendation |
|---|---|---|---|
| `execute-skill` | merged | none | **SAFE-REMOVE** |
| `phase2c-mickey-report` | content in main (byte-identical) | none | **SAFE-REMOVE** |
| `ssh-audit-gitignore` | superseded (branch .gitignore older than main) | none | **SAFE-REMOVE** |
| `lab-cnn-classifier-plan` | merged | **`data/vocalmat_full/` 627M / 12,180 files (gitignored)** | KEEP — sole copy of training set |
| `contour-masked-vae-pipeline` | merged | **`results/masked_patches/` 16G** (13G `patches.npz`) + VAE `.pt` + 20 scripts | KEEP — sole copy |
| `vae-pytorch-pivot` | merged | `results/vae_5970_lab/` 1.3G + 10 memos + image-VAE code | KEEP — sole copy |
| `parallel-analytics-h06` | merged | `results/` 216M (q1–q9) + ~14 scripts | KEEP — sole copy |
| `contour-vae-denoised-retrain` | merged | denoised-retrain result handoff + eval scripts | KEEP — sole negative-result record |
| `deepsqueak-classification-audit` | merged | `results/ds_audit_2026-05-17/` + 7 scripts + 4 handoffs | KEEP |
| `deepsqueak-vae-wrapper` | merged | MATLAB VAE-train wrappers + runbook | KEEP |
| `cnn-progression-slide` | merged | 5 slide-builder scripts | KEEP |
| `lab-noise-triage` | merged | 6 cluster-verdict/soft-notch scripts | KEEP |
| `couple-vs-wild-analysis` | merged | `scripts/couple_vs_wild_analysis.py` | KEEP |
| `traditional-taxonomy-gallery` | merged | gallery generator + output | KEEP |
| `vocalmat-classifier-test` | merged | `results/vocalmat_test/` + `scripts/vocalmat_test/` (incl. `run_inference.py`) | KEEP |
| `feature-viewer-html` | merged | 2 viewer scripts | KEEP (low value) |
| `syllable-splitter` | merged | **untracked `postprocessing/syllable_splitter.py`, `cnn_prob_splitter.py`** | **REVIEW** — touches a Red-Flag `postprocessing/` path |
| `mickey-monday-briefing` | merged | 1 HTML briefing | REVIEW |
| `mickey-report-readability-v2` | merged | untracked `reports/` | REVIEW |
| `corpus-skill` | merged | `.claude/skills/corpus/` (possible sole skill source) | REVIEW |
| `reminder-2026-05-19-html-reports` | 1 trivial commit (stale reminder) | none | REVIEW → likely SAFE |

**Summary:** 3 SAFE-REMOVE, 13 KEEP-unique-data, ~5 REVIEW. **Branch-merged ≠ work-preserved**
— 18/21 branches are merged commit-wise, yet 13 hold the only copy of real deliverables as
never-committed files. **Recommend: remove only the 3 truly-empty SAFE-REMOVE; do not touch any
KEEP worktree until its unique data is backed up (commit-by-path or rsync to rig).**

## H4 — Root clutter

**7 PLAN_/HANDOFF_ at root:**
- `git mv` → `docs/plans/`: `PLAN_lab_cnn_classifier.md`, `PLAN_shape_representation_v2.md` (both TRACKED).
- plain `mv` → `docs/plans/`: `PLAN_geometric_shape_clustering_vae.md` (untracked, current); `PLAN_test_vocalmat_classifier.md` → `docs/plans/completed/`.
- plain `mv` → `docs/handoffs/`: `HANDOFF_TO_CLAUDE_CODE.md` (rename → `2026-05-15_presentation-deck-assets.md`), `HANDOFF_contour_masked_vae_orchestrator.md`.
- **DELETE:** `PLAN_contour_masked_vae_pipeline.md` — **byte-identical** to tracked `docs/plans/PLAN_contour_masked_vae_pipeline.md` (verified `diff` exit 0).

**27 root `.md` total:** keep at root → `README.md, CLAUDE.md, AGENTS.md, DECISIONS.md, IMPLEMENTATION_PROGRESS.md` (+ `PROJECTS.md` pending dedup-check vs `docs/human/PROJECTS.md`). Remaining 16 `ROADMAP*`/misc are tracked → `git mv` to `docs/plans/` (watch name overlaps: bootsnap, knowledge-activation, parts-finder).

**95 MB bundles (all untracked):** the `mickey_meeting_20260427/` and `usv_teaching_20260427/` dirs are just the **extracted copies** of their `.tar.gz`. Recommend `.gitignore` all 4 (keep tarballs as canonical; extracted dirs are redundant).

## H6 — New tooling

| Item | Now | Recommendation |
|---|---|---|
| `.codex/skills/` (4) | tracked | keep |
| `.codex/agents/` (6 .toml) | untracked | **commit** — portable shared agent specs |
| `.codex/hooks/` (scripts) | untracked | **commit** (gitignore `__pycache__/` + `*.bak`) — portable, no secrets |
| `.codex/config.toml` | untracked | **gitignore** — hardcodes `/home/light/…` (other user) |
| `.codex/hooks.json` | untracked | **gitignore** — hardcodes this machine's WSL UNC + `powershell.exe` |
| `.agents/skills/{kcheck,sync}/` | tracked | keep |
| `.claude/scheduled_tasks.lock` | untracked | **gitignore** — ephemeral lock/state (sessionId/pid) |

## H5 — Untracked data/results (DATA-SAFETY)

**⚠ `git add results/` would stage 93,121 files / 3.7 GB** (92,767 `.json` + 169 `.npz` +
88 `.csv` + 37 `.parquet` + 4 `.pt`). `git add USV_Detections/` → 85 files. This is the exact
`78d1c70` failure mode. **Highest-value safety action: gitignore the bulk reproducible
`results/` subdirs by path** so a stray `git add` can never swallow them.

**Confirmed protected:** `results/lab_classifier_v1/best.pt` is **tracked** (production model).

**⚠ IRREPLACEABLE, currently NO git protection (untracked AND not ignored) — data-loss risk:**
1. `results/vae_5970_lab/vae_best.pt` (12M) — trained VAE checkpoint
2. `results/vae_5970_lab/vae_final.pt` (12M)
3. `results/vae_ds_5970_lab/vae_ds_best.pt` (18M) — DS-port VAE checkpoint
4. `results/vae_ds_5970_lab/vae_ds_final.pt` (18M)
5. `USV_Detections/noise_labeled_files/131204_1400_m1fm1_chunk_000.json … _013.json` (14) — **human noise labels** (113 siblings already tracked → unprotected gap in a curated set)
6. `USV_Detections/rejected_detections/131204_1400_m3fm3_chunk_170/` (3 files) — **human reject decisions** (225 siblings tracked)
7. `USV_Detections/131204_…_chunk_170/_saved_tracking.json` + `_chunk_243/_saved_tracking.json` — **human review state**

**Reproducible → gitignore:** bulk `results/` batch+analysis dirs (~93K files), `presentation/`
(160M), `raven_tables_lab_131204/` (already ignored), `reports/` (REVIEW — Mickey deliverable).
**REVIEW (no reproducer found):** `results/syllable_split/cnn_rerun*/*.npz` (169, 48M).

---

## Recommended execution order (each step = present list → OK → act → `git diff --cached --stat`)
1. **H5 gitignore the bulk reproducibles** (biggest safety win — neuters the 93K-file stage risk).
2. **H5 protect the 7 irreplaceable items** (track human-labels by path; back up the 60M `.pt`).
3. **H4 moves** (`git mv` plans/roadmaps; delete the 1 dup; gitignore the 95M bundles).
4. **H6 tooling** (commit `.codex/{hooks,agents}`; gitignore config/hooks.json/lock).
5. **H2 tests** (commit 6 live shape tests + targets; resolve the DANN coupled unit).
6. **H3 worktrees** (remove the 3 SAFE-REMOVE; defer KEEP/REVIEW until data backed up).
