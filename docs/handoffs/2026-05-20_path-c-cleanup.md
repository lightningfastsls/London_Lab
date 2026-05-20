# Path C — post-verdict cleanup handoff

**Date:** 2026-05-20
**Predecessor:** `2026-05-19_path-c-cluster-execution.md` (the Step-1-onwards execution doc)
**Status:** Phase 5 verdict is **CLEAN** — see `results/phase5_cross_cohort/index.html` and `diagnostic_result.json`. This handoff lists everything that still needs cleanup after the verdict.

The verdict is the headline deliverable. None of the tasks below are blockers — they're correctness, documentation, and hygiene. Roughly ordered by priority.

---

## 1. Fix the source bugs that were patched at runtime

We worked around two bugs in `scripts/train_contour_vae_v2.py` and `scripts/assemble_combined_patches.py` with post-hoc scripts. The source bugs are still present; the next run on a fresh dataset will hit them again.

### 1a. `scripts/train_contour_vae_v2.py` lines 846–850 — broken latents merge

**Current (buggy):**
```python
latents_df = pd.DataFrame({
    "patch_idx": manifest["patch_idx"].astype(np.int32).values,
    **latent_cols,
})
prov = manifest[["patch_idx", "wav_stem", "call_id", "window_idx"]]
latents_df = latents_df.merge(prov, on="patch_idx", how="left")
```

**Bug**: when the manifest's `patch_idx` is not globally unique (true for any combined-cohort manifest), the merge cross-joins. For combined 4-cohort, 69,293 rows became 98,945 rows with (z ↔ wav_stem) mapping scrambled.

**Fix** (replace the merge with direct column assignment, same pattern as `re_encode_latents.py`):
```python
latents_df = pd.DataFrame({
    "patch_idx": np.arange(len(manifest), dtype=np.int64),
    **latent_cols,
    "wav_stem": manifest["wav_stem"].reset_index(drop=True).values,
    "call_id": manifest["call_id"].reset_index(drop=True).values,
    "window_idx": manifest["window_idx"].reset_index(drop=True).values,
})
if "cohort" in manifest.columns:
    latents_df["cohort"] = manifest["cohort"].reset_index(drop=True).values
```

Effort: ~5 lines, one Edit.

### 1b. `scripts/assemble_combined_patches.py` — manifest patch_idx non-unique after concat

**Current**: my memmap-streaming rewrite concatenates per-cohort manifests verbatim. Each per-cohort manifest has `patch_idx` 0..N-1 within its cohort. After concat, `patch_idx` 0 appears 4 times (one per cohort).

**Fix**: after `combined_manifest = pd.concat(per_cohort_manifests, ignore_index=True)`, renumber:
```python
combined_manifest["patch_idx_per_cohort"] = combined_manifest["patch_idx"]
combined_manifest["patch_idx"] = np.arange(len(combined_manifest), dtype=np.int64)
```

Effort: 2 lines, one Edit.

### 1c. Verification after both fixes

Re-run the full pipeline on a small subset (e.g. just 5970 + 3452 cohorts, which together are ~13K patches and will train in minutes):
```bash
cd /data/shachar/contour_vae   # or local equivalent
.venv/bin/python scripts/assemble_combined_patches.py \
    --cohort-dirs results/masked_patches/5970_focus results/masked_patches/3452_focus \
    --cohort-names 5970 3452 \
    --output-dir results/masked_patches/combined_test
# ... train ... phase5 ...
```
Assert: `latents.parquet` has the same row count as `patches_manifest.parquet`, both have unique `patch_idx`.

---

## 2. Document the verdict for the knowledge graph

### 2a. Write a vault note

Per `2026-05-19_path-c-cluster-execution.md` (Step 4 — "Documentation"):

> **CLEAN** (no dim fires) | Contour mask successfully removes cage signature even cross-cohort. Write `notes/2026-05-19_path-c-cleared-cross-cohort.md` recording the verdict. Phase 5 is the project's headline result.

Procedure (per `CLAUDE.md`): NEVER write directly to `notes/`. Instead:
1. Drop the note in `inbox/` (e.g. `inbox/2026-05-20_path-c-cleared.md`)
2. Run `/reduce` to route through the pipeline
3. Then `/reflect` will link it to relevant topic maps (cohort-confound, VAE-methodology, etc.)

Suggested content:
- **Headline**: CLEAN verdict, 0 of 32 dims fire, max |Cohen's d| = 0.77, max |r| = 0.36
- **What this confirms**: contour-mask methodology removes cage signature across N=4 cohorts (5970, 3452, 9252, lab_131204)
- **What it doesn't say**: the model is sufficient for downstream behavior analysis — that's a separate question
- **Wikilinks**: `[[vae-cage-confound-diagnostic]]`, `[[contour-mask-methodology]]`, etc.

### 2b. Update predecessor handoff to mark complete

Edit `docs/handoffs/2026-05-19_path-c-cluster-execution.md` — top status block:
```diff
- **Status:** Ready to execute on **rig** (preferred) or ELSC HUJI cluster (fallback).
- **Blocked on:** Step 0 user re-labeling against patched 5970 comparison (re-extract in progress).
+ **Status:** COMPLETE — Phase 5 verdict CLEAN (2026-05-20). All 5 steps executed. See `2026-05-20_path-c-cleanup.md` for residual cleanup.
+ **Blocked on:** Nothing.
```

---

## 3. Add regression tests for the bugs we hit

The bugs in §1 escaped because nothing tested for "multiple manifest concat preserves patch_idx uniqueness" or "latents.parquet has same row count as patches". Add two tests:

### 3a. `tests/test_assemble_combined_patches.py` (new file)

Test that assemble produces a manifest with unique `patch_idx`:
```python
def test_combined_manifest_patch_idx_globally_unique(tmp_path):
    # build two fake cohort dirs with overlapping patch_idx ranges
    # run assemble_combined_patches.main()
    # assert combined.patch_idx.nunique() == len(combined)
```

### 3b. `tests/test_train_contour_vae_v2.py` (new file or appended)

Test that the latents-export logic produces same row count as input patches:
```python
def test_latents_row_count_matches_patches(tmp_path):
    # build a tiny manifest with non-unique cohort-relative patch_idx
    # call the latents-writing function in isolation
    # assert len(latents) == len(manifest)
```

---

## 4. Pull results back to main repo (Step 5 of the predecessor handoff)

Currently all artifacts live on the rig at `/data/shachar/contour_vae/`. To preserve them in the project repo:

```bash
# From local worktree root:
mkdir -p models/contour_vae_combined results/contour_vae_combined results/phase5_cross_cohort

scp shachar@<RIG_IP>:/data/shachar/contour_vae/models/contour_vae_combined/best.pt \
    models/contour_vae_combined/
scp shachar@<RIG_IP>:/data/shachar/contour_vae/models/contour_vae_combined/hyperparams.json \
    models/contour_vae_combined/

scp shachar@<RIG_IP>:/data/shachar/contour_vae/results/contour_vae_combined/latents.parquet \
    results/contour_vae_combined/
scp shachar@<RIG_IP>:/data/shachar/contour_vae/results/contour_vae_combined/training_log.csv \
    results/contour_vae_combined/
scp -r shachar@<RIG_IP>:/data/shachar/contour_vae/results/contour_vae_combined/reconstructions \
    results/contour_vae_combined/

# Phase 5 results are already local (we SCPd them during the verdict step).
```

Decide which artifacts to commit. Recommendations:
- ✅ Commit: `hyperparams.json`, `training_log.csv`, Phase 5 JSON + CSVs, HTML report
- ⚠️ Optional: `latents.parquet` (15 MB, can be regenerated from `best.pt`)
- ⚠️ Optional: `best.pt` (31 MB — depending on your git LFS / commit-size policy)
- ❌ Don't commit: `last.pt` (epoch 108), `patches.npz` (16+ GB)

Add `.gitignore` rules for the big artifacts so they don't get swept in by future `git add`.

---

## 5. Clean up backup files

Locally and on rig, the workflow left `.prefix_bak`, `.broken_bak`, and `.duplicate_patch_idx_bak` files marking each stage we patched in place. Now that the verdict is confirmed CLEAN, they're rollback insurance with diminishing returns.

**On rig:**
```bash
ssh shachar@<RIG_IP> '
find /data/shachar/contour_vae -name "*.prefix_bak" -o -name "*.broken_bak" -o -name "*.duplicate_patch_idx_bak" | xargs -r ls -lh
# review the list, then:
find /data/shachar/contour_vae -name "*.prefix_bak" -delete
find /data/shachar/contour_vae -name "*.broken_bak" -delete
find /data/shachar/contour_vae -name "*.duplicate_patch_idx_bak" -delete
'
```

Frees ~16 GB on the rig's /data.

**Locally (worktree):**
```bash
WT=.claude/worktrees/contour-masked-vae-pipeline
find "$WT/results/" -name "*.prefix_bak" -ls
# review, then delete
find "$WT/results/" -name "*.prefix_bak" -delete
```

Frees ~3 GB local.

Both safe because the post-fix files are validated by the Phase 5 verdict.

---

## 6. SSH credential audit

The handoff at `2026-05-19_ssh-credential-audit.md` (in this directory) has the full procedure. Run it once to know which files mention the rig's SSH coordinates and whether each is gitignored.

Quick start:
```bash
RIG_IP=<your-rig-ip> RIG_USER=<your-rig-user> RIG_HOST=<your-rig-hostname> \
    bash /tmp/ssh_audit.sh
```

Action on the report: anything in the TRACKED section should be replaced with `<RIG>` placeholders if the worktree might ever be pushed publicly.

---

## 7. Optional — Path C v2 follow-ups

If the project sponsor wants to push the methodology further, possible next paths (none in scope here, just noting):

- **Tighter mask sweep**: re-run Phase 5 with masks at ±2, ±3, ±4, ±5, ±6 kHz to find the minimum bandwidth that still passes the diagnostic. Current bandwidth is ±5 kHz.
- **Per-cohort downsampling**: lab_131204 dominates the combined dataset (55,863 of 69,293 = 80.6%). Repeating with stratified or weighted sampling could test whether dominance affects the verdict.
- **Out-of-distribution test**: encode a held-out cohort (a new strain or lab) and check whether its latents fall in-distribution with the 4 we trained on.

These are scope-creep relative to "did Path C work" — they're "what else could we learn with this tool".

---

## Validation — when is "done" done?

When all the following are true:
- §1 fixes applied + py_compile + targeted re-run passes
- §2a vault note exists in `notes/` (via reduce pipeline)
- §2b predecessor handoff marked COMPLETE
- §3 regression tests added + passing
- §4 artifacts in the main repo where you want them, `.gitignore` updated
- §5 backups deleted
- §6 SSH audit report shows no surprises (or accepted-risk decisions documented)

After that the worktree can be merged or retired. The Path C question is answered.
