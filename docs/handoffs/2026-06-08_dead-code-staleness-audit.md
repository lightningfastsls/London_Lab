# Handoff — Dead / Stale / Unneeded Code Audit (ultracode)

**Created:** 2026-06-08 · **For:** a fresh ultracode (multi-agent Workflow) session
**Predecessor:** 2026-06-08 repo handoff cleanup (data + non-USV removal) — see
`docs/REPO_CLEANUP_DONE_2026-06-08.html`. That session cleaned **data and clutter**;
this one targets **code that is dead, stale, duplicated, or unneeded** so a successor
inherits a lean, legible toolchain.

---

## Goal

Find and (after approval) remove/consolidate code that no longer earns its place:
unreferenced scripts, dead modules/functions, duplicate/near-duplicate code, stale
config, docs pointing at removed code, and abandoned experiment trails. **Audit-first:
produce a tiered, risk-ranked plan; do NOT delete in the fan-out.** Mirror the
predecessor's method (read-only audit → plan → safety-critic → staged by-path commits).

## How to run it (suggested Workflow shape)

This is explicitly an **ultracode** task — use the `Workflow` tool. A good structure
(reuse the predecessor's, in `…/workflows/scripts/repo-cleanup-audit-*.js`):

- **Phase 1 — Audit (parallel, read-only `Explore` agents)**, one per dimension below.
  Each returns structured findings: `path · classification(dead|stale|duplicate|keep|
  needs-confirmation) · risk · evidence · proposed_action`.
- **Phase 2 — Synthesize** into tiers (T0 = zero-risk e.g. unreferenced one-off scripts
  with no importers/doc refs; T1 = consolidate duplicates; T2 = remove dead modules
  needing confirmation; T3 = refactor oversized files).
- **Phase 3 — Safety-critic** (adversarial): for every "dead" claim, try to prove it's
  actually reachable (dynamic import, importlib, subprocess invocation, entry in a
  config/JSON, referenced only in a docstring example, used by a notebook). Default
  skeptical — a script invoked only via `subprocess`/`importlib.util.spec_from_file`
  will look unimported to a naive grep.

### Audit dimensions

1. **Unreferenced scripts** — 154 top-level `scripts/*.py` (+51 in `experiments/`,
   `labeling/`). For each: is it imported anywhere, referenced in any doc/README/
   handoff, invoked by another script via subprocess, or named in CLAUDE.md's
   canonical pipeline? Zero hits across all = dead-candidate. Beware subprocess/CLI
   invocation by string.
2. **Duplicate / near-duplicate code** — strong smell here: multiple
   `cnn_wholefile_roc*.py` (`_expanded`, `_native_first`), `render_*_patches.py`
   (vocalmat_style / human_view / v1_faithful / vocalmat_gt_sample), `cnn_*_slide.py`
   generators, `*_2dpca.py` variants. Identify which is canonical vs superseded.
3. **Dead modules/functions in `src/usv_spectrogram/`** (96 modules) — submodules or
   public functions never imported outside their own tests. Use a reachability sweep
   from the real entry points (run_batch_detection, run_app, train_*, the apps).
4. **Retired subsystems** — the Streamlit **param_lab/** (esp. `app_legacy.py`, 662
   LOC, has a REFACTORING_SUMMARY.md) is deprecated (PyQt6 is canonical). The three
   Streamlit launchers (`usv_labeling_tool.py`, `usv_parameter_lab.py`,
   `noise_review_tool.py`) are tombstones with INCONSISTENT behavior (one still runs,
   two `os._exit` at import). Decide: salvage `metrics.py`/`sweep.py`/
   `heuristic_detect.py` into the PyQt app, then archive the rest.
5. **Stale tests beyond the 11 already archived** — `tests/archive/` already holds 11
   dead-collection tests (this session). But ~82 tests still FAIL at runtime against
   missing/changed scripts (`analyze_detection_confidence.py`, `train_lab_classifier.py`
   both absent; `analyze_acoustic_features` expects a function never written;
   `sis_benchmark` script bug). Triage: is the *test* stale or the *code* missing?
6. **Docs referencing removed code** — `docs/scripts-index.md` is stale (claims ~76
   entry points; there are 154; ~half undocumented). README/project-structure.md
   counts drift. Handoffs reference scripts that may now be gone.
7. **`archive/cleaning_legacy/`** — is anything in `archive/` still imported by LIVE
   code? (If so it's not really archived.) Conversely, is live code now duplicating
   what's in archive?
8. **Stale git worktrees** — 18 under `.claude/worktrees/`, ALL dirty (uncommitted
   work). Each is a frozen branch tip. Audit which branches are merged/abandoned;
   propose `git worktree remove` + branch deletion for dead ones (per-worktree, after
   checking the dirty changes are disposable).
9. **Unused dependencies** — diff `requirements*.txt` against actual imports across
   `src/` + `scripts/`. Flag installed-but-unimported packages.
10. **Oversized files (refactor, not delete)** — `app/main_window.py` (1823),
    `dataset/assembler.py` (1490), `classification/cross_population.py` (1484),
    `classification/repertoire_stats.py` (1142). Propose splits; do NOT auto-refactor.

## Concrete leads already found (this session)

- `scripts/train_lab_classifier.py` — the training CLI for the **production** lab
  classifier (`results/lab_classifier_v1/best.pt`) — is MISSING from the live tree;
  survives only in `archive/cleaning_legacy/stack1/scripts/`. Either a real gap
  (restore it) or the model is frozen and the training path is dead. Decide.
- `src/usv_spectrogram/features/` package — referenced by archived tests but does not
  exist; `omer_vectorize`/`amvoc_autoencoder` were never built (TDD specs only).
- `run_app_safe.py` vs `run_app.py` — likely consolidatable.
- `usv_language/` — a second tracked package at repo root, NOT under `src/`, NOT
  installed by `pip install -e .`, no packaging. `information_theory.py` lives here.
  Decide: integrate into the package, or document as a standalone sibling.
- Only ~10 files carry TODO/FIXME/XXX/HACK/DEPRECATED markers — low comment-debt;
  the real debt is whole unused files, not inline cruft.

## Binding constraints (flatten into every agent prompt)

- **Audit-only fan-out. Delete nothing in agents.** Tag HEAD first:
  `git tag pre-deadcode-audit-<date> HEAD`.
- **NEVER `git add -A` / `git add .`** — stage by exact path. A PreToolUse hook
  (`.claude/hooks/guard_destructive.py`) hard-blocks bulk staging, `reset --hard`,
  `clean -f`, and recursive `rm` outside a safe allowlist. Do NOT append
  `# ALLOW_DESTRUCTIVE` to tunnel it (the auto-mode classifier flags that as bad-faith);
  delete tracked code via `git rm <explicit paths>` (no `-r`), which is allowed.
- **LOCKED — never modify:** `src/usv_spectrogram/corpus.py` (canonical STFT/sample-rate
  constants), `ExtractionConfig` values, `scripts/run_batch_detection.py`,
  `app/core/sliding_inference.py`, `postprocessing/`, `models/hard_neg_retrain/*`
  (production CNN). These are NOT dead even if lightly referenced.
- **Tests-as-spec:** do not modify test expectations or delete pre-implementation spec
  tests without discussion. Prefer MOVE to `tests/archive/` over delete (see the
  existing `tests/archive/conftest.py` + README pattern). Verify a test's target is
  genuinely gone (grep the whole tree, not just `src/`) before archiving it.
- **Reachability ≠ grep-for-name.** Confirm "unused" against: subprocess calls,
  `importlib`, dynamic `__import__`, config/JSON references, entry-point tables, and
  docstring CLI examples. The predecessor's safety-critic caught exactly these.
- **`pytest` must still collect 0 errors and not lose passing tests** after any change.
  Baseline today: 1584 collected, 0 collection errors, ~82 pre-existing runtime
  failures (do NOT "fix" by deleting their tests without triage).
- **Per-path commits, each revertible.** Use the `Co-Authored-By: Claude …` trailer.

## Definition of done

1. An HTML audit report (`docs/CODE_DEADWEIGHT_AUDIT_<date>.html`) — feedback rule:
   user-facing outputs are HTML, and the message MUST include the
   `file://wsl.localhost/...` URL.
2. A tiered, risk-ranked removal/consolidation plan with per-item evidence and a
   safety-critic verdict.
3. (After user approval, per tier) staged by-path commits removing/consolidating, with
   `pytest` collection still clean and no regressions vs the baseline.
4. Anything ambiguous (production-adjacent, possibly-live) surfaced as a decision, not
   deleted.

## Useful prior artifacts

- Predecessor audit JSON (scripts/src/tests inventories already mapped):
  `~/.claude/jobs/ccb0c8ba/tmp/audit_result.json` (if still present) — or re-run the
  workflow at `…/workflows/scripts/repo-cleanup-audit-wf_*.js`.
- `docs/REPO_CLEANUP_DONE_2026-06-08.html`, `docs/DATA_REGENERATION_RECIPES.md`,
  `tests/archive/README.md`, `scripts/reclaim_tier4_data.sh` (pattern for a safe,
  self-verifying, DRY_RUN-able cleanup script).
- Revert safety net for the data cleanup: tag `pre-cleanup-2026-06-08`.
