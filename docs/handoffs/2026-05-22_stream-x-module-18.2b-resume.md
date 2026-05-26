# Stream X resume — Module 18.2b Full Data Preparation

**Date**: 2026-05-22
**Predecessor**: Streams M and V from `docs/handoffs/2026-05-22_post-18.2a-orchestrator.md` — both shipped.
**This handoff**: a thin pointer; the original orchestrator doc has the full Stream X spec.

## State entering this session

- `origin/main` includes both merges:
  - `bb990b1b` — Merge PR #1 (Module 18.1 + 18.2a, cleaning gate GO verdict baselined)
  - `6b7a03d2` — Merge PR #2 (diagnostic-VAE epoch-scaling vault note)
- Local `main` may still have user WIP — confirm before pulling.
- Worktree `worktree-lab-cnn-classifier-plan` still exists locally and on origin (not deleted after merge).
- Sample data: 2,196 PNGs / 113 MB in `.claude/worktrees/lab-cnn-classifier-plan/data/vocalmat_sample/` (verify before relying).

## What to do

**Read the canonical Stream X spec** at `docs/handoffs/2026-05-22_post-18.2a-orchestrator.md` (the "Stream X" section). Do not re-derive — every constraint (recording-level grouping, class imbalance 24.5×, ROADMAP decisions D2/D3/D4/D5, files-NOT-to-touch list) is already there.

Then invoke:

```
/implement Module 18.2b Full Data Preparation
```

## Orchestration reminders

1. Step 0 of `/implement`: spawn `test-architect` for pre-implementation tests on `resample.py` and `dataset.py`. Do NOT silently skip.
2. Spawn the background full download (`scripts/cnn_download_vocalmat_sample.py --full` or whatever the ROADMAP specifies) BEFORE coding — it's I/O-bound, code is cognitive, parallelize.
3. Code `resample.py` first (smallest, most testable in isolation). Then `dataset.py`.
4. Wait for download; verify counts against manifest.
5. Spawn `master-reviewer` after implementation.
6. Pre-implementation: run `/kcheck` before touching the cleaning pipeline.

## Binding constraints (not optional)

- `corpus.SAMPLE_RATE_HZ = 300_000` is fixed. Never modify `corpus.py`. The classifier pipeline operates at 250 kHz via `classifier.TARGET_SAMPLE_RATE_HZ`.
- Recording-level grouping: a recording in train MUST NOT appear in val or test. The cage-confound issue otherwise reappears via data leakage.
- `mult_steps` has only 74 files on OSF. Per ROADMAP D5: keep all 12 classes + class-weighted CE + focal loss + oversampling; revisit only if v1 per-class precision < 0.20.
- Production CNN is `models/hard_neg_retrain/best_model.pt` for any inference-mode validation.

## NOT to touch

- `src/usv_spectrogram/corpus.py` (HIGH-risk canary)
- `src/usv_spectrogram/app/core/sliding_inference.py` (HIGH-risk)
- `src/usv_spectrogram/app/core/{notch,denoise}.py`
- `src/usv_spectrogram/postprocessing/normalization.py`
- `scripts/run_batch_detection.py`
- `src/usv_spectrogram/classifier/cleaning_pipeline.py` (frozen for behavior changes)
- `src/usv_spectrogram/classifier/diagnostics.py` (partial unfreeze — docstrings/deprecations OK, behavior changes require Tier-3 review)
- The existing `tests/classifier/test_*.py` spec files

## Done means

ROADMAP §18.2b exit criteria all checked:
- Full download complete + manifest reconciled
- `resample.py` + `dataset.py` exist with passing tests
- Class balance verified
- Module docs written
- master-reviewer says SHIP
