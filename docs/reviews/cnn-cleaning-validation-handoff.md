# Implementation Handoff: CNN Cleaning Validation Gate (Module 18.1)

**Module:** Cleaning Validation Gate (`ROADMAP_lab_cnn_classifier.md` §18.1)
**Review Tier:** 3 (DSP + statistical methodology)
**Date:** 2026-05-21
**Branch / worktree:** `.claude/worktrees/lab-cnn-classifier-plan/`

## What Changed

Module 18.1 ships a falsifiable, mechanically-decidable cleaning gate for
the lab CNN classifier (Phase 18). The gate measures how much cage-confound
signal survives each of 6 cleaning ablations using 4 diagnostics (notch
injection, per-band Cohen's d, K-NN same-cohort rate, raw-pixel PCA d).
Implementation closed in this session and passed master-reviewer (Tier 3)
after a 10-item fix batch.

- 4-layer cleaning stack (soft-notch -> baseline subtraction -> global MAD ->
  per-recording Z-score) with each layer independently toggleable, wrapped
  in `clean_spectrogram(spec, cfg, recording_id)`.
- `CleaningConfig` namedtuple (NOT a frozen dataclass — see Key Invariants
  below) carries the 4 toggles + `baseline_mode` + `tonal_library_path` +
  `sample_rate_hz` (default 250_000, VocalMat-aligned).
- 4 diagnostics returning `DiagnosticResult(name, value, threshold,
  threshold_direction, passed, details)` with thresholds hardcoded to spec:
  0.30 / 0.30 / 0.85 / 1.50.
- Pattern 4 CLI (`scripts/cnn_cleaning_validation.py`) running the 6
  ablations and rendering a Markdown report with go/no-go footer.

## Files Created

| File | LOC | Purpose |
|------|-----|---------|
| `src/usv_spectrogram/classifier/__init__.py` | 49 | Package init + public re-exports + `TARGET_SAMPLE_RATE_HZ` |
| `src/usv_spectrogram/classifier/cleaning_pipeline.py` | 387 | `CleaningConfig` + `clean_spectrogram` 4-layer stack |
| `src/usv_spectrogram/classifier/diagnostics.py` | 773 | 4 diagnostics + diagnostic VAE + ablation registry |
| `scripts/cnn_cleaning_validation.py` | 465 | CLI driver, 6-ablation matrix, Markdown report |
| `docs/modules/cnn-cleaning-validation.md` | 213 | Module documentation |

## Files Modified (review-fix batch only)

- `docs/architecture/patterns.md` — Pattern 1 "Variant: namedtuple
  subclasses" sub-section added (documents the `CleaningConfig` deviation).
- `tests/classifier/test_cleaning_pipeline.py:51` — `parents[3]` ->
  `parents[2]` to match the corrected `test_diagnostics.py:58`.

## Test Results

```
.venv/bin/python -m pytest tests/classifier/ -v
31 passed, 0 failed in 5.93 s
```

- 14 tests for `cleaning_pipeline.py`, 17 for `diagnostics.py`.
- Three documented test-spec amendments (band alignment, noise-floor
  threshold, REPO_ROOT path) all user-approved 2026-05-21 — recorded in
  the module doc under "Test-spec amendments (2026-05-21)".
- The 10-item fix batch (see review file) did NOT break any existing test.

## Key Invariants for Future Modifiers

1. **Cage-tone scaling** — `_inject_cage_tone` scales the injection to
   `INJECTION_SIGMA * local_std` (σ = 2.0). DO NOT revert to a fixed `+N dB`
   offset. The fixed offset saturates normalised-input ablations
   (`mad_only`, `zscore_only`, `all_layers`), producing false-FAIL migration
   on the gate's most important measurement. The fallback for constant-input
   bands uses `_INJECTION_FALLBACK = 0.1`, safe on both [0,1] and dB
   domains.
2. **`notch_injection_test` trains on combined (A + B)** — the diagnostic
   VAE is trained on the concatenation of cohort A and cohort B
   spectrograms; never on cohort A alone. Training on A only biases the
   latent space toward A's features and inflates apparent migration. Both
   the module docstring and the function docstring of `notch_injection_test`
   spell this out — if you find a "cohort A only" claim anywhere, the
   docstring is stale, not the code.
3. **CleaningConfig is a namedtuple subclass, NOT a frozen dataclass** —
   the `test_cleaning_config_is_immutable_after_creation` test probes the
   contract using `object.__setattr__`, which bypasses
   `@dataclass(frozen=True)` at the C-level path. Switching to a frozen
   dataclass will silently break the immutability contract that test asserts.
4. **Layer order is fixed** — `soft-notch -> baseline subtraction -> global
   MAD -> per-recording Z-score`. Re-ordering is a behaviour change and is
   enforced by `test_clean_spectrogram_layer_order_*`. The module-private
   `_apply_*` functions are the patch boundary for that test — DO NOT
   rename or inline them.
5. **Global MAD reproduces `app/core/sliding_inference.py` byte-for-byte** —
   the math (`_MAD_VMIN_SCALE=2.0`, `_MAD_VMAX_SCALE=4.0`, clip-then-normalize,
   `vmax > vmin` guard) is the cross-phase training-grid invariant
   (`feedback-cnn-inference-global-mad`). DO NOT drift these values.
6. **No production-file modifications** — `corpus.py`,
   `app/core/sliding_inference.py`, `app/core/notch.py`, `app/core/denoise.py`,
   `postprocessing/normalization.py`, `scripts/run_batch_detection.py`
   are intentionally untouched. The cleaning module imports
   `notch.TonalLibrary` and `denoise.subtract_temporal_baseline` when
   available and falls back to in-module reproductions when not.

## Known Deferred Items

These two ROADMAP exit criteria are correctly deferred to Module 18.2
because they require the VocalMat dataset which 18.2 owns:

- **Real-data CLI run.** The current CLI always falls back to synthetic
  data even when `--vocalmat-sample` is supplied
  (`scripts/cnn_cleaning_validation.py:399-412`). 18.2 will provide the
  data loader.
- **`docs/handoffs/cleaning-validation-report.md`.** This is the real-data
  go/no-go report. It cannot exist until the previous deferred item lands.

## Cross-References

- ROADMAP: `ROADMAP_lab_cnn_classifier.md` §18.1
- Review: `docs/reviews/cnn-cleaning-validation-review.md`
- Module doc: `docs/modules/cnn-cleaning-validation.md`
- Decision notes:
  - `notes/cage acoustics drive between-cohort spectrogram separation more than biology.md`
  - `notes/falsifiable cleaning gates with numeric thresholds beat vibes-based judgment.md`
  - `notes/notch-injection migration measures cleaning quality better than passive cohort sampling.md`
- Pattern reference: `docs/architecture/patterns.md` §1 (Config Dataclass)
  including the new "namedtuple variant" sub-section.
- Progress log entry: `IMPLEMENTATION_PROGRESS.md` (dated 2026-05-21).
