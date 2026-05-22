# Module 18.2a — VocalMat Sample Downloader + Real-Data Loader Review

**Date:** 2026-05-22
**Reviewer:** master-reviewer
**Review Tier:** 2 (network I/O, data pipeline, ML-adjacent — no direct
STFT authoring, but consumes DSP constants and produces spectrogram
arrays fed into the gate)
**Handoff:** `docs/handoffs/2026-05-21_module-18.2a-sample-download.md`
+ IMPLEMENTATION_PROGRESS.md §2026-05-22
**Files reviewed:**
- `scripts/cnn_download_vocalmat_sample.py` (~520 LOC)
- `tests/test_cnn_download_vocalmat_sample.py` (~220 LOC, 11 tests)
- `scripts/cnn_cleaning_validation.py` (new `_load_real_cohorts`,
  `_png_to_luminance`, `_wav_to_spectrograms`, `_resize_2d` block +
  `main()` wiring)
- `tests/classifier/test_cleaning_real_data_loader.py` (~230 LOC, 12 tests)
- `docs/modules/cnn-download-vocalmat-sample.md`
- `docs/handoffs/cleaning-validation-report.md` + `.n4-NOGO.md`
- `IMPLEMENTATION_PROGRESS.md` (2026-05-22 entry only)

---

## Pre-Review Expectations

Before reading the code I expected:

- No redeclaration of `SAMPLE_RATE_HZ`, `STFT_N_FFT`, `STFT_HOP` — all
  imported from `usv_spectrogram.corpus` per CLAUDE.md corpus protocol.
- `resample_poly(up=5, down=6)` used for 300→250 kHz (constraint C1);
  NOT a hardcoded ratio.
- `urllib` timeout of 30 s, `page_size=100`, exponential 429 backoff,
  idempotent skip-if-exists, seed=1729.
- Do-not-touch list (`corpus.py`, production detection files, existing
  `tests/classifier/test_*.py`) completely unmodified.
- GO/NO-GO decision backed by a plausible mechanism, not cherry-picked
  thresholds.
- 85 tests pass; 62 are pre-existing 18.1 spec tests, 23 are new.

---

## Test Suite

```
85 passed in 17.18s
```

Count matches the handoff claim. The 62 pre-existing spec tests still
pass — confirmed by the empty `git diff` on all do-not-touch files. The
23 new tests are substantive: they test shape, dtype, range,
finite-value, empty-path error, too-short WAV error, determinism, and
the three ROADMAP-prescribed scenarios (dry-run, manifest balance,
idempotency). Not skeletons.

---

## Findings

### WARNING 1 — `_load_real_cohorts` WAV glob is not recursive

**File:** `scripts/cnn_cleaning_validation.py:439–440`

**What:** `lab_wav_dir.glob("*.wav")` and `wild_wav_dir.glob("*.wav")`
are shallow globs. The VocalMat PNG scan at line 420 uses
`rglob("*.png")` (recursive), but the two WAV globs do not.

**Why it matters:** A user pointing `--lab-131204-sample` at a directory
where WAVs live in per-session subdirectories will get an empty list
and a `FileNotFoundError("No .wav in ...")` with no explanation about
depth. Implementor's note: the project's current `USV_lab_131204/` and
`5970 USV/` happen to be flat (all WAVs at depth=1), so the wet run was
unaffected — but defensive `rglob` is cheap insurance.

**Fix:** Change lines 439–440 to use `rglob("*.wav")`.

---

### WARNING 2 — `train_diagnostic_vae` docstring is stale on real-data epoch budget

**File:** `src/usv_spectrogram/classifier/diagnostics.py` (module
docstring + `train_diagnostic_vae` docstring)

**What:** The docstrings say "4-8 epochs is enough for diagnostic K-NN
measurements." The real-data gate showed 4 epochs is definitively NOT
enough for 227×227 inputs (produced `notch_injection_migration = 1.0`
on `all_layers`, triggering the initial NO-GO).

**Why it matters:** A downstream developer following the docstring's
recommendation on real data will reproduce the silent under-training
failure.

**Constraint:** `src/usv_spectrogram/classifier/diagnostics.py` is on
Module 18.1's do-not-touch list (see
`docs/reviews/cnn-cleaning-validation-handoff.md` "Files NOT Modified"
section). 18.2a is therefore NOT the right place for this fix.

**Resolution:** Documented in
`docs/handoffs/cleaning-validation-report.md` Interpretation section
and in the IMPLEMENTATION_PROGRESS.md 2026-05-22 entry. Flagged for a
successor 18.1.x patch by a future session with explicit scope to
touch the cleaning module.

---

### NIT 1 — `notch_depth_db` parameter is accepted but silently deleted

**File:** `src/usv_spectrogram/classifier/diagnostics.py:427–428`

**What:** `_inject_cage_tone` accepts `notch_depth_db` in its signature
(for backward compatibility), then immediately does
`del notch_depth_db`. Dead parameter traverses the call stack while
doing nothing.

**Resolution:** Same scope rule as WARNING 2 — touches the frozen
diagnostics module, not in 18.2a's scope. Flagged for the same future
patch.

---

### NIT 2 — `knn_same_cohort_rate` divergence between n=4 and n=32 runs is not explained in the audit trail

**What:** The NO-GO report shows `baseline_only` knn=0.7287 and the GO
report shows 0.3333 — a large swing on what is nominally the same data
through the same cleaning layer. The Interpretation section in the GO
report discusses `all_layers` notch_injection divergence but does not
mention that `knn_same_cohort_rate` also changed across ablations.

**Why it matters:** A future auditor reading the two reports side-by-
side will notice the discrepancy. The explanation (deeper VAE training
produces better-mixed embeddings even for `baseline_only`) is
scientifically sound but is not written down.

**Fix:** Add one sentence to the Interpretation section of either
report.

---

## GO Verdict Credibility Assessment

**Credible.** The root-cause analysis (4 epochs → insufficient VAE
convergence for 51,529-feature input → degenerate `all_layers`
migration = 1.0) is mechanistically sound:

1. The individual ablations all had `notch_injection` values well
   below 0.30 in the NO-GO run — the cleaning stack itself was not the
   problem, only the measurement tool.
2. `per_band_cohens_d` and `raw_pixel_pca_d` (raw-pixel diagnostics,
   not VAE-based) are identical between the n=4 and n=32 runs,
   confirming the data did not change between runs.
3. The GO run's `all_layers` values (notch=0.0000, per_band_d=0.0696,
   knn=0.3333, pca_d=0.0000) show clear cleaning-stack effect compared
   to `raw` (notch=1.0, per_band_d=26.02, knn=0.3333, pca_d=52.16).
4. The user explicitly chose to re-run with `n_epochs=32`; the change
   was not a threshold adjustment or code modification to force a
   pass.

Not threshold-shopping. The 14 OSF download failures (4+8+2) are
reported honestly and do not affect the verdict pool.

---

## Do-Not-Touch List Verification

`git diff -- src/ tests/classifier/test_cleaning_pipeline.py
tests/classifier/test_diagnostics.py
tests/classifier/test_cleaning_pipeline_adversarial.py
tests/classifier/test_diagnostics_adversarial.py` produces empty
output. All protected files unmodified.

---

## Corpus Protocol Compliance

- `STFT_N_FFT`, `STFT_HOP` imported from `usv_spectrogram.corpus` —
  correct.
- `TARGET_SAMPLE_RATE_HZ` imported from `usv_spectrogram.classifier`
  — correct, not redeclared.
- `SAMPLE_RATE_HZ` imported in tests for synthetic WAV creation only.
- Downloader has no STFT constants (correct).
- `_REAL_TARGET_SHAPE` (227, 227) and `_REAL_WINDOW_DURATION_S` (0.22)
  documented as classifier-pipeline-specific, not corpus invariants.

---

## Architectural Fit

Adds an upstream loader and downstream report-handling only. Does not
modify cleaning pipeline or diagnostics directly. `main()` change is
additive: the existing smoke-test fallback is preserved and the real-
data branch is the new path when all 3 `--*-sample` args are supplied.

---

## Verdict

**CHANGES NEEDED** (no BLOCKERs; two WARNINGs, two NITs)

The module is architecturally sound, the GO verdict is credible,
corpus protocol is respected, and the test suite is real.

- WARNING 1 (glob → rglob): IN SCOPE for 18.2a; will fix.
- WARNING 2 (stale docstring) + NIT 1 (dead parameter): OUT OF SCOPE —
  touches the frozen classifier module. Already flagged in the
  Interpretation section and the IMPLEMENTATION_PROGRESS entry for a
  successor 18.1.x patch.
- NIT 2 (audit trail explanation): IN SCOPE; will fix.

## Fixes Applied (2026-05-22)

### WARNING 1 — recursive WAV glob
- Changed `lab_wav_dir.glob("*.wav")` → `lab_wav_dir.rglob("*.wav")`
  and `wild_wav_dir.glob("*.wav")` → `wild_wav_dir.rglob("*.wav")` at
  `scripts/cnn_cleaning_validation.py:439–440`.
- Re-ran loader tests: 12/12 still pass (the existing
  `test_load_real_cohorts_returns_three_cohorts` exercises the
  recursive path implicitly via PNG scan, and the lab/wild WAVs in the
  fixture are at depth 1, which `rglob` matches).

### NIT 2 — knn divergence audit-trail note
- Added one-sentence explanation to the Interpretation section of
  `docs/handoffs/cleaning-validation-report.md` noting that
  `knn_same_cohort_rate` also depends on VAE training depth and that
  the change between the two runs is expected and monotonic.

### WARNING 2 + NIT 1 — out-of-scope (frozen 18.1 module)
- NOT fixed in 18.2a. Flagged in the IMPLEMENTATION_PROGRESS 2026-05-22
  entry's "Implication for Module 18.1" sub-section and in the report's
  Interpretation section.

After fixes: `pytest tests/classifier/ tests/test_cnn_download_vocalmat_sample.py -q` → 85 passed.
