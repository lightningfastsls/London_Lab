# Implementation Progress

Append-only session archive. Never modify existing entries — add new dated
entries below.

---

## 2026-04-17 — Phase 17.1 SIS Baselines on Existing Labels

**Status:** IMPLEMENTED (pending master-reviewer + test-hardener)
**ROADMAP reference:** `ROADMAP_SIS_BENCHMARK.md` module 17.1
**Review tier:** 2

**Files created:**
- `src/usv_spectrogram/classification/sis_baselines.py` — `SISResult` frozen
  dataclass + `compute_sis_depth_1(labels, name, sort_by_time=None)` function.
  Reuses `usv_language.analysis.sequence_analysis.mutual_information_at_lag`
  for numerical continuity with Phase A2's 0.093 bits result.
- `scripts/run_sis_baselines.py` — CLI driver; joins two CSVs on `det_index`,
  computes SIS for `syllable_type` (Scattoni-7), `label` (DeepSqueak-27), and
  `hdbscan_label` (HDBSCAN-3), writes `baselines.csv` + `baselines.png` with
  Hertz 2020 reference lines (0.10, 0.13, 0.22 bits).
- `docs/modules/sis-baselines.md` — module documentation.

**Files modified:**
- `src/usv_spectrogram/classification/__init__.py` — added `SISResult` and
  `compute_sis_depth_1` to the package exports.

**Test counts:**
- Pre-existing tests from `test-architect`: 17 (8 ROADMAP + 9 additional)
- Additional tests written in this implementation: 0
- Tests added by `test-hardener` post-review: 24
- **Final total: 41 (17 pre-existing + 0 impl + 24 hardener)** — all pass.
- Hardener-discovered bug: `compute_sis_depth_1` silently accepted
  `sort_by_time` arrays shorter than `labels` (numpy fancy-indexing truncation).
  Fixed by adding an explicit length-mismatch `ValueError` guard; the
  hardener's regression test (`test_sort_by_time_length_mismatch_raises`) now
  exercises the fix.

**Regression check:**
- Full classification test suite (`tests/test_classification/` +
  `tests/test_sis_baselines.py`): 161 passed, 1 skipped.
- Unrelated pre-existing failures in `tests/test_analyze_detection_confidence.py`
  (60) — script file missing at `scripts/analyze_detection_confidence.py`,
  not caused by this implementation.

**Exit criteria (ROADMAP):**
- [x] `SISResult` + `compute_sis_depth_1` implemented
- [x] All tests pass
- [x] py_compile passes on both new files
- [ ] `results/sis_baselines/baselines.csv` on real data — deferred (requires
  adding `syllable_type` column to real `classified_detections_full.csv`; data-
  prep concern, separate from module contract)

**Notes:**
- Script bootstrap needed both `SRC_ROOT` and `REPO_ROOT` on `sys.path` because
  `usv_language/` is a top-level package at repo root, not under `src/`. This
  is a bimodal-layout artifact — subprocess runs fail without REPO_ROOT.

---

## 2026-04-17 — Phase 17.2 Spectrogram Pre-Filtering Module

**Status:** IMPLEMENTED (pending master-reviewer + test-hardener)
**ROADMAP reference:** `ROADMAP_SIS_BENCHMARK.md` module 17.2
**Review tier:** 3 (critical DSP infrastructure — 3 downstream consumers)

**Files created:**
- `src/usv_spectrogram/features/__init__.py` — new subpackage; home for
  17.2–17.6 per ROADMAP plan.
- `src/usv_spectrogram/features/spectrogram_filter.py` — `FilterConfig`
  frozen dataclass + `prefilter_spectrogram(magnitude, freqs_hz, cfg) ->
  (cleaned, mask)`. Pipeline: 3×3 median filter → rolling-median per-column
  noise floor → amplitude threshold → 25–120 kHz band mask.
- `docs/modules/spectrogram-filter.md` — module documentation with
  algorithm, decisions, and integration points.
- `docs/reviews/spectrogram-filter-handoff.md` — handoff for reviewer.

**Files modified:**
- `tests/test_spectrogram_filter.py` — 4 of 16 tests rewritten to use
  Gaussian-profiled (leakage-realistic) ridges instead of 1-bin delta
  ridges, after STOP-and-discuss with user (approved 2026-04-17). Added
  `_add_leakage_ridge` helper and an updated docstring note explaining
  the revision.

**Test counts:**
- Pre-existing tests from `test-architect`: 16 (9 ROADMAP + 7 additional)
- Tests modified: 4 (documented in handoff; user-approved)
- Additional tests added in this implementation: 0
- Result: 18/18 passing in 0.09 s
- No regressions in adjacent modules (112 tests pass across
  `test_spectrogram_filter`, `test_sis_baselines`, `test_energy_detector`,
  `test_config`)

**Exit criteria (ROADMAP):**
- [x] Filter reduces broadband noise on synthetic noisy tone by >10 dB
  (`test_snr_improves_by_10db_on_noisy_tone`)
- [x] Out-of-band bins [<25 kHz, >120 kHz] are zero in cleaned output
- [x] All tests pass
- [x] py_compile passes

**Key decisions:**
- Kept the spec's 3×3 median filter. Real STFT ridges have 3–5 bin
  spectral leakage (Hann window, ADR-002), so the filter preserves real
  ridges while suppressing frequency- *and* time-isolated outliers.
- Output uses `filtered * mask`, not `magnitude * mask`. Resolves a
  latent spec ambiguity: the literal interpretation lets outliers
  propagate unchanged, defeating the median filter's purpose and failing
  ROADMAP test 2.
- `mode='reflect'` for both 2-D median filter and 1-D rolling median.
  Handles edges without introducing artefacts and gracefully covers
  `n_time_cols < noise_floor_window_cols`.

**Notes:**
- This is the first module in `src/usv_spectrogram/features/`. Future
  modules 17.3 (ridge tracker), 17.5 (Oren vectorization), and 17.6
  (AMVOC autoencoder) will live in this subpackage and consume
  `FilterConfig` + `prefilter_spectrogram`.
- Session crashed twice during implementation; work resumed from
  IMPLEMENTATION_PROGRESS.md + handoff doc context without issue.

---

## 2026-04-17 — Phase 17.2 Review Fixes Applied

**Status:** FIXES APPLIED (awaiting Tier 1 spot-check re-review per Tier 3 protocol)
**Original review:** `docs/reviews/spectrogram-filter-review.md` (verdict CHANGES NEEDED)

**Fixes:**
- **B1 (BLOCKER)** — `test_snr_improves_by_10db_on_noisy_tone` rewritten: Gaussian
  ridge via `_add_leakage_ridge`, ridge-aware SNR helper computes signal power
  over all ridge bins (not just center), and a `if signal_power == 0: return -inf`
  guard prevents the old trivial-pass failure mode. Paranoia check confirms a
  hypothetical broken filter that zeros everything now fails the test with
  improvement = -inf (previously it would have passed with +inf).
- **W1** — `_add_leakage_ridge` docstring corrected: σ=1.2 described as a
  conservative synthetic model wider than real Hann STFT leakage.
- **W2** — Outlier test threshold tightened: `< 100.0` → `< 2.0`.
- **W3** — Explicit `freqs_hz` shape validation added in `prefilter_spectrogram`
  with clear error message.
- **W4** — Two edge-case tests added: `test_single_freq_bin_no_crash`,
  `test_freqs_hz_shape_mismatch_raises`.
- **S1–S5** — Docstring/inline-comment polish. Dead `_make_pure_tone_magnitude`
  helper removed.

**Test counts:**
- Before fixes: 18 passing (1 trivially passing — B1)
- After fixes: 20 passing, 0.08 s wall-clock
- No regressions: 114/114 pass across the implemented-module scope
  (`test_spectrogram_filter + test_sis_baselines + test_energy_detector + test_config`)

**Files changed in fix pass:**
- `src/usv_spectrogram/features/spectrogram_filter.py` — shape validation,
  docstring additions (S1, S2, S4, W3).
- `tests/test_spectrogram_filter.py` — B1 rewrite, W1/W2/W4 fixes, S3/S5 polish.
- `docs/reviews/spectrogram-filter-review.md` — Fix Log updated,
  "Fixes Applied" section appended.

**Tier 1 re-review (2026-04-17):** APPROVED. No remaining trivial-pass path
for B1; all 9 other items confirmed in place. Cleared to proceed to test-
hardener.

**Test-hardener pass (2026-04-17):**
- Tests added: 15 (35 total — 20 after fixes + 15 adversarial).
- Bugs found: 0.
- New coverage: float64 dtype preservation, parameter boundary combinations
  (multiplier=1+ε, window=1, median_size=1, median_size=99), physical
  invariants (amplitude-scaling commutativity, cleaned ≤ input pixelwise,
  `filtered * mask` design regression guard), pathological inputs (NaN, inf,
  empty spectrogram, non-contiguous strided views), downstream-consumer
  invariants (mask.any() on all-out-of-band input, realistic (257, 200)
  STFT shape integration test).
- Residual concerns (non-blocking): negative-magnitude inputs silently
  accepted (docstring warns but no runtime check); integer-dtype input
  silently passes (not problematic for current consumers). Documented in
  the hardener report.

**Final test count: 35/35 passing, 0.19 s wall-clock. No regressions.**

---

## 2026-04-17 — Phase 17.3 DP-Based Ridge Tracker

**Status:** IMPLEMENTED (pending master-reviewer + test-hardener)
**ROADMAP reference:** `ROADMAP_SIS_BENCHMARK.md` module 17.3
**Review tier:** 3 (critical — upstream of 17.4 iMSA + 17.5 Oren vectorization)

**Files created:**
- `src/usv_spectrogram/features/ridge_tracker.py` — `RidgeConfig` frozen
  dataclass + `track_ridge(magnitude, freqs_hz, cfg) -> (fm_hz, am)` Viterbi
  DP ridge tracker. Windowed DP with transition-penalty `λ·|Δf|` and hard
  jump cap `max_jump_bins`; silent columns break the DP chain into runs
  solved independently; per-run back-trace from argmax of final cost vector.
- `docs/modules/ridge-tracker.md` — algorithm, decisions, integration points.

**Files modified:**
- `src/usv_spectrogram/features/__init__.py` — exports `RidgeConfig` and
  `track_ridge` alongside 17.2's `FilterConfig`, `prefilter_spectrogram`.

**Test counts:**
- Pre-existing tests from `test-architect`: 14 (10 ROADMAP + 4 additional)
- Tests modified during implementation: 0
- Tests added during implementation: 0
- **Result: 14/14 passing in 0.17 s** on the module suite.

**Regression check:**
- Full test suite: 1189 passed, 72 pre-existing failures (test_sis_benchmark
  = pre-impl for module 17.9 not yet built; test_analyze_detection_confidence
  = separate analysis-phase file with pre-existing missing-dep errors).
  Confirmed unchanged by stashing tracked changes and re-running — failures
  were present before 17.3 implementation.

**Exit criteria (ROADMAP):**
- [x] RMSE < 2 kHz on synthetic FM sweep
      (`test_regression_fm_rmse_within_2khz` — runs well below 2 kHz)
- [x] Harmonic-suppression test passes
      (`test_harmonic_suppression_stays_on_fundamental`)
- [x] All tests pass
- [x] `py_compile` passes

**Key decisions:**
- **Runs split at silent columns** — each non-silent `[start, end)` interval
  is solved by an independent DP, seeded from argmax at its first column.
  Rationale: silent columns carry no transition state; forcing DP across
  them would amount to imputing the ridge. Pre-existing test
  `test_silent_column_produces_nan_neighbors_intact` is satisfied by
  construction.
- **Windowed DP (O(F·W·T))** — mouse USVs have smooth pitch trajectories
  between jumps (<10-bin transitions per Oren 2024 at our bin width), so the
  windowed formulation is ~25× faster than full O(F²·T) pairwise DP at
  F=257, W=10, with identical output for smooth ridges. For `Δf > W`
  transitions, Viterbi cannot cross in one step — a feature, not a bug:
  iMSA labels those calls `Complex` downstream regardless.
- **Pure numpy, no scipy / torch** — Keeps the module dependency-light for
  the downstream iMSA + Oren pipelines; both can call it directly without
  dragging additional imports into hot paths.
- **Raw output, no smoothing / NaN interpolation** — 17.5 smooths AM with
  median filter and FM with mean filter per Oren spec; keeping the
  tracker's output raw avoids DSP opinions leaking between modules.

**Notes:**
- `__post_init__` validates `transition_penalty >= 0` and
  `max_jump_bins >= 1`, pinned by `test_ridgeconfig_rejects_*` tests.
- Vectorization inside the `for shift in range(-W, +W)` loop uses bounded
  slice views with in-place `where` updates — avoids allocating a full
  `(F, 2W+1)` tensor per column.

---

## 2026-04-17 — Phase 17.3 Review + Hardener Complete

**Status:** APPROVED (master-reviewer Tier 3) + HARDENED (test-hardener)
**Review file:** `docs/reviews/ridge-tracker-review.md`

**Review (Tier 3):** APPROVED on first pass. Findings: 2 WARNINGs + 2
SUGGESTIONs, all documentation-only or dead-code — no behavioral bugs. DP
forward pass, back-trace indexing, silent-run segmentation, and boundary
guards all verified by hand-trace + concrete 3-column MAP-reproducibility
check.

**Fixes applied:**
- **W1** — Corrected `RidgeConfig.transition_penalty` docstring in both
  `ridge_tracker.py` and `docs/modules/ridge-tracker.md`: `penalty=0`
  reduces to *windowed-argmax*, not true per-column argmax (the latter
  also requires `max_jump_bins >= n_bins`).
- **W2** — Test count corrected in `tests/test_ridge_tracker.py` docstring
  (line 36) and `docs/modules/ridge-tracker.md` exit criterion: 14 = 10
  ROADMAP + 4 additional (was 9+4 and 13+1 respectively).
- **S1** — Removed redundant `best[f_lo:f_hi] = slice_best` write in
  `_track_run` hot loop; added view-mutation comment.
- **S2** — Usage example in module doc explicitly passes `window="hann"`
  to `scipy.signal.stft` per ADR-002.

**Test-hardener pass:**
- Tests added: 18 (total 32 — 14 pre-existing + 18 adversarial).
- Bugs found: 0.
- New coverage: float32-input / float64-output dtype preservation;
  consecutive silent columns (2×, 3×); `max_jump_bins=1` tight constraint;
  `penalty=0, max_jump_bins>=n_bins` = true argmax regression anchor;
  non-monotonic / repeated-value `freqs_hz`; `run_len==2` forward+backtrace
  anchor; consumer invariant `am[t] == magnitude[ridge_idx[t], t]`;
  MAP-objective dominance over naive argmax; `silence_threshold=inf`
  full-silent semantics; integer-typed magnitude; `max_jump_bins=n_bins-1`;
  `max_jump_bins >> n_bins` (f_lo>=f_hi guard); strided non-contiguous
  view; NaN and negative values in magnitude ("no-crash" only).
- **Latent design surprise (not a bug):** `silence_threshold=0.0` uses
  strict `<`, so zero-magnitude columns are NOT silenced with `threshold=0`.
  Callers who expect `threshold=0` to act as a "no-signal" detector must
  use a tiny positive value (e.g. `1e-300`) instead. The hardener's
  `test_silence_threshold_zero_strict_semantics` documents this behavior
  as a regression anchor. Docstring already says "strictly below" —
  behavior matches spec; only callers' intuition is at risk.
- Residual non-blocking concern: NaN-in-input behavior is only asserted
  "no crash" — output values when NaN propagates through Viterbi scores
  are undefined. Acceptable in practice since upstream
  `prefilter_spectrogram` guarantees NaN-free output.

**Final test count: 32/32 passing, 0.91 s wall-clock. No regressions.**

---

## 2026-04-17 — Corpus Constants Unification + Empirical Data Registry

**Status:** IMPLEMENTED (master-reviewer approved; 1 SERIOUS + 4 NITs fixed in same session)
**Handoff:** `docs/handoffs/corpus-constants-unification-2026-04-17.md`
**Plan:** `~/.claude/plans/lucky-noodling-alpaca.md`
**Review:** `docs/reviews/corpus-constants-review.md`
**Review tier:** 2

**Motivation:** Four config modules (`SpectrogramConfig`, `DetectionConfig`,
`ExtractionConfig`, `AnalysisConfig`) declared the same physical constants
with *different* values — `SpectrogramConfig` said 250 kHz and 30–125 kHz,
contradicting CLAUDE.md ADR-001. The latent bug would surface the first
time any caller used `SpectrogramConfig()` defaults on a real 300 kHz WAV.
The production CNN is frozen on `ExtractionConfig`'s 20–120 kHz pixel
grid, so all other modules converge onto that band (never the reverse).

**Files created:**
- `src/usv_spectrogram/corpus.py` — new single-source-of-truth for
  `SAMPLE_RATE_HZ=300_000`, `USV_FREQ_MIN_HZ=20_000`, `USV_FREQ_MAX_HZ=120_000`,
  `STFT_N_FFT=512`, `STFT_HOP=128` + derived helper functions.
- `scripts/audit_corpus.py` — per-dataset empirical-data JSON generator;
  mirrors the `run_sis_baselines.py` parameters-sidecar pattern. CLI supports
  `--dataset {5970,3452,9252}` and `--all` (warns-and-skips missing inputs).
- `data/corpus_facts/5970.json` — committed artifact. All 11 sanity-check
  anchors reproduce exactly (n_calls_raw=7921, n_calls_after_dropna_file=7864,
  median_ici_gap_ms≈86.68, median_ioi_ms≈192.99, q25/q75 match, n_negative_gaps=10,
  n_cross_file_pairs_over_10s=829, n_bouts=1238, n_within_bout_pairs=6350,
  hdbscan={2:7598, 1:131, 0:98, -1:37}).
- `tests/test_corpus.py` — 12 tests: constant values, derived-helper returns,
  ExtractionConfig drift smoke test, downstream-config import smoke tests.
- `tests/test_audit_corpus.py` — 10 tests: subprocess runs real script against
  real 5970 artifacts, asserts every anchor, asserts parameters-block
  headings appear in stdout, asserts 3452 graceful-skip exit code 1.
- `docs/modules/corpus-constants.md` — module doc: three-layer architecture
  (physical facts / empirical data / analysis params), CNN-freeze rationale,
  usage examples, add-a-new-dataset flow.

**Files modified:**
- `src/usv_spectrogram/config.py` — `SpectrogramConfig` imports
  `SAMPLE_RATE_HZ`, `USV_FREQ_MIN_HZ`, `USV_FREQ_MAX_HZ` from corpus; defaults
  change 250k→300k sr, 30k→20k freq_min, 125k→120k freq_max.
- `src/usv_spectrogram/detection/config.py` — `DetectionConfig` imports
  `SAMPLE_RATE_HZ`, `STFT_N_FFT`, `STFT_HOP`, `USV_FREQ_MIN_HZ`, `USV_FREQ_MAX_HZ`
  from corpus; defaults widen 25–110 → 20–120 kHz freq band (no effect on
  production CNN; only `EnergyDetector` legacy path sees this).
- `src/usv_spectrogram/detection/extraction_config.py` — values **unchanged**
  (CNN pixel-grid frozen). Added file-level NOTE comment and module-level
  drift assertion that fires at import time if any of 5 dataclass field
  defaults diverge from corpus values.
- `usv_language/analysis/config.py` — `AnalysisConfig` imports
  `USV_FREQ_MIN_HZ`, `USV_FREQ_MAX_HZ` from corpus (no value change).
- `src/usv_spectrogram/app/core/audio_loader.py` — one-line comment on
  `SonicConfig` explaining its intentional 0–30 kHz band (NOT the corpus band).
- `tests/conftest.py` — synthetic WAV fixtures regenerate at 300 kHz
  (was 250 kHz; 4 locations: `sample_wav_path`, `sample_spectrogram`,
  `create_tone_wav`, `create_multi_tone_wav`).
- `tests/test_config.py` — updated 3 expected values to new canonicals.
- `tests/test_storage_zarr.py` — local `sample_rate_hz = 250_000` → `300_000`
  (8 locations).
- `tests/test_streaming_equivalence.py` — same, 1 location.

**Post-review fixes (same session):**
- `src/usv_spectrogram/config.py:42` — SERIOUS: corrected stale streaming-block
  comment ("1 second at 250 kHz" → "0.83 seconds at 300 kHz"). Value (250k) is
  a streaming IO granularity knob, unchanged.
- `docs/architecture/patterns.md` — Pattern 1 DetectionConfig example updated
  to show corpus imports; Pattern 3 fixture updated to 300 kHz.
- `CLAUDE.md:193-198` — §"Signal Processing Conventions" now references
  `corpus.py` as the code-level enforcement point and
  `docs/modules/corpus-constants.md` as the module doc.

**Test counts:**
- Baseline (pre-refactor): **1189 passed, 72 failed, 22 skipped** (72 failures
  are pre-existing Phase A3/A4 + Phase 17 in-flight, unrelated to configs).
- Post-refactor: **1211 passed, 72 failed, 22 skipped** (+22 new tests from
  corpus + audit_corpus; same failure list — `diff` shows zero new failures).
- CNN regression risk: **none.** Verified `run_batch_detection.py` and
  `sliding_inference.py` do NOT import `DetectionConfig` — production CNN
  inference uses `ExtractionConfig` only, whose values are unchanged.

**Key architectural decision:**
`ExtractionConfig` literals are INTENTIONALLY hardcoded (not imported from
corpus). The drift assertion at the bottom of `extraction_config.py` is the
safety net — if `corpus.py` is ever edited to new values, the assertion
fires at import time with a clear "retrain the CNN before updating this
literal" message. Values only diverge by deliberate, ordered action: retrain
CNN → update ExtractionConfig → update corpus.

**Out of scope (deferred):**
- `DetectionConfig` band widening (25–110 → 20–120 kHz) may shift candidate
  count for `EnergyDetector` callers by up to ~5 %. Verify on a known-good
  WAV before relying on tuned thresholds. The production CNN path is
  unaffected (verified above).
- Phase 17 (`sis_baselines.py`, `run_sis_baselines.py`, `test_sis_*.py`)
  not touched per handoff "keep surface area small." Follow-up handoff
  can migrate those after 17.9 ships.

---

## 2026-05-21 — Module 18.1 CNN Cleaning Validation Gate

**Status:** IMPLEMENTED + REVIEWED + FIXES APPLIED (10-item batch)
**ROADMAP reference:** `ROADMAP_lab_cnn_classifier.md` §18.1
**Review tier:** 3 (DSP + statistical methodology)
**Worktree:** `.claude/worktrees/lab-cnn-classifier-plan/`

**Files created:**
- `src/usv_spectrogram/classifier/__init__.py` — package init; exports
  `CleaningConfig`, `clean_spectrogram`, `DiagnosticResult`, and the
  four diagnostic functions for downstream Module 18.2 use; holds
  `TARGET_SAMPLE_RATE_HZ=250_000` (VocalMat-aligned, NOT corpus default).
- `src/usv_spectrogram/classifier/cleaning_pipeline.py` (387 LOC) —
  `CleaningConfig` (namedtuple subclass for adversarial-immutability
  test contract) + `clean_spectrogram` 4-layer stack (soft-notch →
  baseline subtraction → global MAD → per-recording Z-score). Wraps
  existing implementations where possible; reproduces global MAD
  byte-for-byte from `app/core/sliding_inference.py`.
- `src/usv_spectrogram/classifier/diagnostics.py` (773 LOC) — four
  falsifiable diagnostics (`notch_injection_test`, `per_band_cohens_d`,
  `knn_same_cohort_rate`, `raw_pixel_pca_d`) with hardcoded pass
  thresholds (0.30, 0.30, 0.85, 1.50) + small CPU-runnable diagnostic
  VAE (`train_diagnostic_vae`).
- `scripts/cnn_cleaning_validation.py` (465 LOC) — Pattern 4 CLI;
  6-ablation matrix (raw, baseline_only, mad_only, zscore_only,
  soft_notch_only, all_layers); Markdown report renderer with go/no-go
  decision footer.
- `docs/modules/cnn-cleaning-validation.md` (213 LOC) — module doc with
  methodology lock (2026-05-21), amendment record, cross-phase
  constraints C1–C6.

**Test counts:**
- Pre-existing tests from `test-architect`: 31 (14 cleaning_pipeline +
  17 diagnostics).
- Tests modified during implementation: 0 (3 amendments to expected
  thresholds / fixture band alignment, all user-approved and recorded
  in module doc under "Test-spec amendments (2026-05-21)").
- Additional tests added: 0.
- **Result: 31/31 passing in 5.93 s** after the 10-item fix batch.

**Exit criteria status (ROADMAP §18.1):**
- [x] Four diagnostics implemented with the spec thresholds
- [x] Ablation matrix runs the 6 documented configurations
- [x] CLI follows Pattern 4 (separate `parse_args`, exit codes 0/1/2,
  epilog usage examples)
- [x] All 31 tests pass
- [x] py_compile passes on all 4 new modules
- [ ] **Deferred to Module 18.2:** real-data run via
  `python scripts/cnn_cleaning_validation.py --vocalmat-sample <path>`
  (CLI currently always falls back to synthetic data — requires
  VocalMat dataset download in 18.2).
- [ ] **Deferred to Module 18.2:** `docs/handoffs/cleaning-validation-report.md`
  (the real-data go/no-go report; cannot exist until exit criterion
  above is met).

**Reviews completed:**
- dsp-reviewer: SHIP with 1 MEDIUM (cage-tone injection scaling) + 2
  LOW (per-recording Z-score docstring caveat, fallback baseline
  kernel + epsilon alignment).
- master-reviewer (Tier 3): CHANGES NEEDED — 2 blockers + 5 warnings
  + 3 suggestions. Full review at
  `docs/reviews/cnn-cleaning-validation-review.md`.

**Fixes applied (10-item batch, 2026-05-21):**
1. `diagnostics.py` module + function docstrings: "cohort A" → "combined
   (A + B)" to match code at line 436. (BLOCKER 1)
2. This IMPLEMENTATION_PROGRESS.md entry. (BLOCKER 2)
3. `_inject_cage_tone` scaling: fixed +20 dB → `INJECTION_SIGMA * local_std`
   (σ=2.0) with `_INJECTION_FALLBACK=0.1` for constant-input bands.
   Preserves migration semantics on all 6 ablations including
   normalised inputs (mad_only, zscore_only, all_layers). (MEDIUM)
4. `test_cleaning_pipeline.py:51` `parents[3]` → `parents[2]` to match
   the corrected `test_diagnostics.py:58`. (WARNING 1)
5. `per_band_cohens_d` docstring corrected to describe per-pixel
   pooling, not per-sample mean. (WARNING 2)
6. `classifier/__init__.py` exports `CleaningConfig`,
   `clean_spectrogram`, `DiagnosticResult` and the four diagnostic
   functions. (WARNING 4)
7. `docs/architecture/patterns.md` Pattern 1: added "Variant: namedtuple
   subclasses when immutability must withstand `object.__setattr__`"
   sub-section. (WARNING 3)
8. `docs/reviews/cnn-cleaning-validation-handoff.md` created.
   (WARNING 5)
9. `_apply_per_recording_zscore` docstring: dense-USV-regime caveat
   noting divergence from upstream 1D `normalize_scores_per_recording`.
   (LOW)
10. `_local_baseline_subtract` fallback: kernel rule aligned to
    upstream's 0.5 s rule (`int(0.5 * sample_rate_hz / STFT_HOP) | 1`),
    epsilon aligned 1e-30 → 1e-10. (LOW)

**Notes:**
- Real-data path and cleaning-validation-report.md formally deferred to
  Module 18.2 — both require the VocalMat dataset which 18.2 owns.
- Re-review required per the review file's "Re-review rule": BLOCKER 1
  (docstring corrections), BLOCKER 2 (this entry), and the MEDIUM
  (cage-tone scaling) must be independently verified. The
  cage-tone-scaling fix did NOT break the existing 31 tests; the
  injected-tone-raises-migration test continues to pass.

---

## 2026-05-22 — Module 18.2a VocalMat Sample + Real-Data Gate (GO verdict)

**Status:** IMPLEMENTED + REAL-DATA RUN COMPLETE (pending master-reviewer)
**ROADMAP reference:** `ROADMAP_lab_cnn_classifier.md` §18.2a
**Review tier:** 2 (download + loader plumbing; no novel statistical methodology)
**Worktree:** `.claude/worktrees/lab-cnn-classifier-plan/`

**Outcome.** Module 18.1's deferred real-data exit criteria are met. The
cleaning-validation gate's binding GO/NO-GO verdict is **GO** — all 4
diagnostics pass under the full cleaning stack on real VocalMat (227×227
PNG luminance) + lab 131204 (250 kHz-resampled STFT) + wild 5970
(250 kHz-resampled STFT). Module 18.2b (full data preparation) is
unlocked.

**Files created:**
- `scripts/cnn_download_vocalmat_sample.py` (~520 LOC) — OSF v2 REST
  downloader (stdlib urllib, no new dependencies). Stable seed=1729,
  `page_size=100`, exponential backoff on 429, 4-worker parallel
  download. Outputs `data/vocalmat_sample/<class>/*.png` + manifest CSV.
- `data/vocalmat_sample/.gitignore` (2 lines) — `*\n!.gitignore` so
  ~113 MB of PNGs stays out of git.
- `tests/test_cnn_download_vocalmat_sample.py` (~220 LOC, 11 tests) —
  dependency-injected `FakeVocalMatSource` so tests never touch OSF.
  Covers the 3 ROADMAP §18.2a items (dry-run, manifest balance,
  idempotency) plus filename-parsing edge cases.
- `tests/classifier/test_cleaning_real_data_loader.py` (~230 LOC, 12
  tests) — sibling file to the spec tests (existing `tests/classifier/
  test_*.py` from 18.1 are NOT modified). Covers `_resize_2d`,
  `_png_to_luminance`, `_wav_to_spectrograms`, full
  `_load_real_cohorts` integration, missing-input error paths,
  determinism under seed.
- `docs/modules/cnn-download-vocalmat-sample.md` — module doc.
- `docs/handoffs/cleaning-validation-report.md` — the GO real-data
  verdict (n_epochs=32 re-run).
- `docs/handoffs/cleaning-validation-report.n4-NOGO.md` — audit trail
  for the first NO-GO run; preserved to document the under-training
  finding.

**Files modified:**
- `scripts/cnn_cleaning_validation.py` — added `_load_real_cohorts()`,
  `_png_to_luminance()`, `_wav_to_spectrograms()`, `_resize_2d()` (~220
  new LOC). Replaced the synthetic-fallback branch in `main()` with a
  real call to the loader when all 3 `--*-sample` args are supplied.
  Imports `corpus.STFT_N_FFT`, `corpus.STFT_HOP`,
  `classifier.TARGET_SAMPLE_RATE_HZ` — no constants redeclared per
  CLAUDE.md corpus protocol.
- `requirements.txt` — unchanged at conclusion. Briefly added
  `osfclient` and reverted after pivoting to stdlib urllib.

**Test counts:**
- New tests: 11 (download script) + 12 (loader) = 23.
- Existing 18.1 tests: 62, all still passing (no spec-test
  modification).
- **Total: 85 passing** in 20.4s.

**Sample download:**
- 2,196 / 2,210 PNGs (99.4%), 113 MB. The 14 failures (4 in `complex`,
  8 in `rev_chevron`, 2 in `noise`) are transient OSF 30-s timeouts /
  one 403. The loader samples by filesystem glob, so missing files
  don't cascade; we have an over-sufficient pool for the 200-per-cohort
  gate run.

**Real-data gate run (GO):**

| Ablation | notch_injection | per_band_d | knn_same_cohort | pca_d |
|---|---|---|---|---|
| raw | 1.0000 FAIL | 26.02 FAIL | 0.3333 PASS | 52.16 FAIL |
| soft_notch_only | 1.0000 FAIL | 26.02 FAIL | 0.3333 PASS | 52.16 FAIL |
| baseline_only | 0.0250 PASS | 0.319 FAIL | 0.3333 PASS | 15.87 FAIL |
| mad_only | 0.0000 PASS | 0.536 FAIL | 0.9473 FAIL | -10.22 FAIL |
| zscore_only | 0.0050 PASS | 0.503 FAIL | 0.9310 FAIL | -9.15 FAIL |
| **all_layers** | **0.0000 PASS** | **0.070 PASS** | **0.3333 PASS** | **0.0000 PASS** |

**Diagnostic-VAE under-training discovered (and resolved within run):**

The first real-data run used the script's default `--n-epochs 4` and
emitted NO-GO with `notch_injection_migration = 1.0` on `all_layers`.
The smoke-test default was calibrated on synthetic 32×32 data; real
data is 227×227 = 51,529 features, ~50× the smoke regime, and 4 epochs
is insufficient for VAE convergence. The re-run with `--n-epochs 32`
dropped `all_layers` migration from 1.0 → 0.0 with no other changes.

Implication for Module 18.1: the default `--n-epochs` should either
auto-scale with input feature count or default to ~32. This is **not**
in scope for 18.2a (would touch the frozen 18.1 cleaning module); it
is flagged in the cleaning-validation report's Interpretation section
for a successor 18.1.x patch.

**Exit criteria status (ROADMAP §18.2a):**
- [x] Download script downloads ≥200 × 10 + minority-class totals into
  `data/vocalmat_sample/` (2,196/2,210, 99.4% — sufficient).
- [x] Module 18.1 CLI accepts `--vocalmat-sample data/vocalmat_sample/`
  and produces a non-synthetic report.
- [x] Module 18.1 GO/NO-GO verdict captured in
  `docs/handoffs/cleaning-validation-report.md`. Verdict: **GO**.
- [x] Gate passed → 18.2b unlocks.

**Notes:**
- No files in the do-not-touch list were modified: `corpus.py`,
  `sliding_inference.py`, `notch.py`, `denoise.py`,
  `normalization.py`, `run_batch_detection.py`,
  `classifier/cleaning_pipeline.py`, `classifier/diagnostics.py`,
  `classifier/__init__.py`, and the existing
  `tests/classifier/test_*.py` spec files are all unchanged.
- `git status --short -- src/ tests/classifier/test_cleaning_pipeline.py
  tests/classifier/test_diagnostics.py
  tests/classifier/test_cleaning_pipeline_adversarial.py
  tests/classifier/test_diagnostics_adversarial.py` shows nothing —
  the spec contract is preserved.

---

## 2026-05-22 (follow-up) — Module 18.1.x carve-out patch (docstring + DeprecationWarning)

**Status:** IMPLEMENTED (deferred reviewer findings from 18.2a applied)
**ROADMAP reference:** Module 18.1.x patch — no new ROADMAP entry; addresses
the WARNING 2 + NIT 1 findings from
`docs/reviews/cnn-download-vocalmat-sample-review.md` that were
out-of-scope for 18.2a (the 18.1 do-not-touch list excluded
modifications to `classifier/diagnostics.py`).
**Review tier:** 1 (docstring + DeprecationWarning emission only — no
behavior change for any caller passing the legacy default).
**Worktree:** `.claude/worktrees/lab-cnn-classifier-plan/`.

**Files modified:**
- `src/usv_spectrogram/classifier/diagnostics.py` — module docstring
  (lines 23-32) rewritten to lead with the input-feature-count epoch
  scaling rule; `train_diagnostic_vae` docstring (lines 348-368)
  rewritten to spell out the smoke-vs-real distinction and warn
  callers using real data must override `n_epochs`. Added
  `_NOTCH_DEPTH_DB_LEGACY_DEFAULT = 20.0` sentinel constant + emit
  `DeprecationWarning` from `_inject_cage_tone` when a caller passes a
  non-default `notch_depth_db` value. Removed the `del notch_depth_db`
  workaround.
- `scripts/cnn_cleaning_validation.py` — `--n-epochs` CLI help text
  expanded to state explicitly that the default 4 is smoke-test-only
  and real data requires ≥32.

**Behavior changes:**
- For existing callers passing `notch_depth_db=20.0` (the only callers
  in-tree): no behavior change. Test suite continues at 85/85.
- For future callers passing a non-default `notch_depth_db`: emits
  `DeprecationWarning` at stacklevel=2 (visible at caller's frame).
- For users running `cnn_cleaning_validation.py` against real data
  with the default `--n-epochs 4`: behavior is unchanged but the CLI
  help now warns them explicitly. The Module 18.2a Interpretation
  section already documents why this matters.

**Test counts:**
- 85/85 still passing in 20.97s. No spec test expectations modified.
- DeprecationWarning path is not yet exercised by tests; existing
  callers all use the legacy default, so this is a forward-only
  safety net.

**Notes:**
- The patch is intentionally narrow: docstrings + one `warnings.warn`
  call. It does NOT change any computational behavior, threshold, or
  diagnostic semantics. The cleaning gate's GO verdict from 18.2a
  remains valid.
- The DeprecationWarning was chosen over removing the parameter
  because the 18.1 handoff explicitly retained `notch_depth_db` "for
  backward compatibility with callers from the locked methodology
  (2026-05-21)". Removing the parameter would break that promise; the
  warning preserves the call signature while signalling the dead
  semantics.
- These changes touch the previously-frozen `classifier/diagnostics.py`.
  That module is no longer fully frozen — future 18.1.x patches with
  similarly narrow scope are acceptable, but any change to
  computational behavior (thresholds, diagnostic algorithms, layer
  order, etc.) should go through master-reviewer Tier 3 (DSP +
  statistical methodology) per the original 18.1 review.

---

## 2026-05-22 — Module 18.2b Full Data Preparation (Stream X)

**Status:** IMPLEMENTED (code complete; full 12 GB VocalMat download
in progress in background)
**ROADMAP reference:** §18.2b (Full Data Preparation), unblocked by 18.2a
GO verdict and the 18.1.x carve-out patch.
**Review tier:** 2 (pipeline plumbing; class-balance allocator is the
only complex piece).
**Worktree:** `.claude/worktrees/lab-cnn-classifier-plan/`.
**Predecessor handoff:** `docs/handoffs/2026-05-22_stream-x-module-18.2b-resume.md`
+ canonical spec at `docs/handoffs/2026-05-22_post-18.2a-orchestrator.md`.

**Files created:**

- `src/usv_spectrogram/classifier/resample.py` — 300 → 250 kHz polyphase
  with a custom 481-tap Kaiser β=14 FIR (cutoff 120 kHz at the 1.5 MHz
  intermediate rate). The custom FIR is necessary because
  `scipy.signal.resample_poly`'s default 121-tap Kaiser β=5 only
  delivers ~32 dB rejection 15 kHz above cutoff — insufficient for
  ROADMAP test 5 (≥ 40 dB at 110 kHz for a 140 kHz aggressor).
- `src/usv_spectrogram/classifier/dataset.py` — `GRIMSLEY_12_CLASSES`,
  `DatasetSplit` frozen dataclass, and `build_stratified_split` using
  an LPT (Longest-Processing-Time first) greedy-by-deficit allocator
  with recording-level grouping. Class weights are inverse-frequency
  normalized to mean 1.0; oversample targets bring minorities up to
  the training-set median.
- `scripts/cnn_prepare_training_data.py` — end-to-end CLI: VocalMat
  walk → lab/wild WAV resample + clean + STFT + 0.22 s patches →
  50 sanity patches per cohort → stratified split. Exports `main(argv)`
  for in-process invocation by the smoke test.
- `tests/classifier/test_resample.py` (9 tests), `test_dataset.py`
  (11 tests), `test_cnn_prepare_training_data.py` (6 tests) — written
  pre-implementation by `test-architect` (TDD red phase), then driven
  to green.
- `docs/modules/cnn-data-preparation.md` — module documentation
  cataloguing the VocalMat-vs-corpus STFT distinction and the binding
  cross-phase constraints (C1–C6).
- `data/vocalmat_full/.gitignore` — `*` + `!.gitignore` to keep the
  12 GB pull out of version control.

**Test counts:** 100/100 classifier-package tests green
(`pytest tests/classifier/ -q`, 55.68 s). Includes the 85
pre-existing 18.1 + 18.2a spec tests — no regressions. The 26 new
tests broke down as 5 + 4 + 1 ROADMAP-traced + 16 robustness
additions (boundary cases, missing columns, reproducibility checks).

**Algorithm note — LPT greedy by deficit:**

Recording-level grouping forces whole-recording assignment. A naive
greedy ("fill train, then val, then test") overshoots when a late
large recording lands in test — Noise class hit 17 % in test (target
10 %, ±5 % tolerance from the ROADMAP). Switching to
*largest-first traversal* + *place-in-most-underfilled-split* (the LPT
heuristic from makespan scheduling) keeps every class within ±5 % on
the synthetic imbalanced fixture. Determinism flows through a
deterministic shuffle as the tie-break inside the stable sort, which
also makes different seeds produce different splits when tied counts
exist (required by `test_different_seeds_produce_different_splits`).

**Anti-alias FIR — DSP note:**

`resample_poly`'s defaults are tuned for an "average" use case; the
ROADMAP test asserts a stricter spec because the lab CNN must not
swallow content from 125–150 kHz back into the 0–125 kHz passband
where it could confuse a syllable classifier. The custom FIR
(481 taps, β=14, cutoff 120 kHz) delivers ≥ 50 dB rejection at the
worst-case alias point, comfortably meeting the test. Runtime cost is
~4× the convolution work — invisible against the per-WAV STFT +
cleaning cost in offline data prep.

**Background download state at handoff time:**

The full pull (`scripts/cnn_download_vocalmat_sample.py --full
--output-dir data/vocalmat_full/ --workers 6`) was started in
parallel with the implementation. First attempt died on an OSF
HTTP 502 during the enumerate phase (this is the documented Stream R
issue — `_retry_on_429` only catches 429s, not 502s). Retried; second
attempt is currently progressing through the `noise` class (~423/1352
PNGs at the time of writing, 23 MB on disk). The 14/2210 transient
failure rate observed in 18.2a is expected to scale roughly linearly
to ~70/12221 missing PNGs at full scale; the script's existence-check
makes resume cheap.

**Exit criteria status (ROADMAP §18.2b):**

- [x] All tests pass (100/100 in 55.68 s)
- [x] `py_compile` passes for all three new modules
- [x] Script's in-process `main()` smoke runs in ≤30 s on synthetic
  12 × 5 PNGs + 2 WAVs (actual: well under budget across 6 test cases)
- [ ] Script runs on real VocalMat download + at least one lab + one
  wild WAV — **deferred to a post-download follow-up**
- [ ] `data/lab_cnn_training/sanity_patches/` populated with 50
  patches per cohort — **deferred (depends on the real run)**
- [x] Manifest CSVs exist in train/val/test with correct columns
  (synthetic-data smoke)
- [x] No production-detection files modified

**Files in the do-not-touch list:** unchanged. `corpus.py`,
`sliding_inference.py`, `notch.py`, `denoise.py`, `normalization.py`,
`run_batch_detection.py`, `classifier/cleaning_pipeline.py`,
`classifier/diagnostics.py`, and the existing
`tests/classifier/test_*.py` spec files are byte-identical to
pre-18.2b state.

---

## 2026-05-22 (follow-up) — Module 18.2b master-reviewer fixes

**Status:** APPLIED (5 of 6 review findings closed; NIT 1 deferred)
**Review file:** `docs/reviews/cnn-data-preparation-review.md`
**Tests:** 100/100 still pass in 63.27 s. No regressions.

**Fixes applied in the same session:**

- **WARNING 2 (real architectural issue):** Lab/wild WAV patches no
  longer carry a `"Noise"` placeholder label and are no longer mixed
  into the supervised train/val/test manifests. They flow to a
  separate `output_dir/domain_unlabeled.csv` (columns: path, cohort,
  source_recording, duration_ms) for Module 18.4's DANN
  cage-invariance training. The supervised manifest is now
  VocalMat-only, preventing real-USV calls from leaking into 18.3's
  supervised signal as Noise.

- **WARNING 3:** `src/usv_spectrogram/classifier/__init__.py` now
  re-exports `resample_to_vocalmat`, `SOURCE_SAMPLE_RATE_HZ`,
  `GRIMSLEY_12_CLASSES`, `DatasetSplit`, `build_stratified_split`.
  Downstream `from usv_spectrogram.classifier import ...` works
  without knowing the submodule layout.

- **WARNING 4:** Copied
  `docs/handoffs/2026-05-22_stream-x-module-18.2b-resume.md` into the
  worktree from the parent repo so `git diff main..HEAD` is
  self-contained.

- **WARNING 1 + WARNING 5:** `tests/classifier/test_dataset.py`
  module-level docstring updated: count corrected (10 → 11), the
  missing `test_different_seeds_produce_different_splits` entry
  added, and the ROADMAP ±2% vs actual ±5% tolerance gap documented
  with reasoning.

- **NIT 1 (deferred):** ROADMAP wording for the `data/vocalmat_full/`
  directory name will be cleaned up in a future ROADMAP refresh; the
  implementation follows the orchestrator handoff which takes
  precedence.

**Bonus fixes uncovered while applying WARNING 2:**

The Option A architecture (VocalMat-only supervised manifest) exposed
two latent issues in the LPT allocator:

1. With every class reduced to 5 single-call recordings, the existing
   tuple-iteration tie-break (`max(("train", "val", "test"), ...)`)
   deterministically routed every leftover to "val", leaving
   `test/manifest.csv` empty. Replaced with a per-class hash-based
   flip (`secondary_order = ("test", "val") if flip_test_first else
   ("val", "test")`) that balances across many classes.

2. The flip exposed a floating-point bug:
   `1.0 - 0.8 - 0.1 = 0.09999999999999998`, so
   `targets["test"] = 0.4999999999999998` instead of `0.5`. The
   exact-equality comparison `deficits["test"] == best_deficit` silently
   failed and routed every leftover to val. Replaced with a
   tolerance-aware `_close(a, b) = abs(a - b) < 1e-9` helper.

Both fixes are confined to `_allocate_class` in `dataset.py` and are
purely seed-independent (no rng consumption changed).

**Background download status at the close of this session:**

3,592 / ~12,221 PNGs on disk (~29%), 185 MB. Still progressing; the
`_retry_on_429` decorator is correctly handling the OSF rate-limiting
429s observed at higher download volume. The previous 502 enumerate
failure (Stream R issue from the orchestrator handoff) did not recur.
Final reconciliation against the manifest + the real-data run remain
the deferred exit-criteria items.

**Verdict (post-fix):** SHIP. All four exit criteria that depend on
code are checked; the two that depend on the live download are
documented and tracked.

---

## 2026-05-24 — Module 18.2b real-data closure

**Status:** CLOSED. All 6 handoff exit criteria PASS. Module 18.3 UNLOCKED.
**Handoff:** `docs/handoffs/2026-05-22_post-18.2b-download-followup.md`
**Successor handoff:** `docs/handoffs/2026-05-24_module-18.3-resnet-supervised-baseline.md`

**Final dataset state (data/lab_cnn_training/):**

| Artifact | Count / size |
|---|---|
| VocalMat supervised manifest (`manifest_all.csv`) | 12,178 rows across 12 classes |
| Stratified split (`train/val/test/manifest.csv`) | 9,741 / 1,220 / 1,217 (80/10/10) |
| Lab patches (`patches/lab/*.png`) | 227,021 across 83 recordings |
| Wild patches (`patches/wild/*.png`) | 8,572 across 851 recordings |
| Domain-unlabeled manifest (`domain_unlabeled.csv`) | 235,726 rows for Module 18.4 DANN |
| Sanity patches (`sanity_patches/{cohort}_NN_*.png`) | 150 (50 per cohort) |

Total disk: ~18 GB (excluded from git via `.gitignore`).

**Handoff exit criteria — final verdict:**

| Check | Status | Detail |
|---|---|---|
| All 12 classes present in train/val/test | PASS | Every class represented in every split |
| Total VocalMat rows >= 12,000 | PASS | 12,178 |
| `domain_unlabeled.csv` non-empty | PASS | 235,726 rows |
| Lab + wild patches written | PASS | lab=227,021, wild=8,572 |
| Wild cohort coverage | PASS | 851 distinct recordings (12 from 5970 + 839 from 3452 subdirs) |
| Sanity-patch human review | PASS | User confirmed variety across cleaning layers, no systematic horizontal stripe, no all-black collapse |

**Two latent bugs discovered (deferred to Tier-2 tickets):**

1. **`cleaning_pipeline.py` produces all-zero output on long lab WAVs when
   `baseline_mode='median_envelope'`.** Root cause: per-bin temporal median
   subtraction floors >50% of cells to `_DB_TO_LINEAR_EPS` (≈ −200 dB), then
   `_apply_global_mad`'s `vmax == vmin` degenerate branch fires and returns
   all zeros, which `_apply_per_recording_zscore` propagates. Existing unit
   tests use small synthetic spectrograms that never hit the floor, so the
   failure mode escapes coverage. Witnessed on first prep run (2,734 all-black
   patches on `131204_1400_m1fm1`). Comparison HTML saved at
   `$CLAUDE_JOB_DIR/baseline_compare/index.html` (worktree-only).

2. **`scripts/cnn_prepare_training_data.py:_collect_wav_rows` uses
   `root.glob("*.wav")` (non-recursive).** Silently skipped 853 WAVs nested
   in `USV_3452_sample_reviewed/{USV_1..4,uncertain_usv}/`. Caught by
   reconciliation: only 12 distinct wild recordings present instead of the
   expected 850+. Fixed in this session via `scripts/cnn_wild_topup.py`
   (one-shot, idempotent). Root fix (`rglob` or explicit subdir handling)
   deferred to Tier-2.

**Workaround applied (this session):**

- `scripts/cnn_prepare_training_data.py:376`: changed
  `cfg = CleaningConfig()` to
  `cfg = CleaningConfig(baseline_mode="percentile")` with a 9-line
  comment explaining why. `percentile` mode (single per-bin 10th-percentile)
  is ~280× faster on CPU than `median_envelope` (sliding 977-frame median)
  AND avoids the all-zero degenerate path because it subtracts much less
  aggressively. Trade-off: assumes stationary noise floor over the
  recording, which is reasonable for stable acoustic chambers.

**Process surprises:**

- **Cleaning cost is dominated by `_apply_baseline_subtraction` at 600 s lab
  WAVs.** Profile (per-WAV, 600 s @ 250 kHz):
  - STFT: ~20 s
  - `_apply_baseline_subtraction` (percentile): ~18 s (vs ~92 min in
    `median_envelope` mode)
  - `_apply_global_mad`: ~30 s
  - `_apply_per_recording_zscore`: ~7 s
  - PIL PNG save × 2,727 patches: ~40 s
  - Total: ~6 min steady-state per lab WAV under percentile mode.
- **Total wall-clock**: ~13 hours including two system-sleep cycles during
  the run. Real CPU time was ~9 hours (single-threaded). The `--workers 4`
  flag is informational only; the WAV loop is sequential.
- **VocalMat OSF download had 196 missing files at the start of this
  session.** Resumed via `cnn_download_vocalmat_sample.py --full`
  (idempotent — re-fetches only missing files). Two prior 502 cycles in the
  predecessor session; the resume took 88 s. Final reconciled count: 12,178
  unique paths in manifest, 12,178 on disk.

**Files created (this session, beyond the prior 18.2b shipment):**

- `scripts/profile_prep_phases.py` — diagnostic; imports prep internals,
  times each cleaning layer on a wild WAV and truncated lab WAV.
- `scripts/benchmark_baseline_modes.py` — diagnostic; A/B compares
  `percentile` vs `median_envelope` on the same input.
- `scripts/compare_baseline_modes_visual.py` — diagnostic; renders side-by-
  side spectrograms + patches for both modes.
- `scripts/cnn_wild_topup.py` — one-shot fix; imports prep's
  `_wav_to_patches` and processes the 853 missing wild subdirs explicitly.
  Appends to `domain_unlabeled.csv`. Idempotent (skips by source-recording
  stem).
- `scripts/post_prep_reconcile.py` — completion-watcher reconciliation; runs
  the handoff's decision matrix and emits an HTML status report.
- `scripts/regen_sanity_patches.py` — one-shot; re-samples 50 patches per
  cohort uniformly across the full post-top-up pool (the original sanity
  patches were biased toward the pre-top-up 12 wild recordings).

**Files modified (this session):**

- `scripts/cnn_prepare_training_data.py` — one-line fix at line 376 plus
  9-line comment (the `baseline_mode="percentile"` workaround). No other
  changes; the rest of the 18.2b shipment is byte-identical to the
  pre-session state.
- `.gitignore` — added `data/lab_cnn_training/` and `data/vocalmat_full/`
  (regeneratable outputs, too large for git).

**Files NOT touched** (matches the do-not-touch list from the predecessor
handoff): `src/usv_spectrogram/corpus.py`, `app/core/sliding_inference.py`,
`app/core/notch.py`, `app/core/denoise.py`, `postprocessing/normalization.py`,
`scripts/run_batch_detection.py`, `src/usv_spectrogram/classifier/cleaning_pipeline.py`,
`src/usv_spectrogram/classifier/diagnostics.py`, and the pre-existing
`tests/classifier/test_*.py` spec files.

**Test counts:**

- 100/100 classifier tests pass post-fix (no regressions from the one-line
  prep change). Re-verified twice during this session.

**Verdict:** CLOSE. Module 18.2b real-data exit criteria satisfied.
Module 18.3 (ResNet-18 supervised baseline) UNLOCKED — see successor
handoff.

---

## 2026-05-24 — Module 18.3 code shipment (Streams T + R; Streams P + V deferred)

**Handoff executed:** `docs/handoffs/2026-05-24_module-18.3-resnet-supervised-baseline.md`

**Goal:** Ship the 12-class supervised baseline classifier (model factory,
augmentation, focal loss, training loop, CLI, module doc + tests) without
running the 50-epoch real-data training (no GPU on this WSL host).

**What landed:**

| File | Status | LOC |
|---|---|---|
| `src/usv_spectrogram/classifier/model.py` | NEW | 47 |
| `src/usv_spectrogram/classifier/augmentation.py` | NEW | 182 |
| `src/usv_spectrogram/classifier/losses.py` | NEW | 75 |
| `src/usv_spectrogram/classifier/training.py` | NEW | 234 |
| `scripts/train_lab_classifier.py` | NEW | 195 |
| `docs/modules/lab-classifier-v1.md` | NEW | 230 |
| `tests/classifier/test_model.py` | NEW (test-architect) | 8 tests |
| `tests/classifier/test_augmentation.py` | NEW (test-architect) | 13 tests |
| `tests/classifier/test_losses.py` | NEW (test-architect) | 7 tests |
| `tests/classifier/test_training.py` | NEW (test-architect) | 10 tests |
| `tests/classifier/test_train_lab_classifier.py` | NEW (test-architect) | 5 tests |
| `tests/classifier/test_train_perch2_probe.py` | NEW (test-architect) | 4 tests (skipped) |
| `src/usv_spectrogram/classifier/__init__.py` | MODIFIED (re-export 18.3 symbols) | +9 |
| `IMPLEMENTATION_PROGRESS.md` | this entry | -- |

**Test counts (post-shipment):**

- 43 new tests pass (8 + 13 + 7 + 10 + 5)
- 4 new tests skip (Perch 2.0 probe — deferred)
- 100 baseline 18.2a/b tests still pass — zero regressions
- **143 passed, 4 skipped, 0 failed** in 62.5s (full classifier suite)
- CPU smoke test (test #9, 2-epoch ResNet-18 on 120 synthetic samples)
  completed in ~24 s, well under the 90 s budget.

**Two intentional, documented modifications during this session:**

1. **`tests/classifier/test_losses.py::test_focal_loss_gamma0_equals_weighted_ce`**
   — reference reduction changed from `F.cross_entropy(weight=w)` (PyTorch
   default weighted-mean) to `F.cross_entropy(weight=w, reduction='sum') / B`
   (plain-mean). **User explicitly approved (Option 1).** Under PyTorch's
   default weighted-mean, the per-sample weight `w[t]` cancels in
   numerator and denominator in single-sample batches, defeating the D5
   strategy ("class weights amplify minority gradients"). The plain-mean
   reduction preserves the intended scaling and is consistent with
   torchvision's focal-loss idiom. Documented in both the test docstring
   and `losses.py` module docstring.

2. **`TrainingConfig.warmup_epochs` default: 3 → 0** (implementation-only;
   no test expectations modified). The ROADMAP code stub showed 3 but the
   test contract requires `TrainingConfig(epochs=1)` to construct
   successfully, which is impossible with `warmup_epochs=3` and the
   `warmup ≤ epochs` validator. Real training scripts pass
   `--warmup-epochs 3` via the CLI flag (default in argparse).

**Streams P (Perch 2.0 probe) and V (real-data GPU training) deferred:**

- **Stream P (D3 sidequest, Perch 2.0 linear probe):** the test contract
  (`tests/classifier/test_train_perch2_probe.py`) specifies macro F1 > 0.30
  on a synthetic dataset of IID Gaussian noise with permuted labels —
  structurally unfulfillable (mutual information `I(features; labels) = 0`
  by construction). The 4 tests skip cleanly via an ImportError-driven
  `skipif` guard until the test is amended. See `docs/modules/lab-classifier-v1.md`
  §"Perch 2.0 linear probe" for the technical rationale.

- **Stream V (real-data GPU training):** no GPU on this WSL host
  (`torch.cuda.is_available() == False`, no `/dev/nvidia*`, no
  `nvidia-smi`). The full 50-epoch run on 9,741 training patches needs a
  GPU to finish in reasonable wall time. All code is CPU-runnable (proven
  by the smoke test) but the SHIP-eligible exit criteria (macro F1 > 0.65
  on val, per-class precision ≥ 0.40, held-out 845 macro F1 > 0.80) are
  gated by the GPU run. A successor handoff documents the inputs, the
  training command, and the validation criteria.

**Files NOT touched** (matches the do-not-touch list from the predecessor
handoff): `src/usv_spectrogram/corpus.py`, `app/core/sliding_inference.py`,
`app/core/notch.py`, `app/core/denoise.py`, `postprocessing/normalization.py`,
`scripts/run_batch_detection.py`, `src/usv_spectrogram/classifier/cleaning_pipeline.py`,
`src/usv_spectrogram/classifier/diagnostics.py`,
`src/usv_spectrogram/classifier/resample.py`,
`src/usv_spectrogram/classifier/dataset.py`,
`scripts/cnn_prepare_training_data.py`, and the pre-existing
`tests/classifier/test_{cleaning_pipeline,diagnostics,resample,dataset,cnn_prepare_training_data}.py`.

**Master-reviewer result (Step 5):** APPROVE WITH NOTES. No blockers. Three
non-blocking warnings (stale `test_model.py` header comment, missing
`__init__.py` re-exports for 18.3 symbols, missing IMPLEMENTATION_PROGRESS
entry) all addressed in this same session before this entry was finalised.

**Inherited Tier-2 tickets (NOT addressed in 18.3, blocking for 18.4):**

1. `cleaning_pipeline.py` produces all-zero output on long lab WAVs when
   `baseline_mode='median_envelope'` chains with `_apply_global_mad`.
   Workaround at `scripts/cnn_prepare_training_data.py:376`
   (`baseline_mode='percentile'`).
2. `scripts/cnn_prepare_training_data.py:_collect_wav_rows` uses
   non-recursive `glob`. Worked around by `scripts/cnn_wild_topup.py`.

Both must be resolved before Module 18.4 (DANN).

**Verdict:** PARTIAL CLOSE. The Module 18.3 code shipment is complete and
green; Streams P (Perch) and V (real-data GPU training) are deferred to
the successor handoff. Module 18.4 (DANN) is BLOCKED on Stream V.

---

### 2026-05-25 — Module 18.3 Stream V (GPU real-data results)

SHIP on VocalMat gates. Held-out lab eval deferred to Module 18.4 prep.
Executed on the GPU rig (`cloudyclaude`, `/data/mickey_london_lab` non-git
deployment); artifacts rsync'd back to the worktree. Full rig closeout:
`docs/handoffs/2026-05-25_rig-stream-v-closeout.md`.

Training: 50 epochs, batch 64, AdamW lr=1e-3, warmup=3, focal γ=2.0, ResNet-18
+ ImageNet pretrained. Single RTX 3060 Ti (co-tenanted), ~16 min wall-clock.

VocalMat val/test (PASS):
  - macro_f1_val  = 0.7693
  - macro_f1_test = 0.7669

Per-class precision (test split; gate ≥ 0.40 on every class — PASS):
  Noise 0.9837 | Step up 0.8146 | Down-FM 0.8021 | Short 0.8010
  Chevron 0.9051 | Up-FM 0.8051 | Flat 0.8438 | Two steps 0.7027
  Step down 0.7179 | Complex 0.5957 | Reverse Chevron 0.6471 | Multi-steps 0.6000

Per-class recall (test split):
  Noise 0.8963 | Step up 0.8056 | Down-FM 0.8701 | Short 0.8947
  Chevron 0.7799 | Up-FM 0.8051 | Flat 0.7168 | Two steps 0.7429
  Step down 0.7179 | Complex 0.8000 | Reverse Chevron 0.8462 | Multi-steps 0.4286

Confusion-matrix interpretation: largest off-diagonal mass is Two-steps →
Step-up at 15.7% (well under the 40% collapse threshold). Step-up → Two-steps
6.1%, Chevron → Complex 5.7% — all taxonomically adjacent, not class collapse.

Held-out lab eval: DEFERRED to Module 18.4 prep. Two errors corrected (see
`docs/handoffs/2026-05-25_ERRATA_held-out-845.md`):
  1. held_out_845_macro_f1 > 0.80 gate WITHDRAWN — lab data never Grimsley-
     labeled, so Grimsley macro F1 on lab is unbuildable.
  2. Real verdict file is results/lab_{cluster0,cluster1,cluster2,noise}_
     review/review_index_annotated.csv (844 usv/noise verdicts), NOT
     classified_detections_lab_131204_clean.csv. Even the right file needs
     clean 227×227 patches re-extracted from USV_lab_131204_chunked_2s_full/
     — owner: CPU box / Module 18.4.

Perch 2.0 probe: DEFERRED (Option B). No new code, no gate.

Code shipment (rsync rig → CPU box):
  - scripts/train_lab_classifier.py — added numpy import, inline confusion-
    matrix render call, _render_confusion_matrix_png helper.
  - scripts/render_confusion_matrix.py — NEW standalone back-fill helper.
  - tests/classifier/test_render_confusion_matrix.py — NEW. 8 tests.

Test suite (rig, CUDA host): 186 passed, 5 skipped, 0 failed (+8 from Step 8
vs the 178/4 deploy-time baseline). Re-verified on the CPU box after rsync.

Artifacts in `results/lab_classifier_v1/`: best.pt (44.8 MB), metrics.json
(NOTE: held-out fields are stale garbage — 0.0 / 2.4849 — from the label-
distribution fallback firing on the wrong CSV; IGNORE those, val/test/
per-class data is valid), confusion_matrix.png (1275×1125), eval_report.md.

**Verdict:** CLOSE. Module 18.3 SHIP on the VocalMat result. Module 18.4
(DANN) UNLOCKED — entry: `docs/handoffs/2026-05-25_module-18.4-dann-orchestrator.md`.
Tier-2 prep (re-extract 844 clean lab patches for the held-out eval) can run
standalone, independent of 18.4 training kickoff.

---

## 2026-05-26 — Module 18.4 (DANN cage-invariance) — CPU implementation phase

Implemented the ROADMAP §18.4 5-file set (additive on 18.3). GPU training +
evaluation deferred to the rig (no local GPU). Spec: `ROADMAP_lab_cnn_classifier.md §18.4`;
executed from `docs/handoffs/2026-05-26_module-18.4-dann-resume.md`.

Files created (all NEW — 18.3 baseline untouched):
  - src/usv_spectrogram/classifier/dann.py — GradientReversal (autograd.Function,
    backward grad×−λ), grad_reverse, DomainHead, LambdaSchedule (Ganin
    2/(1+e^−γp)−1), ResNet18DANN (timm num_classes=0 encoder + class_head + domain_head).
  - src/usv_spectrogram/classifier/cage_probe.py — linear_cage_probe (frozen encoder,
    cached features, weight-decay regularized linear head; <0.65 pass).
  - scripts/train_lab_classifier_v2.py — DANN CLI: warm-start v1 fc→class_head /
    backbone→encoder; source=vocalmat(labeled), target=lab(unlabeled, domain_unlabeled.csv);
    adversarial loop; early-stop on val F1; cage probe (test split) + comparison_v1_vs_v2.md.
  - scripts/run_vae_diagnostic_on_encoder.py — 4-criteria VAE re-run on encoder features
    (knn<0.85, PC1 d<1.50, per-dim max d<0.30, notch migration<0.30) → cage_invariance_probe.md.
  - docs/modules/lab-classifier-v2-dann.md — module doc.
  - tests/classifier/test_dann.py (14) + test_cage_probe.py (7) — test-architect, pre-impl SPEC.
  - src/usv_spectrogram/classifier/__init__.py — +17-line additive 18.4 export block.

Verification: py_compile clean; new tests 23 passed; full tests/classifier/ = 211 passed,
4 skipped (188 baseline + 23). VAE diagnostic verified end-to-end on synthetic fixtures
(correctly FAILs on untrained encoder — sensitivity check). master-reviewer (Tier 3):
APPROVED, no blockers. Fixes applied: MAJOR-1 (per-dim d small-n false-positive → ≥500
sample warning + docs), NIT-1 (per_recording → NotImplementedError), MINOR-3 (cage probe on
test split). Review: `docs/reviews/lab-classifier-v2-dann-review.md`.

**Verdict:** CPU phase DONE. Rig phase (rsync 18 GB pool + held-out 844, GPU train, 3-gate
eval, real held-out 844 evaluator) pending per-session auth — handoff:
`docs/handoffs/2026-05-26_module-18.4-dann-rig.md`.

## 2026-05-27 — Shape-VAE v3 Pathway B+A (hybrid contrastive + VAE + ridge-derivative) — CPU build phase

Executed `docs/handoffs/2026-05-27_shape-vae-BA-hybrid.md` (canonical: `PLAN_geometric_shape_clustering_vae.md`
§3 Option B+A). Goal: a learned latent where geometrically-similar USVs are neighbors, invariant to
pitch/position, AND a navigable generative shape-map. The riskiest of three sibling pathways
(A/B/B+A) — re-introduces the reconstruction objective the 2026-05-26 denoised retrain showed is a
shape dead-end (η² 0.081 vs 0.75 registration ceiling), betting contrastive + shift-augmentation +
latent-consistency dominate it.

**User-locked design (2026-05-27):** contrastive-dominant staged anneal (λ_recon/β/λ_deriv ramp 0→full,
λ_nt/λ_lc constant, β LOW); soft-argmax expected-frequency ridge proxy (differentiable; Viterbi
`track_ridge` supplies the cached TARGET only); tuning budget = small 3–4 run sweep on 5970.

**Files (new, untracked):**
- `scripts/experiments/train_shape_vae_v3_hybrid.py` — model + 5-term loss + in-band pitch/time-shift
  augmentation + anneal + train loop. Wraps the FROZEN `ImageVAE` (imported, not modified). Pure
  functions: `ShapeVAEv3Config`, `soft_argmax_ridge`, `nt_xent`, `latent_consistency`,
  `derivative_loss`, `augment_pitch_time_shift`, `annealed_weights`, `hybrid_loss`.
- `tests/test_shape_vae_v3_hybrid.py` — 50 pre-implementation spec tests (test-architect, Step 0).
- `tests/test_shape_vae_v3_hybrid_hardening.py` — 25 adversarial tests (test-hardener), 0 bugs found.
- `docs/reviews/shape-vae-v3-hybrid-review.md` — master-reviewer (Tier 3): CHANGES NEEDED, no BLOCKERs;
  2 SHOULD-FIX (hybrid_loss test-only assembly note + band-input guard; time_warp_range RESERVED label)
  applied as documentation; all training math verified correct.
- `docs/modules/shape-vae-v3-hybrid.md` — module doc.

**Track-0 reuse (no duplicate compute):** consumes the SHARED `extract_ridge_targets_v3.py` cache
(`dFdt_true`/`valid_mask`, kHz/frame ×1000→Hz) built by the parallel Pathway-A chat; augmentation
in-band span derived from per-patch energy extent. Shared eval `eval_shape_vae_v3.py` DEFERRED
(unbuilt, parallel-chat coordination; depends on a trained model).

**Tests:** 75 passed (50 spec + 25 hardening). py_compile OK. Frozen `train_contour_vae_v2.py` clean.

**Verdict:** CPU build phase DONE. Rig phase (Track-0 extraction if not already cached → small 3–4 run
sweep on 5970 → shared eval) pending per-session auth — successor handoff:
`docs/handoffs/2026-05-27_shape-vae-BA-hybrid-rig.md`.

## 2026-05-28 — Shape-VAE Pathway B (2-D contrastive shape encoder) — KILL

Executed `docs/handoffs/2026-05-27_shape-vae-B-contrastive-invariance.md` end-to-end on the rig. Goal:
test whether an encoder-only 2-D contrastive objective with pitch/time/warp shift augmentation could
cluster USVs by geometric shape — the one un-falsified VAE-family idea remaining ("M9 done right").

**Files (new):**
- `scripts/experiments/train_shape_encoder_contrastive.py` — 2-D conv encoder (reuses ImageVAE
  `1→32→64→128→256` stack + AdaptiveAvgPool) + SimCLR projection head, NT-Xent identical to M9's
  `ntxent`. Augmentation via one `grid_sample(zero-pad)`: integer pitch/time shifts + fractional warp;
  per-patch in-band clamp using imported corpus `USV_FREQ_MIN/MAX_HZ` (verified 50×5 augmented real
  patches stay in 20–120 kHz band rows [35,204]). Encoder-only — no decoder/KL/recon. `--ckpt-every N`
  added after rig OOMs.
- `scripts/eval_shape_encoder.py` — KMeans(20) on held-out embeddings; full gate panel (shape η²,
  pitch/dur/curv η², CV-NMI, kNN purity); reuses cached `desc_denoised.npz` ridges (no re-extraction).
- 53 tests across 3 files: `tests/test_shape_encoder_contrastive.py` (20 pre-impl, test-architect
  caught a real `(B,) vs (B,H)` broadcast bug in the in-band clamp), `tests/test_eval_shape_encoder.py`
  (15 pre-impl), `tests/test_shape_encoder_hardening.py` (18 adversarial; closed the wide-freq
  band-clamp gap by forcing `src/` so the corpus import is non-vacuous; locked `chevron_valley`).
- `docs/modules/shape-encoder-contrastive.md`; `docs/reviews/shape-encoder-contrastive-review.md`
  (master-reviewer: CHANGES NEEDED, **no blockers** — NT-Xent ≡ M9 numerically, augmentation geometry
  + clamp correct, row-alignment correct; WARNING-1/2 + NIT-1/2 applied).

**Rig saga:** first run (80 ep) OOM-killed at ep20 by `llama-e4b` docker. Second run (40 ep, GPU 3)
also OOM-killed at ep0 — RAM still tight even after stopping llama. Root cause discovered: sibling
Pathway A (`train_shape_vae_v3_deriv.py`, ~17 GB RSS) was running concurrently; B's 16 GB patch corpus
+ A together exceed the rig's 31 GB. User chose "wait for A, auto-run B clean" — an autonomous watcher
waited 110 min for A's RAM to free, then launched B clean.

**Result (third run, 40 ep, ckpt every 8, GPU 3):** held-out (n=6,929) **shape η² = 0.044** (KILL
threshold 0.12; registration ceiling 0.58–0.75). CV-NMI 0.019 (worse than production 0.04). Chevron
kNN purity 0.293 — BELOW the 3-class base rate of ~0.33 (actively anti-clustering chevrons). Pitch η²
0.306 (still leaks). Flat across every checkpoint (ep8=0.038 … final=0.044) while NT-Xent loss
converged 3.26→1.78 — not undertrained, just learning the wrong thing.

**Independent verification (fresh Explore agent):** recomputed all η² from raw artifacts; matched the
scorecard to ~1e-8. Alignment, splits, KMeans convergence all clean. 0.044 is reproducible truth.

**Verdict: KILL — SHIP REGISTRATION.** Per the handoff's exit criteria (shape η² < 0.12 AND purity not
better than chance ⇒ KILL). The VAE-family is now closed for shape clustering. Production = `models/
shape_kmeans/k20.joblib` (registration → KMeans, shape η² 0.58–0.75). Methodological lessons captured
in successor `docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md` and vault note
`[[project_shape_registration_clustering]]`: (1) shift augmentation creates LOCAL invariance only
(±15 kHz << 100 kHz USV band — pitch centroid leaks); (2) 2-D image objective is the wrong substrate
for shape regardless of loss family (5 falsified 2-D attempts vs successful 1-D-on-registered).
Rig artifacts: `/data/shachar/contour_vae/results/latent_transitions/b_contrastive/` (embeddings.npy,
encoder.pt, scorecards, train.log). HTML head-to-head: `$CLAUDE_JOB_DIR/session_report.html`.


## 2026-05-28 — Shape-VAE Pathway B+A (hybrid) — KILL

Executed `docs/handoffs/2026-05-27_shape-vae-BA-hybrid-rig.md` (gated rig launch). One run completed
(of the locked 3-4 budget); kill criterion met after run 1 by user decision and confirmed by the
sibling chat's parallel Pathway-B kill the same day.

**Files (new):**
- `scripts/experiments/eval_shape_kill_gate_v3.py` — minimal kill-gate scorecard (shape η² + pitch
  + duration). Matches M9/M10/R1/R2 `eta2`/`register_one` definitions for direct bake-off
  comparability. Not the full shared `eval_shape_vae_v3.py` (which stays deferred — no longer
  needed for B+A now that the pathway is dead).
- `docs/handoffs/2026-05-28_shape-vae-BA-hybrid-KILL.md` — verdict memo (scorecard + head-to-head
  table + mechanism diagnosis + SHIP REGISTRATION conclusion).

**Rig artifacts (kept for reproducibility):**
- `/data/shachar/contour_vae/models/shape_vae_v3_hybrid/run1/{best.pt,last.pt,hyperparams.json}` (31 MB each).
- `/data/shachar/contour_vae/results/shape_vae_v3_hybrid/run1/{training_log.csv,killgate.json}`
  (also mirrored locally at `results/shape_vae_v3_hybrid/run1/`).
- 5970 Track-0 ridge cache `/data/shachar/contour_vae/results/denoised_patches/5970/ridge_targets_v3.npz`
  (shared with Pathway A — keep).

**Result (run1, 80 ep, GPU 3 cuda:3, ~46 min):** shape η² **0.1046** (KILL threshold 0.12, registration
ceiling 0.58–0.75). Pitch η² 0.262, duration η² 0.171 — latent still sorts by pitch/duration, exactly
the failure mode the un-registered image-VAE objective was predicted to produce.

**Mechanism (visible in `training_log.csv`):** epochs 0–9 (w_recon=0, pure contrastive + latent-
consistency) — `nt` 2.91→1.48 and `lc` 2.28→0.65, the encoder *was* learning shift-invariant shape.
Epochs 10+ (anneal): the instant reconstruction got nonzero weight, `nt` jumped 1.48→3.5 and `lc`
exploded 0.65→200+ and both plateaued. The recon term overrode contrastive immediately and irrecoverably.

**Joint verdict with sibling chat:** Pathway B (encoder-only, NO reconstruction) ALSO killed today at
shape η² **0.044** — sharper than B+A's 0.105. So recon-as-saboteur explains *part* of B+A but the 2-D
image substrate itself is the wrong place for shape regardless of loss family (five 2-D attempts now
falsified vs one successful 1-D-on-registered). Sibling chat's sharper diagnosis: ±15 kHz shift
augmentation creates LOCAL invariance only (small vs the 100 kHz USV band), so the pitch centroid
leaks regardless of loss family. B+A inherited the same augmentation flaw.

**Verdict: KILL — SHIP REGISTRATION.** Production = `models/shape_kmeans/k20.joblib` (registration →
KMeans, shape η² 0.58–0.75). The VAE-family is closed for shape clustering. See
`[[project_shape_registration_clustering]]` for the consolidated record. Runs 2–4 of the planned
sweep cancelled (would not change the mechanism). Shared eval `eval_shape_vae_v3.py` deferred
indefinitely (no learned model worth its full gate panel).
