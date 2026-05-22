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
