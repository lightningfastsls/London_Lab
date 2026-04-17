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
