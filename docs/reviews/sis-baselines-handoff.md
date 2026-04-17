# Implementation Handoff: SIS Baselines (17.1)

**Module:** SIS Baselines on Existing Labels (Phase 17.1)
**Review Tier:** 2 (per `ROADMAP_SIS_BENCHMARK.md`)
**Date:** 2026-04-17
**Branch:** `main`

## What Changed

- New reusable depth-1 SIS computation in
  `src/usv_spectrogram/classification/sis_baselines.py` (frozen-dataclass
  result + single public function `compute_sis_depth_1`).
- New CLI driver `scripts/run_sis_baselines.py` that reads the Phase A2 +
  HDBSCAN CSVs, joins on `det_index`, computes SIS for all three existing
  labelings, and emits `baselines.csv` + `baselines.png`.
- Package export surface updated in `classification/__init__.py`.
- Module documentation in `docs/modules/sis-baselines.md`.
- Dated entry appended to `IMPLEMENTATION_PROGRESS.md`.

## Files Changed

- `src/usv_spectrogram/classification/sis_baselines.py` (NEW) — `SISResult`
  frozen dataclass + `compute_sis_depth_1`. 98 lines.
- `scripts/run_sis_baselines.py` (NEW) — CLI driver. 167 lines.
- `docs/modules/sis-baselines.md` (NEW) — public interface docs + CLI usage.
- `IMPLEMENTATION_PROGRESS.md` (NEW) — first dated entry.
- `src/usv_spectrogram/classification/__init__.py` (MODIFIED) — added
  `SISResult` + `compute_sis_depth_1` to imports and `__all__`.

## Key Decisions Made

**1. Reused `usv_language.analysis.sequence_analysis.mutual_information_at_lag`
rather than re-implementing.** Preserves exact numerical continuity with the
Phase A2 result (0.093 bits on Scattoni-7). The ROADMAP explicitly required
this: *"reuse this; do not reimplement"*.

**2. Computed marginal entropy directly from `np.bincount` rather than
marginalizing the joint.** The two approaches are mathematically equivalent
but differ in floating-point noise. Direct computation makes the test
`test_conditional_entropy_identity` (which requires `H = MI + H_cond` to
within 1e-9) hold precisely.

**3. Divide-by-zero guard for `entropy_reduction_pct` returns `0.0` when
`H = 0`.** The single-label degenerate case has MI=0 and H=0 — the natural
ratio 0/0 would be NaN. `test_single_label_sequence_returns_zero_mi_and_entropy`
explicitly asserts the field is not NaN. Returning `0.0` is semantically
correct (no reduction because there was no entropy to begin with).

**4. Script bootstrap adds both `SRC_ROOT` and `REPO_ROOT` to `sys.path`.**
The repository has a bimodal layout: `usv_spectrogram/` lives under `src/`,
but its sibling `usv_language/` is a top-level package at repo root. The
standard Pattern-8 bootstrap adds only `src/`, which caused two subprocess
tests (`test_script_produces_baselines_csv_with_3_rows`,
`test_script_exit_code_zero`) to fail with `ModuleNotFoundError: usv_language`.
Adding REPO_ROOT fixed both.

**5. Missing label columns in the input CSVs produce a warning and are
skipped, not a hard failure.** This matters for the real dataset: the current
`classified_detections_full.csv` does not yet have a `syllable_type` column
(only `label`). The contract `test_script_produces_baselines_csv_with_3_rows`
provides all three columns and expects three rows; this still passes. The
robustness allows the script to produce a partial `baselines.csv` on real
data while data-prep work is ongoing, rather than blocking the whole
benchmark.

**6. Factorized with `pd.factorize(sort=True)` for deterministic label→int
mapping.** Two runs on the same data must produce identical `SISResult`
fields including `n_labels`.

## What I'm Unsure About

- **Robustness of the join on `det_index`.** Tests supply `det_index` in both
  CSVs; real CSVs also have it. But there's no verification that `det_index`
  is unique on either side — a duplicate row in either CSV would double-count.
  I did not add an explicit `.drop_duplicates()` because it's not specified
  and might mask a real data-prep bug.
- **Plot layout.** The Hertz reference-line text annotations are placed at
  `x = len(names) - 0.5`, which works for 3 bars but could collide on wider
  charts. The module is only used for ≤3 labelings today, so this is a small
  risk.
- **The `syllable_type` data-prep gap for the real dataset.** Out of scope for
  17.1 but it means the full decision-gate exit criterion
  (`Scattoni-7 MI ≈ 0.093 bits`) cannot be verified today. Flagging for the
  reviewer: should this block approval, or is it correct to treat data prep
  as separate work?

## Test Results

```
.venv/bin/python -m pytest tests/test_sis_baselines.py -v
================ 17 passed in 11.96s ================

# Regression check (classification package)
.venv/bin/python -m pytest tests/test_classification/ tests/test_sis_baselines.py
================ 161 passed, 1 skipped in 12.40s ================
```

**Pre-existing tests from test-architect:** 17 (8 ROADMAP-mandated + 9
gap-pattern additional)
**New tests written during this implementation:** 0 (the pre-implementation
spec was already comprehensive; hardener will add more after review)

## ROADMAP Exit Criteria Status

- [x] `SISResult` + `compute_sis_depth_1` implemented per spec
- [x] All tests pass
- [x] `py_compile` passes on both new files
- [x] Driver script produces `baselines.csv` + `baselines.png` on synthetic
  inputs (verified by `test_script_produces_baselines_csv_with_3_rows`)
- [x] Bar chart includes Hertz reference lines (0.10 / 0.13 / 0.22)
- [ ] **`results/sis_baselines/baselines.csv` on real 5970 data** — deferred.
  Requires adding `syllable_type` column to real `classified_detections_full.csv`
  (data-prep task, not a module-contract failure).
- [ ] **Scattoni-7 MI ≈ 0.093 bits reproducibility check** — also deferred,
  same reason.

## Docs Written/Updated

- `docs/modules/sis-baselines.md` — created
- `IMPLEMENTATION_PROGRESS.md` — created with first entry
- `docs/architecture/patterns.md` — no new pattern introduced
- No new decision note in `notes/` — implementation is a thin wrapper, no
  non-obvious architectural decisions

## Anti-Greenwashing Attestation

- I did not modify `tests/test_sis_baselines.py` during implementation (verified
  by `git diff tests/test_sis_baselines.py` — no changes).
- All expected values in the pre-implementation tests were hand-computed by
  the test-architect from first principles (e.g., periodic binary `MI = 1.0
  bit`, IID `MI < 0.01 bits`). None are copy-pasted from failing output.
- The `entropy_reduction_pct` divide-by-zero guard returns `0.0` as a
  semantically meaningful value, not a silent hack to make a test pass.
