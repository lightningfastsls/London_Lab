# SIS Baselines (17.1) Module Review

**Date:** 2026-04-17
**Reviewer:** master-reviewer (Sonnet 4.6)
**Handoff:** `docs/reviews/sis-baselines-handoff.md`
**Review Tier:** 2 (confirmed — no DSP, no ML training, pure information-theoretic combinatorics)
**Module:** `src/usv_spectrogram/classification/sis_baselines.py` + `scripts/run_sis_baselines.py`

---

## Expected Constraints (Written Before Finding Problems)

From the ROADMAP and ADR references:

- **DSP parameters:** None apply. This module operates on integer label sequences only. ADR-001 and ADR-002 are explicitly out of scope (confirmed by module doc).
- **Reuse requirement:** ROADMAP 17.1 explicitly mandates `mutual_information_at_lag` from `usv_language.analysis` — do not reimplement. This is the single most critical spec requirement.
- **Data flow:** Labels arrive pre-sorted or are sorted by `(file, begin_time_s)` before MI computation. Sequential order matters — an unordered sequence would produce a different MI.
- **Frozen dataclass:** `SISResult` must be `frozen=True` per Pattern 1.
- **Script CLI:** `run_sis_baselines.py` must follow Pattern 4 with Pattern 8 bootstrap.
- **Exit criteria from ROADMAP:** All 8 ROADMAP test cases must pass; `baselines.csv` must have 3 rows on synthetic input.
- **Most likely failure modes:** (a) reimplementing MI instead of reusing, (b) processing labels in wrong order, (c) divide-by-zero when H=0, (d) script bootstrap not finding `usv_language` package.

---

## Test Verification

```
.venv/bin/python -m pytest tests/test_sis_baselines.py -v --tb=short
17 passed in 10.70s
```

Claimed count (17) matches actual output. All 8 ROADMAP test cases and 9 gap-pattern tests pass.

**Anti-greenwashing verification:** `git log --oneline -5 tests/test_sis_baselines.py` shows a single commit (`7c5d537b`) — the pre-implementation test-architect commit. The file was not touched during implementation. Confirmed.

---

## Findings

### BLOCKERS

None.

---

### WARNINGS

**W1 — Import path diverges from ROADMAP spec (minor, informational)**

**What:** The ROADMAP's code snippet in module 17.1 (line 46) specifies `from usv_language.analysis.information_theory import mutual_information_at_lag`. The implementation imports from `from usv_language.analysis.sequence_analysis import mutual_information_at_lag`.

**Where:** `src/usv_spectrogram/classification/sis_baselines.py:22`

**Why it matters:** The function is canonically *defined* in `sequence_analysis`; `information_theory.py` re-imports it from there (`usv_language/analysis/information_theory.py:26-28`). Both import paths resolve to the same function object. The "do not reimplement" requirement is fully satisfied.

**Fix:** No functional change required — adding a short comment pointing at the canonical source is sufficient.

---

**W2 — `det_index` uniqueness is unchecked; duplicate rows in either CSV would silently corrupt MI**

**What:** `_load_merged` merges on `det_index` with a left join but does not call `.drop_duplicates()` on either side before or after the join.

**Where:** `scripts/run_sis_baselines.py:102`

**Why it matters:** A duplicate `det_index` in either CSV would inflate MI by repeating transitions. Real CSVs do not have duplicates today, but the script provides no guard and no warning.

**Fix:** Add a warning (not a hard failure) after loading each CSV when duplicates are detected. Warns the user while preserving the intended behavior.

---

**W3 — Silent sort fallback when `file`/`begin_time_s` columns are absent**

**What:** `_load_merged` sorts by `(file, begin_time_s)` only if those columns are present. If both are absent, no sort occurs and MI is computed on unordered data without warning.

**Where:** `scripts/run_sis_baselines.py:104-106`

**Fix:** Emit a warning when sort keys are missing or partially present.

---

**W4 — ROADMAP exit criterion for real-data reproducibility is deferred; decision gate cannot fire**

**What:** The ROADMAP's exit criterion states: "Scattoni-7 MI ≈ 0.093 bits (matches prior Phase A2 result — reproducibility check)." This criterion is deferred because the real `classified_detections_full.csv` lacks a `syllable_type` column.

**Reviewer opinion:** The `syllable_type` data-prep gap is correctly out of scope for *this module's contract* — the module works correctly on any input that has the column. However, the ROADMAP exit criterion should not disappear. Add a dated follow-up in `ops/reminders.md`.

---

### SUGGESTIONS

**S1 — `mutual_information_at_lag` inner loop uses Python-level iteration (O(K²))** — not relevant at K≤27; flag for module 17.8 (SIM optimizer), which may call the function millions of times.

**S2 — Plot annotation placement (`x = len(names) - 0.5`) may clip on narrow outputs** — low risk for 3 fixed labelings; non-blocking.

**S3 — `SISResult` has no `__post_init__` validation** — the sole constructor produces correct values; adding validation is low priority.

---

## Math Trace

**Periodic binary sequence, N=500:**
- `counts = [250, 250]`, `probs = [0.5, 0.5]` → `marginal_h = 1.0` bit
- Transitions: 0→1 × 249, 1→0 × 249; joint = `[[0,0.5],[0.5,0]]`
- `MI = 1.0` bit (hand-computed from `P(i,j) * log2(P(i,j)/(P(i)P(j)))`)
- `conditional_h = 0.0`, `pct = 100.0` ✓

**Empty, length-1, single-label guards** — all return zero fields, no NaN. Conditional entropy identity holds to float precision because `cond_h = marginal_h − mi` is stored directly.

---

## Spec Compliance Checklist

| ROADMAP requirement | Status |
|---|---|
| `SISResult` frozen dataclass with 7 fields | PASS |
| `compute_sis_depth_1` function signature | PASS |
| `sort_by_time` parameter | PASS |
| `pd.factorize(sort=True)` for labels | PASS |
| `mutual_information_at_lag` reused, not reimplemented | PASS |
| Driver: `--classified-csv`, `--umap-csv`, `--output-dir` args | PASS |
| Driver: join on `det_index` | PASS |
| Driver: `baselines.csv` with one row per labeling | PASS |
| Driver: `baselines.png` with Hertz reference lines | PASS |
| Pattern 1 / Pattern 4 / Pattern 8 | PASS |
| All 8 ROADMAP test cases | PASS (17 total) |
| `baselines.csv` on real data (exit criterion) | DEFERRED (W4) |
| Scattoni-7 MI ≈ 0.093 bits (exit criterion) | DEFERRED (W4) |

---

## Documentation Status

| Doc | Status |
|-----|--------|
| `docs/modules/sis-baselines.md` | EXISTS, accurate |
| `docs/architecture/patterns.md` | NO UPDATE NEEDED |
| Decision notes (`notes/`) | NOT NEEDED |
| `IMPLEMENTATION_PROGRESS.md` | APPENDED |

---

## Verdict

**APPROVED**

No blockers. All 17 tests pass. The pre-implementation test spec was not modified. The single required reuse (`mutual_information_at_lag`) is satisfied. Math traces verify for the periodic-binary and degenerate cases.

The two deferred exit criteria (real-data run, 0.093 bit reproducibility check) are correctly documented as data-prep work outside the module's contract. The three warnings (W1–W3) are quality improvements none of which affect correctness for the current use case; W4 is a follow-up task to track.

---

## Fixes Applied (2026-04-17)

In response to this review, the implementor applied the following fixes in the same session:

**W1 — Fixed.** Added comment at `src/usv_spectrogram/classification/sis_baselines.py:22` pointing at the canonical source module and noting that `information_theory.py` re-exports the same function.

**W2 — Fixed.** `scripts/run_sis_baselines.py:_load_merged` now emits a `[warn]` line on stderr when `det_index` duplicates are detected in either CSV (non-blocking; lets the user see data-prep bugs instead of silently corrupting MI).

**W3 — Fixed.** `_load_merged` now emits a `[warn]` line when `file` or `begin_time_s` sort keys are missing, so silent unordered MI is impossible.

**W4 — Tracked.** Added a dated entry to `ops/reminders.md` recording the data-prep task: add `syllable_type` column to real `classified_detections_full.csv` and rerun the script to verify the 0.093 bit reproducibility check.

**S1–S3 — Deferred** as recommended by the reviewer (low priority, non-blocking).

Verification: `.venv/bin/python -m pytest tests/test_sis_baselines.py -v` — 17 passed post-fix.

---

## Test Hardener Results (2026-04-17)

After review approval, `test-hardener` was spawned and added 24 additional tests (17 → 41 total) covering: the new W2/W3 warning paths, three-symbol periodic MI (log2(3) bits), skewed binary entropy, Scattoni ballpark regression, `sort_by_time` edge cases (ties, float arrays, length mismatch), factorize determinism with shuffled input, CLI edge cases (overwrite, partial labelings, auto-created nested output dir, all-columns-absent exit code), and `SISResult` dataclass semantics (asdict keys, equality, frozen-against-new-attribute, frozen-against-deletion).

**Bug discovered + fixed:** `compute_sis_depth_1` silently accepted `sort_by_time` arrays shorter than `labels`. `np.argsort` on the shorter array produced indices that `numpy` fancy-indexing applied to the longer `labels` array without error, silently truncating the sequence. `n_calls` reported the full length while MI was computed on only the first N elements.

**Fix applied** (`src/usv_spectrogram/classification/sis_baselines.py` — guard before `np.argsort`):

```python
if sort_by_time is not None:
    time_arr = np.asarray(sort_by_time)
    if time_arr.shape[0] != n_calls:
        raise ValueError(
            f"sort_by_time length ({time_arr.shape[0]}) does not match "
            f"labels length ({n_calls})"
        )
    order = np.argsort(time_arr, kind="stable")
    labels = labels[order]
```

The hardener's regression test (`test_sort_by_time_length_mismatch_raises`) was unskipped after the fix — it now exercises the `ValueError` path and passes.

**Final test count:** 41 passed, 0 skipped, 0 failed.
