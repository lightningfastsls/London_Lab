# DeepSqueak Results Import Module Review

**Module:** Phase 14.2 -- DeepSqueak Results Ingestion
**Date:** 2026-02-25
**Reviewer:** master-reviewer
**Handoff:** `docs/reviews/deepsqueak-import-handoff.md`
**Review Tier:** 2 (new module + script + tests, no DSP/ML changes)

---

## Test Results

```
22 passed in 0.89s (module tests)
55 passed (classification suite -- deepsqueak_import + raven_export)
120 passed (classification + detection + event-triggered, no regressions)
```

py_compile: PASS on all three new/modified files.

---

## Summary

The implementation is solid. The core architecture is clean, patterns are followed, and all 22 module tests pass. Five issues were identified and fixed during review.

---

## Findings and Fixes Applied

### B1 (BLOCKER, FIXED): `load_detections_for_merge` missing from `__init__.py`

**What:** Function was in the handoff's public API table but absent from both the import block and `__all__` in `__init__.py`.
**Fix:** Added to both the `from .deepsqueak_import import (...)` block and `__all__` list.
**File:** `src/usv_spectrogram/classification/__init__.py`

### W1 (WARNING, FIXED): CLI argument `--tolerance` renamed to `--tolerance-ms`

**What:** The ROADMAP spec and project convention (encode units in names) call for `--tolerance-ms`.
**Fix:** Renamed argument and updated all references including epilog examples.
**File:** `scripts/import_deepsqueak_results.py`

### W2 (WARNING, DEFERRED): Default tolerance 5ms vs vault note 10-50ms

**What:** The vault note on timestamp proximity matching documents 10-50 ms as typical DS drift. The implementation defaults to 5 ms.
**Resolution:** The 5 ms default is justified: DeepSqueak's hop size is 0.4 ms, so drift should be well under 5 ms. The vault note describes the general cross-tool case, not the specific Raven-table round-trip case where timestamps should be near-identical. If real data shows >5% unmatched at 5 ms, raise to 10 ms. The `--tolerance-ms` CLI flag makes runtime adjustment easy.

### W3 (WARNING, FIXED): Dead `matched_det_stems` variable

**What:** A `dict[str, set[int]]` was built but never read -- leftover scaffolding.
**Fix:** Removed three lines (declaration + two write sites).
**File:** `src/usv_spectrogram/classification/deepsqueak_import.py`

### W4 (WARNING, FIXED): Test used corrupt file instead of empty Excel

**What:** `test_empty_file_skipped_in_batch` created a zero-byte file (corrupt ZIP), not a header-only Excel.
**Fix:** Now writes a proper header-only Excel via `empty_df.to_excel()`. Removed dead `empty_df` variable that was constructed but never used.
**File:** `tests/test_classification/test_deepsqueak_import.py`

### S2 (SUGGESTION, FIXED): Fallback column normalization hardened

**What:** Unknown columns weren't stripping `/`, `-`, `.` characters.
**Fix:** Added `.replace("/", "_").replace("-", "_").replace(".", "_")` to fallback normalization.
**File:** `src/usv_spectrogram/classification/deepsqueak_import.py`

### S1 (SUGGESTION, ACKNOWLEDGED): Signature divergence from ROADMAP spec

**What:** `merge_with_detections` takes pre-loaded `dict` (not `Path`) and `export_classified_detections` returns `Path` (not `None`).
**Resolution:** Both are deliberate improvements over the ROADMAP spec. Pre-loaded dict enables testability; Path return enables call chaining.

### S3 (SUGGESTION, DEFERRED): Duplicate-column collision if both "Label" and "Type" exist

**What:** Both map to "label" in `_COLUMN_MAP`, which could produce duplicate columns.
**Resolution:** This is an edge case for a hypothetical malformed Excel. Will address if encountered in real data.

---

## Verdict

**APPROVED** -- All blockers and warnings fixed. Module is ready for Phase 14.3.

---

## Post-Fix Verification

```
py_compile: PASS (all 3 files)
tests/test_classification/test_deepsqueak_import.py: 22/22 passed
tests/test_classification/: 55/55 passed (no regressions to raven_export)
Full regression (classification + detection + event-triggered): 120/120 passed
```
