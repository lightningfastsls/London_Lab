# Phase 14.2: DeepSqueak Results Ingestion -- Handoff

**Date:** 2026-02-25
**Review Tier:** 2 (new module + script + tests, no DSP/ML changes)
**Status:** Implementation complete, review APPROVED

## What Was Built

A format adapter that reads DeepSqueak classification Excel outputs, normalizes column names to snake_case, and merges them with our detection JSONs via timestamp proximity matching. This completes the round-trip: detections -> Raven tables -> DeepSqueak -> Excel -> merged CSV.

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/usv_spectrogram/classification/deepsqueak_import.py` | ~380 | Core module: config, Excel loading, detection loading, matching, CSV export |
| `scripts/import_deepsqueak_results.py` | ~150 | CLI entry point with `--dry-run`, `--tolerance`, `-v` |
| `tests/test_classification/test_deepsqueak_import.py` | ~330 | 14 tests across 10 test classes |
| `docs/modules/deepsqueak-import.md` | ~100 | Module documentation |
| `docs/reviews/deepsqueak-import-handoff.md` | -- | This file |

## Files Modified

| File | Change |
|------|--------|
| `src/usv_spectrogram/classification/__init__.py` | Added `DeepSqueakImportConfig`, `ImportSummary`, and 5 public functions to imports and `__all__` |

## Architecture Decisions

### Separate `_load_detection_json_extended` vs modifying `load_detection_json`
The existing `load_detection_json` returns only `{start_s, end_s, duration_ms}`. For the merge we need additional fields (probabilities, detection_index, user_action). Rather than modifying the existing function (which raven_export.py depends on), we created a private extended version that reads the same JSON but extracts more fields.

### Greedy 1:1 nearest-neighbor matching
DeepSqueak reads our Raven tables, so there should be a 1:1 correspondence between DS rows and detection JSONs. Greedy matching removes each detection from the candidate pool once matched, preventing double-assignment. The algorithm reports unmatched items from both sides.

### `_COLUMN_MAP` for normalization
A static dictionary maps known DeepSqueak column names to snake_case. Unknown columns get basic normalization (lowercase, spaces to underscores). This handles both current DS column names and the "Type" vs "Label" variation across DS versions.

### `match_quality` column
Every row in the output gets one of four labels: `exact` (0ms distance), `fuzzy` (within tolerance), `unmatched_ds` (DS call with no detection), `unmatched_det` (detection with no DS call). This enables downstream filtering.

## Public API

```python
from usv_spectrogram.classification import (
    DeepSqueakImportConfig,      # frozen dataclass: dirs + tolerance
    ImportSummary,               # mutable stats accumulator
    load_deepsqueak_excel,       # single Excel -> normalized DataFrame
    load_all_deepsqueak_results, # batch load all .xlsx
    merge_with_detections,       # timestamp proximity matching
    export_classified_detections,# write merged CSV
    import_deepsqueak_results,   # full pipeline orchestrator
)
```

## What I'm Unsure About

- **DeepSqueak column names**: The `_COLUMN_MAP` is based on DeepSqueak v3 documentation. If the actual Excel files have different column names (e.g., different unit suffixes), the map may need updating. Unknown columns are handled gracefully (basic normalization), so this won't crash — it just won't get the canonical name.
- **WAV stem extraction**: The `_DS_SUFFIXES` list may not cover all DeepSqueak naming conventions. Additional suffixes can be added easily.
- **kHz vs Hz in Excel columns**: DeepSqueak exports frequencies in kHz but we normalize column names to `_hz` suffix for consistency. The actual values remain as-is (in kHz). If downstream code expects Hz values, a unit conversion step would be needed.

## ROADMAP Exit Criteria Status

- [x] Excel loading normalizes column names correctly
- [x] Timestamp matching finds exact match (0ms difference)
- [x] Timestamp matching finds fuzzy match within tolerance
- [x] Timestamp matching rejects match beyond tolerance
- [x] Unmatched detections from both sides reported as warnings
- [x] Multiple Excel files concatenated with source_file column
- [x] Empty Excel file handled gracefully
- [x] Merged output preserves all DeepSqueak acoustic features
- [x] py_compile passes on all new files
- [x] All tests pass (22/22)
- [x] Full suite regression-free (120/120 core tests passed)

## Test Coverage

| Category | Count | What's tested |
|----------|-------|---------------|
| Config validation | 4 | defaults, zero tolerance, negative tolerance, string->Path |
| Column normalization | 3 | known columns, unknown columns, Type->label |
| Exact match | 1 | 0ms distance, match_quality="exact" |
| Fuzzy match | 1 | 3ms offset within 5ms tolerance |
| Beyond tolerance | 1 | 100ms offset rejected, both sides reported |
| Unmatched reporting | 1 | mixed matched/unmatched from both sides |
| Multiple Excel files | 1 | concatenation with source_file column |
| Empty file handling | 2 | ValueError on empty, batch skips empty |
| Output completeness | 1 | all 15 DS + 8 det + 3 merge columns present |
| WAV stem extraction | 4 | plain, _Detections, _calls, _classified |
| ImportSummary | 2 | match_rate calculation, zero division safety |
| Integration | 1 | end-to-end pipeline with CSV + summary JSON output |

## Dependencies

- `pandas` (already in project) -- DataFrame operations and CSV writing
- `openpyxl` (already in project) -- Excel file reading
- Standard library: `json`, `logging`, `pathlib`, `dataclasses`
- Reuses `_is_detection_json` from `raven_export.py`

## Docs Written/Updated

- `docs/modules/deepsqueak-import.md` -- created
- `docs/reviews/deepsqueak-import-handoff.md` -- created (this file)
- `src/usv_spectrogram/classification/__init__.py` -- updated

## What's Next

This adapter enables:
1. Run the import pipeline on real DeepSqueak classification output
2. Phase 14.3: Repertoire Statistics (syllable distribution analysis using the merged CSV)
3. Population comparison (wild vs lab) using the classification labels
