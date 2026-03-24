# Phase 14.1: Raven Selection Table Export Adapter — Handoff

**Date:** 2026-02-23
**Review Tier:** 2 (new module + script + tests, no DSP/ML changes)
**Status:** Implementation complete, all tests passing

## What Was Built

A format adapter that converts individual detection JSON files (from the USV detection app) into Raven Pro selection table format — the standard bioacoustics annotation interchange format used by Raven Pro, DeepSqueak, and Audacity.

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/usv_spectrogram/classification/__init__.py` | 20 | Package init with `__all__` exports |
| `src/usv_spectrogram/classification/raven_export.py` | ~280 | Core module: config, discovery, conversion, TSV writing |
| `scripts/export_raven_tables.py` | ~140 | CLI entry point with `--dry-run` and `-v` |
| `tests/test_classification/__init__.py` | 0 | Test subpackage init |
| `tests/test_classification/test_raven_export.py` | ~340 | 33 tests across 9 test classes |

No existing files were modified.

## Architecture Decisions

### Use `core_time` not `saved_region`
Detection JSONs contain both `core_time` (actual USV boundaries) and `saved_region` (boundaries + context padding). Raven tables need the actual USV start/end, so we read only `core_time.start_s` and `core_time.end_s`.

### Fixed frequency bounds
Mouse USVs occupy 25-125 kHz. Since our detection pipeline doesn't extract per-syllable frequency bounds, we write the full band into every Raven row. This is standard practice — Raven's frequency columns are used for visualization zoom, not analysis.

### One table per WAV
The output naming convention `{wav_stem}.Table.1.selections.txt` follows Raven's own convention for selection tables associated with a sound file.

### Directory-based WAV mapping
Detection subdirectories are matched to WAV stems by name. This handles both exact matches (`rec_001/` → `rec_001.wav`) and prefix matches (`rec_001_session_abc/` → `rec_001.wav`).

## Public API

```python
from usv_spectrogram.classification import (
    RavenExportConfig,    # frozen dataclass: dirs + freq bounds
    ExportSummary,        # mutable stats accumulator
    load_detection_json,  # single JSON → {start_s, end_s, duration_ms}
    discover_wav_detection_mapping,  # dir tree → {wav_stem: [json_paths]}
    detections_to_raven_table,       # list[dict] → pd.DataFrame
    export_raven_tables,             # full pipeline → list[Path]
)
```

## What I'm Unsure About

- **Prefix matching edge cases**: The longest-stem-wins prefix matching handles `rec_001` vs `rec_001_retry` correctly, but could still be surprising if a WAV stem is an accidental prefix of an unrelated directory name.
- **`export_summary.json` written by library vs CLI**: Currently the library writes it in `export_raven_tables()`. The CLI dry-run path skips this (calls `discover_wav_detection_mapping` directly). If someone uses the library API with dry-run intent, they'd need to call discovery separately.

## ROADMAP Exit Criteria Status

- [x] `export_raven_tables.py --dry-run` reports correct mapping from real `USV_Detections/`
- [x] Full export produces one `.txt` per WAV with correct Raven format
- [ ] Exported `.txt` files loadable in Raven Pro (Raven Pro not available for testing)
- [x] `export_summary.json` shows total WAVs, total detections, any unmapped directories
- [x] All tests pass (33/33)
- [x] py_compile passes on all new files

## Test Coverage (33 tests)

| Category | Count | What's tested |
|----------|-------|---------------|
| Config validation | 4 | defaults, low>=high, negative freq, string→Path |
| JSON loading | 3 | valid, malformed, missing core_time |
| Skip logic | 5 | _saved_tracking.json, .png, .csv, valid .json, .txt |
| Discovery/mapping | 6 | matching, empty dirs, unmapped dirs, missing det dir, missing wav dir, prefix match |
| Conversion | 6 | sort order, 1-indexing, freq bounds, columns, view/channel, rounding |
| TSV output | 2 | header format, data row values |
| Integration | 4 | end-to-end, output dir creation, malformed skip, missing dir error |
| ExportSummary | 2 | to_dict roundtrip, empty_detection_dirs vs unmapped_dirs |
| CLI dry-run | 1 | --dry-run writes no files and returns exit 0 |

## Verification Results

- `py_compile`: All 3 new Python files compile cleanly
- `pytest tests/test_classification/` : 33/33 passed
- `pytest tests/` (full suite): 309 passed, 0 failures, no regressions
- Pre-existing: 8 notion_notes test errors (unrelated, `anthropic` not installed)

## CLI Usage

```bash
# Standard export
python scripts/export_raven_tables.py \
    --detections-dir USV_Detections \
    --wav-dir "5970 USV" \
    --output-dir raven_tables

# Dry run (mapping + counts only, no file writes)
python scripts/export_raven_tables.py \
    --detections-dir USV_Detections \
    --wav-dir "5970 USV" \
    --output-dir raven_tables \
    --dry-run -v
```

## Dependencies

- `pandas` (already in project) — DataFrame construction and TSV writing
- Standard library only: `json`, `logging`, `pathlib`, `dataclasses`
- No DSP, numpy, or scipy needed

## Docs Written/Updated

- `docs/modules/raven-export.md` -- created
- `docs/reviews/raven-export-handoff.md` -- created (this file)
- `docs/reviews/raven-export-review.md` -- created
- `docs/architecture/patterns.md` -- no update needed (existing patterns followed)
- `DECISIONS.md` -- no new ADRs needed

## What's Next

This adapter enables the next classification pipeline steps:
1. Run `export_raven_tables.py` on real `USV_Detections/` data
2. Import the `.Table.1.selections.txt` files into DeepSqueak for syllable classification
3. Build the reverse adapter (DeepSqueak → our format) once classification labels exist
