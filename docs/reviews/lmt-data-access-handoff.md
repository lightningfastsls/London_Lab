# LMT Data Access Layer — Handoff

**Date:** 2026-02-24
**Review Tier:** 2 (new module with data access, timestamp math, alignment logic; no DSP/ML changes)
**Status:** Implementation complete, all tests passing

## What Was Built

A Python API for loading behavioral event annotations from Live Mouse Tracker (LMT) SQLite databases and aligning them with USV detections. Bridges LMT's 30 fps frame-based coordinate system with our 300 kHz WAV and spectrogram coordinate systems.

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/usv_spectrogram/lmt/__init__.py` | 19 | Package init with `__all__` exports |
| `src/usv_spectrogram/lmt/db_loader.py` | ~165 | Database loader: BehavioralEvent, AnimalInfo, LMTDatabaseLoader |
| `src/usv_spectrogram/lmt/synchronizer.py` | ~145 | Coordinate sync: SyncConfig, LMTSynchronizer, alignment |
| `tests/test_lmt.py` | ~290 | 21 tests across 3 test classes |

No existing files were modified.

## Architecture Decisions

### Read-only database access
LMT databases are precious experimental data. File databases open with `?mode=ro` URI flag. In-memory databases (testing) bypass this since they're ephemeral.

### Variable ANIMAL schema handling
LMT ANIMAL table has 3-9 columns across different database versions. The loader reads columns by name (via `cursor.description`) and returns None for any missing optional fields (genotype, sex, strain).

### Detection dicts, not DetectedUSV
The alignment API accepts generic dicts with `start_time`/`end_time` keys rather than coupling to our `DetectedUSV` dataclass. This makes the API reusable with any detection format.

### Time range filtering in SQL
When `time_range` is specified, seconds are converted to frame numbers before the SQL query. This lets SQLite's index do the filtering rather than loading all events into Python.

### Event specificity ranking
For the `dominant_event` selection, events are ranked: pairwise (partner_id set) > specific behavioral action > general behavioral state > environmental. This heuristic prioritizes the most informative behavioral context for USV analysis.

## Public API

```python
from usv_spectrogram.lmt import (
    LMTDatabaseLoader,  # context manager for SQLite access
    LMTSynchronizer,    # coordinate conversion + alignment
    BehavioralEvent,    # frozen dataclass: event with timing
    AnimalInfo,         # frozen dataclass: animal metadata
    SyncConfig,         # frozen dataclass: sync parameters
)
```

## What I'm Unsure About

- **`time_offset_s` abstraction**: The plan opted for a simple constant offset rather than parsing `"USV seq"` trigger events from the database. This is correct for now, but when we integrate with real data, we may need a `TriggerParser` that reads `"USV seq"` events and computes the offset automatically.
- **`IDANIMALC`/`IDANIMALD` ignored**: Multi-animal events (Group3, Nest4) involve animals C and D, but the current API only exposes `animal_id` (A) and `partner_id` (B). Sufficient for pairwise analysis but would need extension for multi-animal studies.

## Test Coverage (21 tests)

| Category | Count | What's tested |
|----------|-------|---------------|
| Loader basics | 2 | open DB, context manager |
| Animal queries | 2 | full schema, minimal (3-col) schema |
| Event queries | 5 | all events, type filter, time range, animal filter, timeline sort |
| Event types | 1 | distinct type listing |
| Coordinate conversion | 4 | frame→seconds, with offset, seconds→sample, seconds→spec frame |
| Alignment | 4 | overlap found, no overlap, multiple events, exact boundary (no overlap) |
| Config validation | 3 | defaults, invalid frame rate, invalid sample rate |

## Verification Results

- `py_compile`: All 4 new Python files compile cleanly
- `pytest tests/test_lmt.py -v`: 21/21 passed
- `pytest tests/` (full suite): 330 passed, 0 failures, no regressions
- Pre-existing: 8 notion_notes test errors (unrelated, `anthropic`/`notion_client` not installed)

## Dependencies

- `sqlite3` (standard library only)
- No external packages required

## Docs Written

- `docs/modules/lmt-data-access.md` -- module documentation
- `docs/reviews/lmt-data-access-handoff.md` -- this file
- `docs/architecture/patterns.md` -- no update needed (existing patterns followed)
- `DECISIONS.md` -- no new ADRs needed (uses existing ADR-001, ADR-002)

## What's Next

This module enables:
1. Loading behavioral annotations from real LMT databases alongside USV detections
2. Enriching USV detections with behavioral context ("what was the mouse doing?")
3. Building USV-behavior correlation analysis (e.g., "do approach events predict specific USV types?")
4. Future: `TriggerParser` to auto-compute `time_offset_s` from `"USV seq"` events
