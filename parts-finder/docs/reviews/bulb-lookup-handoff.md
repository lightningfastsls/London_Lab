# Implementation Handoff: Bulb Type Lookup

**Module:** Bulb Type Lookup (second category resolver after OilLookup)
**Review Tier:** 2 (Standard — new module, new DB method, schema change)
**Date:** 2026-02-24
**Branch:** main

## What Changed

- **Expanded VehicleSpecs bulb schema** from 4 generic fields (`headlight_bulb`, `fog_light_bulb`, `tail_light_bulb`, `turn_signal_bulb`) to 8 granular lamp positions (`low_beam_bulb`, `high_beam_bulb`, `front_turn_bulb`, `rear_turn_bulb`, `tail_brake_bulb`, `reverse_bulb`, `fog_bulb`, `license_plate_bulb`)
- **Added `BulbResult`** frozen dataclass in `models.py` with `from_vehicle_specs()` factory and `replaceable_positions` property that filters out LED and empty positions
- **Added `find_specs_by_model_year()`** DB method that queries by make + model + year, ignoring engine_code (for tier 2 fallback)
- **Created `BulbLookup`** class in new `lookup/bulbs.py` package with two-tier cascade: exact match -> model-year fallback
- **21 new tests** covering BulbResult construction, LED filtering, tier 1, tier 2, and cascade integration

## Files Changed

- `parts-finder/src/parts_finder/models.py` (MODIFIED) — replaced 4 bulb fields with 8 in VehicleSpecs; added `_BULB_POSITION_FIELDS` mapping dict and `BulbResult` frozen dataclass
- `parts-finder/src/parts_finder/db.py` (MODIFIED) — updated `_SCHEMA` (4->8 bulb columns), `_SPECS_FIELDS` list; added `find_specs_by_model_year()` method
- `parts-finder/src/parts_finder/lookup/__init__.py` (NEW) — empty package init
- `parts-finder/src/parts_finder/lookup/bulbs.py` (NEW) — `BulbLookup` class, `_has_bulb_data()` helper, `_BULB_SPEC_FIELDS` tuple
- `parts-finder/tests/test_bulb_lookup.py` (NEW) — 21 tests across 6 test classes

## Key Decisions Made

1. **BulbResult lives in `models.py`, not `bulbs.py`** — Follows the OilResult pattern where all domain result types live in the models module. The lookup module imports from models.
2. **`_BULB_POSITION_FIELDS` dict as single source of truth** — Maps BulbResult field names (e.g. `"low_beam"`) to VehicleSpecs field names (e.g. `"low_beam_bulb"`). Used by both `from_vehicle_specs()` and `replaceable_positions`, so adding a new position only requires updating one dict.
3. **Two tiers, not three** — Unlike OilLookup's three-tier cascade, bulbs skip brand-level aggregation. Oil has natural brand patterns (most Toyotas use 0W-20); bulbs are chassis-specific (Corolla H7 vs Camry H11), so brand aggregation would give meaningless results.
4. **Model-year fallback rationale** — Different engine variants on the same chassis (e.g., 2021 Corolla 2ZR-FE vs 1ZR-FE) share identical light housings. This tier relaxes only the engine_code constraint.
5. **`_resolve_names` duplicated from OilLookup** — Identical pattern. Could be extracted to a shared utility, but premature abstraction with only 2 consumers. Flag for refactoring if a third lookup class appears.
6. **"LED" as a sentinel value** — Factory-LED positions store the string `"LED"` in the database. The `replaceable_positions` property excludes these because aftermarket bulbs can't replace factory LEDs (different voltage, mounting, heat management).

## What I'm Unsure About

- **`_resolve_names` duplication** — Identical logic exists in `OilLookup._resolve_names` and `BulbLookup._resolve_names`. Currently OK with two consumers but worth noting for future refactor if more lookup classes come.
- **`find_specs_by_model_year` LIMIT 1 behavior** — If multiple engine variants exist for the same model-year in the DB, which row is returned depends on SQLite insertion order. For bulbs this is fine (same chassis = same bulbs), but it's worth documenting.
- **No index for model-year query** — The existing `idx_specs_lookup` indexes `(make, model, engine_code)`. The new `find_specs_by_model_year` query doesn't use `engine_code`, so this index is partially useful. For the current data volume this is fine, but a dedicated `(make, model)` index may be needed at scale.

## Test Results

```
pytest parts-finder/tests/ -v
133 passed, 0 failed (21 new bulb tests + 112 existing)
```

## ROADMAP Exit Criteria Status

- [x] VehicleSpecs expanded to 8 granular bulb positions
- [x] BulbResult dataclass with `replaceable_positions` filtering
- [x] `from_vehicle_specs()` factory method
- [x] `find_specs_by_model_year()` DB method
- [x] BulbLookup two-tier cascade (exact -> model-year)
- [x] LED positions filtered from replaceable results
- [x] Full test coverage (construction, properties, tier 1, tier 2, cascade, edge cases)
- [x] py_compile passes on all files
- [x] All 133 tests pass (0 regressions)

## Docs Written/Updated

- No existing docs structure in `parts-finder/` — N/A
- No `DECISIONS.md` in parts-finder yet — schema expansion rationale captured in this handoff
- No `IMPLEMENTATION_PROGRESS.md` in parts-finder yet — N/A
