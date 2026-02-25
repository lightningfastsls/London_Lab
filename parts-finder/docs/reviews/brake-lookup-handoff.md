# Implementation Handoff: Brake Parts Lookup

**Module:** Brake Parts Lookup (fourth category resolver after OilLookup, BulbLookup, CoolantLookup)
**Review Tier:** 2 (Standard — new module, new DB method, new model type)
**Date:** 2026-02-24
**Branch:** main

## What Changed

- **Added `BrakeResult`** frozen dataclass in `models.py` with `from_vehicle_specs()` classmethod that resolves cross-references at construction time. Fields for front/rear pad/disc OEM numbers, brake fluid type, and cross-reference tuples.
- **Added `_BRAKE_SPEC_FIELDS` and `_BRAKE_OEM_FIELDS`** module-level constants for field mapping and data-presence checking.
- **Added `find_specs_by_model_year_for_brakes()`** DB method — same as `find_specs_by_model_year` but ORDER BY prefers records with `front_brake_pad_oem != ''`.
- **Created `BrakeLookup`** class in new `lookup/brakes.py` with two-tier cascade: exact match -> model-year fallback. No brand default tier (brake OEM numbers are too vehicle-specific to aggregate by brand).
- **18 new tests** covering BrakeResult dataclass, exact match, cross-references, model-year fallback, and cascade integration.

## Files Changed

- `parts-finder/src/parts_finder/models.py` (MODIFIED) — added `BrakeResult` frozen dataclass, `_BRAKE_SPEC_FIELDS` mapping, `_BRAKE_OEM_FIELDS` tuple
- `parts-finder/src/parts_finder/db.py` (MODIFIED) — added `find_specs_by_model_year_for_brakes()` method
- `parts-finder/src/parts_finder/lookup/brakes.py` (NEW) — `BrakeLookup` class, `_has_brake_data()` helper
- `parts-finder/tests/test_brake_lookup.py` (NEW) — 18 tests across 5 test classes

## Key Decisions Made

1. **Two-tier cascade only (no brand default)** — Unlike oil and coolant where brand-level defaults are meaningful (all Toyota use SLLC coolant, most Toyota use 0W-20 oil), brake pad OEM numbers vary per model, year, and trim. A brand-level default would be misleading.
2. **Cross-references resolved at construction time via `db` parameter** — `BrakeResult.from_vehicle_specs()` takes a `db` parameter (typed as `object` to avoid circular import) and calls `db.find_crossrefs(oem)` for each non-empty OEM number. This keeps the DB layer generic while letting BrakeResult be a complete, self-contained result.
3. **Tuple fields for cross-references** — Frozen dataclasses require hashable default values. `tuple[ProductCrossRef, ...]` works; `list` would not. This also enforces immutability throughout.
4. **`has_data` excludes brake fluid alone** — Brake fluid type without pad/disc OEM numbers isn't actionable for cross-referencing. Only OEM numbers count for "has data."
5. **"disc" vs "rotor" terminology** — User-facing fields use "disc" (industry standard: brake disc), while DB columns use "rotor" (existing schema convention). `_BRAKE_SPEC_FIELDS` maps between them.
6. **`_resolve_names` still duplicated (4th consumer)** — The coolant handoff flagged this as overdue for extraction at 3 consumers. Now at 4 consumers. Should be extracted to a shared utility in a dedicated refactoring task.

## Limitations (Documented, Not Implemented)

- **No disc diameter** — The `vehicle_specs` table has no `front_disc_dia_mm` / `rear_disc_dia_mm` columns. Needs a future schema migration.
- **No brake type** — No `front_brake_type` / `rear_brake_type` columns (ventilated/solid/drum). Also needs schema work.

## What I'm Unsure About

- **`_resolve_names` refactor is now critical** — Four identical copies exist (oil, bulbs, coolant, brakes). Each is ~15 lines. Should be extracted to `lookup/_shared.py` or a `BaseLookup` class.
- **Cross-ref resolution in the classmethod** — Passing `db` to a dataclass factory is slightly unusual. Alternative: resolve cross-refs in the lookup class and pass them to the constructor. Current approach is more cohesive (all BrakeResult construction logic in one place) but couples the model to the DB interface.
- **No index for brake model-year query** — Same as bulb and coolant: the existing `idx_specs_lookup` indexes `(make, model, engine_code)`. Fine at current data volume but worth noting if the DB grows.

## Test Results

```
pytest parts-finder/tests/ -v
234 passed, 0 failed (18 new brake tests + 216 existing)
```

## Exit Criteria Status

- [x] BrakeResult frozen dataclass with from_vehicle_specs classmethod
- [x] has_data property (excludes fluid-only)
- [x] _BRAKE_SPEC_FIELDS and _BRAKE_OEM_FIELDS constants
- [x] find_specs_by_model_year_for_brakes() DB method
- [x] BrakeLookup two-tier cascade (exact -> model_year)
- [x] Cross-reference resolution via db.find_crossrefs()
- [x] Partial results (some OEM fields empty, still returns BrakeResult)
- [x] Full test coverage (dataclass, exact match, cross-refs, model-year, cascade, edge cases)
- [x] py_compile passes on all files
- [x] All 234 tests pass (0 regressions)

## Docs Written/Updated

- `parts-finder/docs/reviews/brake-lookup-handoff.md` — this file
