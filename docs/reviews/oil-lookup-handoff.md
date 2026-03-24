# Implementation Handoff: Oil Specification Lookup Module

**Module:** Oil Specification Lookup (parts-finder)
**Review Tier:** 2 (new module + new DB queries + schema change + tests, no DSP/ML)
**Date:** 2026-02-24
**Branch:** main

## What Was Built

A three-tier cascade lookup module that resolves oil specifications for a vehicle from the parts database. Given a `VehicleRecord` (from the government API), it progressively relaxes matching criteria until a result is found or all tiers are exhausted:

1. **Exact match** (Tier 1) — make + model + year + engine_code
2. **Engine family fallback** (Tier 2) — make + engine prefix (e.g. "2ZR" from "2ZR-FE") + year
3. **Brand default** (Tier 3) — most common oil spec aggregated across all models for the make

Also added three oil-related fields to the `VehicleSpecs` schema that were present in the source data (`oil-finder-free.jsx` OIL_DB) but not yet captured: API/ACEA spec codes, OEM approvals, and change interval.

## Files Changed

| File | Action | Lines | Purpose |
|------|--------|-------|---------|
| `parts-finder/src/parts_finder/models.py` | MODIFIED | 237 | Added 3 oil fields to `VehicleSpecs`; new `OilResult` frozen dataclass with `from_vehicle_specs` and `from_brand_default` classmethods |
| `parts-finder/src/parts_finder/db.py` | MODIFIED | 249 | Added 3 columns to `_SCHEMA` DDL; updated `_SPECS_FIELDS`; new `find_specs_by_engine_family()` and `find_brand_default_oil()` methods |
| `parts-finder/data/oil_specs_sample.csv` | MODIFIED | 4 | Added `oil_spec`, `oil_oem_approval`, `oil_change_interval_km` columns to all 3 existing rows |
| `parts-finder/src/parts_finder/oil_lookup.py` | NEW | 159 | `extract_engine_family()` function + `OilLookup` class with 3-tier cascade |
| `parts-finder/tests/test_oil_lookup.py` | NEW | 316 | 23 tests across 6 test classes |

No existing test files were modified. No test expectations were changed.

## Key Decisions Made

### 1. Engine family extraction is conservative (dash-only)
Only dash-separated engine codes are split (e.g. "2ZR-FE" -> "2ZR"). Dashless codes like "G4FJ" or "EA888" return `None` rather than guessing at a prefix boundary. This means Tier 2 is skipped for dashless codes, falling through to brand default. **Rationale:** Incorrect family matching would silently return wrong specs (e.g. wrong oil capacity), which is worse than admitting "I don't know" and falling to aggregate data.

### 2. Brand default sets capacity to 0.0
`OilResult.from_brand_default()` explicitly zeros out `capacity_l` because oil capacity varies per engine — aggregating it would produce a misleading number. Zero signals "verify this yourself." Viscosity and spec code, by contrast, are often consistent across a brand's lineup and can be usefully aggregated.

### 3. Empty oil_viscosity treated as "no match"
Each cascade tier checks that the matched `VehicleSpecs` has a non-empty `oil_viscosity` before accepting it. A record that exists in the DB (e.g. for brake/filter data) but has no oil data doesn't short-circuit the cascade — it falls through to the next tier. This prevents exact-match records with incomplete data from blocking more informative matches.

### 4. Name resolution prefers English, falls back to Hebrew
`_resolve_names()` uses `make_english`/`model_english` from the `VehicleRecord` (populated by `NameMapper` enrichment) and falls back to `make_hebrew` if English isn't available. A warning is logged when Hebrew fallback is used, since DB matches with Hebrew names are unlikely.

### 5. OilResult lives in models.py, not oil_lookup.py
Follows the existing frozen-dataclass-in-models pattern (`VehicleRecord`, `VehicleSpecs`, `ProductCrossRef` all live in `models.py`). The lookup module imports from models, not the reverse.

### 6. New DB queries live in PartsDatabase
`find_specs_by_engine_family()` and `find_brand_default_oil()` are methods on `PartsDatabase`, keeping data access centralized. Same pattern as existing `find_specs()` / `find_crossrefs()`.

## What I'm Unsure About

- **Engine family LIKE query performance**: `WHERE engine_code LIKE '2ZR%'` uses a prefix pattern which SQLite can optimize with the existing index on `(make, model, engine_code)`. But the query only filters on `make` and `engine_code` (not `model`), so the index may not be as effective. For the current data volume this is irrelevant, but worth noting if the DB grows significantly.

- **Brand default aggregation ties**: `find_brand_default_oil()` uses `ORDER BY COUNT(*) DESC LIMIT 1`. If two viscosity/spec combinations are equally common, SQLite picks one non-deterministically. This could cause different results on different runs with the same data. Probably fine in practice (brand lineups tend to converge on one spec), but worth being aware of.

- **OEM approval not included in brand default**: `from_brand_default()` sets `oem_approval=""` because OEM approvals are engine-specific and can't be aggregated. But the `from_brand_default` classmethod signature doesn't even accept it as a parameter — if we later want to include a "most common OEM approval" that would need a signature change.

## Test Coverage (23 tests, 6 classes)

| Class | Count | What's Tested |
|-------|-------|---------------|
| `TestExtractEngineFamily` | 5 | dash code, multi-dash, no-dash->None, empty->None, leading-dash->None |
| `TestOilResultConstruction` | 3 | from_vehicle_specs fields, from_brand_default capacity=0, frozen immutability |
| `TestOilLookupExactMatch` | 4 | returns result, correct source string, confidence="exact", no match->None |
| `TestOilLookupEngineFamilyFallback` | 3 | "2ZR" matches "2ZR-FE", correct confidence, dashless skips tier |
| `TestOilLookupBrandDefaultFallback` | 3 | most common spec returned, capacity=0, correct confidence |
| `TestOilLookupCascade` | 5 | exact short-circuits, full cascade, no make->None, empty engine->brand default, empty oil data cascades |

## Verification Results

- `py_compile`: All 4 Python files (models.py, db.py, oil_lookup.py, test_oil_lookup.py) compile cleanly
- `pytest parts-finder/tests/test_oil_lookup.py -v`: **23/23 passed** (0.07s)
- `pytest parts-finder/tests/ -v`: **104/104 passed** (1.25s) — zero regressions
- Import data tests pass without modification (dynamic field discovery via `_SPECS_FIELDS` + dataclass type annotations propagates new columns automatically)

## Public API

```python
from parts_finder.oil_lookup import OilLookup, extract_engine_family
from parts_finder.models import OilResult

# Standalone utility
extract_engine_family("2ZR-FE")  # -> "2ZR"
extract_engine_family("G4FJ")    # -> None

# Cascade lookup
with PartsDatabase("parts.db") as db:
    oil = OilLookup(db)
    result = oil.lookup(vehicle_record)
    if result:
        print(f"{result.viscosity} ({result.confidence})")
        # e.g. "0W-20 (exact)" or "5W-30 (brand_default)"

# OilResult construction
OilResult.from_vehicle_specs(specs, confidence="exact")
OilResult.from_brand_default(make="Toyota", viscosity="0W-20",
                             spec="API SP", change_interval_km=15000)
```

## Dependencies

- Standard library only: `logging`, `dataclasses`, `sqlite3`
- Internal: `parts_finder.db.PartsDatabase`, `parts_finder.models.{VehicleRecord, VehicleSpecs, OilResult}`
- No new external packages required

## Docs Written/Updated

- `docs/reviews/oil-lookup-handoff.md` — this file
- No module doc created (parts-finder doesn't have a `docs/modules/` convention yet)
- No DECISIONS.md or patterns.md updates needed (no architectural departures)

## What's Next

This module is a building block for the parts-finder API layer. Next steps:
1. Wire `OilLookup` into the FastAPI endpoint that accepts a plate number
2. Build similar lookup modules for other categories (filters, brakes, bulbs) following the same cascade pattern
3. Populate the DB with production data from the `oil-finder-free.jsx` OIL_DB source
