# FastAPI Backend (Phase 6.1)

## Overview

HTTP API layer that wraps the Parts Finder domain into a single `POST /api/plate-lookup` endpoint. Takes an Israeli license plate number and returns vehicle identification plus parts recommendations across 7 product categories.

## Architecture

```
POST /api/plate-lookup {"plate": "1234567"}
  │
  ├── Pydantic validates request body
  ├── normalize_plate() → 400 if invalid
  ├── Cache check → return cached response if hit
  ├── engine.lookup(plate) → LookupResult
  │     ├── PlateClient resolves plate (async HTTP to data.gov.il)
  │     ├── NameMapper enriches Hebrew → English
  │     └── 5 category lookups: oil, coolant, bulbs, brakes, filters
  ├── build_response(result) → PlateLookupResponse
  │     ├── Translates domain models to Pydantic schemas
  │     └── Computes coverage (count non-None categories out of 7)
  └── Cache store + return JSON
```

## Key Files

| File | Purpose |
|------|---------|
| `src/parts_finder/api/__init__.py` | Package init |
| `src/parts_finder/api/schemas.py` | Pydantic request/response models (API contract) |
| `src/parts_finder/api/response_builder.py` | Domain → Pydantic translation layer |
| `src/parts_finder/api/app.py` | FastAPI application factory (`create_app`) |
| `src/parts_finder/api/routes.py` | Route handler for `POST /api/plate-lookup` |
| `src/parts_finder/lookup/filters.py` | FilterLookup (new, 2-tier cascade) |
| `tests/test_api.py` | 8 API endpoint tests |
| `tests/test_filters.py` | 16 FilterLookup unit tests |

## Domain Layer Changes

### FilterResult (models.py)

New frozen dataclass mirroring `BrakeResult` pattern:
- 4 OEM fields: `oil_filter_oem`, `air_filter_oem`, `cabin_filter_oem`, `fuel_filter_oem`
- 4 cross-reference tuples for aftermarket equivalents
- `has_data` property and `from_vehicle_specs` classmethod

### FilterLookup (lookup/filters.py)

Two-tier cascade following established pattern:
1. **Exact match** — make + model + year + engine_code
2. **Model-year fallback** — make + model + year (any engine)

### LookupResult (lookup_engine.py)

Added `filters: Optional[FilterResult] = None` — backwards-compatible.

### PartsDatabase (db.py)

- Added `find_specs_by_model_year_for_filters()` — purpose-specific query preferring rows with `air_filter_oem != ''`
- Added `check_same_thread` parameter to constructor (default `True`, used in tests)

## API Response Format

```json
{
  "vehicle": {
    "plate": "1234567",
    "make": "Toyota",
    "model": "Corolla",
    "year": 2021,
    "engine_code": "2ZR-FE",
    "fuel_type": "petrol"
  },
  "categories": {
    "oil": { "viscosity": "0W-20", "capacity_l": 4.2, ... },
    "oil_filter": { "oem_part_number": "04152-YZZA1", "aftermarket": [...] },
    "air_filter": { "oem_part_number": "17801-21050", "aftermarket": [...] },
    "cabin_filter": { "oem_part_number": "87139-02020", "aftermarket": [...] },
    "brakes": { "front_pad_oem": "04465-02220", ... },
    "bulbs": { "low_beam": "H11", "high_beam": "HB3", ... },
    "coolant": { "spec": "Toyota SLLC", "color": "pink", ... }
  },
  "coverage": "7/7 categories matched",
  "unmatched_categories": [],
  "data_source": "database"
}
```

## Error Handling

| Error | HTTP Status | Source |
|-------|------------|--------|
| Invalid plate format | 400 | `ValueError` from `normalize_plate()` |
| Plate not in registry | 404 | `PlateNotFoundError` from `PlateClient` |
| Government API down | 503 | `GovApiError` from `PlateClient` |
| Empty plate body | 422 | Pydantic validation |

## Caching

In-memory dict keyed on normalized plate string → `(PlateLookupResponse, timestamp)`.
Uses `time.monotonic()` for TTL. Configured via `AppConfig.cache_ttl_minutes` (default 60 minutes).

**TTL note:** The ROADMAP says "cache indefinitely" since vehicles don't change. The implementation uses a 60-minute TTL instead, for two practical reasons: (1) prevents stale data if the DB is re-seeded while the server is running, and (2) bounds memory growth for long-running processes. To approximate "indefinite" caching, increase `cache_ttl_minutes` to a large value (e.g. 525600 for one year). Expired entries are evicted lazily on the next lookup for the same plate.

## Design Decisions

1. **Separate Pydantic models** — domain dataclasses stay frozen and pure; API models handle validation/serialization
2. **Response builder** — centralises all domain→API translation in one module
3. **`@asynccontextmanager` lifespan** — modern FastAPI pattern (not deprecated `on_event`)
4. **`data_source: "database"` always** — AI fallback is Phase 6.2 (not implemented yet)
5. **60-minute cache TTL** — see Caching section above for rationale

## Test Coverage

- 8 API tests covering all 7 ROADMAP test cases + Pydantic validation
- 16 FilterLookup tests following exact same structure as `test_brake_lookup.py`
- Full suite: **330 tests passing** (304 existing + 24 new + 2 bonus)
