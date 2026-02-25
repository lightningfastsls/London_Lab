# FastAPI Backend — Implementation Handoff

## What Was Built

Phase 6.1 of the Parts Finder project: FastAPI HTTP layer exposing `POST /api/plate-lookup` returning 7 product categories, plus the missing filter domain layer needed to complete the category set.

## Files Changed/Created

### Domain Layer (modified existing code)

| File | Change |
|------|--------|
| `models.py` | Added `FilterResult` dataclass, `_FILTER_SPEC_FIELDS`, `_FILTER_OEM_FIELDS` |
| `db.py` | Added `find_specs_by_model_year_for_filters()`, `check_same_thread` param |
| `lookup/filters.py` | **NEW** — `FilterLookup` class (2-tier cascade) |
| `lookup/__init__.py` | Added re-exports for all 4 lookup classes |
| `lookup_engine.py` | Added `filters` to `LookupResult`, wired `FilterLookup` |

### API Layer (all new)

| File | Purpose |
|------|---------|
| `api/__init__.py` | Package init |
| `api/schemas.py` | 13 Pydantic models (request, response, errors) |
| `api/response_builder.py` | `build_response()` — domain→Pydantic translation |
| `api/app.py` | `create_app()` — application factory with lifespan |
| `api/routes.py` | `plate_lookup()` — POST endpoint with caching + error handling |

### Tests

| File | Tests |
|------|-------|
| `tests/test_filters.py` | 16 tests (dataclass, exact, crossrefs, cascade, guards) |
| `tests/test_api.py` | 9 tests (full flow, partial, 400/404/503, cache, TTL expiry, schema) |

### Documentation

| File | Purpose |
|------|---------|
| `docs/modules/fastapi-backend.md` | Module documentation |
| `docs/reviews/fastapi-backend-handoff.md` | This file |

## Patterns Followed

- **FilterLookup** follows the exact same 2-tier cascade pattern as `BrakeLookup`
- **FilterResult** follows the exact same cross-reference resolution pattern as `BrakeResult`
- **Purpose-specific DB query** (`find_specs_by_model_year_for_filters`) follows coolant/brakes pattern
- **Test factories** (`_make_specs`, `_make_vehicle`) reused from existing test files
- **Response builder** separates domain→API translation (no Pydantic in domain code)

## Validation Results

- `py_compile` passes for all 10 new/modified files
- **331 tests passing** (304 existing + 16 filter + 9 API + 2 from db param)
- Zero test failures
- No existing tests broken by domain layer changes

## Known Limitations

1. **`data_source` is always `"database"`** — AI fallback (Phase 6.2) not implemented
2. **No CORS middleware** — Phase 7.1 will add it for frontend
3. **No rate limiting or auth** — internal tool, not needed yet
4. **In-memory cache only** — fine for single-process; need Redis for multi-worker

## Review Tier

**Tier 2** (ROADMAP spec) — new module with external interface (HTTP API).

## What to Review

1. **FilterResult.from_vehicle_specs** — cross-reference resolution follows BrakeResult pattern exactly
2. **Response builder coverage computation** — counts non-None categories out of 7
3. **Cache TTL logic** — uses `time.monotonic()`, removes stale entries on check
4. **Error mapping** — ValueError→400, PlateNotFoundError→404, GovApiError→503
5. **API test mocking** — PlateClient patched at import site, TestClient with direct state injection
