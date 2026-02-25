# FastAPI Backend — Review

## Review Tier

**Tier 2** (ROADMAP spec) — new module with external interface (HTTP API).

## Verdict: PASS (after fixes)

All warnings addressed. 331 tests passing. No blockers.

## Findings

### Warnings (all fixed)

| # | Finding | Severity | Fix |
|---|---------|----------|-----|
| W1 | `coolant_type="Toyota SLLC"` in full-flow test doesn't match `_COOLANT_SPECS` key `"SLLC"` — test passed via brand-default fallback, not the DB path | Medium | Changed to `coolant_type="SLLC"` + added `capacity_l==6.4` assertion to verify DB path |
| W2 | `demo_lookup.py` not updated with `_format_filters()` — Filters still listed as future category | Medium | Added `_format_filters()`, updated `_FUTURE_CATEGORIES` (3→2), updated format_report tests in `test_lookup_engine.py` (placeholder count 3→2, no-match count 4→5) |
| W3 | `ai_model` in AppConfig defaults to Sonnet; ROADMAP Phase 6.2 specifies Haiku for AI fallback | Low | Changed to `claude-3-5-haiku-20241022` (dormant until Phase 6.2) |
| W4 | Cache uses 60-minute TTL; ROADMAP says "cache indefinitely" | Low | Documented TTL rationale in `docs/modules/fastapi-backend.md` — practical reasons: DB re-seeding, memory bounds |
| W5 | Cache TTL expiry branch untested | Medium | Added `test_expired_cache_triggers_refetch` — backdates cache timestamp, verifies PlateClient called again |

### Suggestions

| # | Finding | Action |
|---|---------|--------|
| S1 | Local `from parts_finder.api.routes import router` inside `create_app()` | Moved to module-level import (no circular dependency risk) |
| S2 | FilterResult placed before BrakeResult in `models.py` (the template it was modeled after) | Skipped — cosmetic reordering risk outweighs benefit; `from __future__ import annotations` makes order irrelevant for type checking |

## Validation After Fixes

- `py_compile` passes for all modified files
- **331 tests passing** (304 existing + 16 filter + 9 API + 2 bonus)
- Zero failures
