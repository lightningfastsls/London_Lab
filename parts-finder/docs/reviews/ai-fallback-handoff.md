# AI Fallback Module — Handoff

**Date:** 2026-02-25
**Review Tier:** 2 (new module with async API integration, response merging logic; no DSP/ML changes)
**Status:** Implementation complete, all tests passing

## What Was Built

An AI fallback layer that calls Claude Haiku to generate vehicle parts specifications when the local database has no data for certain categories. Results are flagged with `source="ai_fallback"` for UI confidence badging, and every database miss is logged to a JSONL file for manual review and progressive database enrichment.

This fulfills the Phase 6.2 placeholder that existed since the initial API design (`schemas.py:134`: "until Phase 6.2 adds AI fallback").

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `parts-finder/src/parts_finder/api/fallback.py` | ~260 | `AIFallbackResult` dataclass + `AIFallback` class: prompt building, API call with retries, response parsing, JSONL miss logging |
| `parts-finder/tests/test_fallback.py` | ~300 | 18 tests across 7 test classes, all mocking the Anthropic API |

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `parts-finder/src/parts_finder/config.py` | +2 fields | `ai_fallback_enabled: bool` and `misses_log_path: str` for runtime control |
| `parts-finder/src/parts_finder/api/response_builder.py` | Modified `build_response()` signature + merge logic | Accepts optional `AIFallbackResult`, merges AI data into unmatched categories, computes tri-state `data_source` |
| `parts-finder/src/parts_finder/api/routes.py` | +10 lines in `plate_lookup()` | Calls fallback after initial response when unmatched categories exist |
| `parts-finder/src/parts_finder/api/app.py` | +7 lines in `_lifespan()` | Initializes `AIFallback` on `app.state` if `ANTHROPIC_API_KEY` env var is set and config allows it |

## Architecture Decisions

### Lazy import of anthropic SDK
The `from parts_finder.api.fallback import AIFallback` import is inside `_lifespan()`, not at module level. This means the app starts cleanly on machines without the `anthropic` package installed when fallback is not needed.

### Two-pass response building
Rather than deeply mutating the response, the route calls `build_response(result)` first (DB-only), then if unmatched categories exist and AI returns data, calls `build_response(result, ai_result=ai_result)` again. This keeps `build_response` a pure function — same inputs always produce the same output.

### Tri-state data_source
`"database"` (all from DB), `"hybrid"` (mix of DB + AI), or `"ai_fallback"` (everything from AI). The hybrid determination checks whether any matched category came from the database while AI filled others.

### Selective category prompting
The prompt only asks Claude for the specific unmatched categories, not all 7. This reduces token usage by ~50-70% on typical partial-miss lookups (e.g., asking for 2 categories instead of 7).

### Retry strategy with break on unknown errors
`anthropic.APIError` and `json.JSONDecodeError` are retried (transient/recoverable), but unexpected exceptions break immediately. This prevents retry loops on bugs in our own parsing code.

### JSONL miss logging
JSON Lines format (one JSON object per line) is append-only, grep-friendly, and trivially parseable. Each line is self-contained with timestamp, vehicle info, unmatched categories, and the raw AI response (or `null` on failure).

## Public API

```python
from parts_finder.api.fallback import AIFallback, AIFallbackResult

# Initialization (done automatically in app lifespan)
fallback = AIFallback(config)

# Usage (done automatically in route handler)
result: AIFallbackResult = await fallback.generate_specs(vehicle, ["oil", "brakes"])

# Check if AI returned anything useful
if result.has_data:
    response = build_response(lookup_result, ai_result=result)
```

## What I'm Unsure About

- **`_CATEGORY_SCHEMAS` as inline strings**: The per-category JSON schema examples are stored as string literals inside `fallback.py`. If these schemas diverge from the actual Pydantic models, there's no compile-time check. An alternative would be generating schema examples from the Pydantic models programmatically, but that adds complexity for 7 stable schemas.
- **No JSON mode flag**: The current implementation relies on the system prompt instructing the model to return JSON. Anthropic's API does not have a native `response_format=json` parameter like OpenAI. If the model occasionally returns markdown-wrapped JSON, the `json.loads()` will fail and trigger a retry. This has been acceptable in testing but could be improved with a response-text sanitizer (strip markdown fences) if it becomes an issue in production.

## Test Coverage (18 tests)

| Category | Count | What's tested |
|----------|-------|---------------|
| Happy path | 2 | Oil+brakes populated; all 7 categories populated |
| API error handling | 2 | APIError returns empty result; retry count matches config |
| Malformed JSON | 2 | Unparseable text returns empty; wrong types skipped |
| Selective categories | 2 | Only requested categories populated; empty unmatched skips API call |
| Miss logging | 3 | Successful response logged; failed response logged with null; multiple misses appended |
| Config wiring | 4 | Default model; custom model; max_retries; misses_path |
| AIFallbackResult | 3 | Empty has no data; with oil has data; frozen immutability |

## Verification Results

- `py_compile`: All 6 modified/new Python files compile cleanly
- `pytest parts-finder/tests/test_fallback.py -v`: 18/18 passed
- `pytest parts-finder/tests/ -v` (full suite): 349 passed, 0 failures, no regressions

## Dependencies

- `anthropic` (already in `requirements.txt`, newly installed in venv)
- No other new external packages

## Docs Written

- `docs/reviews/ai-fallback-handoff.md` — this file

## What's Next

This module enables:
1. **UI confidence badges** — frontend can display "AI-generated" warning when `data_source` is `"ai_fallback"` or `"hybrid"`
2. **Database self-improvement** — `data/misses.jsonl` can be reviewed periodically, verified AI results inserted into the DB, progressively reducing the miss rate
3. **Monitoring** — miss log analysis to identify which makes/models/categories are most commonly unmatched, guiding data collection priorities
4. **Future: structured output** — when Anthropic adds a JSON mode API parameter, the retry-on-parse-failure path can be simplified
