# AI Fallback Module Review

**Module:** AI Fallback (Phase 6.2)
**Review Date:** 2026-02-25
**Reviewer:** master-reviewer
**Handoff Doc:** `parts-finder/docs/reviews/ai-fallback-handoff.md`
**Review Tier:** 2 (Standard -- new async API integration, response merging logic; no DSP/ML changes)

---

## Executive Summary

The implementation is structurally sound. The `AIFallback` class is well-designed: lazy import, graceful degradation on all error paths, selective prompting that reduces token usage, JSONL miss logging, and correct retry semantics. The `AIFallbackResult` frozen dataclass follows the established project pattern. All 18 original tests pass and there are zero regressions.

Five warnings and three suggestions were identified. All five have been fixed (see Fixes Applied below). The most significant find was W3: the `build_response(result, ai_result=...)` merge path had zero test coverage, and writing those tests exposed a real bug in the tri-state `data_source` computation (now fixed).

**Verdict: PASSED (after fixes)**

---

## Findings Summary

### WARNINGS (all fixed)

| # | Finding | File | Status |
|---|---------|------|--------|
| W1 | No markdown-fence sanitization before `json.loads()` | `fallback.py:156-162` | FIXED |
| W2 | `message.content[0]` IndexError treated as unexpected error | `fallback.py:156` | FIXED |
| W3 | `build_response(result, ai_result=...)` path had zero test coverage | `response_builder.py:178-214` | FIXED |
| W4 | `data_source` field description stale in Swagger UI | `schemas.py:132-138` | FIXED |
| W5 | `misses_log_path` not validated for empty string | `config.py:53-56` | FIXED |

### SUGGESTIONS (all applied)

| # | Finding | Status |
|---|---------|--------|
| S1 | `import asyncio` at method scope 11 times | FIXED -- moved to module level |
| S2 | Tests import private `_EMPTY_RESULT` | FIXED -- uses `AIFallbackResult()` |
| S3 | `len(matched) > 0` vacuous guard | FIXED -- removed |

---

## Fixes Applied

### W1: Markdown-fence sanitization
- **File:** `parts-finder/src/parts_finder/api/fallback.py:158-160`
- **Change:** Added fence stripping before `json.loads()`. If `raw_text` starts with triple backticks, the first line (fence opener) and trailing fence are stripped.
- **Test added:** `TestMalformedJSON::test_markdown_fenced_json_is_parsed` -- mocks API returning `` ```json\n{...}\n``` ``, asserts result is parsed correctly.

### W2: Empty content guard
- **File:** `parts-finder/src/parts_finder/api/fallback.py:156-158`
- **Change:** Added explicit `if not message.content:` guard that raises `json.JSONDecodeError`, routing empty-content responses through the retry path (not the immediate `break`).
- **Test added:** `TestMalformedJSON::test_empty_content_list_retries` -- mocks empty `message.content = []`, asserts `call_count == max_retries` (retried, not broken).

### W3: build_response AI merge test coverage + bug fix
- **File created:** `parts-finder/tests/test_response_builder.py` (~170 lines, 7 tests)
- **Tests added:**
  1. `test_db_only_data_source_database` -- no AI result -> "database"
  2. `test_ai_only_data_source_ai_fallback` -- DB empty, AI fills -> "ai_fallback"
  3. `test_mixed_data_source_hybrid` -- DB + AI -> "hybrid"
  4. `test_ai_does_not_overwrite_db_match` -- DB oil preserved over AI oil
  5. `test_unmatched_categories_shrinks_after_ai_fill` -- 7 -> 5 after AI fills 2
  6. `test_coverage_string_reflects_ai_contribution` -- "3/7" counts AI
  7. `test_ai_result_with_no_useful_data` -- empty AI -> "database"
- **Bug found and fixed:** `response_builder.py:197-206` -- the tri-state `data_source` check used `getattr(ai_result, name, None) is None` to detect DB-only categories, but this failed when AI also returned data for a category the DB already had. Fixed by tracking `db_provided` set before AI merge and using `elif db_provided:` for the hybrid check.

### W4: Schema description update
- **File:** `parts-finder/src/parts_finder/api/schemas.py:132-138`
- **Change:** Replaced stale "Always 'database' until Phase 6.2 adds AI fallback" with accurate tri-state description.

### W5: Config validation
- **File:** `parts-finder/src/parts_finder/config.py:55-56`
- **Change:** Added `if not self.misses_log_path.strip(): raise ValueError(...)` to `__post_init__`.
- **Tests added:** `test_empty_misses_path_raises` and `test_whitespace_misses_path_raises` in `test_config.py`.

### S1: Module-level asyncio import
- **File:** `parts-finder/tests/test_fallback.py:17`
- **Change:** Moved `import asyncio` from 11 method-scope locations to module-level import block.

### S2: Remove private `_EMPTY_RESULT` import
- **File:** `parts-finder/tests/test_fallback.py:25, 434, 451`
- **Change:** Replaced `_EMPTY_RESULT` with `AIFallbackResult()` in all test assertions.

### S3: Remove vacuous guard
- **File:** `parts-finder/src/parts_finder/api/response_builder.py:200`
- **Change:** Removed `len(matched) > 0 and` from hybrid check (was always true when `used_ai=True`). Subsequently replaced entire check with simpler `elif db_provided:` as part of W3 bug fix.

---

## Verification Results

- `pytest parts-finder/tests/ -v`: **360 passed, 0 failures** (was 349 before; +11 new tests)
- No regressions across any existing test module
