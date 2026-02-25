# Seed Database Module Review

**Module:** Seed Database (JSX Oil Data Extraction Pipeline)
**Review Date:** 2026-02-25
**Reviewer:** master-reviewer
**Handoff Doc:** `parts-finder/docs/reviews/seed-database-handoff.md`
**Review Tier:** 2 (Standard)

---

## Executive Summary

The implementation is correct and functionally complete. All 55 tests pass. The pipeline successfully extracts 193 entries from `oil-finder-free.jsx`, maps fuel types without errors, and imports cleanly into the SQLite database through the existing `import_data.py` pipeline. No blockers were found.

There are three warnings and four suggestions. The most important warning is a latent but real fragility in `js_object_to_json()`: the unquoted-key regex will produce invalid JSON if any string value ever contains a bare word immediately followed by a colon (e.g. `"API SP: note"`). This does not affect the current JSX file — a manual scan found zero such values — but it is one copy-paste away from a silent parse failure. The second warning is that `iter_oil_entries()` accepts `year_from` and `year_to` parameters that are never used, creating a misleading interface. The third is that the E2E test fixture opens `PartsDatabase` without a context manager or explicit `close()`, which deviates from the established pattern in every other test file.

**Verdict: CHANGES NEEDED**

---

## Test Run

```
pytest parts-finder/tests/ -v --tb=short
304 passed, 0 failed (55 new seed tests + 249 existing)
```

All 55 new tests pass. Zero regressions across the full suite.

---

## Section-by-Section Findings

### 1. `seed_from_jsx.py` — `js_object_to_json()` regex converter

**Status: PASS with one warning**

The two-pass regex approach is sound for the current OIL_DB structure. The unquoted-key pass correctly handles the `viscosity:`, `spec:`, `oem:`, `capacity:`, `interval:`, and `note:` field names. The trailing-comma pass handles nested objects. The `//` comment strip handles the `// Electric vehicle…` comment style if one were ever added inside OIL_DB.

A manual scan of all quoted string values in the current `oil-finder-free.jsx` confirms **zero** string values contain a bare-word-colon pattern. All spec values (`"API SP"`, `"ACEA C3"`, `"API CK-4, ACEA C3"`) use commas or spaces, not colons. All OEM values (`"VW 504.00/507.00"`, `"BMW LL-04"`) use slashes or hyphens. The current data is safe.

**WARNING-1: The key-quoting regex will corrupt any string value containing a bare word followed by a colon**

The regex:
```python
text = re.sub(r'(?<!["\w])([A-Za-z_]\w*)\s*:', r'"\1":', text)
```
operates on the entire JS text as a flat string — it does not distinguish between keys and values. If any string value contains `word:` (not preceded by `"` or a word character), the regex will inject double-quotes into the middle of the string, producing invalid JSON.

- **Where:** `parts-finder/scripts/seed_from_jsx.py` — the key-quoting `re.sub` call
- **Why it matters:** The failure mode is a `json.JSONDecodeError` with a confusing message. No test catches this because no current value triggers it.
- **Fix:** Add a test case documenting the limitation. Optionally tighten the regex to only match at line-start positions.

---

### 2. `seed_from_jsx.py` — `iter_oil_entries()` year parameters

**Status: PASS with one warning**

`iter_oil_entries()` accepts `year_from` and `year_to` as parameters but never uses them. The year range is applied later in `export_csv()`. The parameters are dead weight.

- **Where:** `parts-finder/scripts/seed_from_jsx.py` — `iter_oil_entries()` signature
- **Fix:** Remove dead parameters from `iter_oil_entries()`. Update callers.

---

### 3. E2E test fixture does not close `PartsDatabase`

**Status: PASS with one warning**

The `seeded_db` fixture opens `PartsDatabase(":memory:")` directly and returns it without closing. Every other test uses the context manager pattern.

- **Where:** `parts-finder/tests/test_seed_from_jsx.py` — `seeded_db` fixture
- **Fix:** Refactor fixture to use `with PartsDatabase(":memory:") as db: yield ...`

---

## Findings Summary

### BLOCKERS

None.

### WARNINGS (must fix)

| # | Finding | File | Fix |
|---|---------|------|-----|
| W1 | Key-quoting regex fragility; no test covers colon-in-value failure mode | `seed_from_jsx.py`, `test_seed_from_jsx.py` | Add test documenting limitation; tighten regex |
| W2 | `iter_oil_entries()` accepts `year_from`/`year_to` but ignores them | `seed_from_jsx.py`, callers | Remove dead parameters |
| W3 | E2E test fixture opens `PartsDatabase` without context manager | `test_seed_from_jsx.py` | Use `with ... as db: yield ...` |

### SUGGESTIONS

| # | Finding | Notes |
|---|---------|-------|
| S1 | `OilEntry` and `SeedReport` are not frozen | Low risk — neither is mutated after construction |
| S2 | Entry-count loop duplicated in orchestrator | Minor; orchestrator could consume `SeedReport` |
| S3 | No error handling for `FileNotFoundError`/`OperationalError` in CLI scripts | Affects CLI usability |
| S4 | `js_object_to_json()` does not strip `/* */` block comments | One-line fix; currently safe but latent risk |

---

## Fixes Applied (2026-02-25)

All three warnings resolved. Tests: 57 passed (was 55, +2 new).

| # | Fix | Detail |
|---|-----|--------|
| W1 | Hardened `js_object_to_json()` regex with string-protection | Quoted strings are extracted into placeholders before key-quoting runs, then restored. This prevents colons inside string values from being corrupted. Also added `/* */` block comment stripping (S4). Added `test_colon_in_string_value_preserved` and `test_block_comments_stripped` tests. `seed_from_jsx.py:92-107` |
| W2 | Removed dead `year_from`/`year_to` from `iter_oil_entries()` | Function signature now `iter_oil_entries(oil_db: dict)`. Updated callers in `seed_from_jsx.main()` and `seed_database.main()`. Renamed test to `test_iter_is_year_agnostic`. `seed_from_jsx.py:175`, `seed_database.py:79` |
| W3 | E2E fixture uses context manager with `yield` | `seeded_db` now uses `with PartsDatabase(":memory:") as db: yield ...`, matching established pattern. `test_seed_from_jsx.py:455` |

Also cleaned up: removed unused `asdict` import from `seed_from_jsx.py`.

---

## Verdict

**PASS** — All warnings addressed. 57/57 tests passing.
