# Oil Specification Lookup Module Review

**Reviewed by:** Master Reviewer (Sonnet 4.6)
**Date:** 2026-02-24
**Module:** `parts-finder` — Oil Specification Lookup
**Tier:** 2 (Standard — new module + schema change + new DB queries + tests)
**Verdict:** APPROVED (all warnings fixed — see Fix Log)

---

## Test Results

- `pytest parts-finder/tests/test_oil_lookup.py -v`: **23/23 passed** (0.10s)
- `pytest parts-finder/tests/ -v`: **104/104 passed** (0.53s) — zero regressions

---

## BLOCKER (must fix before next module)

None.

---

## WARNINGS (fix soon)

### W1. LIKE injection via unescaped `%` and `_` in engine prefix

**File:** `parts-finder/src/parts_finder/db.py:180-189` (`find_specs_by_engine_family`)

**Problem:** `find_specs_by_engine_family` appends `%` to the engine prefix and passes it directly to a LIKE query:

```python
(make, f"{engine_prefix}%", year, year),
```

`extract_engine_family` strips only at the dash and does not sanitize the prefix it returns. If the government API ever returns an engine code containing `%` or `_` (SQLite LIKE wildcards), the prefix will carry them into the query. For example, `extract_engine_family("2%R-FE")` returns `"2%R"`, and `"2%R%"` in LIKE matches any engine code of the form `2<any chars>R<any chars>`, which could return a record for a completely unrelated engine. This is confirmed executable: in a manual trace the query `WHERE engine_code LIKE '2%R%'` matches `2ZR-FE`.

Real-world engine codes from `data.gov.il` are unlikely to contain `%` or `_`, but no upstream validation blocks it. VehicleRecord's `engine_code` field accepts any string from the API.

**Fix:** Escape LIKE wildcards in the prefix before building the pattern, or validate that `extract_engine_family` only returns alphanumeric/hyphen-free prefixes:

Option A — escape in `find_specs_by_engine_family`:
```python
# Escape LIKE special characters before adding the wildcard suffix
escaped_prefix = engine_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
row = self._conn.execute(
    "SELECT * FROM vehicle_specs"
    " WHERE make = ? AND engine_code LIKE ? ESCAPE '\\'"
    " AND year_from <= ? AND year_to >= ?"
    " LIMIT 1",
    (make, f"{escaped_prefix}%", year, year),
)
```

Option B — validate in `extract_engine_family` (simpler, preferred):
```python
import re
_SAFE_PREFIX_RE = re.compile(r'^[A-Za-z0-9]+$')

def extract_engine_family(engine_code: str) -> str | None:
    ...
    prefix = code.split("-", 1)[0]
    if not prefix or not _SAFE_PREFIX_RE.match(prefix):
        return None
    return prefix
```

Option B is preferred because it also protects future callers who might use the prefix for other purposes.

---

### W2. `test_specs_with_no_oil_data_cascades` comment is misleading about engine family tier behavior

**File:** `parts-finder/tests/test_oil_lookup.py:296-312` (`test_specs_with_no_oil_data_cascades`)

**Problem:** The test comment says:

```python
# Should cascade past the exact (empty oil) match into engine family
# "3ZR" matches "3ZR-FE" but that's the empty one, so further
# falls to brand_default (from the 2ZR-FE record)
self.assertIn(result.confidence, ("engine_family", "brand_default"))
```

The comment says `("engine_family", "brand_default")` is acceptable, implying the engine family tier might return a result. But manual tracing shows the engine family query for prefix `"3ZR"` with `LIKE '3ZR%'` only matches `3ZR-FE`, which has empty `oil_viscosity` — so that tier also falls through. The actual result is always `"brand_default"`. The assertion is too permissive: it would pass even if the engine family tier incorrectly returned a result for a record with empty oil data — a bug the test is supposed to catch.

More critically, the assertion `assertIn(result.confidence, ("engine_family", "brand_default"))` does not actually verify that the cascade behaved correctly — it would pass if a bug caused tier 2 to short-circuit with an empty-oil match (which would be wrong). The test should assert `assertEqual(result.confidence, "brand_default")`.

**Fix:**
```python
def test_specs_with_no_oil_data_cascades(self) -> None:
    """VehicleSpecs with empty oil_viscosity is treated as no match at every tier."""
    self.db.insert_specs(_make_specs(
        model="Corolla", engine_code="3ZR-FE",
        year_from=2019, year_to=2023,
        oil_viscosity="",
        oil_capacity_l=0.0,
    ))
    vehicle = _make_vehicle(engine_code="3ZR-FE")
    result = self.oil.lookup(vehicle)
    # Tier 1: exact match (3ZR-FE) has empty oil -> skipped
    # Tier 2: engine family "3ZR" LIKE '3ZR%' also matches 3ZR-FE -> still empty -> skipped
    # Tier 3: brand_default uses 2ZR-FE record -> 0W-20
    self.assertIsNotNone(result)
    self.assertEqual(result.confidence, "brand_default")
    self.assertEqual(result.viscosity, "0W-20")
```

---

### W3. Missing test: `find_specs_by_engine_family` and `find_brand_default_oil` not tested directly in `test_db.py`

**File:** `parts-finder/tests/test_db.py`

**Problem:** The two new `PartsDatabase` methods (`find_specs_by_engine_family` and `find_brand_default_oil`) are exercised only indirectly through `test_oil_lookup.py`. The `test_db.py` file tests all pre-existing DB methods directly but has no direct unit tests for the new methods. This means:

1. A bug in `find_specs_by_engine_family` (e.g., wrong column filter, year range logic) would only surface through the cascade, making it harder to locate.
2. The non-deterministic tie-breaking behavior in `find_brand_default_oil` (noted by the implementer) is not documented through a test.
3. The `WHERE oil_viscosity != ''` filter in `find_brand_default_oil` is not tested with records that have empty viscosity mixed in.

The pattern in `test_db.py` is to test all public DB methods directly (see `TestInsertAndFindSpecs`, `TestInsertAndFindCrossRefs`). The new methods break this pattern.

**Fix:** Add a `TestNewOilQueries` class to `test_db.py` with at minimum:

```python
class TestNewOilQueries(unittest.TestCase):
    def setUp(self):
        self.db = PartsDatabase(":memory:")

    def tearDown(self):
        self.db.close()

    def test_find_specs_by_engine_family_matches_prefix(self):
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        result = self.db.find_specs_by_engine_family("Toyota", "2ZR", 2021)
        self.assertIsNotNone(result)
        self.assertEqual(result.engine_code, "2ZR-FE")

    def test_find_specs_by_engine_family_no_match_returns_none(self):
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        result = self.db.find_specs_by_engine_family("Toyota", "1KD", 2021)
        self.assertIsNone(result)

    def test_find_specs_by_engine_family_respects_year_range(self):
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE", year_from=2019, year_to=2023))
        self.assertIsNone(self.db.find_specs_by_engine_family("Toyota", "2ZR", 2018))
        self.assertIsNone(self.db.find_specs_by_engine_family("Toyota", "2ZR", 2024))

    def test_find_brand_default_oil_returns_most_common(self):
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE", oil_viscosity="0W-20", ...))
        self.db.insert_specs(_make_specs(engine_code="1ZR-FE", oil_viscosity="5W-30", ...))
        self.db.insert_specs(_make_specs(engine_code="3ZR-FE", oil_viscosity="0W-20", ...))
        result = self.db.find_brand_default_oil("Toyota")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "0W-20")  # most common

    def test_find_brand_default_oil_skips_empty_viscosity(self):
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE", oil_viscosity=""))
        result = self.db.find_brand_default_oil("Toyota")
        self.assertIsNone(result)

    def test_find_brand_default_oil_no_make_returns_none(self):
        result = self.db.find_brand_default_oil("Nonexistent")
        self.assertIsNone(result)
```

---

## SUGGESTIONS (nice to have)

| # | Issue | File | Fix |
|---|-------|------|-----|
| S1 | `OilResult.confidence` is a plain `str` with no validation — a typo like `"excact"` would silently produce a bad result. Consider a `Literal["exact", "engine_family", "brand_default"]` type annotation or a small `VALID_CONFIDENCES` frozenset check in `from_vehicle_specs`. | `parts-finder/src/parts_finder/models.py:179` | `confidence: Literal["exact", "engine_family", "brand_default"]` |
| S2 | The `test_multi_dash_returns_first_segment` test name claims to test a multi-dash code but uses `"1KD-FTV"`, which has exactly one dash. A genuinely multi-dash code (e.g. `"2ZZ-GE-VVTi"`) would be more self-documenting, though the current behavior is correct. | `parts-finder/tests/test_oil_lookup.py:56-57` | Rename test to `test_single_dash_returns_prefix` or use `"2ZZ-GE-VVTi"` to actually test multi-dash. |
| S3 | `find_brand_default_oil` aggregates `oil_change_interval_km` as a group-by key alongside `oil_viscosity` and `oil_spec`. This means records with the same viscosity and spec but different intervals are counted separately, potentially diluting the vote for the "true" most common viscosity. Consider grouping only by `oil_viscosity` and taking `MAX(COUNT(*))`, or document the three-column grouping as intentional. | `parts-finder/src/parts_finder/db.py:201-207` | Add a comment explaining why three columns are grouped together. |
| S4 | `OilLookup._resolve_names()` returns `(None, None)` when `make` is falsy but returns `(make, None)` when only `model` is missing. The `lookup()` caller checks `make is None` (line 74) but not `model is None` for Tier 1. This is correct since Tier 1 guards with `if model and engine_code`. Worth adding a comment at the model-None guard site to explain why `None` model is acceptable (Tier 1 skips, Tier 2 and 3 proceed make-only). | `parts-finder/src/parts_finder/oil_lookup.py:84` | Add inline comment: `# model=None skips Tier 1 (exact match); Tier 2 uses make+prefix only` |
| S5 | `IMPLEMENTATION_PROGRESS.md` has no entry for this module. The oil lookup is the first `parts-finder` module, so adding a dated entry would establish traceability for future parts-finder modules. | `IMPLEMENTATION_PROGRESS.md` | Add a dated entry for parts-finder / oil lookup module completion. |

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | 0 | — |
| WARNING | 3 | W1, W2, W3 |
| SUGGESTION | 5 | S1, S2, S3, S4, S5 |

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/oil-lookup.md`) | MISSING | Handoff notes "parts-finder doesn't have a `docs/modules/` convention yet" — acceptable for first module, but if more parts-finder modules are added this gap compounds |
| `docs/architecture/patterns.md` | UP TO DATE | 3-tier cascade is parts-finder-specific; no new USV pipeline pattern was established |
| `DECISIONS.md` | UP TO DATE | No new ADRs needed; cascade design is documented in handoff |
| `IMPLEMENTATION_PROGRESS.md` | NOT UPDATED | No entry exists for oil lookup or parts-finder module family |

---

## Code Quality Assessment

The implementation is well-structured and the core design decisions are sound. Specific positive observations:

**What's correct and well-done:**

- `_SPECS_FIELDS` in `db.py` is perfectly aligned with the `VehicleSpecs` dataclass field order (verified by enumeration — 32/32 fields match). The new `oil_spec` and `oil_oem_approval` fields slot in at the correct positions.
- The schema DDL column order in `_SCHEMA` matches `_SPECS_FIELDS`. This matters because `INSERT OR REPLACE` uses named columns (not positional), so misalignment would not cause data corruption, but it does mean the code is consistent and readable.
- The `WHERE oil_viscosity != ''` filter in `find_brand_default_oil` correctly prevents aggregating empty-spec records into the default.
- The `if specs is None or not specs.oil_viscosity` guard in `_try_exact` and `_try_engine_family` correctly handles the "record exists but no oil data" case — confirmed by manual trace.
- `OilResult.from_brand_default` correctly zeroes `capacity_l` with a clear rationale. The deliberate zero is documented both in code and in the handoff.
- The conservative `extract_engine_family` design (dash-only, `None` on dashless) is correct. Dashless codes like "G4FJ" could accidentally match short prefixes from unrelated families. The current behavior correctly falls through to brand_default.
- `frozen=True` on all three dataclasses (`VehicleRecord`, `VehicleSpecs`, `OilResult`) is consistent with Pattern 1 (Config Dataclass Pattern).
- The `_resolve_names` fallback chain (English -> Hebrew with warning) is correct — Hebrew names are unlikely to match English-populated DB records.

**The implementer's "unsure about" items — assessment:**

- **Engine family LIKE query performance**: Accurate observation. The index is on `(make, model, engine_code)`, and the Tier 2 query filters on `make` and `engine_code LIKE prefix%` without `model`. SQLite will use the `make` part of the index but not do a full prefix scan on `engine_code`. For tens of thousands of records this is fine; not a concern for the current scope.
- **Brand default aggregation ties**: Accurate observation. Non-deterministic with equal counts. Acceptable in practice but worth a comment (see S3).
- **OEM approval not in brand default**: Design is correct. The `from_brand_default` signature excluding `oem_approval` is a feature, not a limitation — it makes the zero-OEM-approval contract explicit and prevents callers from accidentally passing aggregated OEM data that would be meaningless.

---

## Verdict

**APPROVED WITH WARNINGS**

Three warnings were found. None blocks correctness for normal inputs.

W1 (LIKE injection) is the highest priority — it is a latent bug triggered only by malformed API data. Given that the government API is the data source and real engine codes are alphanumeric-only, the probability of a real-world trigger is low but not zero. Fix before exposing the API endpoint to production traffic.

W2 (misleading test assertion) is a test-quality issue — the `assertIn` is too permissive and would mask a specific bug in the cascade. Fix in the next commit.

W3 (missing direct DB tests) is a coverage gap that follows the established pattern in `test_db.py`. Fix when adding the next DB method or parts-finder module.

---

## Fix Documentation Requirement

After applying all fixes listed above, the implementor MUST:
1. Add a "## Fixes Applied" section to this review file (`docs/reviews/oil-lookup-review.md`)
2. For each fix: state what was changed, which file:line, and why
3. Re-run the affected tests and record pass/fail counts
4. Update `IMPLEMENTATION_PROGRESS.md` with a dated entry noting the fixes
5. Re-run master-reviewer OR self-verify against each BLOCKER/WARNING above

---

## Fix Log

Track resolution of findings here. Implementor updates this section after fixing issues.

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| W1 | FIXED | oil_lookup.py:32-35 | 2026-02-24 | Added `_SAFE_PREFIX_RE = re.compile(r"^[A-Za-z0-9]+$")` guard in `extract_engine_family`. Prefixes with `%`, `_`, or other non-alphanumeric chars now return `None`. Added test `test_like_wildcard_in_prefix_returns_none`. |
| W2 | FIXED | test_oil_lookup.py:301-317 | 2026-02-24 | Changed `assertIn(..., ("engine_family", "brand_default"))` to `assertEqual(result.confidence, "brand_default")`. Updated comment to accurately trace the cascade path. |
| W3 | FIXED | test_db.py:190-262 | 2026-02-24 | Added `TestEngineFamilyAndBrandDefaultQueries` class with 7 tests: prefix match, no match, year range boundaries, empty DB, most common brand default, empty viscosity skip, unknown make. |
| S1 | FIXED | models.py:179 | 2026-02-24 | Changed `confidence: str` to `confidence: Literal["exact", "engine_family", "brand_default"]`. Also updated `from_vehicle_specs` parameter type. |
| S2 | FIXED | test_oil_lookup.py:56-57 | 2026-02-24 | Changed test input from `"1KD-FTV"` (single dash) to `"2ZZ-GE-VVTi"` (genuine multi-dash). |
| S3 | FIXED | db.py:200-201 | 2026-02-24 | Added docstring paragraph explaining why GROUP BY uses all three columns. |
| S4 | FIXED | oil_lookup.py:85 | 2026-02-24 | Added inline comment: `# Tier 1: exact match (requires model; Tiers 2 & 3 use make only)`. |
| S5 | DEFERRED | | | IMPLEMENTATION_PROGRESS.md tracks USV project modules; parts-finder doesn't have its own tracking file yet. Will address when parts-finder gets a ROADMAP. |

### Post-fix verification

```
py_compile: models.py, db.py, oil_lookup.py, test_oil_lookup.py, test_db.py — all OK
pytest parts-finder/tests/ -v: 112/112 passed (0.66s) — 0 regressions
  New tests: 7 in test_db.py + 1 in test_oil_lookup.py = 8 added
```
