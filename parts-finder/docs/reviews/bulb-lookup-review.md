# Bulb Type Lookup Module Review

**Module:** Bulb Type Lookup
**Review Date:** 2026-02-24
**Reviewer:** master-reviewer
**Handoff Doc:** `parts-finder/docs/reviews/bulb-lookup-handoff.md`
**Review Tier:** 2 (Standard)

---

## Executive Summary

The implementation is clean, consistent with the OilLookup pattern, and all 133 tests pass with zero regressions. The core logic is correct. There are no blockers. There are three warnings, two of which are minor documentation gaps, and one is a genuine behavioral ambiguity in the tier-2 fallback that deserves a test and a docstring fix. There are also two suggestions for follow-on polish.

**Verdict: PASS_WITH_NOTES**

---

## Test Run

```
pytest parts-finder/tests/ -v
133 passed, 0 failed
```

All 21 new bulb tests pass. No regressions against the 112 existing tests.

---

## Section-by-Section Findings

### 1. `models.py` — VehicleSpecs schema expansion + BulbResult

**Status: PASS**

The 4-to-8 field migration is clean. All new fields follow the existing naming convention (`<position>_bulb`), default to `""`, and are marked `NOT NULL DEFAULT ''` in the schema. The `frozen=True` dataclass constraint is correctly applied to `BulbResult`.

The `_BULB_POSITION_FIELDS` dict as a single source of truth is a good design choice. Both `from_vehicle_specs()` and `replaceable_positions` iterate over it, so adding a ninth position in the future only requires one dict entry. The mapping logic at line 286 (`**{pos: getattr(specs, specs_field) for pos, specs_field in _BULB_POSITION_FIELDS.items()}`) is concise and correct.

**WARNING-1: `BulbResult.source` has no `""` (empty string) exclusion test for "LED" case sensitivity**

The docstring says the sentinel is the string `"LED"` (all caps). The `replaceable_positions` filter at line 277 checks `not in ("", "LED")`. This is correct, but there is no test verifying that `"led"` or `"Led"` is NOT filtered (i.e., the system trusts the data to be normalized). This is a data contract assumption that should be documented. It is not a bug in the current code — it is an undocumented invariant.

- Where: `models.py:277` and `models.py:247-248` (the docstring)
- Why it matters: If a future import script or data entry writes `"led"` instead of `"LED"`, those positions will appear in `replaceable_positions` as a bulb type, which is wrong.
- Fix (documentation-level): Add a sentence to the `BulbResult` class docstring: "Values must be normalized before insertion — `'LED'` must be stored in all-caps; mixed-case variants will not be filtered by `replaceable_positions`." Optionally add a test asserting `BulbResult(low_beam="led").replaceable_positions == {"low_beam": "led"}` to make the contract visible rather than hidden.

**No issues with the backward-compatibility concern.** The old 4 fields (`headlight_bulb`, `fog_light_bulb`, `tail_light_bulb`, `turn_signal_bulb`) were purely in the schema definition — the handoff confirms they were replaced, not added alongside. Since this project has no existing persisted database with real data yet (it is still in Phase 2 population), there is no migration burden. The `IF NOT EXISTS` in the schema means the new schema only applies to new databases. Any existing database would need to be recreated or migrated, but given the project state this is the correct approach and not a risk.

---

### 2. `db.py` — Schema update + `find_specs_by_model_year()`

**Status: PASS with one warning**

The schema at lines 46-53 correctly lists all 8 new columns with `TEXT NOT NULL DEFAULT ''`. The `_SPECS_FIELDS` list at lines 82-94 is in the same order as the schema columns, which is required by the `SELECT *` + positional mapping pattern used elsewhere. The order was verified to match.

The `find_specs_by_model_year()` method at lines 196-217 is correctly parameterized (no SQL injection risk — all four parameters are passed as `?` placeholders). The query logic `WHERE make = ? AND model = ? AND year_from <= ? AND year_to >= ?` is correct and consistent with how `find_specs()` handles the year range.

**WARNING-2: `find_specs_by_model_year()` has no direct unit test in `test_db.py`**

The existing `test_db.py` has a dedicated test class `TestEngineFamilyAndBrandDefaultQueries` that directly tests `find_specs_by_engine_family()` and `find_brand_default_oil()` at the DB layer (independent of the lookup class). There is no equivalent for `find_specs_by_model_year()`. The method is exercised indirectly through `TestBulbLookupModelYearFallback`, but a direct DB-layer test is missing.

- Where: `tests/test_db.py` — no `find_specs_by_model_year` coverage
- Why it matters: The handoff itself flags this method's LIMIT 1 behavior as an uncertainty. A direct test would document the boundary behavior (year range boundaries, model mismatch returns None, behavior when multiple engine variants exist).
- Fix: Add a test class `TestModelYearQuery` to `test_db.py` with at minimum: year-range boundary test (year outside range returns None), model mismatch returns None, and a test with two different engine_codes for same model-year confirming LIMIT 1 returns *a* result (documenting the ambiguity, as the existing `test_year_range_overlap_returns_first_match` does for `find_specs`).

**Note on the LIMIT 1 ambiguity (handoff uncertainty item 2):** The behavior is acceptable for bulbs because the assumption is same-chassis = same-bulbs, but the handoff correctly notes this. The `find_specs()` method in `test_db.py` has an explicit `test_year_range_overlap_returns_first_match` test that documents the LIMIT 1 behavior as a known data quality concern. The new method should have the same documentation-as-test. Without it, the behavior is only implied.

---

### 3. `lookup/bulbs.py` — BulbLookup class

**Status: PASS with one warning**

The cascade logic is correct and well-structured. The guard at line 74 (`if model and engine_code`) correctly requires both model AND engine_code for tier 1, then the guard at line 80 (`if model`) requires only model for tier 2. This matches the OilLookup tier-1 guard exactly.

The `_has_bulb_data()` helper correctly short-circuits on any non-empty bulb field. The `_try_exact()` and `_try_model_year()` private methods are clean and each do exactly one thing.

**WARNING-3: Tier-2 can return a result when the exact match exists but has no bulb data**

Consider this scenario:
1. DB contains Corolla 2ZR-FE with all bulb fields empty (no data yet).
2. DB also contains Corolla 1ZR-FE with bulb data populated.
3. Vehicle has engine_code="2ZR-FE".

The cascade proceeds:
- Tier 1: `find_specs("Toyota", "Corolla", 2021, "2ZR-FE")` returns the 2ZR-FE record. `_has_bulb_data` returns False. Tier 1 returns None.
- Tier 2: `find_specs_by_model_year("Toyota", "Corolla", 2021)` — this uses LIMIT 1. It may return *either* the 2ZR-FE (empty) or the 1ZR-FE (populated) depending on insertion order. If it returns the empty 2ZR-FE row, tier 2 also returns None, even though a populated record exists.

This is a genuine edge case the handoff does not mention. It is not a bug in the current code — the LIMIT 1 behavior is deterministic for a given database state — but it is a surprising result that could surface during data population.

- Where: `lookup/bulbs.py:128-132`, `db.py:208-213`
- Why it matters: A vehicle with exact-match record but empty bulb data will silently return None from tier 2 if the LIMIT 1 happens to return the empty record. The user sees "no data" when there is actually data for a sibling engine variant.
- Fix (documentation-level): Add a comment to `_try_model_year` and to `find_specs_by_model_year` noting that if multiple rows exist for a make/model/year, the method returns the first in insertion order, which may not be the most data-complete record. Suggest a future improvement: `ORDER BY (CASE WHEN low_beam_bulb != '' THEN 0 ELSE 1 END) LIMIT 1`. This is not a required fix for the current phase but should be tracked.

**The `_resolve_names` duplication is acceptable.** The handoff correctly notes this. With exactly two consumers and identical logic, extraction would be premature. The duplication is visible in both files and the note-to-refactor is in the handoff. No action required now.

---

### 4. `lookup/__init__.py` — Package init

**Status: PASS**

Empty init file. Correct for a new package. No issues.

---

### 5. `tests/test_bulb_lookup.py` — 21 new tests

**Status: PASS with gaps noted**

The test count of 21 is accurate. The 6 test classes cover the right dimensions:

- `TestBulbResultConstruction` — construction, factory, source field, empty defaults, immutability
- `TestReplaceablePositions` — empty filter, LED filter, all-LED, all-empty, mixed
- `TestBulbLookupExactMatch` — hit, source field, miss
- `TestBulbLookupModelYearFallback` — different engine same model, source field, different model miss
- `TestBulbLookupCascade` — short-circuit at tier 1, falls through to tier 2, no-make None, no-model None, no-bulb-data None, empty-engine skips tier 1

**Gap 1: No test verifying that tier 2 correctly returns model_year source (not exact) when tier 1 hits the DB but finds no bulb data.** The test `test_specs_with_no_bulb_data_returns_none` at line 268 tests this — but only for a case where the vehicle's engine_code exactly matches. It does not test the case where tier 1 matches (finds a record) but `_has_bulb_data` returns False, and then tier 2 also matches the same record (same LIMIT 1 problem). This is the edge case from WARNING-3 above.

**Gap 2: No test for `replaceable_positions` returning a copy vs a live reference.** This is minor — since `BulbResult` is frozen and the dict is constructed fresh on each property call, there is no mutation risk. But it's worth noting as a non-issue that reviewers might wonder about.

**No test anti-greenwashing detected.** All assertions are meaningful and test-specific data flows. No expected values appear artificially adjusted.

---

### 6. OilLookup pattern comparison

**Status: CONSISTENT**

BulbLookup follows the OilLookup pattern in all key dimensions:

| Dimension | OilLookup | BulbLookup | Consistent? |
|-----------|-----------|------------|-------------|
| Result dataclass in models.py | OilResult | BulbResult | Yes |
| Lookup class in dedicated module | oil_lookup.py | lookup/bulbs.py | Yes (new subpackage) |
| `_resolve_names` method | Identical | Identical | Yes (duplication acknowledged) |
| Cascade guard pattern | `if model and engine_code` | Same | Yes |
| `_has_data` check pattern | `not specs.oil_viscosity` | `not _has_bulb_data(specs)` | Slight difference — see below |
| DB method naming | `find_specs_by_engine_family` | `find_specs_by_model_year` | Consistent style |
| In-memory DB in tests | Yes | Yes | Yes |
| setUp/tearDown pattern | Yes | Yes | Yes |

**The `_has_data` check difference is intentional and correct.** OilLookup checks a single field (`oil_viscosity`) because all oil data lives or dies with the viscosity field. BulbLookup needs to check all 8 bulb fields because any one of them may be populated independently. The `_has_bulb_data` helper correctly uses `any()`. This is not a deviation from OilLookup — it is the appropriate generalization.

**The subpackage structure (`lookup/`) is a minor deviation from OilLookup's flat placement.** `oil_lookup.py` lives at the package root, while `BulbLookup` lives in `lookup/bulbs.py`. This is forward-looking (anticipates more lookup classes) and is a reasonable choice. However, it creates an asymmetry: `from parts_finder.oil_lookup import OilLookup` vs `from parts_finder.lookup.bulbs import BulbLookup`. A future refactor should decide whether to move OilLookup into `lookup/oil.py` or keep BulbLookup at the root. This is not a problem now, but should be documented as a pending decision.

---

## Assessment of the 3 Handoff Uncertainty Items

### Uncertainty 1: `_resolve_names` duplication
**Assessment: Correctly deferred.** Two consumers is the right threshold to avoid premature abstraction. The duplication is low-risk because `_resolve_names` is simple (8 lines, no state), both copies are identical, and the refactor trigger ("if a third lookup class appears") is specific and actionable. No action required.

### Uncertainty 2: `find_specs_by_model_year` LIMIT 1 behavior
**Assessment: Correctly flagged, but needs a test to document it.** The handoff says "for bulbs this is fine (same chassis = same bulbs)" — this is true in the happy case, but see WARNING-3 above. The deeper issue is that LIMIT 1 with no ORDER BY is non-deterministic in the presence of multiple rows. For the current data volume this is fine, but a test that explicitly documents the LIMIT 1 ambiguity (as `test_year_range_overlap_returns_first_match` does in `test_db.py`) would prevent future confusion. The handoff's analysis is sound; it just needs to be materialized as a test.

### Uncertainty 3: No index for model-year query
**Assessment: Correctly analyzed, no action needed now.** The existing `idx_specs_lookup` on `(make, model, engine_code)` is a composite index. SQLite can still use the leftmost prefix `(make, model)` for the new query, even though `engine_code` is absent from the WHERE clause. The index is therefore partially useful — it narrows the scan to the make+model subset before the year range filter is applied. The handoff's concern about a dedicated `(make, model)` index is valid at scale but premature for the current data volume. The note is correct and no action is needed now.

---

## SQL Safety Audit

No SQL injection risks found. All three new query methods (`find_specs()`, `find_specs_by_engine_family()`, `find_specs_by_model_year()`) use parameterized queries with `?` placeholders throughout. Column names are never interpolated from user input. The only string-formatted SQL in the codebase is in `import_data.py` (the `table_name` variable in the COUNT query), which is constructed from a controlled internal value, not user input.

---

## Handoff Accuracy

All claims in the handoff match the implementation:

- "8 granular lamp positions" — confirmed in `models.py:142-149` and `db.py:46-53`
- "BulbResult frozen dataclass" — confirmed at `models.py:242`
- "`from_vehicle_specs()` factory" — confirmed at `models.py:281-288`
- "Two-tier cascade: exact -> model-year" — confirmed in `lookup/bulbs.py:72-85`
- "LED filtering via `replaceable_positions`" — confirmed at `models.py:273-278`
- "21 new tests" — confirmed by count and test run
- "133 total tests passing" — confirmed

One minor inaccuracy: the handoff says `oil_lookup.py` has `_resolve_names` and "identical logic exists in `OilLookup._resolve_names` and `BulbLookup._resolve_names`". This is accurate. However, the handoff also says `_resolve_names` "could be extracted to a shared utility" — this is true, but the natural extraction point would be a base class or a standalone function in `lookup/__init__.py` (which is currently empty). This is not a criticism, just an observation for when the refactor happens.

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/bulb_lookup.md`) | MISSING | No `docs/modules/` directory exists in parts-finder; the handoff acknowledges this. Not a blocker given the project is pre-release, but warrants a WARNING. |
| `DECISIONS.md` in parts-finder | MISSING | Handoff correctly notes this. Schema expansion rationale (4 vs 8 fields, "LED" sentinel, two-tier vs three-tier) is captured only in the handoff file. Should be promoted to a DECISIONS.md before the API layer is built. |
| `IMPLEMENTATION_PROGRESS.md` in parts-finder | MISSING | Handoff acknowledges. Not yet needed given single-developer context. |
| `patterns.md` in parts-finder | MISSING (no `docs/architecture/` dir) | The lookup class pattern is not documented anywhere outside the code and this handoff. |

None of these missing docs are blockers at this stage. The sub-project is still young and documentation infrastructure is being built incrementally.

---

## Findings Summary

### BLOCKERS
None.

### WARNINGS

**WARNING-1: "LED" sentinel normalization is an undocumented invariant**
- File: `parts-finder/src/parts_finder/models.py:247-248` (docstring), `parts-finder/src/parts_finder/models.py:277`
- Problem: The `replaceable_positions` filter uses exact string match `"LED"`. Mixed-case variants like `"led"` or `"Led"` will not be filtered. This is correct behavior, but it is an undocumented contract on data entry.
- Fix: Add a sentence to the `BulbResult` class docstring stating that LED sentinel values must be stored as `"LED"` (all caps). Optionally add a test that exposes the case-sensitivity as a known behavior, not a bug.

**WARNING-2: `find_specs_by_model_year()` has no direct unit test in `test_db.py`**
- File: `parts-finder/tests/test_db.py` — missing test class
- Problem: All other query methods (`find_specs`, `find_specs_by_engine_family`, `find_brand_default_oil`) have direct DB-layer tests. The new method is exercised only through the BulbLookup integration tests.
- Fix: Add `TestModelYearQuery` class to `test_db.py` covering: match (year inside range), no-match (year outside range), no-match (wrong model), and LIMIT 1 ambiguity documentation test.

**WARNING-3: Tier-2 fallback LIMIT 1 ordering is non-deterministic for data-partial scenarios**
- File: `parts-finder/src/parts_finder/db.py:208-213`, `parts-finder/src/parts_finder/lookup/bulbs.py:128-132`
- Problem: If multiple engine variants exist for the same make/model/year and the first row returned by LIMIT 1 has empty bulb fields, tier 2 returns None even though a sibling row has data. The handoff's "same chassis = same bulbs" assumption holds for complete data but is unreliable during incremental population.
- Fix (short-term): Add an ORDER BY clause to `find_specs_by_model_year` that prefers populated rows: `ORDER BY (CASE WHEN low_beam_bulb != '' THEN 0 ELSE 1 END)`. Fix (documentation-level): Add a comment in both `db.py` and `bulbs.py` explaining the LIMIT 1 assumption and its limitation. The existing analogous test `test_year_range_overlap_returns_first_match` in `test_db.py` should be mirrored for this method.

### SUGGESTIONS

**SUGGESTION-1: Asymmetric module placement for OilLookup vs BulbLookup**
- `oil_lookup.py` is at the package root; `BulbLookup` is in `lookup/bulbs.py`. This asymmetry will grow with each new lookup class. Decide now whether to move `oil_lookup.py` into `lookup/oil.py` (preferred — more consistent) or move `bulbs.py` out of the subpackage.

**SUGGESTION-2: `lookup/__init__.py` could re-export the public classes**
- Currently empty. Consider adding `from parts_finder.lookup.bulbs import BulbLookup` to make the import path `from parts_finder.lookup import BulbLookup` consistent with the eventual `from parts_finder.lookup import OilLookup` (if OilLookup is moved).

---

## Verdict

**PASS_WITH_NOTES**

No blockers. Three warnings should be addressed before the API layer (Phase 3) is built — WARNING-2 (missing DB test) and WARNING-3 (LIMIT 1 behavior documentation) in particular, because Phase 3 will surface the LIMIT 1 ambiguity in production scenarios and "we'll fix it later" becomes harder once the API contract is established. WARNING-1 (LED case sensitivity) is a documentation fix only and can be done at any time.

The implementation is otherwise correct, well-structured, and consistent with the established OilLookup pattern.

---

## Fixes Applied

### W1 — LED case-sensitivity docstring
Added a `.. note::` to the `BulbResult` class docstring in `models.py:250-252` stating that `"LED"` must be stored in all-caps and mixed-case variants will not be filtered.

### W2 — Direct DB tests for `find_specs_by_model_year`
Added `TestModelYearQuery` class to `test_db.py` with 7 tests: `test_match_within_year_range`, `test_year_boundaries`, `test_model_mismatch_returns_none`, `test_ignores_engine_code`, `test_multiple_engines_returns_a_result`, `test_prefers_populated_bulb_row`, `test_empty_db_returns_none`.

### W3 — ORDER BY for LIMIT 1 data-partial safety
Added `ORDER BY (CASE WHEN low_beam_bulb != '' THEN 0 ELSE 1 END)` to the `find_specs_by_model_year()` query in `db.py:211`. The `test_prefers_populated_bulb_row` test validates this: inserts an empty row first, a populated row second, and asserts the populated row is returned.

### Post-fix test results
```
pytest parts-finder/tests/ -v
189 passed, 0 failed (28 bulb-related tests + 161 existing)
```

## Fix Log

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| W1 | FIXED | models.py:250-252 | 2026-02-24 | Docstring note added |
| W2 | FIXED | test_db.py (TestModelYearQuery, 7 tests) | 2026-02-24 | All 7 pass |
| W3 | FIXED | db.py:211 (ORDER BY clause) | 2026-02-24 | test_prefers_populated_bulb_row validates |
