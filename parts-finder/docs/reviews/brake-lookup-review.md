# Brake Parts Lookup Module Review

**Module:** Brake Parts Lookup
**Review Date:** 2026-02-24
**Reviewer:** master-reviewer
**Handoff Doc:** `parts-finder/docs/reviews/brake-lookup-handoff.md`
**Review Tier:** 2 (Standard)

---

## Executive Summary

The implementation is correct, consistent with the BulbLookup/CoolantLookup pattern, and all 234 tests pass with zero regressions. The cascade logic is sound, the two-tier design is well-justified, and the cross-reference resolution works correctly. There are no blockers.

There are three warnings and three suggestions. The most important warning is a direct repeat of a gap that was found, flagged, and fixed in the previous two lookup reviews: `find_specs_by_model_year_for_brakes()` has no direct DB-layer test in `test_db.py`. The second warning is a semantic duplication trap: `_BRAKE_OEM_FIELDS` in `models.py` and `BrakeResult.has_data` express the same "four fields count as data" semantics using different field names with no shared link, meaning they can silently diverge.

The four-way `_resolve_names` duplication graduates from SUGGESTION to WARNING because the coolant review explicitly stated it should be extracted before a fourth lookup class was added, and this is that fourth lookup class.

**Verdict: CHANGES NEEDED**

---

## Test Run

```
pytest parts-finder/tests/ -v
234 passed, 0 failed (18 new brake tests + 216 existing)
```

All 18 new brake tests pass. No regressions.

---

## Section-by-Section Findings

### 1. `models.py` — BrakeResult dataclass and constants

**Status: PASS with one warning**

`BrakeResult` is correctly implemented as a `frozen=True` dataclass. All OEM fields default to `""`, cross-reference fields default to `()` (required for hashability in a frozen dataclass), and the `source` field correctly uses `Literal["exact", "model_year"]` — consistent with `BulbResult` and at two tiers rather than three.

The `from_vehicle_specs()` classmethod correctly uses `_BRAKE_SPEC_FIELDS` to map BrakeResult field names to VehicleSpecs field names, resolving the "disc vs. rotor" terminology divergence transparently. The cross-reference resolution loop (`oem_to_crossref`) is clear and correct.

The `has_data` property correctly excludes `brake_fluid_type` from the "data present" test and is documented with an explicit rationale.

**WARNING-1: `_BRAKE_OEM_FIELDS` and `BrakeResult.has_data` express the same semantics twice without a shared link**

`_BRAKE_OEM_FIELDS` contains VehicleSpecs-level field names and its comment says: "The four OEM fields used to check 'has data' (excludes brake_fluid_type)."

But `BrakeResult.has_data` does NOT use `_BRAKE_OEM_FIELDS`. It references BrakeResult's own field names inline. Meanwhile `_BRAKE_OEM_FIELDS` is only consumed by `_has_brake_data()` in `brakes.py` (which operates on a `VehicleSpecs`).

These two are semantically equivalent — they both encode the rule "pad and disc OEM numbers count as data; brake fluid alone does not" — but they are connected by human convention only, not by code. If a future developer adds a fifth brake OEM field (e.g., a caliper OEM), they would need to update both independently with no static link guiding them.

- **Where:** `parts-finder/src/parts_finder/models.py` (constants and `has_data` property)
- **Why it matters:** Silent divergence risk. If `has_data` is updated but `_BRAKE_OEM_FIELDS` is not (or vice versa), the lookup class's "should I cascade?" gate disagrees with the result object's "do I have actionable data?" gate.
- **Fix:** Update the comment on `_BRAKE_OEM_FIELDS` to accurately state its actual role: "VehicleSpecs field names checked to decide whether a specs record contains useful brake data for lookup cascade decisions." Remove the claim that it is "used to check 'has data'" since `BrakeResult.has_data` does not use it. Optionally: add a code comment inside `BrakeResult.has_data` cross-referencing `_BRAKE_OEM_FIELDS`: "# mirrors _BRAKE_OEM_FIELDS check; keep in sync if fields are added."

---

### 2. `db.py` — `find_specs_by_model_year_for_brakes()`

**Status: PASS with one warning**

The new method is correctly implemented. The SQL is parameterized, the year range logic is identical to the other `find_specs_by_model_year_*` variants, and the `ORDER BY (CASE WHEN front_brake_pad_oem != '' THEN 0 ELSE 1 END) LIMIT 1` clause correctly applies the established pattern of preferring data-populated rows over empty rows.

The ORDER BY was applied proactively — the implementer did not wait for a reviewer to flag it. That is a genuine improvement over the coolant review, where the ORDER BY was added as a fix.

**WARNING-2: `find_specs_by_model_year_for_brakes()` has no direct unit test in `test_db.py`**

This is the third consecutive occurrence of this gap:

- Bulb review WARNING-2: `find_specs_by_model_year()` — flagged, fixed by adding `TestModelYearQuery` (7 tests)
- Coolant review WARNING-1: `find_specs_by_model_year_for_coolant()` — flagged, fixed by adding `TestCoolantModelYearQuery` (6 tests)
- This review: `find_specs_by_model_year_for_brakes()` — same gap, no fix applied proactively

`test_db.py` now contains `TestModelYearQuery` and `TestCoolantModelYearQuery`, but no `TestBrakeModelYearQuery`. The method is exercised only indirectly through the cascade integration tests.

- **Where:** `parts-finder/tests/test_db.py` — missing `TestBrakeModelYearQuery` class
- **Why it matters:** The critical ORDER BY preference behavior — "return brake-populated row over empty row" — is not tested at the DB layer. A refactor that accidentally drops the ORDER BY clause would only be caught indirectly.
- **Fix:** Add a `TestBrakeModelYearQuery` class to `test_db.py` mirroring `TestCoolantModelYearQuery`. Required tests: `test_match_within_year_range`, `test_year_boundaries`, `test_model_mismatch_returns_none`, `test_ignores_engine_code`, `test_prefers_populated_brake_row`, `test_empty_db_returns_none`. The `test_prefers_populated_brake_row` test is the critical one — insert empty row first, populated row second, confirm the populated row is returned.

---

### 3. `lookup/brakes.py` — BrakeLookup class

**Status: PASS with one graduated warning**

The cascade structure is clean and mirrors BulbLookup exactly at the structural level. The tier guards (`if model and engine_code` for tier 1, `if model` for tier 2) are correct and consistent. The `_has_brake_data()` helper is correctly placed at module level, following the `_has_bulb_data()` pattern. The absence of a tier 3 brand default is well-justified and correctly documented.

**WARNING-3 (graduated from SUGGESTION): `_resolve_names` duplication is now at 4 consumers**

The coolant review (SUGGESTION-1) stated: "This review categorizes it as SUGGESTION-1 rather than WARNING because the code is harmless-as-is, but it should be tracked and executed before a fourth lookup class is added." The fourth lookup class is now present without the extraction having been done.

The four identical copies (~21 lines each) live at:
- `parts-finder/src/parts_finder/oil_lookup.py`
- `parts-finder/src/parts_finder/lookup/bulbs.py`
- `parts-finder/src/parts_finder/lookup/coolant.py`
- `parts-finder/src/parts_finder/lookup/brakes.py`

- **Fix:** Create `parts-finder/src/parts_finder/lookup/_shared.py` with a module-level function `resolve_vehicle_names(vehicle: VehicleRecord) -> tuple[str | None, str | None]`. Each lookup class's `_resolve_names` becomes a one-line call to this function.

---

### 4. `tests/test_brake_lookup.py` — 18 new tests

**Status: PASS with minor gap noted**

The 5 test classes cover the right dimensions: BrakeResult dataclass, exact match, cross-references, model-year fallback, and cascade integration. No test anti-greenwashing detected. All assertions are meaningful. Factory helpers are consistent with prior test patterns.

**Minor gap:** `has_data` test only covers `front_pad_oem` and `rear_disc_oem` individually — `rear_pad_oem` and `front_disc_oem` are not individually tested.

---

### 5. Cross-Reference Resolution Pattern (`db: object` typing)

**Status: PASS with architectural note**

The `db: object` typing loses static type safety — mypy/pyright cannot verify the passed object has `find_crossrefs()`. The circular import problem is real (`models.py` cannot import `PartsDatabase` from `db.py`).

Two cleaner alternatives exist:
1. **Protocol:** Define a `CrossRefResolver` Protocol in `models.py` — no circular import needed.
2. **Move resolution to the lookup class** — removes DB dependency from the model entirely.

Neither is a blocker. The Protocol approach is worth applying before the API layer is built.

---

### 6. Pattern Compliance Comparison

| Dimension | OilLookup | BulbLookup | CoolantLookup | BrakeLookup | Compliant? |
|-----------|-----------|------------|---------------|-------------|------------|
| Result type in `models.py` | OilResult | BulbResult | CoolantResult | BrakeResult | Yes |
| Lookup in `lookup/` subpackage | No (root) | Yes | Yes | Yes | Yes |
| `_resolve_names` method | Identical | Identical | Identical | Identical | Yes (4x dup: W3) |
| Cascade guard pattern | Yes | Yes | Yes | Yes | Yes |
| Purpose-specific DB method | N/A | Yes | Yes | Yes | Yes |
| ORDER BY preference in tier 2 | N/A | Yes (fix) | Yes (fix) | Yes (proactive) | Yes |
| Direct DB-layer test for tier 2 | N/A | Yes (fix) | Yes (fix) | **Missing** | **No** |

---

### 7. Two-Tier vs. Three-Tier Cascade Justification

**Status: APPROVED**

The rationale for omitting a brand-default tier is sound. Brake pad OEM numbers are vehicle-model-specific — Toyota does not have a "default brake pad." The model-year fallback (ignoring engine_code) is also correctly justified: different engine variants on the same chassis share the same brake system.

---

## Handoff Accuracy

All claims in the handoff match the implementation. The handoff accurately self-identifies both open concerns (`_resolve_names` duplication and `db: object` typing). This review agrees with both assessments and graduates the `_resolve_names` concern to WARNING.

---

## Findings Summary

### BLOCKERS

None.

### WARNINGS (must fix before next module)

| # | Finding | File | Fix |
|---|---------|------|-----|
| W1 | `_BRAKE_OEM_FIELDS` comment misleading — two parallel "has data" definitions can silently diverge | `models.py` | Fix comment; add cross-reference comment inside `has_data` |
| W2 | `find_specs_by_model_year_for_brakes()` has no direct DB-layer test — repeats gap fixed in both prior reviews | `test_db.py` | Add `TestBrakeModelYearQuery` with 6 tests |
| W3 | `_resolve_names` duplication at 4 consumers — prior review's extraction condition is now met | All four lookup modules | Extract to `lookup/_shared.py` |

### SUGGESTIONS

| # | Finding | Notes |
|---|---------|-------|
| S1 | `has_data` test coverage incomplete — only 2 of 4 OEM fields tested individually | Add `rear_pad_oem` and `front_disc_oem` assertions |
| S2 | `db: object` should be a Protocol for static type safety | Define `CrossRefResolver` Protocol in `models.py` |
| S3 | `lookup/__init__.py` has no re-exports | Consider adding re-exports for all 4 lookup classes |

---

## Fix Documentation Requirement

After applying all fixes listed above, the implementor MUST:
1. Add a "## Fixes Applied" section to this review file
2. For each fix: state what was changed, which file, and why
3. Re-run the affected tests and record pass/fail counts
4. Re-run master-reviewer OR self-verify against each WARNING above

---

## Verdict

**CHANGES NEEDED**

Three warnings must be addressed before the next lookup module is implemented:

- **W1** is a documentation fix (~5 minutes) that prevents silent divergence between two parallel "has data" definitions.
- **W2** is a test addition (~30 minutes) that applies the same DB-layer test pattern fixed in the prior two reviews.
- **W3** is a refactor (~1-2 hours) that was explicitly deferred to "before the fourth lookup class" — that condition is now met.

None of the three is a behavioral bug. All three address code hygiene, test coverage, and documentation concerns that compound with each new lookup module.

---

## Fixes Applied

### W1: Fixed `_BRAKE_OEM_FIELDS` comment and added cross-reference in `has_data`

- **File:** `parts-finder/src/parts_finder/models.py`
- **What:** Rewrote the `_BRAKE_OEM_FIELDS` comment to accurately describe its role ("VehicleSpecs field names checked by `_has_brake_data()` in `brakes.py` to decide whether a specs record contains useful brake data for cascade decisions"). Added a cross-reference comment inside `BrakeResult.has_data`: "Mirrors `_BRAKE_OEM_FIELDS` check on VehicleSpecs — keep in sync if brake OEM fields are added or removed."
- **Why:** Prevents silent divergence between the two parallel "has data" definitions.

### W2: Added `TestBrakeModelYearQuery` to `test_db.py`

- **File:** `parts-finder/tests/test_db.py`
- **What:** Added 6 tests mirroring `TestCoolantModelYearQuery`: `test_match_within_year_range`, `test_year_boundaries`, `test_model_mismatch_returns_none`, `test_ignores_engine_code`, `test_prefers_populated_brake_row`, `test_empty_db_returns_none`.
- **Why:** The critical ORDER BY preference behavior is now tested directly at the DB layer, not just indirectly through cascade integration tests.

### W3: Extracted `_resolve_names` to `lookup/_shared.py`

- **File (new):** `parts-finder/src/parts_finder/lookup/_shared.py`
- **Files (modified):** `oil_lookup.py`, `lookup/bulbs.py`, `lookup/coolant.py`, `lookup/brakes.py`
- **What:** Created `resolve_vehicle_names(vehicle, logger)` as a module-level function in `_shared.py`. The `logger` parameter preserves each caller's log identity (e.g., warning messages still appear under `parts_finder.lookup.brakes`, not `parts_finder.lookup._shared`). Each lookup class's `_resolve_names` is now a one-line delegation.
- **Why:** Eliminates 4-way code duplication (~21 lines x 4 = ~84 lines removed). Future lookup modules import from `_shared` instead of copying.

### S1: Completed `has_data` test coverage

- **File:** `parts-finder/tests/test_brake_lookup.py`
- **What:** Added `rear_pad_oem` and `front_disc_oem` individual assertions to `test_has_data_true_when_any_oem_set`, so all four OEM fields are now tested individually.
- **Why:** Closes the minor gap flagged in section 4 of the review.

### Test Results After Fixes

```
pytest parts-finder/tests/ -v
295 passed, 0 failed
```

All 3 warnings and 1 suggestion resolved. Zero regressions.
