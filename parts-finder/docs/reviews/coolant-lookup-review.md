# Coolant Lookup Module Review

**Module:** Coolant Specification Lookup
**Review Date:** 2026-02-24
**Reviewer:** master-reviewer
**Handoff Doc:** `parts-finder/docs/reviews/coolant-lookup-handoff.md`
**Review Tier:** 2 (Standard)

---

## Executive Summary

The implementation is correct, well-structured, and consistent with the OilLookup/BulbLookup pattern. The three-tier cascade logic is sound, the compatibility matrix chemistry is accurate, and all 210 tests pass with zero regressions. There are no blockers. There are three warnings and three suggestions. The most important warning is a type annotation gap that allows invalid `source` values to reach `CoolantResult` without compile-time detection. The second most important is a missing direct DB-layer test for `find_specs_by_model_year_for_coolant`, which exactly mirrors a gap that was flagged and fixed in the bulb review. There is also a test coverage gap for HOAT in the OAT mixing warning test.

**Verdict: CHANGES NEEDED**

---

## Test Run

```
pytest parts-finder/tests/ -v
210 passed, 0 failed (20 new coolant tests + 190 existing)
```

All 20 new coolant tests pass. No regressions.

---

## Section-by-Section Findings

### 1. `models.py` — CoolantResult dataclass

**Status: PASS with one note**

`CoolantResult` follows the OilResult/BulbResult pattern correctly. It is `frozen=True`, carries all required fields, and has both `from_vehicle_specs()` and `from_brand_default()` factory classmethods. The `from_brand_default()` method correctly hard-codes `capacity_l=0.0` and `source="brand_default"`, which is the same design as `OilResult.from_brand_default()`. The `spec_info` 4-tuple parameter type `tuple[str, str, str, str]` is correct and matches `CoolantSpec` field order.

The `Literal["exact", "model_year", "brand_default"]` type annotation on the `source` field is correct and consistent with BulbResult's `Literal["exact", "model_year"]`.

**NOTE: `from_vehicle_specs` source annotation matches the Literal but the caller passes an unconstrained `str`.**
This is a type system gap — see WARNING-2 below for the full explanation. The model itself is correctly annotated; the problem is in the lookup class.

---

### 2. `db.py` — `find_specs_by_model_year_for_coolant()`

**Status: PASS with one warning**

The new method is correctly implemented. The SQL is parameterized (no injection risk), the year range logic is identical to `find_specs_by_model_year()`, and the `ORDER BY (CASE WHEN coolant_type != '' THEN 0 ELSE 1 END)` clause correctly addresses the same LIMIT 1 data-partial scenario that was fixed in the bulb review (W3 in the bulb review). The fix was applied proactively here, which is correct.

However, the pattern established in the bulb review's W2 fix — adding a direct DB-layer test class for the new query method — was not followed for this method.

**WARNING-1: `find_specs_by_model_year_for_coolant()` has no direct unit test in `test_db.py`**

The bulb review identified the exact same gap for `find_specs_by_model_year()`, categorized it as WARNING-2, and it was fixed by adding `TestModelYearQuery` to `test_db.py`. That fix is now visible at `test_db.py:267`. The equivalent fix was not applied for the new coolant-specific method. The method is exercised indirectly through `TestCoolantLookupModelYearFallback`, but a direct DB-layer test is missing.

- Where: `parts-finder/tests/test_db.py` — no `find_specs_by_model_year_for_coolant` coverage
- Why it matters: Without a direct test, the ORDER BY preference behavior is not documented at the DB level. A future refactor of the query might break the LIMIT 1 guarantee silently. The pattern that was established in the bulb review is now inconsistent.
- Fix: Add a `TestCoolantModelYearQuery` class to `test_db.py` mirroring `TestModelYearQuery` but testing the coolant-specific behavior: year range match, year boundaries, model mismatch returns None, ignores engine_code, and critically a `test_prefers_populated_coolant_row` test that inserts an empty row first and a populated row second, confirming `coolant_type != ''` ordering returns the populated row.

---

### 3. `lookup/coolant.py` — CoolantLookup class

**Status: PASS with one warning**

The cascade structure is clean and correct. The tier guards (`if model and engine_code` for tier 1, `if model` for tier 2) are identical to BulbLookup and follow the established pattern. The `_specs_to_result()` helper elegantly consolidates the DB-to-result conversion for both tier 1 and tier 2, which is a small improvement over BulbLookup's `_try_exact()` and `_try_model_year()` each calling `BulbResult.from_vehicle_specs()` directly. This is a good refactoring pattern.

The `_resolve_coolant_type()` normalizer is correctly implemented. The two-pass approach (exact key match first, then substring match against spec names) is pragmatic for messy real-world DB values.

The `_build_mixing_warning()` function is correctly implemented. The chemistry of the compatibility matrix is reasonable for the 5 technology types covered (see Section 5 for knowledge base audit).

**WARNING-2: `_specs_to_result` source parameter is typed as `str`, not `Literal["exact", "model_year"]`**

At `coolant.py:233-237`, `_specs_to_result()` takes `source: str`. This signature is weaker than necessary. The method passes `source` directly to `CoolantResult.from_vehicle_specs()`, whose signature requires `Literal["exact", "model_year", "brand_default"]`. If a caller passes an invalid string (e.g., a typo like `"excact"`), `mypy`/`pyright` will not catch it at the `_specs_to_result()` call site because the parameter accepts any `str`.

Compare to `OilResult.from_vehicle_specs()` which takes `confidence: Literal["exact", "engine_family", "brand_default"]` — that type flows through without weakening because the OilLookup tier methods call the classmethod directly with a literal string.

The contrast with the established OilLookup/BulbLookup pattern is visible: in those lookup classes, the `source`/`confidence` literal is passed directly to the result factory method at each call site, making the type flow transparent. The `_specs_to_result()` helper is a design improvement for DRY purposes, but it loses the Literal narrowing.

- Where: `parts-finder/src/parts_finder/lookup/coolant.py:236`
- Why it matters: Type checkers cannot verify that `source="excact"` is invalid at the call sites (`coolant.py:208` and `coolant.py:218`). Silent typos could produce CoolantResult objects with invalid source values at runtime.
- Fix: Change `source: str` to `source: Literal["exact", "model_year"]` at `coolant.py:236`. Add `from typing import Literal` to the imports. Note: `"brand_default"` is intentionally excluded from this parameter type because brand defaults are handled entirely in `_try_brand_default()` which calls `CoolantResult.from_brand_default()` directly (which hard-codes `source="brand_default"`).

---

### 4. `tests/test_coolant_lookup.py` — 20 new tests

**Status: PASS with one gap**

The test structure is correct and the 6 test classes cover the right dimensions:

- `TestCoolantResultConstruction` — construction, factory, zero-capacity brand default, immutability
- `TestMixingWarning` — OAT, IAT, HOAT partial, unknown technology
- `TestCoolantLookupExactMatch` — hit, source field, make/model miss
- `TestCoolantLookupModelYearFallback` — different engine same model, source field
- `TestCoolantLookupBrandDefault` — known brand (Toyota), unknown brand returns None
- `TestCoolantLookupCascade` — short-circuit at tier 1, full cascade to brand default, no-make None, empty coolant skips, empty engine skips tier 1, mixing warning present

The `setUp`/`tearDown` pattern with `PartsDatabase(":memory:")` is correct and consistent. No test anti-greenwashing detected.

**WARNING-3: `test_oat_warning` does not verify that HOAT appears in the OAT incompatibility warning**

The OAT entry in `COMPATIBILITY_MATRIX` is `{"OAT"}`. Therefore `_ALL_TECHNOLOGIES - {"OAT"}` = `{"IAT", "HOAT", "P-OAT", "Si-OAT"}`. The warning produced is:

```
Do NOT mix with HOAT, IAT, P-OAT or Si-OAT coolant
```

The test at `test_coolant_lookup.py:97-105` checks for `IAT`, `P-OAT`, and `Si-OAT` but does NOT check for `HOAT`. This is a test coverage gap: one of the four incompatible types is unverified. The OAT warning is currently correct, but the test cannot detect a future regression where HOAT is accidentally added to OAT's compatible set.

- Where: `parts-finder/tests/test_coolant_lookup.py:97-105`
- Why it matters: Coolant mixing warnings are safety-relevant. The test should verify ALL incompatible types listed in the warning, not a subset.
- Fix: Add `self.assertIn("HOAT", warning)` to `test_oat_warning`. Optionally add a `test_si_oat_warning()` test that verifies Si-OAT's incompatible set (`{"IAT", "HOAT", "P-OAT"}`) and a `test_p_oat_warning()` test, so each technology type has its full incompatibility list verified.

---

### 5. Knowledge Base Accuracy Audit

**Status: PASS — all 9 specs and 15 brand defaults are accurate**

The coolant specifications in `_COOLANT_SPECS` were checked against published manufacturer standards:

| Key | Spec Name | Technology | Color | Assessment |
|-----|-----------|------------|-------|------------|
| G13 | TL 774 J (G13) | Si-OAT | purple | Correct. VW G13 is Si-OAT, purple/violet. |
| LC-18 | BMW LC-18 | HOAT | blue/green | Correct. BMW LC-18 / Hoechst OAT + silicate = HOAT. Blue/green is correct. |
| 325.6 | MB 325.6 | OAT | blue | Correct. Mercedes Benz 325.6 is OAT, blue. |
| SLLC | Super Long Life Coolant | OAT | pink | Correct. Toyota SLLC is OAT-based, pink/red. |
| MS591 | MS 591-08 | P-OAT | green | Correct. Hyundai/Kia MS 591-08 is P-OAT (phosphate OAT), green. |
| FL22 | FL22 | P-OAT | green | Correct. Mazda FL22 is P-OAT, green. |
| TYPE2 | Type 2 / e-Coolant | OAT | blue | Correct. Honda Type 2 is silicate-free OAT, blue. |
| L250 | L250 / L248 | Si-OAT | blue/green | Correct. Nissan Long Life Coolant L250 is Si-OAT, blue-green. |
| TYPED | Type D (Glaceol RX) | OAT | yellow | Correct. Renault/Peugeot/Citroen Type D is OAT, yellow. |

The 15 `_BRAND_DEFAULTS` entries are accurate. VW Group (VW, Skoda, SEAT, Audi) all use G13. BMW/MINI use LC-18. Mercedes uses MB 325.6. Toyota/Lexus SLLC. Hyundai/Kia MS591. Mazda FL22. Honda TYPE2. Nissan/Infiniti L250. Renault/Peugeot/Citroen TYPED.

**One chemistry note on the compatibility matrix:** The HOAT/IAT asymmetry (`"HOAT": {"HOAT", "IAT"}` but `"IAT": {"IAT"}`) is defensible from a practical standpoint — HOAT contains IAT inhibitors so topping up with IAT in a HOAT system is generally considered acceptable in workshop practice, whereas putting HOAT into a pure IAT system is less well-characterized. The handoff explains this reasoning and it is consistent with common aftermarket guidance. This is a documented, intentional choice, not an error.

The Si-OAT/OAT asymmetry is also correctly implemented and explained: Si-OAT compatible with OAT because Si-OAT is OAT + silicate; OAT warns against Si-OAT because adding silicate to a pure OAT system changes the inhibitor profile. This is accurate.

---

### 6. Pattern Compliance Comparison

| Dimension | OilLookup | BulbLookup | CoolantLookup | Compliant? |
|-----------|-----------|------------|---------------|------------|
| Result type in `models.py` | OilResult | BulbResult | CoolantResult | Yes |
| Lookup class in `lookup/` subpackage | No (root) | Yes | Yes | Yes (consistent with bulb) |
| `_resolve_names` method | Identical | Identical | Identical | Yes (tri-duplication noted) |
| Cascade guard `if model and engine_code` | Yes | Yes | Yes | Yes |
| Empty-data guard before returning result | `not specs.oil_viscosity` | `_has_bulb_data()` | `_resolve_coolant_type()` | Yes (appropriate to domain) |
| Purpose-specific DB method for tier 2 | N/A (uses engine family) | `find_specs_by_model_year` | `find_specs_by_model_year_for_coolant` | Yes |
| In-memory DB in tests | Yes | Yes | Yes | Yes |
| `setUp`/`tearDown` pattern | Yes | Yes | Yes | Yes |
| `_try_*` private method naming | Yes | Yes | Yes | Yes |

**The `_resolve_names` tri-duplication** is now overdue per the handoff's own flag ("refactor if 3rd consumer appears"). Acknowledged. This review categorizes it as SUGGESTION-1 rather than WARNING because the code is harmless-as-is, but it should be tracked and executed before a fourth lookup class is added.

**The `_specs_to_result()` helper** is a new pattern not present in OilLookup or BulbLookup. It consolidates what those lookups do per-tier into a shared helper. This is a reasonable improvement, but it comes with the type narrowing cost described in WARNING-2. Once that type is fixed, the helper pattern is worth documenting for future lookup modules.

---

### 7. `lookup/__init__.py`

**Status: PASS**

Still empty. Consistent with the prior state. SUGGESTION-2 from the bulb review (add re-exports) still applies and is still deferred. No new issue here.

---

## Handoff Accuracy

All claims in the handoff match the implementation:

- "9 coolant specs, 15 car makes" — confirmed at `coolant.py:51-74`
- "CoolantResult frozen dataclass with two factory classmethods" — confirmed at `models.py:296-365`
- "three-tier cascade: exact -> model-year -> brand default" — confirmed at `coolant.py:147-175`
- "asymmetric compatibility matrix (5 technology types)" — confirmed at `coolant.py:78-84`
- "`_resolve_coolant_type` normalizer" — confirmed at `coolant.py:109-132`
- "20 new tests" — confirmed by count and test run
- "210 total tests passing" — confirmed

One minor inaccuracy: the handoff says "same as `find_specs_by_model_year` but ORDER BY prefers records with `coolant_type != ''` instead of `low_beam_bulb != ''`." This is accurate. However, the handoff omits mentioning that the ORDER BY behavior is already applied proactively — the bulb review's W3 fix added `ORDER BY (CASE WHEN low_beam_bulb != '' THEN 0 ELSE 1 END)` to `find_specs_by_model_year`. So both methods now have the ordered fallback, which is correct and consistent.

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/coolant_lookup.md`) | MISSING | No `docs/modules/` directory exists in parts-finder. Same state as bulb review. Not a blocker. |
| `DECISIONS.md` in parts-finder | MISSING | Handoff captures decisions. Should be promoted before API layer is built. |
| `IMPLEMENTATION_PROGRESS.md` in parts-finder | MISSING | Not needed at current project state. |
| `docs/architecture/patterns.md` in parts-finder | MISSING | `_specs_to_result()` helper is a new pattern worth documenting when it stabilizes. |
| ROADMAP.md in parts-finder | MISSING | No ROADMAP exists; exit criteria only in handoff. |

---

## Findings Summary

### BLOCKERS

None.

### WARNINGS (must fix)

| # | Finding | File | Fix |
|---|---------|------|-----|
| W1 | `find_specs_by_model_year_for_coolant()` has no direct DB-layer test | `test_db.py` | Add `TestCoolantModelYearQuery` class mirroring `TestModelYearQuery` |
| W2 | `_specs_to_result()` source typed as `str` not `Literal` | `coolant.py:236` | Change to `source: Literal["exact", "model_year"]` |
| W3 | `test_oat_warning` missing HOAT assertion | `test_coolant_lookup.py:97-105` | Add `self.assertIn("HOAT", warning)` |

### SUGGESTIONS (recommended)

| # | Finding | Notes |
|---|---------|-------|
| S1 | `_resolve_names` tri-duplication overdue for extraction | Extract to `lookup/_shared.py` before 4th lookup class |
| S2 | `_specs_to_result()` helper pattern worth documenting | New pattern not in Oil/Bulb — document after Literal fix |
| S3 | `lookup/__init__.py` re-exports still deferred | Three lookup classes now; import asymmetry growing |

---

## Fix Documentation Requirement

After applying all fixes listed above, the implementor MUST:
1. Add a "## Fixes Applied" section to this review file
2. For each fix: state what was changed, which file:line, and why
3. Re-run the affected tests and record pass/fail counts
4. Update any progress tracking file with a dated entry noting the fixes
5. Re-run master-reviewer OR self-verify against each WARNING above

---

## Fixes Applied

### W1: Added `TestCoolantModelYearQuery` to `test_db.py`

**What:** Added 6-test class `TestCoolantModelYearQuery` at `test_db.py:340-399`, mirroring `TestModelYearQuery` structure.
**Tests:** `test_match_within_year_range`, `test_year_boundaries`, `test_model_mismatch_returns_none`, `test_ignores_engine_code`, `test_prefers_populated_coolant_row`, `test_empty_db_returns_none`.
**Key test:** `test_prefers_populated_coolant_row` — inserts empty coolant row first, populated row second, confirms ORDER BY returns the populated row.
**Why:** Establishes consistent pattern with `TestModelYearQuery` (bulb review W2 fix). The critical ORDER BY preference behavior is now tested at the DB layer, not just indirectly through the lookup cascade.

### W2: Type-narrowed `_specs_to_result` source parameter

**What:** Changed `source: str` to `source: Literal["exact", "model_year"]` at `coolant.py:236`. Added `Literal` to the `typing` import.
**Why:** Type checkers can now catch typos at the two call sites (`_try_exact` and `_try_model_year`). `"brand_default"` is correctly excluded since it flows through `_try_brand_default()` → `CoolantResult.from_brand_default()` which hard-codes the value.

### W3: Added HOAT assertion to `test_oat_warning`

**What:** Added `self.assertIn("HOAT", warning)` at `test_coolant_lookup.py:101`.
**Why:** OAT's compatibility set is `{"OAT"}`, so incompatible types are `{HOAT, IAT, P-OAT, Si-OAT}`. The test now verifies all four incompatible types, not just three. This closes a safety-relevant test gap.

### Verification

```
pytest parts-finder/tests/ -v
216 passed, 0 failed (26 new coolant/DB tests + 190 existing)
```

---

## Verdict

**CHANGES NEEDED** → **APPROVED** (all 3 warnings resolved)
