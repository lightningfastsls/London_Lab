# Implementation Handoff: Coolant Specification Lookup

**Module:** Coolant Specification Lookup (third category resolver after OilLookup and BulbLookup)
**Review Tier:** 2 (Standard — new module, new DB method, new model type)
**Date:** 2026-02-24
**Branch:** main

## What Changed

- **Added `CoolantResult`** frozen dataclass in `models.py` with `from_vehicle_specs()` and `from_brand_default()` factory methods. Carries spec name, technology type, color, capacity, aftermarket match, mixing warning, and source tier.
- **Added `find_specs_by_model_year_for_coolant()`** DB method — same as `find_specs_by_model_year` but ORDER BY prefers records with `coolant_type != ''` instead of `low_beam_bulb != ''`.
- **Created `CoolantLookup`** class in new `lookup/coolant.py` with three-tier cascade: exact match -> model-year fallback -> brand default. Unique features: brand knowledge base (9 coolant specs, 15 car makes), compatibility matrix (5 technology types), and mixing-warning generation.
- **20 new tests** covering CoolantResult construction, mixing warnings, tier 1, tier 2, tier 3, and cascade integration.

## Files Changed

- `parts-finder/src/parts_finder/models.py` (MODIFIED) — added `CoolantResult` frozen dataclass with two factory classmethods
- `parts-finder/src/parts_finder/db.py` (MODIFIED) — added `find_specs_by_model_year_for_coolant()` method
- `parts-finder/src/parts_finder/lookup/coolant.py` (NEW) — `CoolantLookup` class, `CoolantSpec` NamedTuple, `_COOLANT_SPECS` knowledge base, `_BRAND_DEFAULTS` mapping, `COMPATIBILITY_MATRIX`, `_build_mixing_warning()`, `_resolve_coolant_type()` normalizer
- `parts-finder/tests/test_coolant_lookup.py` (NEW) — 20 tests across 6 test classes

## Key Decisions Made

1. **CoolantResult lives in `models.py`, not `coolant.py`** — Follows the established OilResult/BulbResult pattern where all domain result types live in the models module.
2. **Brand knowledge base is hardcoded, not in the DB** — Coolant technology standards are manufacturer-wide conventions (all VW group uses G13, all Toyota uses SLLC). These don't vary per-vehicle like oil capacity does, so a hardcoded dict is the right data structure. The DB stores `coolant_type` abbreviations; the knowledge base decodes them.
3. **Purpose-specific `find_specs_by_model_year_for_coolant()` method** — The existing `find_specs_by_model_year()` ORDER BY prefers bulb-populated records. If the DB has two records for the same model — one with bulb data but no coolant, another with coolant but no bulb — the bulb query returns the wrong one for coolant purposes. This follows the established pattern noted in the bulb handoff: "refactor flag if 3rd consumer appears."
4. **Three tiers matching oil's cascade depth** — Tier 3 (brand default) is viable because coolant specs are manufacturer-wide. Unlike bulbs (where brand aggregation is meaningless), coolant type is determined by the carmaker's chosen inhibitor technology.
5. **Asymmetric compatibility matrix** — Si-OAT can mix with OAT (it's OAT + silicate), but OAT warns against Si-OAT (adding silicate to a pure OAT system is undesirable). Similarly, HOAT is compatible with IAT (it contains IAT inhibitors), but IAT warns against HOAT. This models real-world coolant chemistry accurately.
6. **`_resolve_names` duplicated (3rd consumer now)** — Identical logic exists in OilLookup, BulbLookup, and now CoolantLookup. The bulb handoff flagged "refactor if 3rd consumer appears." This is now due for extraction to a shared utility when the next module is planned.
7. **`_resolve_coolant_type` as a normalizer** — The DB stores whatever abbreviation was entered ("G13", "TL 774 J", etc.). This function normalizes via exact key match first, then substring match against spec names. Pragmatic approach for messy user-entered data.

## What I'm Unsure About

- **`_resolve_names` refactor is now overdue** — Three identical copies exist. Should be extracted to a shared utility (e.g., `lookup/_shared.py` or a base class) in the next module implementation. The duplication is harmless but violates DRY at 3 consumers.
- **`_resolve_coolant_type` substring matching** — The fallback substring match (checking if the normalized input appears in the spec name) could produce false positives with very short inputs. In practice, DB values are always spec abbreviations, but this is worth noting.
- **Compatibility matrix completeness** — The matrix covers 5 common technology types. Some newer formulations (e.g., Dex-Cool, Asian-vehicle-specific formulas) could be added. The current set covers the 15 brands in `_BRAND_DEFAULTS`.
- **No index for coolant model-year query** — Same situation as the bulb query: the existing `idx_specs_lookup` indexes `(make, model, engine_code)`. The coolant model-year query doesn't use `engine_code`. Fine at current data volume.

## Test Results

```
pytest parts-finder/tests/ -v
210 passed, 0 failed (20 new coolant tests + 190 existing)
```

## ROADMAP Exit Criteria Status

- [x] CoolantResult frozen dataclass with factory methods
- [x] `find_specs_by_model_year_for_coolant()` DB method
- [x] Brand knowledge base (9 specs, 15 makes)
- [x] Compatibility matrix (5 technology types)
- [x] Mixing-warning generation
- [x] CoolantLookup three-tier cascade (exact -> model-year -> brand default)
- [x] `_resolve_coolant_type` normalizer for DB values
- [x] Full test coverage (construction, mixing warnings, tier 1, tier 2, tier 3, cascade, edge cases)
- [x] py_compile passes on all files
- [x] All 210 tests pass (0 regressions)

## Docs Written/Updated

- `parts-finder/docs/reviews/coolant-lookup-handoff.md` — this file
- No existing DECISIONS.md or IMPLEMENTATION_PROGRESS.md in parts-finder/ — key decisions captured in this handoff
