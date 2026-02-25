# Implementation Handoff: Filtron XLS Parser

**Module:** Filtron Excel catalog parser (Phase 4.1 — first vendor parser)
**Review Tier:** 2 (Standard — new sub-package, new dataclass, 49 tests)
**Date:** 2026-02-24
**Branch:** main

## What Changed

- **Created `parsers/` sub-package** with `__init__.py` marker, establishing the pattern for future vendor parsers (Bosch, Mahle, etc.)
- **Added `FiltronRecord`** non-frozen dataclass as intermediate representation for a single catalog row (make, model, year range, engine code, filter type, Filtron/OEM/MANN part numbers)
- **Added `FiltronParser`** class with automatic header detection (scans rows 1..10 for >= 3 recognized columns), year range parsing, filter type resolution (explicit → part-prefix inference → "unknown"), and multi-part OEM cell splitting
- **Added `to_crossrefs()`** static method producing `ProductCrossRef` instances — up to 2 per record (Filtron + MANN), multiplied by each OEM part number
- **Added `to_specs_updates()`** static method grouping records by vehicle identity and collecting OEM parts into category-specific fields (oil_filter_oem, air_filter_oem, etc.)
- **49 new tests** across 10 test classes, all using synthetic Excel workbooks

## Files Changed

- `parts-finder/src/parts_finder/parsers/__init__.py` (NEW) — empty package init
- `parts-finder/src/parts_finder/parsers/filtron_parser.py` (NEW) — `FiltronRecord` dataclass, `FiltronParser` class, `COLUMN_ALIASES` (~40 entries), `FILTER_TYPE_MAP` (~25 entries), `_PART_PREFIX_TO_TYPE` mapping, helper functions (`parse_year_range`, `resolve_filter_type`, `split_multi_part`)
- `parts-finder/tests/test_filtron_parser.py` (NEW) — 49 tests across 10 classes using synthetic openpyxl workbooks

## Key Decisions Made

1. **`FiltronRecord` is non-frozen** — Unlike the domain models (`VehicleSpecs`, `ProductCrossRef`) which are frozen, the intermediate record is mutable so parsing helpers can build it up incrementally during row processing. It never leaves the parser boundary.
2. **`to_specs_updates()` returns `list[dict]`, not `list[VehicleSpecs]`** — Filtron catalogs lack `fuel_type` (a required `VehicleSpecs` field). Returning dicts lets the import orchestrator decide how to merge partial data.
3. **Slash is NOT a delimiter for OEM part splitting** — Part numbers like `04E115561H` don't contain slashes, but cross-reference codes like `OX 388D / OX 381D` do. Splitting on slash would corrupt MANN/Bosch part numbers. Only comma and semicolon are used as delimiters.
4. **Filter type fallback chain: explicit → prefix → "unknown"** — The `resolve_filter_type()` function tries the `FILTER_TYPE_MAP` lookup first, then infers from Filtron part-number prefix (OP→oil, AP/AK→air, K→cabin, PP→fuel). Records with "unknown" type are excluded from both `to_crossrefs()` and `to_specs_updates()`.
5. **Header auto-detection with 3-column minimum** — A row qualifies as header if >= 3 cells map to recognized column aliases AND at least one is `make` or `filtron_part`. This handles catalogs with title/logo rows above the real header.
6. **English + Polish column aliases** — Filtron is a Polish company (filtron.eu). Their catalogs may use Polish headers ("filtr oleju", "rok produkcji") or English ones. Both are mapped to canonical field names.

## What I'm Unsure About

- **Actual Filtron catalog format** — This parser was built against the plan's specification, not a real file. Column aliases and filter type maps may need expansion once a real `.xlsx` is available.
- **Merged cells** — The parser assumes no merged cells spanning multiple data rows. If Filtron catalogs use merged cells for make/model grouping, `openpyxl`'s `read_only=True` mode may return `None` for merged sub-cells.
- **`OPEN_ENDED_YEAR = datetime.date.today().year`** — Evaluated at import time. In a long-running process this would be stale across year boundaries, but for a CLI import tool this is fine.

## Test Results

```
pytest parts-finder/tests/ -v
189 passed, 0 failed (49 new filtron parser tests + 140 existing)
```

## Exit Criteria Status

- [x] `parsers/` sub-package created
- [x] `FiltronRecord` intermediate dataclass
- [x] `FiltronParser.parse()` with header auto-detection
- [x] Year range parsing (dash, open-ended, "from", single, en-dash, invalid)
- [x] Filter type resolution (explicit + prefix inference)
- [x] Multi-part OEM cell splitting (comma/semicolon, not slash)
- [x] `to_crossrefs()` producing `ProductCrossRef` instances (Filtron + MANN dual entries)
- [x] `to_specs_updates()` grouping by vehicle identity
- [x] 10 test classes (49 tests total), all using synthetic workbooks
- [x] py_compile passes
- [x] All 189 tests pass (0 regressions)

## Docs Written/Updated

- This handoff doc (`docs/reviews/filtron-parser-handoff.md`)
