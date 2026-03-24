# Seed Database — Handoff Document

## What Was Built

Three files that extract ~193 vehicle oil specs from `oil-finder-free.jsx` and load them into the Parts Finder SQLite database.

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/seed_from_jsx.py` | ~210 | Core parser: JS→JSON conversion, fuel type extraction, CSV export |
| `scripts/seed_database.py` | ~75 | Orchestrator: JSX → CSV → SQLite (calls seed_from_jsx + import_data) |
| `tests/test_seed_from_jsx.py` | ~290 | 55 tests: unit, integration, acceptance, end-to-end |

**No existing files modified.**

## Key Design Decisions

1. **Regex-based JS→JSON** — No JS runtime dependency. Two regex passes handle unquoted keys and trailing commas. Works because OIL_DB uses a limited JS subset.

2. **Priority-ordered fuel extraction** — 11 rules checked in sequence (Electric → e-Skyactiv X → eHybrid → PHEV → Mild Hybrid → Hybrid → Diesel → Petrol → BMW D/T suffixes → sport keywords → fallback petrol).

3. **EV entries skipped** — 3 Electric entries (Kona, Ioniq, Niro) have N/A values and no oil data. Silently excluded.

4. **OEM "None" → ""** — JS string "None" in Toyota/Suzuki entries mapped to empty string for clean DB storage.

5. **engine_desc as engine_code** — Descriptive labels like "1.6T Petrol" used verbatim since OIL_DB has no real engine codes.

## Test Results

```
55 passed in 0.52s
```

Test coverage:
- 20 fuel type extraction cases (every keyword path + priority ordering)
- 7 JS→JSON conversion cases (unquoted keys, trailing commas, comments, unicode)
- 4 block extraction cases (happy path, missing, nested braces, let/const)
- 6 entry iteration cases (normal, EV skip, OEM mapping)
- 4 CSV export cases (headers, values, dir creation, count)
- 8 acceptance tests on real JSX (190+ entries, 3 EVs, valid fuels, Škoda diacritic)
- 6 end-to-end round-trip tests (JSX → CSV → import → query Toyota/BMW/EV/OEM)

## Usage

```bash
# Extract CSV only
python scripts/seed_from_jsx.py

# Full pipeline: JSX → CSV → SQLite
python scripts/seed_database.py --db data/parts.db

# Custom year range
python scripts/seed_database.py --db parts.db --year-from 2019 --year-to 2024
```

## Output

- **193 rows** exported (196 total - 3 EVs)
- 9 makes, 47 models
- CSV at `data/seed/oil_db_seed.csv`
- Compatible with existing `import_data.py` pipeline (no changes needed)
