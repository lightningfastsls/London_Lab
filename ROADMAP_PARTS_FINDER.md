# Parts Finder — Implementation Roadmap

> **Product:** License Plate → Vehicle ID → Spare Parts Lookup for the Israeli market.
> **Categories:** Oil Specs · Oil Filters · Air Filters · Cabin Filters · Brake Pads & Discs · Bulbs · Coolant
> **Client:** Tevel Group
> **Source plan:** `parts-finder-plan.docx` (February 2026)
>
> All code lives under `parts-finder/` in this repo as a portable project-within-a-project.
> When the target repo is available, move the entire `parts-finder/` directory there.
> `/implement` blocks reference paths relative to `parts-finder/`.
>
> **Existing asset:** `oil-finder-free.jsx` (repo root) — working React prototype with static oil database
> covering Toyota, Hyundai, Kia, VW, Škoda, BMW, Mercedes-Benz, Mazda, Suzuki (~80+ vehicle configs).
> This is the starting point for the frontend (Phase 7.1) and its `OIL_DB` object can seed the SQLite database.

---

## How to Use This File

1. Work through modules **in order** within each phase (dependencies are noted)
2. Each module has a **`/implement` command** — copy-paste into Claude Code
3. Phase gates must pass before starting the next phase
4. **Data compilation** (manually looking up specs in manufacturer catalogs) happens alongside code work. The code provides import tools; the data itself is compiled from free public sources.

## Status Key

- **DONE** — Implemented and tested
- **READY** — Dependencies met, can start
- **BLOCKED** — Waiting on dependency or external input
- **FUTURE** — Not yet prioritized

## Directory Structure

```
parts-finder/
├── src/
│   └── parts_finder/
│       ├── __init__.py
│       ├── config.py              # AppConfig frozen dataclass
│       ├── plate_client.py        # data.gov.il API client
│       ├── name_mapper.py         # Hebrew → English vehicle names
│       ├── models.py              # Data models (frozen dataclasses)
│       ├── db.py                  # SQLite database layer
│       ├── lookup/                # Category-specific lookup modules
│       │   ├── __init__.py
│       │   ├── oil.py
│       │   ├── filters.py
│       │   ├── bulbs.py
│       │   ├── coolant.py
│       │   └── brakes.py
│       ├── api/                   # FastAPI backend
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── routes.py
│       │   └── fallback.py        # Claude AI fallback
│       └── parsers/               # Data import parsers
│           ├── __init__.py
│           └── filtron_parser.py
├── scripts/
│   ├── import_data.py             # CSV → DB import CLI
│   ├── validate_api.py            # Test plate API with known vehicles
│   └── compile_bulbs.py           # OSRAM data compilation helper
├── tests/
│   ├── conftest.py
│   ├── test_plate_client.py
│   ├── test_name_mapper.py
│   ├── test_db.py
│   ├── test_lookup/
│   │   ├── test_oil.py
│   │   ├── test_filters.py
│   │   ├── test_bulbs.py
│   │   ├── test_coolant.py
│   │   └── test_brakes.py
│   └── test_api.py
├── data/                          # Compiled lookup data (CSV/JSON)
│   ├── hebrew_names.json
│   ├── oil_specs.csv
│   ├── bulb_types.csv
│   ├── filter_crossref.csv
│   ├── coolant_specs.csv
│   └── brake_parts.csv
├── requirements.txt
└── README.md
```

---

## Phase 1: Foundation & License Plate Lookup

### 1.1 Project Scaffold & Config

**What:** Create the parts-finder directory structure, Python package layout, config dataclass, and requirements.
**Status:** READY
**Review Tier:** 1
**Depends on:** None

/implement Parts Finder Scaffold

Create the parts-finder project scaffold with a frozen config dataclass and package structure.

**Context:** This is a portable project living inside the USV repo temporarily. All paths are under `parts-finder/`. Follow the frozen dataclass pattern from `src/usv_spectrogram/detection/config.py`. Use FastAPI for backend, sqlite3 for database, httpx for async HTTP calls to data.gov.il.

**Files to create:**

1. `parts-finder/src/parts_finder/__init__.py` (NEW) — Package init with version

```python
__version__ = "0.1.0"
```

2. `parts-finder/src/parts_finder/config.py` (NEW) — Application configuration

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class AppConfig:
    """Configuration for the Parts Finder application."""

    # data.gov.il API
    gov_api_base_url: str = "https://data.gov.il/api/3/action/datastore_search"
    vehicle_resource_id: str = "053cea08-09bc-40ec-8f7a-156f0677aff3"
    api_timeout_s: float = 10.0

    # Database
    db_path: str = "parts_finder.db"

    # Cache
    cache_vehicle_lookups: bool = True   # Vehicle data doesn't change — cache indefinitely

    # AI Fallback
    ai_fallback_enabled: bool = True
    ai_model: str = "claude-haiku"       # ~$0.001/query for edge cases
    ai_max_retries: int = 2

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    def __post_init__(self) -> None:
        if self.api_timeout_s <= 0:
            raise ValueError("api_timeout_s must be positive")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be 1-65535")
```

3. `parts-finder/requirements.txt` (NEW) — Dependencies

```
fastapi>=0.109.0
uvicorn>=0.27.0
httpx>=0.27.0
anthropic>=0.40.0
openpyxl>=3.1.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

4. `parts-finder/tests/conftest.py` (NEW) — Test configuration with path bootstrap

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

**Test plan:**
```
1. Import parts_finder package — no import errors
2. Create AppConfig with defaults — all fields have sensible values
3. Create AppConfig with invalid port — raises ValueError
4. Create AppConfig with negative timeout — raises ValueError
5. Verify AppConfig is frozen — assignment raises FrozenInstanceError
```

**Exit criteria:**
- [ ] Directory structure created matching the layout above
- [ ] `AppConfig` dataclass is frozen with validation
- [ ] `requirements.txt` lists all needed dependencies
- [ ] `conftest.py` bootstraps import path
- [ ] All tests pass
- [ ] py_compile passes on all new files

---

### 1.2 data.gov.il API Client

**What:** Async Python client that queries the Israeli vehicle registry by license plate number and returns structured vehicle data.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 1.1

/implement data.gov.il API Client

Build an async HTTP client that queries the Israeli government vehicle registry API to resolve license plate numbers to vehicle identification data.

**Context:** The Israeli Ministry of Transport publishes the national vehicle registry (~4M records) as open data via a CKAN REST API. No API key needed, no rate limits published, free for commercial use per Government Resolution No. 1933.

**API Details:**
- Endpoint: `GET https://data.gov.il/api/3/action/datastore_search`
- Query params: `resource_id=053cea08-09bc-40ec-8f7a-156f0677aff3`, `filters={"mispar_rechev":"<PLATE>"}`, `limit=1`
- Response is JSON with `result.records` array containing vehicle data

**Key response fields to extract:**

| API Field | Hebrew | Maps To | Use |
|-----------|--------|---------|-----|
| `tozeret_nm` | שם תוצרת | make_hebrew | Make in Hebrew (e.g., טויוטה) |
| `kinuy_mishari` | כינוי מסחרי | model_hebrew | Model in Hebrew (e.g., קורולה) |
| `shnat_yitzur` | שנת ייצור | year | Production year |
| `degem_manoa` | דגם מנוע | engine_code | **Golden key** for all part lookups |
| `sug_delek_nm` | סוג דלק | fuel_type | Fuel type |
| `misgeret` | מסגרת | vin | VIN (fallback if engine code missing) |
| `ramat_gimur` | רמת גימור | trim | Trim level |

**Files to create:**

1. `parts-finder/src/parts_finder/models.py` (NEW) — Vehicle data model

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class VehicleRecord:
    """Vehicle identification data from data.gov.il registry."""
    plate_number: str
    make_hebrew: str
    model_hebrew: str
    year: int
    engine_code: str          # degem_manoa — the golden key for part lookups
    fuel_type: str
    vin: Optional[str] = None
    trim: Optional[str] = None
    make_english: Optional[str] = None   # Populated by name_mapper
    model_english: Optional[str] = None  # Populated by name_mapper

    @property
    def lookup_key(self) -> str:
        """Canonical key for specs database lookup: make|model|year_range|engine."""
        return f"{self.make_english or self.make_hebrew}|{self.model_english or self.model_hebrew}|{self.year}|{self.engine_code}"
```

2. `parts-finder/src/parts_finder/plate_client.py` (NEW) — Async API client

```python
class PlateClient:
    """Async client for the data.gov.il vehicle registry API."""

    def __init__(self, config: AppConfig):
        ...

    async def lookup(self, plate_number: str) -> VehicleRecord:
        """Query data.gov.il by license plate. Returns VehicleRecord or raises."""
        # 1. Validate plate_number (7-8 digits, strip hyphens)
        # 2. Build query URL with filters={"mispar_rechev": plate_number}
        # 3. GET request with httpx.AsyncClient
        # 4. Parse response — check result.success, extract result.records[0]
        # 5. Map Hebrew field names to VehicleRecord fields
        # 6. Handle: no records found, API errors, timeouts
        ...
```

Key implementation notes:
- Plate numbers should be normalized: strip hyphens, leading zeros, whitespace
- The API returns Hebrew field values — we store them as-is and translate later via name_mapper
- `degem_manoa` (engine code) is THE critical field. If missing, log a warning and populate from VIN decoding later
- Use `httpx.AsyncClient` for non-blocking I/O
- Handle the API returning multiple records (shouldn't happen for a unique plate, but be defensive)
- **⚠ NHTSA VIN decoder does NOT work for Israeli-market vehicles** — the VIN `misgeret` field from data.gov.il is useful as a secondary identifier, but do NOT use the US NHTSA API for decoding it. If VIN decoding is needed as a fallback, use the VIN prefix (WMI) to identify manufacturer only, or explore EU-focused VIN services.

**Test plan:**
```
1. Test plate number normalization (hyphens, spaces, leading zeros stripped)
2. Test successful lookup with mocked API response — returns correct VehicleRecord
3. Test plate not found — raises appropriate error (PlateNotFoundError)
4. Test API timeout — raises TimeoutError with descriptive message
5. Test API error response (success=false) — raises ApiError
6. Test missing engine_code in response — VehicleRecord created with engine_code="" and warning logged
7. Test response with multiple records — uses first record
```

**Exit criteria:**
- [ ] `PlateClient.lookup()` returns `VehicleRecord` from mocked API response
- [ ] All 7 error/edge cases handled
- [ ] Plate number normalization handles common Israeli formats
- [ ] All tests pass (using mocked HTTP, no real API calls in tests)
- [ ] py_compile passes

---

### 1.3 Hebrew-English Name Mapper

**What:** Translation table mapping Hebrew manufacturer/model names to canonical English equivalents, covering the ~20 most common Israeli vehicle brands.
**Status:** READY
**Review Tier:** 1
**Depends on:** Phase 1.1

/implement Hebrew Name Mapper

Build a lookup module that translates Hebrew vehicle manufacturer and model names to canonical English equivalents.

**Context:** The data.gov.il API returns vehicle names in Hebrew (e.g., "טויוטה" for Toyota, "קורולה" for Corolla). We need canonical English names for cross-referencing with international manufacturer catalogs (MANN-FILTER, Brembo, OSRAM, etc.).

The Israeli market is dominated by ~20 brands: Toyota, Hyundai, Kia, Mazda, Nissan, Suzuki, Honda, Mitsubishi, Škoda, VW, Seat, Renault, Peugeot, Citroën, BMW, Mercedes, Audi, Subaru, Chevrolet, Dacia. Japanese/Korean brands make up ~60-70% of the market.

**Files to create:**

1. `parts-finder/data/hebrew_names.json` (NEW) — Mapping table

```json
{
  "makes": {
    "טויוטה": "Toyota",
    "יונדאי": "Hyundai",
    "קיה": "Kia",
    "מאזדה": "Mazda",
    "ניסאן": "Nissan",
    "סוזוקי": "Suzuki",
    "הונדה": "Honda",
    "מיצובישי": "Mitsubishi",
    "סובארו": "Subaru",
    "שברולט": "Chevrolet",
    "סקודה": "Škoda",
    "פולקסווגן": "Volkswagen",
    "סיאט": "SEAT",
    "רנו": "Renault",
    "פיג'ו": "Peugeot",
    "סיטרואן": "Citroën",
    "ב.מ.וו": "BMW",
    "מרצדס": "Mercedes-Benz",
    "אאודי": "Audi",
    "דאצ'יה": "Dacia",
    "לקסוס": "Lexus",
    "אופל": "Opel",
    "פורד": "Ford",
    "ג'יפ": "Jeep"
  },
  "models": {
    "קורולה": "Corolla",
    "יאריס": "Yaris",
    "קאמרי": "Camry",
    "ראב 4": "RAV4",
    "לנד קרוזר": "Land Cruiser",
    "i10": "i10",
    "i20": "i20",
    "i30": "i30",
    "טוסון": "Tucson"
  }
}
```

(Note: The models table will grow over time. Start with the top sellers; the lookup should gracefully handle missing translations.)

2. `parts-finder/src/parts_finder/name_mapper.py` (NEW) — Translation module

```python
class NameMapper:
    """Translates Hebrew vehicle names to canonical English."""

    def __init__(self, mapping_path: Path):
        # Load hebrew_names.json
        ...

    def translate_make(self, hebrew_make: str) -> str:
        """Return English make name, or original Hebrew if not found."""
        ...

    def translate_model(self, hebrew_model: str) -> str:
        """Return English model name, or original Hebrew if not found."""
        ...

    def enrich_vehicle(self, record: VehicleRecord) -> VehicleRecord:
        """Return a new VehicleRecord with English names populated."""
        # Since VehicleRecord is frozen, create a new instance with translated names
        ...
```

Implementation notes:
- Hebrew matching should be fuzzy: strip whitespace, handle alternate spellings
- If no translation found, return the original Hebrew string (don't fail)
- Log untranslated names so we can add them to the mapping later (miss tracking)
- The mapping file is the source of truth — easy to extend without code changes

**Test plan:**
```
1. Test known make translation (טויוטה → Toyota)
2. Test known model translation (קורולה → Corolla)
3. Test unknown make — returns original Hebrew, logs warning
4. Test unknown model — returns original Hebrew, logs warning
5. Test enrich_vehicle — returns new VehicleRecord with English fields populated
6. Test whitespace handling in Hebrew input
7. Test empty string input — returns empty string
```

**Exit criteria:**
- [ ] `hebrew_names.json` covers top 20+ Israeli vehicle brands
- [ ] `NameMapper.translate_make()` and `translate_model()` work for all mapped names
- [ ] Unknown names degrade gracefully (return original, log miss)
- [ ] All tests pass
- [ ] py_compile passes

---

## Phase 1 Gate

- [ ] `PlateClient` can query data.gov.il and return `VehicleRecord` (tested with mocks)
- [ ] `NameMapper` translates top 20 Israeli brands Hebrew → English
- [ ] Project structure is clean and portable
- [ ] All tests passing

---

## Phase 2: Vehicle Specs Database

### 2.1 Database Schema & Models

**What:** SQLite database schema for storing vehicle specifications across all 7 product categories, plus a product cross-reference table mapping OEM parts to aftermarket equivalents.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 1.1

/implement Vehicle Specs Database

Build the SQLite database layer with schema for vehicle specs (all 7 categories) and product cross-references.

**Context:** This is the core lookup database. Once we know a vehicle's make, model, year, and engine code (from data.gov.il), we look up specs here. The database is essentially READ-ONLY at runtime — it's populated offline via data import scripts, then queried by the API.

Use plain `sqlite3` (no ORM) with frozen dataclasses for the data models. This keeps the code simple and consistent with the project's existing patterns.

**Files to create:**

1. `parts-finder/src/parts_finder/models.py` (MODIFY — add spec models after VehicleRecord)

```python
@dataclass(frozen=True)
class VehicleSpecs:
    """Full specifications for a vehicle across all 7 product categories."""
    make: str
    model: str
    year_from: int
    year_to: int
    engine_code: str
    fuel_type: str

    # Oil
    oil_viscosity: str = ""          # e.g., '0W-20'
    oil_spec: str = ""               # e.g., 'API SP, ILSAC GF-6A'
    oil_capacity_l: float = 0.0
    oil_change_km: int = 0

    # Filters (OEM part numbers)
    oil_filter_oem: str = ""         # e.g., '04152-YZZA1'
    air_filter_oem: str = ""
    cabin_filter_oem: str = ""

    # Brakes
    brake_front_type: str = ""       # e.g., 'disc-ventilated'
    brake_front_dia_mm: int = 0
    brake_pad_front_oem: str = ""
    brake_pad_rear_oem: str = ""
    brake_disc_front_oem: str = ""
    brake_disc_rear_oem: str = ""

    # Bulbs (ECE types per position)
    bulb_low_beam: str = ""          # e.g., 'H11'
    bulb_high_beam: str = ""         # e.g., 'HB3'
    bulb_front_turn: str = ""
    bulb_rear_turn: str = ""
    bulb_tail_brake: str = ""
    bulb_reverse: str = ""
    bulb_fog: str = ""
    bulb_license: str = ""

    # Coolant
    coolant_spec: str = ""           # e.g., 'Toyota SLLC (pink)'
    coolant_capacity_l: float = 0.0

@dataclass(frozen=True)
class ProductCrossRef:
    """Maps an OEM part number to aftermarket equivalents."""
    oem_part_number: str
    category: str                    # 'oil_filter', 'air_filter', 'cabin_filter', 'brake_pad', 'brake_disc'
    brand: str                       # e.g., 'MANN-FILTER'
    brand_part_number: str           # e.g., 'W 712/83'
    notes: str = ""
```

2. `parts-finder/src/parts_finder/db.py` (NEW) — Database layer

```python
class PartsDatabase:
    """SQLite database for vehicle specs and product cross-references."""

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS vehicle_specs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        make TEXT NOT NULL,
        model TEXT NOT NULL,
        year_from INTEGER NOT NULL,
        year_to INTEGER NOT NULL,
        engine_code TEXT NOT NULL,
        fuel_type TEXT DEFAULT '',
        -- Oil fields
        oil_viscosity TEXT DEFAULT '',
        oil_spec TEXT DEFAULT '',
        oil_capacity_l REAL DEFAULT 0,
        oil_change_km INTEGER DEFAULT 0,
        -- Filter OEM part numbers
        oil_filter_oem TEXT DEFAULT '',
        air_filter_oem TEXT DEFAULT '',
        cabin_filter_oem TEXT DEFAULT '',
        -- Brake fields
        brake_front_type TEXT DEFAULT '',
        brake_front_dia_mm INTEGER DEFAULT 0,
        brake_pad_front_oem TEXT DEFAULT '',
        brake_pad_rear_oem TEXT DEFAULT '',
        brake_disc_front_oem TEXT DEFAULT '',
        brake_disc_rear_oem TEXT DEFAULT '',
        -- Bulb ECE types
        bulb_low_beam TEXT DEFAULT '',
        bulb_high_beam TEXT DEFAULT '',
        bulb_front_turn TEXT DEFAULT '',
        bulb_rear_turn TEXT DEFAULT '',
        bulb_tail_brake TEXT DEFAULT '',
        bulb_reverse TEXT DEFAULT '',
        bulb_fog TEXT DEFAULT '',
        bulb_license TEXT DEFAULT '',
        -- Coolant
        coolant_spec TEXT DEFAULT '',
        coolant_capacity_l REAL DEFAULT 0,
        -- Constraints
        UNIQUE(make, model, year_from, year_to, engine_code)
    );

    CREATE TABLE IF NOT EXISTS product_crossref (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        oem_part_number TEXT NOT NULL,
        category TEXT NOT NULL,
        brand TEXT NOT NULL,
        brand_part_number TEXT NOT NULL,
        notes TEXT DEFAULT '',
        UNIQUE(oem_part_number, brand, brand_part_number)
    );

    CREATE INDEX IF NOT EXISTS idx_specs_lookup
        ON vehicle_specs(make, model, engine_code);

    CREATE INDEX IF NOT EXISTS idx_crossref_oem
        ON product_crossref(oem_part_number);
    """

    def __init__(self, db_path: str):
        # Connect and create schema if needed
        ...

    def find_specs(self, make: str, model: str, year: int, engine_code: str) -> Optional[VehicleSpecs]:
        """Find specs matching the vehicle. Matches on make+model+engine_code where year is in range."""
        ...

    def find_crossrefs(self, oem_part_number: str) -> list[ProductCrossRef]:
        """Find all aftermarket equivalents for an OEM part number."""
        ...

    def insert_specs(self, specs: VehicleSpecs) -> None:
        """Insert or update a vehicle specs record."""
        ...

    def insert_crossref(self, crossref: ProductCrossRef) -> None:
        """Insert a product cross-reference."""
        ...
```

Key design notes:
- The `vehicle_specs` table uses a composite unique index on (make, model, year_from, year_to, engine_code)
- Year matching uses range: `year_from <= query_year <= year_to` (many specs apply across model years)
- Engine code is the PRIMARY disambiguation key (same model with different engines needs different filters)
- All category fields default to empty string — we populate categories incrementally as data is compiled
- Cross-ref table is normalized: one OEM part can map to multiple aftermarket brands

**Test plan:**
```
1. Create database — schema created without errors
2. Insert VehicleSpecs — record stored and retrievable
3. find_specs exact match — returns correct record
4. find_specs year range match — year within year_from..year_to returns record
5. find_specs no match — returns None
6. Insert ProductCrossRef — stored and retrievable by OEM part number
7. find_crossrefs — returns all brands for one OEM part
8. Duplicate insert — handled gracefully (upsert or ignore)
9. Database file is created at configured path
```

**Exit criteria:**
- [ ] SQLite schema creates cleanly
- [ ] CRUD operations work for both tables
- [ ] Year-range matching works for specs lookup
- [ ] Cross-reference lookup returns all brands for an OEM part
- [ ] All tests pass (using temp database files)
- [ ] py_compile passes

---

### 2.2 Data Import CLI

**What:** Command-line script to import vehicle specs and cross-references from CSV files into the SQLite database.
**Status:** READY
**Review Tier:** 1
**Depends on:** Phase 2.1

/implement Data Import CLI

Build a CLI script that imports CSV data files into the parts finder SQLite database.

**Context:** The specs database is populated offline from manually compiled CSV files. Each product category has its own CSV. The import script reads CSVs and inserts into the database, handling duplicates and validation. Follow the Script CLI Pattern from `docs/architecture/patterns.md`.

**Files to create:**

1. `parts-finder/scripts/import_data.py` (NEW) — CLI importer

```python
"""Import CSV data into the parts finder database.

Usage:
    python import_data.py --db parts_finder.db --specs data/oil_specs.csv
    python import_data.py --db parts_finder.db --crossref data/filter_crossref.csv
    python import_data.py --db parts_finder.db --all data/
"""
```

The script should:
- Accept `--specs <csv>` for vehicle_specs CSV files
- Accept `--crossref <csv>` for product_crossref CSV files
- Accept `--all <directory>` to import all CSVs in a directory
- Validate CSV headers match expected columns
- Report: N inserted, N updated, N skipped (duplicates), N errors
- Use `INSERT OR REPLACE` for idempotent re-imports

2. `parts-finder/data/oil_specs_sample.csv` (NEW) — Sample data for testing

```csv
make,model,year_from,year_to,engine_code,fuel_type,oil_viscosity,oil_spec,oil_capacity_l,oil_change_km
Toyota,Corolla,2019,2023,2ZR-FE,petrol,0W-20,"API SP, ILSAC GF-6A",4.2,15000
Toyota,Corolla,2019,2023,1NR-FE,diesel,5W-30,"ACEA C3",4.5,10000
Hyundai,i30,2017,2023,G4FJ,petrol,5W-30,"API SP",4.0,15000
```

**Test plan:**
```
1. Import sample CSV — records appear in database
2. Re-import same CSV — no duplicates created
3. Import with missing columns — clear error message
4. Import with invalid data types — row skipped, error logged
5. --all mode imports multiple CSVs from directory
6. Report shows correct counts (inserted, updated, skipped)
```

**Exit criteria:**
- [ ] CSV import works for both specs and crossref tables
- [ ] Idempotent — re-running doesn't create duplicates
- [ ] Validation catches malformed CSVs
- [ ] All tests pass
- [ ] py_compile passes

---

## Phase 2 Gate

- [ ] Database schema handles all 7 product categories
- [ ] Import CLI can populate database from CSVs
- [ ] Sample data loads correctly
- [ ] All tests passing

---

## Phase 3: Oil & Bulb Lookup (Quick Wins)

### 3.1 Oil Specification Lookup

**What:** Module that takes a VehicleRecord and returns oil specifications (viscosity, API/ACEA spec, capacity, change interval) from the specs database.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 2.1

/implement Oil Specification Lookup

Build the oil specification lookup module — the first category-specific lookup. This is the highest-coverage, simplest category (oil specs are highly standardized).

**Context:** Oil specifications are defined by viscosity grade (SAE, e.g., "0W-20") and quality standard (API SP, ACEA C3, or OEM-specific like VW 504.00). Given a vehicle's make, model, year, and engine code, we look up the required spec from our database. If no exact match, fall back to common defaults for the brand group.

Key domain knowledge for oil specs:
- Japanese/Korean (~65% of Israeli market): Mostly API SP / ILSAC GF-6A, typically 0W-20 or 5W-30
- European VW/Audi/Škoda/SEAT: VW-specific specs (504.00/507.00)
- BMW: BMW LL-04 (long life) and LL-17 FE+ (newer models)
- Mercedes: MB 229.52 or 229.71
- The spec determines compatibility, NOT the brand (Castrol, Shell, Mobil are interchangeable IF spec matches)

**💡 Existing data:** The `oil-finder-free.jsx` prototype (repo root) contains a complete `OIL_DB` JavaScript object with ~80+ vehicle configurations across 9 Israeli-market brands. This data was compiled from OEM owner's manuals and service documentation. It can be exported to CSV and imported into the SQLite database as the initial oil specs dataset — saving significant compilation effort for Phase 3.1. The data includes viscosity, API/ACEA spec, OEM spec, capacity, and change interval for each engine variant.

**Files to create:**

1. `parts-finder/src/parts_finder/lookup/oil.py` (NEW)

```python
@dataclass(frozen=True)
class OilResult:
    """Oil specification result for a vehicle."""
    viscosity: str             # e.g., '0W-20'
    spec: str                  # e.g., 'API SP, ILSAC GF-6A'
    capacity_l: float          # e.g., 4.2
    change_interval_km: int    # e.g., 15000
    source: str                # 'database' or 'fallback'
    confidence: str            # 'exact', 'year_range', 'engine_family', 'brand_default'

class OilLookup:
    def __init__(self, db: PartsDatabase):
        ...

    def lookup(self, vehicle: VehicleRecord) -> Optional[OilResult]:
        """Look up oil spec for a vehicle. Tries exact match, then progressively looser matches."""
        # 1. Exact: make + model + year + engine_code
        # 2. Year range: make + model + engine_code (year within range)
        # 3. Engine family: make + engine_code prefix (e.g., '2ZR' matches '2ZR-FE', '2ZR-FAE')
        # 4. Brand default: common spec for this make (e.g., Toyota → 0W-20 API SP)
        ...
```

**Test plan:**
```
1. Exact match lookup — returns correct viscosity, spec, capacity
2. Year range match — vehicle year within year_from..year_to
3. Engine family fallback — partial engine code match
4. No match — returns None
5. Result includes confidence level indicating match quality
6. Oil capacity is non-negative float
```

**Exit criteria:**
- [ ] Lookup works with exact, year-range, and engine-family matching
- [ ] Confidence level accurately reflects match quality
- [ ] All tests pass
- [ ] py_compile passes

---

### 3.2 Bulb Type Lookup

**What:** Module that returns the ECE bulb type for each lamp position on a vehicle. Simplest category — all bulbs are universal ECE/SAE types.
**Status:** READY
**Review Tier:** 1
**Depends on:** Phase 2.1

/implement Bulb Type Lookup

Build the bulb lookup module. Unlike other categories, bulbs are universal — an H7 is an H7 regardless of brand. The customer just needs to know the socket type per lamp position.

**Context:** All automotive bulbs conform to ~30-40 ECE/SAE standard types (H1, H4, H7, H11, HB3, W5W, P21W, etc.). Each has a unique physical base making it non-interchangeable. The data model is simply: `{make, model, year, lamp_position} → {ECE_bulb_type}`.

Standard lamp positions (typically 8-12 per vehicle):
- Low beam, High beam, Front fog, Front turn signal
- Rear turn signal, Tail/brake, Reverse, License plate
- (Optional: DRL, side marker, interior)

Note: Modern LED-equipped positions (common on 2020+ vehicles) should be flagged as "LED (non-replaceable)" since there's no aftermarket bulb to sell.

**💡 Data compilation shortcut:** Teoalida.com sells pre-compiled OSRAM (6M+ rows) and Philips (30M+ rows) bulb databases as CSV files for $30-$50. This could bypass the entire manual bulb compilation effort — ready-made structured data with make/model/year/position/bulb-type. Evaluate before doing manual OSRAM scraping. See: teoalida.com (search for "car bulb database").

**Files to create:**

1. `parts-finder/src/parts_finder/lookup/bulbs.py` (NEW)

```python
@dataclass(frozen=True)
class BulbResult:
    """Bulb types for all positions on a vehicle."""
    low_beam: str          # e.g., 'H11' or 'LED'
    high_beam: str         # e.g., 'HB3'
    front_turn: str        # e.g., 'WY21W'
    rear_turn: str
    tail_brake: str
    reverse: str
    fog: str
    license_plate: str
    source: str            # 'database' or 'fallback'

    @property
    def replaceable_positions(self) -> dict[str, str]:
        """Return only positions with replaceable (non-LED) bulbs."""
        ...

class BulbLookup:
    def __init__(self, db: PartsDatabase):
        ...

    def lookup(self, vehicle: VehicleRecord) -> Optional[BulbResult]:
        """Look up bulb types for all positions."""
        ...
```

**Test plan:**
```
1. Lookup known vehicle — returns all bulb positions
2. LED position handling — flagged as non-replaceable
3. replaceable_positions property — filters out LED entries
4. No match — returns None
5. All positions have valid ECE type codes or 'LED'
```

**Exit criteria:**
- [ ] Lookup returns bulb types for all standard positions
- [ ] LED positions correctly identified
- [ ] `replaceable_positions` filters correctly
- [ ] All tests pass
- [ ] py_compile passes

---

## Phase 3 Gate

- [ ] Oil lookup works with multi-level fallback matching
- [ ] Bulb lookup identifies all standard lamp positions
- [ ] Both modules handle missing data gracefully
- [ ] All tests passing

---

## Phase 4: Filter Lookup

### 4.1 Filtron XLS Parser

**What:** Parser for the Filtron (MANN+HUMMEL) downloadable Excel catalog that contains structured filter-to-vehicle data with cross-references. This is the single best free structured dataset for automotive filters.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 2.1

/implement Filtron XLS Parser

Build a parser for the Filtron Excel catalog — the foundational dataset for oil, air, and cabin filter lookups.

**Context:** Filtron (owned by MANN+HUMMEL) publishes a downloadable Excel file with their complete filter catalog including vehicle applications and cross-references. This is the ONLY freely downloadable structured dataset found across all filter manufacturers. It covers ~97% of European vehicles.

The Excel file contains columns for:
- Vehicle make, model, year range
- Engine code / engine displacement
- Filter type (oil, air, cabin, fuel)
- Filtron part number
- OEM part number (cross-reference)
- MANN-FILTER equivalent part number

**Files to create:**

1. `parts-finder/src/parts_finder/parsers/filtron_parser.py` (NEW)

```python
class FiltronParser:
    """Parse the Filtron Excel catalog into structured records."""

    def __init__(self, xls_path: Path):
        self.xls_path = xls_path

    def parse(self) -> list[dict]:
        """Parse the Excel file and return normalized records.

        Each record has: make, model, year_from, year_to, engine_code,
        filter_type, filtron_part, oem_part, mann_part
        """
        # 1. Read with openpyxl
        # 2. Identify header row (may not be row 1)
        # 3. Normalize column names
        # 4. Extract records, skip empty rows
        # 5. Normalize make/model names to English
        # 6. Split year ranges (e.g., "2019-2023" → year_from=2019, year_to=2023)
        ...

    def to_specs_updates(self, records: list[dict]) -> list[dict]:
        """Convert parsed records to vehicle_specs update format (OEM part numbers)."""
        ...

    def to_crossrefs(self, records: list[dict]) -> list[ProductCrossRef]:
        """Convert parsed records to ProductCrossRef entries (OEM → aftermarket)."""
        ...
```

Implementation notes:
- The Excel structure may vary between releases — make header detection flexible
- Some cells may contain multiple part numbers separated by commas or semicolons
- Filter types need mapping: Filtron's type codes → our categories (oil_filter, air_filter, cabin_filter)
- The parser should be robust to dirty data (missing cells, extra whitespace, alternate date formats)

**Test plan:**
```
1. Parse a small test Excel file with known data — correct record count
2. Year range parsing handles various formats ("2019-2023", "2019-", "from 2019")
3. OEM part number extraction is clean (no extra whitespace)
4. Filter type mapping is correct (oil, air, cabin)
5. to_crossrefs produces valid ProductCrossRef objects
6. Empty rows and malformed rows are skipped with warnings
```

**Exit criteria:**
- [ ] Parser reads Filtron XLS and produces structured records
- [ ] Records convert cleanly to database-ready formats
- [ ] Robust to common data quality issues
- [ ] All tests pass (using a small synthetic test Excel file)
- [ ] py_compile passes

---

### 4.2 Filter Lookup Module

**What:** Lookup module for oil, air, and cabin filters — returns OEM part numbers and aftermarket cross-references for a vehicle.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 2.1, Phase 4.1

/implement Filter Lookup Module

Build the filter lookup module covering oil filters, air filters, and cabin filters. Uses OEM part numbers from the specs database and cross-references to aftermarket brands.

**Context:** The strategy is: we store the OEM filter part number per vehicle. The cross-reference table then maps OEM → MANN, Hengst, MAHLE, Bosch, etc. The user sees: "Your oil filter OEM part: 04152-YZZA1 → MANN W 712/83 / Hengst H90W30 / MAHLE OC 613".

**Files to create:**

1. `parts-finder/src/parts_finder/lookup/filters.py` (NEW)

```python
@dataclass(frozen=True)
class FilterResult:
    """Filter lookup result for one filter type."""
    filter_type: str                      # 'oil_filter', 'air_filter', 'cabin_filter'
    oem_part_number: str                  # e.g., '04152-YZZA1'
    aftermarket: list[ProductCrossRef]    # List of brand alternatives
    source: str                           # 'database' or 'fallback'

@dataclass(frozen=True)
class AllFiltersResult:
    """Combined filter results for a vehicle."""
    oil_filter: Optional[FilterResult] = None
    air_filter: Optional[FilterResult] = None
    cabin_filter: Optional[FilterResult] = None

    @property
    def coverage(self) -> str:
        """e.g., '2/3 filters found'"""
        found = sum(1 for f in [self.oil_filter, self.air_filter, self.cabin_filter] if f)
        return f"{found}/3 filters found"

class FilterLookup:
    def __init__(self, db: PartsDatabase):
        ...

    def lookup(self, vehicle: VehicleRecord) -> AllFiltersResult:
        """Look up all 3 filter types for a vehicle."""
        # 1. Get vehicle specs from DB
        # 2. For each filter type, extract OEM part number from specs
        # 3. For each OEM part, find cross-references in product_crossref table
        # 4. Return combined result
        ...
```

**Test plan:**
```
1. Lookup with all 3 filters in DB — returns complete AllFiltersResult
2. Lookup with partial data (only oil filter) — returns partial result, others None
3. Cross-references found for OEM part — aftermarket list populated
4. No cross-references — aftermarket list is empty, OEM part still returned
5. coverage property reports correctly ("2/3 filters found")
6. No specs match — returns AllFiltersResult with all None
```

**Exit criteria:**
- [ ] All 3 filter types looked up from specs database
- [ ] Cross-references resolved from product_crossref table
- [ ] Partial results handled gracefully
- [ ] All tests pass
- [ ] py_compile passes

---

## Phase 4 Gate

- [ ] Filtron parser handles real-world Excel format
- [ ] Filter lookup returns OEM + aftermarket parts for all 3 filter types
- [ ] All tests passing

---

## Phase 5: Coolant & Brake Lookup

### 5.1 Coolant Lookup with Compatibility

**What:** Coolant specification lookup with a mixing-compatibility matrix to warn users about incompatible coolant types.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 2.1

/implement Coolant Lookup

Build the coolant specification lookup module with mixing-compatibility warnings.

**Context:** Coolant specs vary significantly by manufacturer — wrong coolant can damage the engine. European brands have well-documented specs (VW G13, BMW LC-18, MB 325.6). Asian brands often use proprietary formulations. The module should warn users if they're mixing incompatible types.

**Key coolant type mapping (from the plan):**

| Brand Group | Spec | Color | Aftermarket Match |
|-------------|------|-------|-------------------|
| VW/Škoda/SEAT | TL 774 J (G13) | Purple | GLYSANTIN G40/G30 |
| BMW | BMW LC-18 | Blue/Green | GLYSANTIN G48 |
| Mercedes | MB 325.6 | Blue | GLYSANTIN G30 |
| Toyota/Lexus | Super Long Life Coolant | Pink | Any OAT pink coolant |
| Hyundai/Kia | MS 591-08 | Green | P-OAT green coolant |
| Mazda | FL22 | Green | FL22-compatible only |
| Honda | Type 2 / e-Coolant | Blue | Honda-spec OAT |
| Nissan | L250 / L248 | Blue/Green | Si-OAT coolant |
| Renault/PSA | Type D (Glaceol RX) | Yellow | GLYSANTIN G33 |

Coolant technologies (NOT mixable across types):
- IAT (Inorganic Acid Technology) — green, oldest type
- OAT (Organic Acid Technology) — various colors, long-life
- HOAT (Hybrid OAT) — mix of IAT+OAT
- P-OAT (Phosphated OAT) — Asian OEMs
- Si-OAT (Silicated OAT) — some Euro+Asian

**Files to create:**

1. `parts-finder/src/parts_finder/lookup/coolant.py` (NEW)

```python
@dataclass(frozen=True)
class CoolantResult:
    """Coolant specification result."""
    spec: str              # e.g., 'Toyota SLLC (pink)'
    technology: str        # 'OAT', 'HOAT', 'P-OAT', 'Si-OAT', 'IAT'
    color: str             # e.g., 'pink'
    capacity_l: float
    aftermarket_match: str # e.g., 'Any OAT pink coolant'
    mixing_warning: str    # e.g., 'Do NOT mix with green IAT coolant'
    source: str

class CoolantLookup:
    # ... includes COMPATIBILITY_MATRIX class-level constant
    ...
```

**Test plan:**
```
1. Lookup Toyota — returns SLLC pink, OAT technology
2. Lookup VW — returns G13 spec, GLYSANTIN G40 aftermarket match
3. Mixing warning generated for incompatible technologies
4. Capacity returned when available
5. Unknown vehicle — returns None
```

**Exit criteria:**
- [ ] Lookup covers all major brand groups from the compatibility table
- [ ] Mixing warnings generated accurately
- [ ] All tests pass
- [ ] py_compile passes

---

### 5.2 Brake Parts Lookup

**What:** Brake pad and disc lookup — the most complex category due to part variety and the WVA numbering system.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 2.1

/implement Brake Parts Lookup

Build the brake parts lookup module for pads and discs, including cross-references to major aftermarket brands.

**Context:** Brakes are the hardest category because of compound variety (ceramic, semi-metallic, organic), shape complexity, and the proprietary WVA numbering system. However, for our scope (pads + discs for common Israeli vehicles), we store OEM part numbers and cross-reference to Brembo, TRW, Textar, Ferodo, Bosch, ATE, Mintex.

The data comes from: Brembo catalog (primary), Ferodo catalog (best cross-references), Textar Brakebook (expert search by dimensions).

**Files to create:**

1. `parts-finder/src/parts_finder/lookup/brakes.py` (NEW)

```python
@dataclass(frozen=True)
class BrakeResult:
    """Brake parts result for a vehicle."""
    # Front
    front_pad_oem: str
    front_pad_crossrefs: list[ProductCrossRef]
    front_disc_oem: str
    front_disc_crossrefs: list[ProductCrossRef]
    front_type: str             # 'disc-ventilated', 'disc-solid'
    front_disc_dia_mm: int      # e.g., 275

    # Rear
    rear_pad_oem: str
    rear_pad_crossrefs: list[ProductCrossRef]
    rear_disc_oem: str
    rear_disc_crossrefs: list[ProductCrossRef]

    source: str

class BrakeLookup:
    def __init__(self, db: PartsDatabase):
        ...

    def lookup(self, vehicle: VehicleRecord) -> Optional[BrakeResult]:
        """Look up brake pads and discs for front and rear axles."""
        ...
```

**Test plan:**
```
1. Lookup known vehicle — returns front and rear pad/disc OEM numbers
2. Cross-references populated from product_crossref table
3. Disc diameter included when available
4. Partial data (e.g., only front pads known) — returns partial result
5. No match — returns None
```

**Exit criteria:**
- [ ] Front and rear brake parts looked up
- [ ] Cross-references to aftermarket brands resolved
- [ ] Partial results handled gracefully
- [ ] All tests pass
- [ ] py_compile passes

---

## Phase 5 Gate

- [ ] All 7 product categories have working lookup modules
- [ ] Coolant mixing compatibility matrix implemented
- [ ] Brake lookup handles partial data
- [ ] All tests passing

---

## Phase 5.5: End-to-End Demo

### 5.5.1 Seed Database from Existing Assets

**What:** Populate the SQLite database with real vehicle data by converting the ~80+ vehicle configs from `oil-finder-free.jsx` (existing React prototype) into importable CSVs. This gives immediate coverage for Toyota, Hyundai, Kia, VW, Skoda, BMW, Mercedes-Benz, Mazda, Suzuki.
**Status:** READY
**Review Tier:** 1
**Depends on:** Phase 2 (database layer)

/implement Seed Database from Existing Assets

Parse the existing React prototype's oil database and convert it into structured CSV data that can be imported into the SQLite database via `import_data.py`.

**Context:** `oil-finder-free.jsx` (repo root) contains an `OIL_DB` JavaScript object with ~80+ vehicle configs covering Toyota, Hyundai, Kia, VW, Skoda, BMW, Mercedes-Benz, Mazda, Suzuki. Each entry has make/model/year-range/engine/oil-viscosity/capacity/spec/filter-part-number. This is a data extraction + transformation task — parse JS object literals into Python dicts, then write CSVs matching the `vehicle_specs` and `product_xref` table schemas from Phase 2.

**Files to create:**

1. `parts-finder/scripts/seed_from_jsx.py` (NEW) — Parser + CSV exporter

```python
"""Parse oil-finder-free.jsx OIL_DB and export to CSV for database import.

Reads the JS source, extracts the OIL_DB object via regex/AST parsing,
normalizes fields, and writes:
  - vehicle_specs.csv (make, model, year_from, year_to, engine_code, fuel_type)
  - oil_specs.csv (vehicle_ref, viscosity, capacity_l, api_spec, change_interval_km)
  - oil_filters.csv (vehicle_ref, oem_part, aftermarket_brand, aftermarket_part)

Usage:
    python scripts/seed_from_jsx.py --input ../../oil-finder-free.jsx --output-dir data/seed/
"""
```

Key parsing approach:
- Read the `.jsx` file as text
- Extract the `OIL_DB` object block (between its opening `{` and matching closing `}`)
- Use `json5` or regex-based parsing to handle JS object literals (trailing commas, unquoted keys, single-quoted strings)
- Normalize Hebrew model names using the existing `hebrew_mapper` module if available
- Write separate CSVs for vehicles, oil specs, and filter cross-references

2. `parts-finder/data/seed/` (NEW directory) — Output location for generated CSVs

3. `parts-finder/scripts/seed_database.py` (NEW) — Orchestrator that runs seed_from_jsx.py output through import_data.py

```python
"""Orchestrate full database seeding: parse JSX -> CSVs -> SQLite import.

Usage:
    python scripts/seed_database.py --jsx ../../oil-finder-free.jsx --db parts_finder.db
"""
```

**Test plan:**
```
1. Parse OIL_DB from oil-finder-free.jsx — extracts >= 30 vehicle entries
2. Each extracted entry has required fields (make, model, year range, viscosity)
3. Generated vehicle_specs.csv is valid CSV with correct headers
4. Generated oil_specs.csv references valid vehicle entries
5. Import into test database succeeds without constraint violations
6. Round-trip: parse -> CSV -> import -> query returns expected oil spec for known vehicle
```

**Exit criteria:**
- [ ] `seed_from_jsx.py` parses `oil-finder-free.jsx` and extracts >= 30 vehicle configs
- [ ] Output CSVs match the Phase 2 database schema
- [ ] `seed_database.py` runs end-to-end: JSX → CSVs → SQLite
- [ ] All tests pass
- [ ] py_compile passes on all new files

### 5.5.2 CLI Demo Script

**What:** A single script (`scripts/demo_lookup.py`) that takes a license plate, calls the live government API, looks up all available categories, and prints a readable report. This is the end-to-end smoke test that proves the whole pipeline works with real data.
**Status:** READY
**Review Tier:** 1
**Depends on:** 5.5.1 + whichever lookup modules are DONE

/implement CLI Demo Script

Build an end-to-end CLI tool that takes a license plate number, resolves it via the government API, looks up all available parts categories in the local database, and prints a formatted report.

**Context:** This is the integration smoke test that proves the full pipeline works: plate → API → vehicle identity → database lookup → human-readable output. Key design principle: **show what we have, acknowledge what's missing** — each category prints its result or `[not yet in database]`. This lets you demo incrementally without needing all 7 categories populated.

**Files to create:**

1. `parts-finder/scripts/demo_lookup.py` (NEW) — CLI entry point

```python
"""End-to-end parts lookup demo.

Takes a license plate, resolves it via data.gov.il, matches against
the local database, and prints a formatted report across all categories.

Usage:
    python scripts/demo_lookup.py --plate 12-345-67 --db parts_finder.db
    python scripts/demo_lookup.py --plate 1234567    # dashes optional
"""
```

Expected output format:
```
Vehicle: Toyota Corolla 2021 (2ZR-FE, petrol)

Oil:
  Viscosity: 0W-20  |  Capacity: 4.2L  |  Spec: API SP
  Change interval: 15,000 km
  OEM filter: 04152-YZZA1 → Mann W 68/3, Wix WL7459
  Confidence: exact (Toyota Corolla 2019-2023 2ZR-FE)

Coolant:
  Type: Toyota SLLC (pink)  |  Capacity: 6.3L
  Warning: Do NOT mix with green IAT coolant

Filters:   [not yet in database]
Brakes:    [not yet in database]
Bulbs:     [not yet in database]
```

2. `parts-finder/src/parts_finder/lookup_engine.py` (NEW) — Orchestration layer

```python
"""Coordinate plate resolution + multi-category database lookup.

Combines the gov API client, hebrew mapper, vehicle matcher, and
all available category lookup modules into a single query pipeline.
Returns a LookupResult with per-category results or 'not available' markers.
"""
```

Key behaviors:
- Accept plate with or without dashes (normalize to digits-only)
- Call `gov_api_client` to resolve plate → vehicle data
- Translate Hebrew names via `hebrew_mapper`
- Match against `vehicle_specs` table using `vehicle_matcher`
- For each implemented category module, query specs; for unimplemented ones, return a placeholder
- Include confidence level in results (exact match vs. fallback)
- Handle errors gracefully: API timeout, no match found, database missing

**Test plan:**
```
1. demo_lookup.py --help prints usage without errors
2. LookupEngine with mocked API + seeded DB returns correct oil spec for known plate
3. LookupEngine gracefully handles API timeout — prints error, doesn't crash
4. LookupEngine with no database match — prints "Vehicle not found in database" with suggestions
5. Output formatting: each category present, missing categories show [not yet in database]
6. Plate normalization: "12-345-67" and "1234567" produce identical lookups
```

**Exit criteria:**
- [ ] `demo_lookup.py` runs end-to-end with a real plate number (requires network + seeded DB)
- [ ] `LookupEngine` orchestrates API → DB lookup with proper error handling
- [ ] Output shows all categories, with placeholders for unimplemented ones
- [ ] Works for at least 3 different makes from the seeded database
- [ ] All tests pass
- [ ] py_compile passes on all new files

### Phase 5.5 Gate

- [ ] Database seeded with >= 30 real vehicle configs (from `oil-finder-free.jsx`)
- [ ] `demo_lookup.py` runs end-to-end with a real plate and returns oil specs
- [ ] Demo works for at least 3 different makes (Toyota, Hyundai, + one European)

---

## Phase 6: FastAPI Backend

### 6.1 API Endpoints

**What:** FastAPI application with a single main endpoint: POST /api/plate-lookup that takes a license plate and returns the full parts recommendation across all 7 categories.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phases 1-5

/implement FastAPI Backend

Build the FastAPI backend that orchestrates plate lookup → vehicle ID → specs → product recommendations.

**Context:** This is the main API that the frontend calls. One endpoint does everything: takes a license plate number, queries data.gov.il, matches to our specs database, resolves cross-references, and returns a structured JSON response with recommendations for all 7 product categories.

Follow FastAPI best practices: Pydantic models for request/response, dependency injection for database/client instances, proper error handling with HTTP status codes.

**Files to create:**

1. `parts-finder/src/parts_finder/api/app.py` (NEW) — FastAPI application factory

```python
from fastapi import FastAPI
from parts_finder.config import AppConfig

def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    app = FastAPI(title="Parts Finder API", version="0.1.0")
    # Initialize PlateClient, PartsDatabase, NameMapper
    # Register routes
    # Add startup/shutdown events for DB connection
    return app
```

2. `parts-finder/src/parts_finder/api/routes.py` (NEW) — API routes

```python
# POST /api/plate-lookup
# Request: { "plate": "1234567" }
# Response: {
#   "vehicle": { make, model, year, engine_code, fuel_type },
#   "categories": {
#     "oil": { viscosity, spec, capacity_l, ... },
#     "oil_filter": { oem_part, aftermarket: [...] },
#     "air_filter": { ... },
#     "cabin_filter": { ... },
#     "brakes": { front: {...}, rear: {...} },
#     "bulbs": { low_beam: "H11", ... },
#     "coolant": { spec, color, capacity_l, mixing_warning }
#   },
#   "coverage": "6/7 categories matched",
#   "unmatched_categories": ["brakes"],
#   "data_source": "database"  // or "ai_fallback"
# }
```

The endpoint flow:
1. Validate plate number format
2. Check cache for this plate (vehicles don't change — cache indefinitely)
3. If cache miss: query data.gov.il via PlateClient
4. Translate Hebrew names via NameMapper
5. Look up specs in database for each category
6. If any category has no match AND ai_fallback enabled → try Claude Haiku
7. Resolve cross-references for filter/brake OEM parts
8. Return structured response with coverage summary

Error responses:
- 400: Invalid plate format
- 404: Plate not found in government registry
- 503: data.gov.il API unavailable

**Test plan:**
```
1. Full flow with mocked API + populated DB — returns all 7 categories
2. Partial coverage — response includes unmatched_categories list
3. Invalid plate format — 400 error
4. Plate not in registry — 404 error
5. data.gov.il unavailable — 503 error (if not cached)
6. Cached plate — no API call made, response from cache
7. Response JSON matches expected schema
```

**Exit criteria:**
- [ ] POST /api/plate-lookup works end-to-end (with mocks)
- [ ] All 7 categories included in response
- [ ] Proper HTTP error codes for all failure modes
- [ ] Caching prevents redundant API calls
- [ ] All tests pass
- [ ] py_compile passes

---

### 6.2 Claude AI Fallback

**What:** When a vehicle isn't in our specs database, use Claude Haiku to generate plausible specifications, flagged as "unverified" in the response.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 6.1

/implement Claude AI Fallback

Build the AI fallback module that uses Claude Haiku to generate vehicle specs when our database has no match.

**Context:** Expected initial miss rate is ~10-15% of lookups. Rather than showing nothing, we query Claude Haiku (~$0.001/query) to generate specs. The AI result is flagged as 'unverified' and displayed with a disclaimer. Every miss is logged so we can manually verify and add to the database within 24-48 hours — the DB self-improves over time.

Use the Anthropic Python SDK. Claude Haiku is chosen for cost ($0.25/MTok input, $1.25/MTok output) and speed.

**Files to create:**

1. `parts-finder/src/parts_finder/api/fallback.py` (NEW)

```python
class AIFallback:
    """Use Claude Haiku to generate vehicle specs when DB has no match."""

    SYSTEM_PROMPT = """You are a vehicle parts specification expert. Given a vehicle's
    make, model, year, and engine code, provide the correct specifications for:
    oil (viscosity, API/ACEA spec, capacity), filters (OEM part numbers if known),
    bulb types (ECE codes per position), coolant (spec and type), and brake info.

    Respond in JSON format. Only include fields you are confident about.
    Mark uncertain fields as null."""

    def __init__(self, config: AppConfig):
        # Initialize Anthropic client
        ...

    async def generate_specs(self, vehicle: VehicleRecord) -> dict:
        """Query Claude Haiku for vehicle specs. Returns partial dict of specs."""
        # 1. Build prompt with vehicle details
        # 2. Call Claude Haiku API
        # 3. Parse JSON response
        # 4. Log the miss for manual review
        # 5. Return specs dict with source='ai_fallback'
        ...

    def log_miss(self, vehicle: VehicleRecord, ai_response: dict) -> None:
        """Log DB miss for manual review and future addition."""
        # Append to a misses log file (CSV or JSON Lines)
        ...
```

**Test plan:**
```
1. Generate specs with mocked Anthropic API — returns valid specs dict
2. API error — returns empty dict, logs error, doesn't crash
3. Malformed AI response — handled gracefully, returns partial result
4. Miss logged to file with vehicle details and timestamp
5. AI specs marked with source='ai_fallback' in all results
```

**Exit criteria:**
- [ ] Fallback generates plausible specs via Claude Haiku (tested with mocks)
- [ ] All responses clearly flagged as 'ai_fallback'
- [ ] Misses logged for manual review
- [ ] Graceful handling of API errors
- [ ] All tests pass
- [ ] py_compile passes

---

## Phase 6 Gate

- [ ] Full API flow works: plate → vehicle → specs → products
- [ ] Caching prevents redundant government API calls
- [ ] AI fallback handles database misses
- [ ] Coverage reporting accurate
- [ ] All tests passing

---

## Phase 7: Frontend & Integration

### 7.1 React Parts Finder UI

**What:** React frontend component that takes a license plate input and displays parts recommendations across all 7 categories.
**Status:** READY
**Review Tier:** 2
**Depends on:** Phase 6.1

/implement React Parts Finder UI

Build the React frontend that provides the license plate input form and displays the parts recommendation results.

**Context:** The existing prototype `oil-finder-free.jsx` (repo root, 506 lines) is a fully working React component with:
- Static `OIL_DB` covering 9 brands (Toyota, Hyundai, Kia, VW, Škoda, BMW, Mercedes-Benz, Mazda, Suzuki)
- Dropdown UI (Make → Model → Engine) — needs to be converted to license plate input
- Polished dark-themed UI with gradient backgrounds, SVG icons, CSS animations, mobile-responsive layout
- Electric vehicle handling ("no engine oil required" message)
- Placeholder `CATALOG` mapping (TBD-* values) ready for Tevel SKU integration
- Branding: "Cloudy Claude · Oil Finder"

**Extend this file, don't rebuild from scratch.** The main changes are:
1. Replace Make/Model/Engine dropdowns with a license plate input field
2. Replace static `OIL_DB` lookup with a fetch() call to `POST /api/plate-lookup`
3. Add 6 more category cards (filters, brakes, bulbs, coolant) alongside the existing oil card
4. Add coverage summary and AI-fallback disclaimer badge

The UI should be simple and mobile-friendly:
1. **Input section:** License plate number field + Search button
2. **Vehicle info card:** Make, model, year, engine displayed after lookup
3. **Category cards:** One card per product category showing specs + recommended parts
4. **Coverage indicator:** "6/7 categories matched" with AI-fallback disclaimer where relevant

**Files to create:**

1. `parts-finder/frontend/src/components/PartsFinder.jsx` (NEW) — Main component
2. `parts-finder/frontend/src/components/CategoryCard.jsx` (NEW) — Reusable card per category
3. `parts-finder/frontend/src/components/VehicleInfo.jsx` (NEW) — Vehicle info display
4. `parts-finder/frontend/src/api/partsApi.js` (NEW) — API client

Design notes:
- Use fetch() for API calls (no need for axios for one endpoint)
- Mobile-first responsive layout
- Hebrew RTL support for Israeli market
- Loading spinner during plate lookup
- Error messages in Hebrew and English
- Each category card shows: spec/type, OEM part number, aftermarket alternatives with brand names
- AI-fallback results shown with a subtle warning banner

**Test plan:**
```
1. Plate input validates format (7-8 digits)
2. Search triggers API call and shows loading state
3. Results display all 7 category cards
4. AI fallback results show disclaimer badge
5. Error states display user-friendly messages
6. Mobile layout renders correctly
```

**Exit criteria:**
- [ ] Plate input + search works
- [ ] All 7 categories displayed in results
- [ ] AI fallback disclaimer visible
- [ ] Mobile responsive
- [ ] Error handling for all API failure modes

---

### 7.2 Product Catalog Mapping (Tevel Integration)

**What:** Connect the specs database to Tevel Group's actual product inventory — map specifications and OEM part numbers to Tevel SKUs with prices.
**Status:** BLOCKED (needs Tevel inventory data)
**Review Tier:** 2
**Depends on:** Phase 6.1, Tevel product data

/implement Product Catalog Mapping

Build the integration layer that maps vehicle specs and OEM part numbers to Tevel Group's actual product catalog (SKUs, prices, stock status).

**Context:** This is where business value lives — turning a specification into a purchasable product. The mapping varies by category:
- **Oil:** Spec (e.g., '0W-20 API SP') → filter Tevel oil catalog by spec → show matching products
- **Filters:** OEM part number → cross-reference table → MANN/Hengst/MAHLE → Tevel SKU
- **Brakes:** OEM part number → cross-reference → Brembo/TRW/Textar → Tevel SKU
- **Bulbs:** ECE type (e.g., 'H7') → any H7 in Tevel bulb catalog (universal fit)
- **Coolant:** Spec (e.g., 'VW G13') → filter Tevel coolant catalog by spec

**BLOCKED:** This module requires Tevel's product catalog data (SKUs, prices, stock status). The code structure can be built now, but actual mapping requires the real catalog.

**Files to create:**

1. `parts-finder/src/parts_finder/catalog.py` (NEW) — Catalog mapping layer

```python
@dataclass(frozen=True)
class TevelProduct:
    """A product from Tevel's catalog."""
    sku: str
    brand: str
    name: str
    price_ils: float           # Price in Israeli Shekels
    in_stock: bool
    category: str

class CatalogMapper:
    """Maps specs/OEM parts to Tevel products."""

    def __init__(self, db: PartsDatabase):
        ...

    def find_oil_products(self, viscosity: str, spec: str) -> list[TevelProduct]:
        ...

    def find_filter_product(self, oem_part: str) -> list[TevelProduct]:
        ...

    def find_bulb_products(self, ece_type: str) -> list[TevelProduct]:
        ...
```

**Exit criteria:**
- [ ] Interface defined for catalog mapping
- [ ] Stub implementation works with test data
- [ ] Ready to integrate when Tevel catalog data is available

---

### 7.3 Admin Dashboard & Miss Tracking

**What:** Simple admin view showing database coverage stats, AI fallback miss rates, and a queue of vehicles that need manual data addition.
**Status:** FUTURE
**Review Tier:** 1
**Depends on:** Phase 6.2

/implement Admin Dashboard

Build a simple admin dashboard (can be a CLI report or minimal web page) showing:
- Total vehicles in specs database
- Coverage by category (% of lookups that get a DB hit vs. AI fallback)
- Top 20 most-missed vehicles (from the AI fallback miss log)
- Suggestions for which vehicles to add to the DB next (highest miss count)

This drives the self-improvement loop: every AI fallback miss becomes a candidate for manual verification and permanent addition to the database.

**Files to create:**

1. `parts-finder/scripts/coverage_report.py` (NEW) — CLI coverage report

**Exit criteria:**
- [ ] Report shows total vehicle configs in DB
- [ ] Report shows miss rate from AI fallback log
- [ ] Top missed vehicles listed for prioritized data entry

---

## Phase 7 Gate

- [ ] Frontend displays all 7 categories from plate lookup
- [ ] Product catalog mapping interface ready (even if data is pending)
- [ ] Miss tracking enables database self-improvement
- [ ] All tests passing
- [ ] End-to-end flow works: plate entry → vehicle ID → specs → products → display

---

## Full Project Gate

- [ ] License plate → vehicle ID works via data.gov.il (free, unlimited)
- [ ] Specs database covers top 200 Israeli vehicle configurations (~90% market)
- [ ] All 7 product categories have working lookup modules
- [ ] FastAPI backend serves structured results
- [ ] AI fallback handles database misses (~$0.001/query)
- [ ] React frontend is mobile-friendly and supports Hebrew RTL
- [ ] Miss tracking enables continuous database improvement
- [ ] Total ongoing cost: ~$0-50/year (AI fallback only)
