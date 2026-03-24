"""Tests for seed_from_jsx — JSX OIL_DB parser and CSV exporter.

Covers:
  - Fuel type extraction (17+ keyword/regex paths)
  - JS→JSON conversion (unquoted keys, trailing commas, comments)
  - OIL_DB block extraction (happy path, missing block)
  - Entry iteration (normal, EV skip, "None" OEM mapping)
  - CSV export (correct headers and values)
  - Acceptance: parse real oil-finder-free.jsx
  - End-to-end round-trip: JSX → CSV → import_csv → db.find_specs
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# --- Bootstrap: add scripts/ and src/ to path ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for p in (SCRIPTS_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from seed_from_jsx import (  # noqa: E402
    CSV_COLUMNS,
    OilEntry,
    SeedReport,
    export_csv,
    extract_fuel_type,
    extract_oil_db_block,
    iter_oil_entries,
    js_object_to_json,
    parse_oil_db,
)


# =====================================================================
# 1. extract_fuel_type — exhaustive keyword coverage
# =====================================================================


class TestExtractFuelType:
    """Test priority-ordered fuel type extraction from engine descriptions."""

    # --- Exact keyword matches (priority 1-8) ---

    def test_electric_returns_none(self):
        assert extract_fuel_type("Electric") is None

    def test_e_skyactiv_x_is_petrol(self):
        """Mazda's SPCCI engine is petrol, not electric."""
        assert extract_fuel_type("2.0 e-Skyactiv X") == "petrol"

    def test_ehybrid_is_hybrid(self):
        """VW naming convention: 'eHybrid' = PHEV but mapped to hybrid."""
        assert extract_fuel_type("3.0 V6 eHybrid") == "hybrid"

    def test_phev_detected(self):
        assert extract_fuel_type("2.5 PHEV") == "phev"

    def test_phev_bmw_330e(self):
        assert extract_fuel_type("330e PHEV") == "phev"

    def test_mild_hybrid_is_hybrid(self):
        assert extract_fuel_type("1.2 Mild Hybrid") == "hybrid"

    def test_generic_hybrid(self):
        assert extract_fuel_type("1.8 Hybrid") == "hybrid"

    def test_diesel_keyword(self):
        assert extract_fuel_type("2.0 Diesel") == "diesel"

    def test_petrol_keyword(self):
        assert extract_fuel_type("1.6T Petrol") == "petrol"

    # --- Regex suffix rules (priority 9-10, BMW/MB) ---

    def test_bmw_diesel_suffix_D(self):
        """BMW '318d 2.0D' — suffix D = diesel."""
        assert extract_fuel_type("318d 2.0D") == "diesel"

    def test_bmw_petrol_suffix_T(self):
        """BMW '320i 2.0T' — suffix T = turbo petrol."""
        assert extract_fuel_type("320i 2.0T") == "petrol"

    def test_mb_diesel_suffix_d(self):
        """Mercedes 'A180d 1.5D' — uppercase D suffix = diesel."""
        assert extract_fuel_type("A180d 1.5D") == "diesel"

    # --- Sport / performance keywords (priority 11) ---

    def test_gti_is_petrol(self):
        assert extract_fuel_type("2.0T GTI") == "petrol"

    def test_amg_is_petrol(self):
        assert extract_fuel_type("A35 AMG 2.0T") == "petrol"

    def test_sport_is_petrol(self):
        assert extract_fuel_type("1.4T Sport") == "petrol"

    def test_v6_is_petrol(self):
        assert extract_fuel_type("3.5 V6") == "petrol"

    def test_v8_is_petrol(self):
        assert extract_fuel_type("M50i 4.4T V8") == "petrol"

    # --- Fallback ---

    def test_unknown_defaults_to_petrol(self):
        """Unrecognized descriptions default to petrol (most common)."""
        assert extract_fuel_type("1.0T") == "petrol"

    # --- Priority ordering ---

    def test_electric_before_hybrid(self):
        """'Electric' exact match takes priority over substring matches."""
        assert extract_fuel_type("Electric") is None

    def test_phev_before_hybrid(self):
        """'PHEV' keyword should match before generic 'Hybrid'."""
        # This tests that PHEV doesn't fall through to Hybrid
        assert extract_fuel_type("2.5 PHEV") == "phev"


# =====================================================================
# 2. js_object_to_json — JS literal conversion
# =====================================================================


class TestJsObjectToJson:
    """Test JS object literal → valid JSON conversion."""

    def test_unquoted_keys_get_quoted(self):
        js = '{ viscosity: "5W-30", capacity: "4.5" }'
        result = json.loads(js_object_to_json(js))
        assert result == {"viscosity": "5W-30", "capacity": "4.5"}

    def test_trailing_comma_removed(self):
        js = '{ "a": 1, "b": 2, }'
        result = json.loads(js_object_to_json(js))
        assert result == {"a": 1, "b": 2}

    def test_nested_trailing_commas(self):
        js = '{ "x": { "y": "z", }, }'
        result = json.loads(js_object_to_json(js))
        assert result == {"x": {"y": "z"}}

    def test_single_line_comments_stripped(self):
        js = '{\n  // comment\n  "key": "val"\n}'
        result = json.loads(js_object_to_json(js))
        assert result == {"key": "val"}

    def test_unicode_preserved(self):
        """Škoda's diacritic must survive the conversion."""
        js = '{ "Škoda": { "model": "Octavia" } }'
        result = json.loads(js_object_to_json(js))
        assert "Škoda" in result

    def test_already_quoted_keys_not_double_quoted(self):
        js = '{ "already_quoted": "value" }'
        result = json.loads(js_object_to_json(js))
        assert result == {"already_quoted": "value"}

    def test_mixed_quoted_and_unquoted(self):
        js = '{ unquoted: "a", "quoted": "b" }'
        result = json.loads(js_object_to_json(js))
        assert result == {"unquoted": "a", "quoted": "b"}

    def test_colon_in_string_value_preserved(self):
        """Values containing 'word:' patterns must not be corrupted."""
        js = '{\n  spec: "API SP: revised",\n  oem: "Requires: synthetic"\n}'
        result = json.loads(js_object_to_json(js))
        assert result["spec"] == "API SP: revised"
        assert result["oem"] == "Requires: synthetic"

    def test_block_comments_stripped(self):
        js = '{\n  /* comment */\n  "key": "val"\n}'
        result = json.loads(js_object_to_json(js))
        assert result == {"key": "val"}


# =====================================================================
# 3. extract_oil_db_block — block extraction
# =====================================================================


class TestExtractOilDbBlock:
    """Test extraction of the OIL_DB = { ... }; block."""

    def test_finds_block(self):
        jsx = 'const OIL_DB = { Toyota: { Corolla: {} } };\nconst x = 1;'
        block = extract_oil_db_block(jsx)
        assert block.startswith("{")
        assert block.endswith("}")
        # Verify it's parseable after conversion
        json.loads(js_object_to_json(block))

    def test_raises_on_missing_block(self):
        with pytest.raises(ValueError, match="Could not find OIL_DB"):
            extract_oil_db_block("const OTHER_DB = {};")

    def test_handles_nested_braces(self):
        jsx = 'const OIL_DB = { a: { b: { c: "d" } } };'
        block = extract_oil_db_block(jsx)
        result = json.loads(js_object_to_json(block))
        assert result["a"]["b"]["c"] == "d"

    def test_let_declaration(self):
        """Should also find 'let OIL_DB = ...'."""
        jsx = 'let OIL_DB = { x: 1 };'
        block = extract_oil_db_block(jsx)
        assert json.loads(js_object_to_json(block)) == {"x": 1}


# =====================================================================
# 4. iter_oil_entries — entry iteration
# =====================================================================


class TestIterOilEntries:
    """Test entry iteration with EV skipping and OEM mapping."""

    @pytest.fixture()
    def sample_db(self) -> dict:
        return {
            "Toyota": {
                "Corolla": {
                    "1.8 Petrol": {
                        "viscosity": "0W-20",
                        "spec": "API SP",
                        "oem": "None",
                        "capacity": "4.2",
                        "interval": "15000",
                    },
                }
            },
            "Hyundai": {
                "Kona": {
                    "Electric": {
                        "viscosity": "N/A",
                        "spec": "N/A",
                        "oem": "N/A",
                        "capacity": "N/A",
                        "interval": "N/A",
                    },
                    "1.6T Petrol": {
                        "viscosity": "5W-30",
                        "spec": "API SP",
                        "oem": "Hyundai/Kia Oil Standard",
                        "capacity": "4.5",
                        "interval": "15000",
                    },
                }
            },
        }

    def test_normal_entry(self, sample_db: dict):
        entries = list(iter_oil_entries(sample_db))
        corolla = [e for e in entries if e.model == "Corolla"]
        assert len(corolla) == 1
        assert corolla[0].fuel_type == "petrol"
        assert corolla[0].viscosity == "0W-20"
        assert corolla[0].capacity_l == 4.2
        assert corolla[0].interval_km == 15000

    def test_ev_skipped(self, sample_db: dict):
        entries = list(iter_oil_entries(sample_db))
        ev_entries = [e for e in entries if e.engine_desc == "Electric"]
        assert len(ev_entries) == 0

    def test_oem_none_mapped_to_empty(self, sample_db: dict):
        entries = list(iter_oil_entries(sample_db))
        corolla = [e for e in entries if e.model == "Corolla"][0]
        assert corolla.oem_approval == ""

    def test_oem_real_value_preserved(self, sample_db: dict):
        entries = list(iter_oil_entries(sample_db))
        kona = [e for e in entries if e.model == "Kona"][0]
        assert kona.oem_approval == "Hyundai/Kia Oil Standard"

    def test_iter_is_year_agnostic(self, sample_db: dict):
        """iter_oil_entries has no year filtering — years applied at CSV export."""
        entries = list(iter_oil_entries(sample_db))
        assert len(entries) == 2

    def test_total_entries(self, sample_db: dict):
        entries = list(iter_oil_entries(sample_db))
        assert len(entries) == 2  # Corolla petrol + Kona petrol (EV skipped)


# =====================================================================
# 5. export_csv — CSV output
# =====================================================================


class TestExportCsv:
    """Test CSV export produces correct headers and values."""

    @pytest.fixture()
    def sample_entries(self) -> list[OilEntry]:
        return [
            OilEntry(
                make="Toyota",
                model="Corolla",
                engine_desc="1.8 Petrol",
                fuel_type="petrol",
                viscosity="0W-20",
                capacity_l=4.2,
                spec="API SP",
                oem_approval="",
                interval_km=15000,
            ),
            OilEntry(
                make="BMW",
                model="3 Series",
                engine_desc="320d 2.0D",
                fuel_type="diesel",
                viscosity="5W-30",
                capacity_l=5.2,
                spec="ACEA C3",
                oem_approval="BMW LL-04",
                interval_km=15000,
            ),
        ]

    def test_csv_headers(self, sample_entries: list[OilEntry], tmp_path: Path):
        out = tmp_path / "test.csv"
        export_csv(sample_entries, out)
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == CSV_COLUMNS

    def test_csv_row_values(self, sample_entries: list[OilEntry], tmp_path: Path):
        out = tmp_path / "test.csv"
        export_csv(sample_entries, out, year_from=2019, year_to=2024)
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        toyota = rows[0]
        assert toyota["make"] == "Toyota"
        assert toyota["model"] == "Corolla"
        assert toyota["year_from"] == "2019"
        assert toyota["year_to"] == "2024"
        assert toyota["engine_code"] == "1.8 Petrol"
        assert toyota["fuel_type"] == "petrol"
        assert toyota["oil_viscosity"] == "0W-20"
        assert toyota["oil_capacity_l"] == "4.2"
        assert toyota["oil_spec"] == "API SP"
        assert toyota["oil_oem_approval"] == ""
        assert toyota["oil_change_interval_km"] == "15000"

    def test_csv_creates_parent_dirs(self, sample_entries: list[OilEntry], tmp_path: Path):
        out = tmp_path / "nested" / "deep" / "test.csv"
        export_csv(sample_entries, out)
        assert out.exists()

    def test_csv_row_count(self, sample_entries: list[OilEntry], tmp_path: Path):
        out = tmp_path / "test.csv"
        count = export_csv(sample_entries, out)
        assert count == 2


# =====================================================================
# 6. Acceptance: parse real oil-finder-free.jsx
# =====================================================================


class TestAcceptanceRealJsx:
    """Parse the actual JSX file and verify expected counts."""

    JSX_PATH = Path(__file__).resolve().parents[2] / "oil-finder-free.jsx"

    @pytest.fixture()
    def oil_db(self) -> dict:
        if not self.JSX_PATH.exists():
            pytest.skip("oil-finder-free.jsx not found at repo root")
        jsx_text = self.JSX_PATH.read_text(encoding="utf-8")
        return parse_oil_db(jsx_text)

    def test_parses_without_error(self, oil_db: dict):
        assert isinstance(oil_db, dict)

    def test_at_least_190_entries(self, oil_db: dict):
        entries = list(iter_oil_entries(oil_db))
        assert len(entries) >= 190, f"Expected >= 190 entries, got {len(entries)}"

    def test_exactly_3_evs_skipped(self, oil_db: dict):
        total = sum(
            len(engines)
            for models in oil_db.values()
            for engines in models.values()
        )
        exported = len(list(iter_oil_entries(oil_db)))
        skipped = total - exported
        assert skipped == 3, f"Expected 3 EVs skipped, got {skipped}"

    def test_known_makes_present(self, oil_db: dict):
        expected_makes = {"Toyota", "Hyundai", "Kia", "Volkswagen", "BMW", "Mercedes-Benz", "Mazda"}
        assert expected_makes.issubset(oil_db.keys())

    def test_skoda_diacritic_preserved(self, oil_db: dict):
        assert "Škoda" in oil_db

    def test_all_fuel_types_valid(self, oil_db: dict):
        valid_fuels = {"petrol", "diesel", "hybrid", "phev"}
        entries = list(iter_oil_entries(oil_db))
        actual_fuels = {e.fuel_type for e in entries}
        assert actual_fuels.issubset(valid_fuels), f"Unknown fuels: {actual_fuels - valid_fuels}"

    def test_all_capacities_positive(self, oil_db: dict):
        entries = list(iter_oil_entries(oil_db))
        for e in entries:
            assert e.capacity_l > 0, f"{e.make} {e.model} {e.engine_desc}: capacity={e.capacity_l}"

    def test_csv_export_roundtrip(self, oil_db: dict, tmp_path: Path):
        """Export to CSV, read it back, verify row count and headers."""
        entries = list(iter_oil_entries(oil_db))
        out = tmp_path / "seed.csv"
        export_csv(entries, out)

        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(entries)
        assert list(reader.fieldnames) == CSV_COLUMNS


# =====================================================================
# 7. End-to-end: JSX → CSV → import_csv → db.find_specs
# =====================================================================


class TestEndToEndRoundTrip:
    """Full pipeline: parse JSX → write CSV → import → query."""

    JSX_PATH = Path(__file__).resolve().parents[2] / "oil-finder-free.jsx"

    @pytest.fixture()
    def seeded_db(self, tmp_path: Path):
        """Parse JSX, export CSV, import into in-memory DB."""
        if not self.JSX_PATH.exists():
            pytest.skip("oil-finder-free.jsx not found at repo root")

        from import_data import import_csv
        from parts_finder.db import PartsDatabase

        # Parse and export
        jsx_text = self.JSX_PATH.read_text(encoding="utf-8")
        oil_db = parse_oil_db(jsx_text)
        entries = list(iter_oil_entries(oil_db))
        csv_path = tmp_path / "seed.csv"
        export_csv(entries, csv_path, year_from=2018, year_to=2025)

        # Import into DB (context manager ensures connection is closed)
        with PartsDatabase(":memory:") as db:
            report = import_csv(db, csv_path, "specs")
            yield db, report, len(entries)

    def test_import_no_errors(self, seeded_db):
        db, report, _ = seeded_db
        assert report.errors == 0, f"Import errors: {report.error_details}"

    def test_import_correct_count(self, seeded_db):
        db, report, entry_count = seeded_db
        assert report.inserted == entry_count

    def test_find_toyota_corolla(self, seeded_db):
        db, _, _ = seeded_db
        result = db.find_specs("Toyota", "Corolla", 2020, "1.8 Petrol")
        assert result is not None
        assert result.oil_viscosity == "0W-20"
        assert result.fuel_type == "petrol"

    def test_find_bmw_320d(self, seeded_db):
        db, _, _ = seeded_db
        result = db.find_specs("BMW", "3 Series", 2022, "320d 2.0D")
        assert result is not None
        assert result.oil_viscosity == "5W-30"
        assert result.fuel_type == "diesel"
        assert result.oil_oem_approval == "BMW LL-04"

    def test_ev_not_in_db(self, seeded_db):
        db, _, _ = seeded_db
        result = db.find_specs("Hyundai", "Kona", 2022, "Electric")
        assert result is None

    def test_oem_none_is_empty_in_db(self, seeded_db):
        db, _, _ = seeded_db
        result = db.find_specs("Toyota", "Corolla", 2020, "1.8 Petrol")
        assert result is not None
        assert result.oil_oem_approval == ""
