"""Tests for PartsDatabase (vehicle specs + part cross-references)."""

from __future__ import annotations

import unittest

from parts_finder.db import PartsDatabase
from parts_finder.models import ProductCrossRef, VehicleRecord, VehicleSpecs


def _make_specs(**overrides: object) -> VehicleSpecs:
    """Return a VehicleSpecs with sensible defaults, with optional overrides."""
    defaults: dict = dict(
        make="Toyota",
        model="Corolla",
        year_from=2019,
        year_to=2023,
        engine_code="1ZR-FE",
        fuel_type="Gasoline",
    )
    defaults.update(overrides)
    return VehicleSpecs(**defaults)


def _make_crossref(**overrides: object) -> ProductCrossRef:
    """Return a ProductCrossRef with sensible defaults."""
    defaults: dict = dict(
        oem_part_number="04152-YZZA1",
        category="oil_filter",
        brand="Mann",
        brand_part_number="W 68/3",
    )
    defaults.update(overrides)
    return ProductCrossRef(**defaults)


class TestDatabaseCreation(unittest.TestCase):
    """Schema creation and basic connectivity."""

    def test_in_memory_db_creates_without_error(self) -> None:
        with PartsDatabase(":memory:") as db:
            self.assertIsNotNone(db)

    def test_schema_tables_exist(self) -> None:
        with PartsDatabase(":memory:") as db:
            tables = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
                " ORDER BY name"
            ).fetchall()
            table_names = [row["name"] for row in tables]
            self.assertIn("vehicle_specs", table_names)
            self.assertIn("product_crossref", table_names)


class TestInsertAndFindSpecs(unittest.TestCase):
    """Insert and query round-trips for VehicleSpecs."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_round_trip_all_fields(self) -> None:
        specs = _make_specs(
            oil_viscosity="0W-20",
            oil_capacity_l=4.2,
            oil_filter_oem="04152-YZZA1",
            front_brake_pad_oem="04465-02220",
            coolant_type="Super Long Life",
            spark_plug_oem="SK20R11",
            spark_plug_gap_mm=1.1,
        )
        self.db.insert_specs(specs)
        result = self.db.find_specs("Toyota", "Corolla", 2021, "1ZR-FE")

        self.assertIsNotNone(result)
        self.assertEqual(result.make, "Toyota")
        self.assertEqual(result.model, "Corolla")
        self.assertEqual(result.year_from, 2019)
        self.assertEqual(result.year_to, 2023)
        self.assertEqual(result.engine_code, "1ZR-FE")
        self.assertEqual(result.oil_viscosity, "0W-20")
        self.assertAlmostEqual(result.oil_capacity_l, 4.2)
        self.assertEqual(result.oil_filter_oem, "04152-YZZA1")
        self.assertEqual(result.front_brake_pad_oem, "04465-02220")
        self.assertEqual(result.coolant_type, "Super Long Life")
        self.assertEqual(result.spark_plug_oem, "SK20R11")
        self.assertAlmostEqual(result.spark_plug_gap_mm, 1.1)

    def test_year_range_match(self) -> None:
        """Query year inside [year_from, year_to] should match."""
        self.db.insert_specs(_make_specs(year_from=2019, year_to=2023))

        # Boundary: exact start
        self.assertIsNotNone(
            self.db.find_specs("Toyota", "Corolla", 2019, "1ZR-FE")
        )
        # Boundary: exact end
        self.assertIsNotNone(
            self.db.find_specs("Toyota", "Corolla", 2023, "1ZR-FE")
        )
        # Middle
        self.assertIsNotNone(
            self.db.find_specs("Toyota", "Corolla", 2021, "1ZR-FE")
        )
        # Outside range
        self.assertIsNone(
            self.db.find_specs("Toyota", "Corolla", 2018, "1ZR-FE")
        )
        self.assertIsNone(
            self.db.find_specs("Toyota", "Corolla", 2024, "1ZR-FE")
        )

    def test_no_match_returns_none(self) -> None:
        self.db.insert_specs(_make_specs())
        result = self.db.find_specs("Honda", "Civic", 2021, "K20C2")
        self.assertIsNone(result)

    def test_year_range_overlap_returns_first_match(self) -> None:
        """Overlapping ranges: LIMIT 1 picks one silently.

        This documents the current behavior — overlapping ranges are a
        data quality concern that the import script (Phase 2b) must prevent.
        """
        self.db.insert_specs(_make_specs(
            year_from=2018, year_to=2020, oil_viscosity="5W-30",
        ))
        self.db.insert_specs(_make_specs(
            year_from=2019, year_to=2023, oil_viscosity="0W-20",
        ))
        # Year 2019 matches both — we get whichever LIMIT 1 returns
        result = self.db.find_specs("Toyota", "Corolla", 2019, "1ZR-FE")
        self.assertIsNotNone(result)
        self.assertIn(result.oil_viscosity, ("5W-30", "0W-20"))

    def test_upsert_replaces_on_conflict(self) -> None:
        """INSERT OR REPLACE should update existing record on UNIQUE conflict."""
        self.db.insert_specs(_make_specs(oil_viscosity="5W-30"))
        self.db.insert_specs(_make_specs(oil_viscosity="0W-20"))

        result = self.db.find_specs("Toyota", "Corolla", 2021, "1ZR-FE")
        self.assertEqual(result.oil_viscosity, "0W-20")

        # Verify only one row exists (not duplicated)
        count = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM vehicle_specs"
        ).fetchone()["n"]
        self.assertEqual(count, 1)


class TestInsertAndFindCrossRefs(unittest.TestCase):
    """Insert and query round-trips for ProductCrossRef."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_round_trip(self) -> None:
        xref = _make_crossref(notes="Direct replacement")
        self.db.insert_crossref(xref)
        results = self.db.find_crossrefs("04152-YZZA1")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].oem_part_number, "04152-YZZA1")
        self.assertEqual(results[0].category, "oil_filter")
        self.assertEqual(results[0].brand, "Mann")
        self.assertEqual(results[0].brand_part_number, "W 68/3")
        self.assertEqual(results[0].notes, "Direct replacement")

    def test_multiple_brands_for_same_oem(self) -> None:
        self.db.insert_crossref(_make_crossref(brand="Wix", brand_part_number="WL7459"))
        self.db.insert_crossref(_make_crossref(brand="Mann", brand_part_number="W 68/3"))
        self.db.insert_crossref(_make_crossref(brand="Bosch", brand_part_number="P 7235"))

        results = self.db.find_crossrefs("04152-YZZA1")
        self.assertEqual(len(results), 3)
        # Results are ordered alphabetically by brand (ORDER BY brand)
        brands = [r.brand for r in results]
        self.assertEqual(brands, ["Bosch", "Mann", "Wix"])

    def test_no_match_returns_empty_list(self) -> None:
        self.db.insert_crossref(_make_crossref())
        results = self.db.find_crossrefs("NONEXISTENT-999")
        self.assertEqual(results, [])


class TestEngineFamilyAndBrandDefaultQueries(unittest.TestCase):
    """Direct tests for find_specs_by_engine_family and find_brand_default_oil."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_engine_family_matches_prefix(self) -> None:
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        result = self.db.find_specs_by_engine_family("Toyota", "2ZR", 2021)
        self.assertIsNotNone(result)
        self.assertEqual(result.engine_code, "2ZR-FE")

    def test_engine_family_no_match_returns_none(self) -> None:
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        result = self.db.find_specs_by_engine_family("Toyota", "1KD", 2021)
        self.assertIsNone(result)

    def test_engine_family_respects_year_range(self) -> None:
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE", year_from=2019, year_to=2023,
        ))
        self.assertIsNone(
            self.db.find_specs_by_engine_family("Toyota", "2ZR", 2018)
        )
        self.assertIsNone(
            self.db.find_specs_by_engine_family("Toyota", "2ZR", 2024)
        )
        self.assertIsNotNone(
            self.db.find_specs_by_engine_family("Toyota", "2ZR", 2019)
        )
        self.assertIsNotNone(
            self.db.find_specs_by_engine_family("Toyota", "2ZR", 2023)
        )

    def test_engine_family_empty_db_returns_none(self) -> None:
        result = self.db.find_specs_by_engine_family("Toyota", "2ZR", 2021)
        self.assertIsNone(result)

    def test_brand_default_returns_most_common(self) -> None:
        # 2 records with 0W-20, 1 with 5W-30 -> 0W-20 wins
        self.db.insert_specs(_make_specs(
            model="Corolla", engine_code="2ZR-FE",
            oil_viscosity="0W-20", oil_spec="API SP",
            oil_change_interval_km=15000,
        ))
        self.db.insert_specs(_make_specs(
            model="Camry", engine_code="2AR-FE",
            year_from=2018, year_to=2023,
            oil_viscosity="0W-20", oil_spec="API SP",
            oil_change_interval_km=15000,
        ))
        self.db.insert_specs(_make_specs(
            model="Land Cruiser", engine_code="1VD-FTV",
            year_from=2015, year_to=2021,
            oil_viscosity="5W-30", oil_spec="ACEA C3",
            oil_change_interval_km=10000,
        ))
        result = self.db.find_brand_default_oil("Toyota")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "0W-20")

    def test_brand_default_skips_empty_viscosity(self) -> None:
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE", oil_viscosity="",
        ))
        result = self.db.find_brand_default_oil("Toyota")
        self.assertIsNone(result)

    def test_brand_default_unknown_make_returns_none(self) -> None:
        self.db.insert_specs(_make_specs())
        result = self.db.find_brand_default_oil("Nonexistent")
        self.assertIsNone(result)


class TestModelYearQuery(unittest.TestCase):
    """Direct tests for find_specs_by_model_year."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_match_within_year_range(self) -> None:
        self.db.insert_specs(_make_specs(year_from=2019, year_to=2023))
        result = self.db.find_specs_by_model_year("Toyota", "Corolla", 2021)
        self.assertIsNotNone(result)
        self.assertEqual(result.model, "Corolla")

    def test_year_boundaries(self) -> None:
        self.db.insert_specs(_make_specs(year_from=2019, year_to=2023))
        # Exact start
        self.assertIsNotNone(
            self.db.find_specs_by_model_year("Toyota", "Corolla", 2019)
        )
        # Exact end
        self.assertIsNotNone(
            self.db.find_specs_by_model_year("Toyota", "Corolla", 2023)
        )
        # Outside range
        self.assertIsNone(
            self.db.find_specs_by_model_year("Toyota", "Corolla", 2018)
        )
        self.assertIsNone(
            self.db.find_specs_by_model_year("Toyota", "Corolla", 2024)
        )

    def test_model_mismatch_returns_none(self) -> None:
        self.db.insert_specs(_make_specs())
        result = self.db.find_specs_by_model_year("Toyota", "Camry", 2021)
        self.assertIsNone(result)

    def test_ignores_engine_code(self) -> None:
        """Should return a record regardless of engine_code."""
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        # Query doesn't specify engine — should still match
        result = self.db.find_specs_by_model_year("Toyota", "Corolla", 2021)
        self.assertIsNotNone(result)

    def test_multiple_engines_returns_a_result(self) -> None:
        """Multiple engine variants for same model-year: LIMIT 1 picks one.

        Documents the same ambiguity as test_year_range_overlap_returns_first_match
        for find_specs. The caller should not depend on which row is returned.
        """
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        self.db.insert_specs(_make_specs(engine_code="1ZR-FE"))
        result = self.db.find_specs_by_model_year("Toyota", "Corolla", 2021)
        self.assertIsNotNone(result)
        self.assertIn(result.engine_code, ("2ZR-FE", "1ZR-FE"))

    def test_prefers_populated_bulb_row(self) -> None:
        """ORDER BY ensures a row with bulb data is returned over an empty one."""
        # Insert empty row first (would be returned without ORDER BY)
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        # Insert populated row second
        self.db.insert_specs(_make_specs(
            engine_code="1ZR-FE", low_beam_bulb="H7",
        ))
        result = self.db.find_specs_by_model_year("Toyota", "Corolla", 2021)
        self.assertIsNotNone(result)
        self.assertEqual(result.low_beam_bulb, "H7")

    def test_empty_db_returns_none(self) -> None:
        result = self.db.find_specs_by_model_year("Toyota", "Corolla", 2021)
        self.assertIsNone(result)


class TestCoolantModelYearQuery(unittest.TestCase):
    """Direct tests for find_specs_by_model_year_for_coolant."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_match_within_year_range(self) -> None:
        self.db.insert_specs(_make_specs(
            year_from=2019, year_to=2023, coolant_type="SLLC",
        ))
        result = self.db.find_specs_by_model_year_for_coolant(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.model, "Corolla")

    def test_year_boundaries(self) -> None:
        self.db.insert_specs(_make_specs(
            year_from=2019, year_to=2023, coolant_type="SLLC",
        ))
        # Exact start
        self.assertIsNotNone(
            self.db.find_specs_by_model_year_for_coolant(
                "Toyota", "Corolla", 2019,
            )
        )
        # Exact end
        self.assertIsNotNone(
            self.db.find_specs_by_model_year_for_coolant(
                "Toyota", "Corolla", 2023,
            )
        )
        # Outside range
        self.assertIsNone(
            self.db.find_specs_by_model_year_for_coolant(
                "Toyota", "Corolla", 2018,
            )
        )
        self.assertIsNone(
            self.db.find_specs_by_model_year_for_coolant(
                "Toyota", "Corolla", 2024,
            )
        )

    def test_model_mismatch_returns_none(self) -> None:
        self.db.insert_specs(_make_specs(coolant_type="SLLC"))
        result = self.db.find_specs_by_model_year_for_coolant(
            "Toyota", "Camry", 2021,
        )
        self.assertIsNone(result)

    def test_ignores_engine_code(self) -> None:
        """Should return a record regardless of engine_code."""
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE", coolant_type="SLLC",
        ))
        result = self.db.find_specs_by_model_year_for_coolant(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNotNone(result)

    def test_prefers_populated_coolant_row(self) -> None:
        """ORDER BY ensures a row with coolant data is returned over an empty one."""
        # Insert empty row first (would be returned without ORDER BY)
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE", coolant_type=""))
        # Insert populated row second
        self.db.insert_specs(_make_specs(
            engine_code="1ZR-FE", coolant_type="SLLC",
        ))
        result = self.db.find_specs_by_model_year_for_coolant(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.coolant_type, "SLLC")

    def test_empty_db_returns_none(self) -> None:
        result = self.db.find_specs_by_model_year_for_coolant(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNone(result)


class TestBrakeModelYearQuery(unittest.TestCase):
    """Direct tests for find_specs_by_model_year_for_brakes."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_match_within_year_range(self) -> None:
        self.db.insert_specs(_make_specs(
            year_from=2019, year_to=2023,
            front_brake_pad_oem="04465-02220",
        ))
        result = self.db.find_specs_by_model_year_for_brakes(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.model, "Corolla")

    def test_year_boundaries(self) -> None:
        self.db.insert_specs(_make_specs(
            year_from=2019, year_to=2023,
            front_brake_pad_oem="04465-02220",
        ))
        # Exact start
        self.assertIsNotNone(
            self.db.find_specs_by_model_year_for_brakes(
                "Toyota", "Corolla", 2019,
            )
        )
        # Exact end
        self.assertIsNotNone(
            self.db.find_specs_by_model_year_for_brakes(
                "Toyota", "Corolla", 2023,
            )
        )
        # Outside range
        self.assertIsNone(
            self.db.find_specs_by_model_year_for_brakes(
                "Toyota", "Corolla", 2018,
            )
        )
        self.assertIsNone(
            self.db.find_specs_by_model_year_for_brakes(
                "Toyota", "Corolla", 2024,
            )
        )

    def test_model_mismatch_returns_none(self) -> None:
        self.db.insert_specs(_make_specs(front_brake_pad_oem="04465-02220"))
        result = self.db.find_specs_by_model_year_for_brakes(
            "Toyota", "Camry", 2021,
        )
        self.assertIsNone(result)

    def test_ignores_engine_code(self) -> None:
        """Should return a record regardless of engine_code."""
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE", front_brake_pad_oem="04465-02220",
        ))
        result = self.db.find_specs_by_model_year_for_brakes(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNotNone(result)

    def test_prefers_populated_brake_row(self) -> None:
        """ORDER BY ensures a row with brake data is returned over an empty one."""
        # Insert empty row first (would be returned without ORDER BY)
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE", front_brake_pad_oem="",
        ))
        # Insert populated row second
        self.db.insert_specs(_make_specs(
            engine_code="1ZR-FE", front_brake_pad_oem="04465-02220",
        ))
        result = self.db.find_specs_by_model_year_for_brakes(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.front_brake_pad_oem, "04465-02220")

    def test_empty_db_returns_none(self) -> None:
        result = self.db.find_specs_by_model_year_for_brakes(
            "Toyota", "Corolla", 2021,
        )
        self.assertIsNone(result)


class TestEnrichmentFlow(unittest.TestCase):
    """End-to-end: VehicleRecord -> find_specs -> find_crossrefs."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")

    def tearDown(self) -> None:
        self.db.close()

    def test_vehicle_record_to_specs_to_crossrefs(self) -> None:
        """Simulate the full lookup chain from plate to parts."""
        # Given a VehicleRecord (from Phase 1a/1b)
        vehicle = VehicleRecord(
            plate="1234567",
            make_hebrew="Toyota",
            model_hebrew="Corolla",
            year=2021,
            engine_code="1ZR-FE",
            fuel_type="Gasoline",
        )

        # And a populated database
        specs = _make_specs(oil_filter_oem="04152-YZZA1")
        self.db.insert_specs(specs)
        self.db.insert_crossref(_make_crossref(
            oem_part_number="04152-YZZA1",
            brand="Mann",
            brand_part_number="W 68/3",
        ))

        # When we look up specs using vehicle identity
        found_specs = self.db.find_specs(
            vehicle.make_hebrew, vehicle.model_hebrew,
            vehicle.year, vehicle.engine_code,
        )
        self.assertIsNotNone(found_specs)
        self.assertEqual(found_specs.oil_filter_oem, "04152-YZZA1")

        # Then we can find aftermarket parts via OEM number
        crossrefs = self.db.find_crossrefs(found_specs.oil_filter_oem)
        self.assertEqual(len(crossrefs), 1)
        self.assertEqual(crossrefs[0].brand, "Mann")

    def test_context_manager_protocol(self) -> None:
        """Verify the with-statement works end-to-end."""
        with PartsDatabase(":memory:") as db:
            db.insert_specs(_make_specs())
            result = db.find_specs("Toyota", "Corolla", 2021, "1ZR-FE")
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
