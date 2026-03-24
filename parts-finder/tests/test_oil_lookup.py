"""Tests for oil specification lookup (four-tier cascade)."""

from __future__ import annotations

import unittest

from parts_finder.db import PartsDatabase
from parts_finder.models import OilResult, VehicleRecord, VehicleSpecs
from parts_finder.oil_lookup import OilLookup, extract_engine_family


def _make_specs(**overrides: object) -> VehicleSpecs:
    """Return a VehicleSpecs with sensible defaults, with optional overrides."""
    defaults: dict = dict(
        make="Toyota",
        model="Corolla",
        year_from=2019,
        year_to=2023,
        engine_code="2ZR-FE",
        fuel_type="petrol",
        oil_viscosity="0W-20",
        oil_capacity_l=4.2,
        oil_spec="API SP",
        oil_oem_approval="Toyota TGMO",
        oil_change_interval_km=15000,
    )
    defaults.update(overrides)
    return VehicleSpecs(**defaults)


def _make_vehicle(**overrides: object) -> VehicleRecord:
    """Return a VehicleRecord with sensible defaults."""
    defaults: dict = dict(
        plate="1234567",
        make_hebrew="טויוטה",
        model_hebrew="קורולה",
        year=2021,
        engine_code="2ZR-FE",
        fuel_type="petrol",
        make_english="Toyota",
        model_english="Corolla",
    )
    defaults.update(overrides)
    return VehicleRecord(**defaults)


# ── extract_engine_family ────────────────────────────────────────────


class TestExtractEngineFamily(unittest.TestCase):
    """Engine family prefix extraction from dash-separated codes."""

    def test_dash_code_returns_prefix(self) -> None:
        self.assertEqual(extract_engine_family("2ZR-FE"), "2ZR")

    def test_multi_dash_returns_first_segment(self) -> None:
        self.assertEqual(extract_engine_family("2ZZ-GE-VVTi"), "2ZZ")

    def test_no_dash_returns_none(self) -> None:
        self.assertIsNone(extract_engine_family("G4FJ"))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(extract_engine_family(""))

    def test_leading_dash_returns_none(self) -> None:
        """A code starting with dash has no valid prefix."""
        self.assertIsNone(extract_engine_family("-FE"))

    def test_like_wildcard_in_prefix_returns_none(self) -> None:
        """Prefixes with SQL LIKE wildcards are rejected as unsafe."""
        self.assertIsNone(extract_engine_family("2%R-FE"))
        self.assertIsNone(extract_engine_family("2_R-FE"))


# ── OilResult construction ───────────────────────────────────────────


class TestOilResultConstruction(unittest.TestCase):
    """OilResult classmethod constructors."""

    def test_from_vehicle_specs(self) -> None:
        specs = _make_specs()
        result = OilResult.from_vehicle_specs(specs, confidence="exact")

        self.assertEqual(result.viscosity, "0W-20")
        self.assertAlmostEqual(result.capacity_l, 4.2)
        self.assertEqual(result.spec, "API SP")
        self.assertEqual(result.oem_approval, "Toyota TGMO")
        self.assertEqual(result.change_interval_km, 15000)
        self.assertEqual(result.confidence, "exact")
        self.assertIn("Toyota", result.source)
        self.assertIn("Corolla", result.source)
        self.assertIn("2ZR-FE", result.source)

    def test_from_brand_default_has_zero_capacity(self) -> None:
        result = OilResult.from_brand_default(
            make="Toyota", viscosity="0W-20", spec="API SP",
            change_interval_km=15000,
        )

        self.assertEqual(result.viscosity, "0W-20")
        self.assertAlmostEqual(result.capacity_l, 0.0)
        self.assertEqual(result.confidence, "brand_default")
        self.assertIn("brand default", result.source)

    def test_frozen(self) -> None:
        result = OilResult.from_brand_default(
            make="Toyota", viscosity="0W-20", spec="API SP",
            change_interval_km=15000,
        )
        with self.assertRaises(AttributeError):
            result.viscosity = "5W-30"  # type: ignore[misc]


# ── Tier 1: exact match ─────────────────────────────────────────────


class TestOilLookupExactMatch(unittest.TestCase):
    """Tier 1 — exact match on make + model + year + engine_code."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs())
        self.oil = OilLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_returns_result(self) -> None:
        vehicle = _make_vehicle()
        result = self.oil.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.viscosity, "0W-20")

    def test_exact_match_correct_source(self) -> None:
        vehicle = _make_vehicle()
        result = self.oil.lookup(vehicle)
        self.assertIn("Toyota Corolla 2019-2023 2ZR-FE", result.source)

    def test_exact_match_confidence(self) -> None:
        vehicle = _make_vehicle()
        result = self.oil.lookup(vehicle)
        self.assertEqual(result.confidence, "exact")

    def test_no_match_returns_none(self) -> None:
        vehicle = _make_vehicle(
            make_english="Honda", model_english="Civic",
            make_hebrew="הונדה", model_hebrew="סיוויק",
            engine_code="K20C2",
        )
        result = self.oil.lookup(vehicle)
        self.assertIsNone(result)


# ── Tier 2: engine family fallback ───────────────────────────────────


class TestOilLookupEngineFamilyFallback(unittest.TestCase):
    """Tier 2 — engine family prefix match."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        # DB has "2ZR-FE" — lookups for "2ZR-FAE" should match via family
        self.db.insert_specs(_make_specs(engine_code="2ZR-FE"))
        self.oil = OilLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_family_prefix_matches(self) -> None:
        """'2ZR-FAE' has family '2ZR' which matches '2ZR-FE' in DB."""
        vehicle = _make_vehicle(engine_code="2ZR-FAE")
        result = self.oil.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.viscosity, "0W-20")

    def test_family_match_confidence(self) -> None:
        # Use a different model so model-year tier doesn't intercept
        vehicle = _make_vehicle(
            model_english="Auris", model_hebrew="אוריס",
            engine_code="2ZR-FAE",
        )
        result = self.oil.lookup(vehicle)
        self.assertEqual(result.confidence, "engine_family")

    def test_dashless_code_skips_tier(self) -> None:
        """Dashless codes like 'G4FJ' cannot extract family — tier 2 skipped."""
        vehicle = _make_vehicle(
            make_english="Hyundai", model_english="i30",
            make_hebrew="יונדאי", model_hebrew="i30",
            engine_code="G4FJ",
        )
        # No Hyundai data in DB — should cascade through to None
        result = self.oil.lookup(vehicle)
        self.assertIsNone(result)


# ── Tier 3: brand default fallback ───────────────────────────────────


class TestOilLookupBrandDefaultFallback(unittest.TestCase):
    """Tier 3 — brand-level default aggregation."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        # Insert several Toyota records to establish a brand default
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
            model="RAV4", engine_code="A25A-FKS",
            year_from=2019, year_to=2024,
            oil_viscosity="0W-20", oil_spec="API SP",
            oil_change_interval_km=15000,
        ))
        self.oil = OilLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_returns_most_common_spec(self) -> None:
        """Querying an unknown model falls back to brand aggregate."""
        vehicle = _make_vehicle(
            model_english="Supra", model_hebrew="סופרה",
            engine_code="B58",  # dashless, no family match
        )
        result = self.oil.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.viscosity, "0W-20")

    def test_brand_default_capacity_is_zero(self) -> None:
        vehicle = _make_vehicle(
            model_english="Supra", model_hebrew="סופרה",
            engine_code="B58",
        )
        result = self.oil.lookup(vehicle)
        self.assertAlmostEqual(result.capacity_l, 0.0)

    def test_brand_default_confidence(self) -> None:
        vehicle = _make_vehicle(
            model_english="Supra", model_hebrew="סופרה",
            engine_code="B58",
        )
        result = self.oil.lookup(vehicle)
        self.assertEqual(result.confidence, "brand_default")


# ── Cascade integration ──────────────────────────────────────────────


class TestOilLookupCascade(unittest.TestCase):
    """Integration: cascade short-circuits at the right tier."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            model="Corolla", engine_code="2ZR-FE",
            oil_viscosity="0W-20", oil_spec="API SP",
            oil_change_interval_km=15000,
        ))
        self.oil = OilLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_short_circuits(self) -> None:
        """When exact match hits, cascade stops at tier 1."""
        vehicle = _make_vehicle()
        result = self.oil.lookup(vehicle)
        self.assertEqual(result.confidence, "exact")

    def test_full_cascade_to_brand_default(self) -> None:
        """Unknown model + dashless engine → brand default."""
        vehicle = _make_vehicle(
            model_english="GR86", model_hebrew="GR86",
            engine_code="FA24",  # dashless
        )
        result = self.oil.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.confidence, "brand_default")

    def test_no_make_returns_none(self) -> None:
        """Vehicle with no usable make name returns None."""
        vehicle = _make_vehicle(
            make_english=None, make_hebrew="",
        )
        result = self.oil.lookup(vehicle)
        self.assertIsNone(result)

    def test_empty_engine_falls_to_model_year(self) -> None:
        """Empty engine_code skips tier 1, hits tier 2 (model-year)."""
        vehicle = _make_vehicle(engine_code="")
        result = self.oil.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.confidence, "model_year")

    def test_specs_with_no_oil_data_cascades(self) -> None:
        """VehicleSpecs with empty oil_viscosity is treated as no match at every tier."""
        self.db.insert_specs(_make_specs(
            model="Corolla", engine_code="3ZR-FE",
            year_from=2019, year_to=2023,
            oil_viscosity="",  # no oil data
            oil_capacity_l=0.0,
        ))
        vehicle = _make_vehicle(engine_code="3ZR-FE")
        result = self.oil.lookup(vehicle)
        # Tier 1: exact match (3ZR-FE) has empty oil -> skipped
        # Tier 2: model-year Corolla 2019-2023 matches 2ZR-FE record -> has oil data
        self.assertIsNotNone(result)
        self.assertEqual(result.confidence, "model_year")
        self.assertEqual(result.viscosity, "0W-20")


if __name__ == "__main__":
    unittest.main()
