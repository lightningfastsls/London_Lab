"""Tests for filter parts lookup (two-tier cascade with cross-references)."""

from __future__ import annotations

import unittest

from parts_finder.db import PartsDatabase
from parts_finder.models import (
    FilterResult,
    ProductCrossRef,
    VehicleRecord,
    VehicleSpecs,
)
from parts_finder.lookup.filters import FilterLookup


def _make_specs(**overrides: object) -> VehicleSpecs:
    """Return a VehicleSpecs with sensible defaults, with optional overrides."""
    defaults: dict = dict(
        make="Toyota",
        model="Corolla",
        year_from=2019,
        year_to=2023,
        engine_code="2ZR-FE",
        fuel_type="petrol",
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


# ── FilterResult dataclass ────────────────────────────────────────


class TestFilterResultDataclass(unittest.TestCase):
    """FilterResult frozen dataclass behavior."""

    def test_frozen(self) -> None:
        """FilterResult is immutable."""
        result = FilterResult(oil_filter_oem="04152-YZZA1", source="exact")
        with self.assertRaises(AttributeError):
            result.oil_filter_oem = "changed"  # type: ignore[misc]

    def test_has_data_true_when_any_oem_set(self) -> None:
        """has_data returns True if at least one OEM field is non-empty."""
        self.assertTrue(FilterResult(oil_filter_oem="04152-YZZA1").has_data)
        self.assertTrue(FilterResult(air_filter_oem="17801-21050").has_data)
        self.assertTrue(FilterResult(cabin_filter_oem="87139-02020").has_data)
        self.assertTrue(FilterResult(fuel_filter_oem="23300-21010").has_data)

    def test_has_data_false_when_empty(self) -> None:
        """has_data returns False when all OEM fields are empty."""
        self.assertFalse(FilterResult().has_data)


# ── Tier 1: exact match ─────────────────────────────────────────


class TestFilterLookupExactMatch(unittest.TestCase):
    """Tier 1 — exact match on make + model + year + engine_code."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            oil_filter_oem="04152-YZZA1",
            air_filter_oem="17801-21050",
            cabin_filter_oem="87139-02020",
            fuel_filter_oem="23300-21010",
        ))
        self.filters = FilterLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_returns_all_oem_fields(self) -> None:
        vehicle = _make_vehicle()
        result = self.filters.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.oil_filter_oem, "04152-YZZA1")
        self.assertEqual(result.air_filter_oem, "17801-21050")
        self.assertEqual(result.cabin_filter_oem, "87139-02020")
        self.assertEqual(result.fuel_filter_oem, "23300-21010")

    def test_exact_match_source(self) -> None:
        vehicle = _make_vehicle()
        result = self.filters.lookup(vehicle)
        self.assertEqual(result.source, "exact")

    def test_partial_data_only_oil_filter(self) -> None:
        """Specs with only oil filter OEM still return a partial result."""
        self.db.insert_specs(_make_specs(
            model="Yaris",
            engine_code="1NZ-FE",
            oil_filter_oem="04152-B1010",
            # All other filter fields empty
        ))
        vehicle = _make_vehicle(
            model_english="Yaris", model_hebrew="יאריס",
            engine_code="1NZ-FE",
        )
        result = self.filters.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.oil_filter_oem, "04152-B1010")
        self.assertEqual(result.air_filter_oem, "")
        self.assertEqual(result.cabin_filter_oem, "")
        self.assertEqual(result.fuel_filter_oem, "")
        self.assertTrue(result.has_data)


# ── Cross-references ─────────────────────────────────────────────


class TestFilterLookupCrossRefs(unittest.TestCase):
    """Cross-references populated from product_crossref table."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            oil_filter_oem="04152-YZZA1",
            air_filter_oem="17801-21050",
            cabin_filter_oem="87139-02020",
        ))
        # Add cross-references for oil filter
        self.db.insert_crossref(ProductCrossRef(
            oem_part_number="04152-YZZA1",
            category="oil_filter",
            brand="MANN-FILTER",
            brand_part_number="W 712/83",
        ))
        self.db.insert_crossref(ProductCrossRef(
            oem_part_number="04152-YZZA1",
            category="oil_filter",
            brand="Bosch",
            brand_part_number="P 7177",
        ))
        # Add cross-reference for air filter
        self.db.insert_crossref(ProductCrossRef(
            oem_part_number="17801-21050",
            category="air_filter",
            brand="MANN-FILTER",
            brand_part_number="C 26 013",
        ))
        self.filters = FilterLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_oil_filter_crossrefs_populated(self) -> None:
        vehicle = _make_vehicle()
        result = self.filters.lookup(vehicle)
        self.assertEqual(len(result.oil_filter_crossrefs), 2)
        brands = {xref.brand for xref in result.oil_filter_crossrefs}
        self.assertEqual(brands, {"MANN-FILTER", "Bosch"})

    def test_air_filter_crossrefs_populated(self) -> None:
        vehicle = _make_vehicle()
        result = self.filters.lookup(vehicle)
        self.assertEqual(len(result.air_filter_crossrefs), 1)
        self.assertEqual(result.air_filter_crossrefs[0].brand, "MANN-FILTER")

    def test_no_crossref_returns_empty_tuple(self) -> None:
        """Cabin/fuel filter have no cross-refs in DB — should be empty tuples."""
        vehicle = _make_vehicle()
        result = self.filters.lookup(vehicle)
        self.assertEqual(result.cabin_filter_crossrefs, ())
        self.assertEqual(result.fuel_filter_crossrefs, ())


# ── Tier 2: model-year fallback ──────────────────────────────────


class TestFilterLookupModelYearFallback(unittest.TestCase):
    """Tier 2 — model-year match (different engine, same chassis)."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        # DB has Corolla with 2ZR-FE — lookup for 1ZR-FE should fallback
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE",
            air_filter_oem="17801-21050",
            cabin_filter_oem="87139-02020",
        ))
        self.filters = FilterLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_different_engine_same_model_matches(self) -> None:
        """Corolla 1ZR-FE not in DB, but 2ZR-FE Corolla has filter data."""
        vehicle = _make_vehicle(engine_code="1ZR-FE")
        result = self.filters.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.air_filter_oem, "17801-21050")

    def test_model_year_source(self) -> None:
        vehicle = _make_vehicle(engine_code="1ZR-FE")
        result = self.filters.lookup(vehicle)
        self.assertEqual(result.source, "model_year")


# ── Cascade integration ──────────────────────────────────────────


class TestFilterLookupCascade(unittest.TestCase):
    """Integration: cascade short-circuits at the right tier."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE",
            air_filter_oem="17801-21050",
        ))
        self.filters = FilterLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_no_match_returns_none(self) -> None:
        """Vehicle with no matching specs returns None."""
        vehicle = _make_vehicle(
            make_english="Honda", model_english="Civic",
            make_hebrew="הונדה", model_hebrew="סיוויק",
            engine_code="K20C2",
        )
        result = self.filters.lookup(vehicle)
        self.assertIsNone(result)

    def test_specs_with_no_filter_data_cascades(self) -> None:
        """Exact match with no filter data cascades to model_year."""
        # Insert a second Corolla engine with NO filter data
        self.db.insert_specs(_make_specs(
            engine_code="1NR-FE",
            # No filter fields set
        ))
        vehicle = _make_vehicle(engine_code="1NR-FE")
        result = self.filters.lookup(vehicle)
        # Tier 1 finds 1NR-FE but has no filter data -> cascade
        # Tier 2 finds 2ZR-FE Corolla with filter data
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "model_year")
        self.assertEqual(result.air_filter_oem, "17801-21050")

    def test_hebrew_only_name_logs_warning(self) -> None:
        """Hebrew-only make name logs a warning about DB match likelihood."""
        vehicle = _make_vehicle(make_english=None)
        with self.assertLogs("parts_finder.lookup.filters", level="WARNING") as cm:
            self.filters.lookup(vehicle)
        self.assertTrue(
            any("Hebrew make name" in msg for msg in cm.output),
        )

    def test_no_make_returns_none(self) -> None:
        """Vehicle with no usable make name returns None."""
        vehicle = _make_vehicle(make_english=None, make_hebrew="")
        result = self.filters.lookup(vehicle)
        self.assertIsNone(result)

    def test_empty_engine_skips_tier1_tries_tier2(self) -> None:
        """Empty engine_code skips tier 1, hits tier 2."""
        vehicle = _make_vehicle(engine_code="")
        result = self.filters.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "model_year")


if __name__ == "__main__":
    unittest.main()
