"""Tests for brake parts lookup (two-tier cascade with cross-references)."""

from __future__ import annotations

import logging
import unittest

from parts_finder.db import PartsDatabase
from parts_finder.models import (
    BrakeResult,
    ProductCrossRef,
    VehicleRecord,
    VehicleSpecs,
)
from parts_finder.lookup.brakes import BrakeLookup


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


# ── BrakeResult dataclass ──────────────────────────────────────────


class TestBrakeResultDataclass(unittest.TestCase):
    """BrakeResult frozen dataclass behavior."""

    def test_frozen(self) -> None:
        """BrakeResult is immutable."""
        result = BrakeResult(front_pad_oem="04465-02220", source="exact")
        with self.assertRaises(AttributeError):
            result.front_pad_oem = "changed"  # type: ignore[misc]

    def test_has_data_true_when_any_oem_set(self) -> None:
        """has_data returns True if at least one OEM field is non-empty."""
        self.assertTrue(BrakeResult(front_pad_oem="04465-02220").has_data)
        self.assertTrue(BrakeResult(rear_pad_oem="04466-02181").has_data)
        self.assertTrue(BrakeResult(front_disc_oem="43512-02330").has_data)
        self.assertTrue(BrakeResult(rear_disc_oem="42431-02250").has_data)

    def test_has_data_false_when_empty(self) -> None:
        """has_data returns False when all OEM fields are empty."""
        self.assertFalse(BrakeResult().has_data)

    def test_has_data_false_with_only_fluid(self) -> None:
        """Brake fluid alone is not actionable — has_data should be False."""
        self.assertFalse(BrakeResult(brake_fluid_type="DOT 4").has_data)


# ── Tier 1: exact match ───────────────────────────────────────────


class TestBrakeLookupExactMatch(unittest.TestCase):
    """Tier 1 — exact match on make + model + year + engine_code."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            front_brake_pad_oem="04465-02220",
            rear_brake_pad_oem="04466-02181",
            front_rotor_oem="43512-02330",
            rear_rotor_oem="42431-02250",
            brake_fluid_type="DOT 4",
        ))
        self.brakes = BrakeLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_returns_all_oem_fields(self) -> None:
        vehicle = _make_vehicle()
        result = self.brakes.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.front_pad_oem, "04465-02220")
        self.assertEqual(result.rear_pad_oem, "04466-02181")
        self.assertEqual(result.front_disc_oem, "43512-02330")
        self.assertEqual(result.rear_disc_oem, "42431-02250")

    def test_exact_match_source(self) -> None:
        vehicle = _make_vehicle()
        result = self.brakes.lookup(vehicle)
        self.assertEqual(result.source, "exact")

    def test_brake_fluid_type_included(self) -> None:
        vehicle = _make_vehicle()
        result = self.brakes.lookup(vehicle)
        self.assertEqual(result.brake_fluid_type, "DOT 4")

    def test_partial_data_only_front_pads(self) -> None:
        """Specs with only front pad OEM still return a partial result."""
        self.db.insert_specs(_make_specs(
            model="Yaris",
            engine_code="1NZ-FE",
            front_brake_pad_oem="04465-52260",
            # All other brake fields empty
        ))
        vehicle = _make_vehicle(
            model_english="Yaris", model_hebrew="יאריס",
            engine_code="1NZ-FE",
        )
        result = self.brakes.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.front_pad_oem, "04465-52260")
        self.assertEqual(result.rear_pad_oem, "")
        self.assertEqual(result.front_disc_oem, "")
        self.assertEqual(result.rear_disc_oem, "")
        self.assertTrue(result.has_data)


# ── Cross-references ───────────────────────────────────────────────


class TestBrakeLookupCrossRefs(unittest.TestCase):
    """Cross-references populated from product_crossref table."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            front_brake_pad_oem="04465-02220",
            rear_brake_pad_oem="04466-02181",
            front_rotor_oem="43512-02330",
            rear_rotor_oem="42431-02250",
            brake_fluid_type="DOT 4",
        ))
        # Add cross-references for front pads
        self.db.insert_crossref(ProductCrossRef(
            oem_part_number="04465-02220",
            category="brake_pad",
            brand="Brembo",
            brand_part_number="P 83 137",
        ))
        self.db.insert_crossref(ProductCrossRef(
            oem_part_number="04465-02220",
            category="brake_pad",
            brand="TRW",
            brand_part_number="GDB3548",
        ))
        # Add cross-reference for front disc
        self.db.insert_crossref(ProductCrossRef(
            oem_part_number="43512-02330",
            category="brake_disc",
            brand="Brembo",
            brand_part_number="09.C398.11",
        ))
        self.brakes = BrakeLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_front_pad_crossrefs_populated(self) -> None:
        vehicle = _make_vehicle()
        result = self.brakes.lookup(vehicle)
        self.assertEqual(len(result.front_pad_crossrefs), 2)
        brands = {xref.brand for xref in result.front_pad_crossrefs}
        self.assertEqual(brands, {"Brembo", "TRW"})

    def test_front_disc_crossrefs_populated(self) -> None:
        vehicle = _make_vehicle()
        result = self.brakes.lookup(vehicle)
        self.assertEqual(len(result.front_disc_crossrefs), 1)
        self.assertEqual(result.front_disc_crossrefs[0].brand, "Brembo")

    def test_no_crossref_returns_empty_tuple(self) -> None:
        """Rear pad/disc have no cross-refs in DB — should be empty tuples."""
        vehicle = _make_vehicle()
        result = self.brakes.lookup(vehicle)
        self.assertEqual(result.rear_pad_crossrefs, ())
        self.assertEqual(result.rear_disc_crossrefs, ())


# ── Tier 2: model-year fallback ────────────────────────────────────


class TestBrakeLookupModelYearFallback(unittest.TestCase):
    """Tier 2 — model-year match (different engine, same chassis)."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        # DB has Corolla with 2ZR-FE — lookup for 1ZR-FE should fallback
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE",
            front_brake_pad_oem="04465-02220",
            rear_brake_pad_oem="04466-02181",
        ))
        self.brakes = BrakeLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_different_engine_same_model_matches(self) -> None:
        """Corolla 1ZR-FE not in DB, but 2ZR-FE Corolla has brake data."""
        vehicle = _make_vehicle(engine_code="1ZR-FE")
        result = self.brakes.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.front_pad_oem, "04465-02220")

    def test_model_year_source(self) -> None:
        vehicle = _make_vehicle(engine_code="1ZR-FE")
        result = self.brakes.lookup(vehicle)
        self.assertEqual(result.source, "model_year")


# ── Cascade integration ────────────────────────────────────────────


class TestBrakeLookupCascade(unittest.TestCase):
    """Integration: cascade short-circuits at the right tier."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE",
            front_brake_pad_oem="04465-02220",
        ))
        self.brakes = BrakeLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_no_match_returns_none(self) -> None:
        """Vehicle with no matching specs returns None."""
        vehicle = _make_vehicle(
            make_english="Honda", model_english="Civic",
            make_hebrew="הונדה", model_hebrew="סיוויק",
            engine_code="K20C2",
        )
        result = self.brakes.lookup(vehicle)
        self.assertIsNone(result)

    def test_specs_with_no_brake_data_cascades(self) -> None:
        """Exact match with no brake data cascades to model_year."""
        # Insert a second Corolla engine with NO brake data
        self.db.insert_specs(_make_specs(
            engine_code="1NR-FE",
            # No brake fields set
        ))
        vehicle = _make_vehicle(engine_code="1NR-FE")
        result = self.brakes.lookup(vehicle)
        # Tier 1 finds 1NR-FE but has no brake data -> cascade
        # Tier 2 finds 2ZR-FE Corolla with brake data
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "model_year")
        self.assertEqual(result.front_pad_oem, "04465-02220")

    def test_hebrew_only_name_logs_warning(self) -> None:
        """Hebrew-only make name logs a warning about DB match likelihood."""
        vehicle = _make_vehicle(make_english=None)
        with self.assertLogs("parts_finder.lookup.brakes", level="WARNING") as cm:
            self.brakes.lookup(vehicle)
        self.assertTrue(
            any("Hebrew make name" in msg for msg in cm.output),
        )

    def test_no_make_returns_none(self) -> None:
        """Vehicle with no usable make name returns None."""
        vehicle = _make_vehicle(make_english=None, make_hebrew="")
        result = self.brakes.lookup(vehicle)
        self.assertIsNone(result)

    def test_empty_engine_skips_tier1_tries_tier2(self) -> None:
        """Empty engine_code skips tier 1, hits tier 2."""
        vehicle = _make_vehicle(engine_code="")
        result = self.brakes.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "model_year")


if __name__ == "__main__":
    unittest.main()
