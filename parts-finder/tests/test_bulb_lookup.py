"""Tests for bulb type lookup (two-tier cascade)."""

from __future__ import annotations

import unittest

from parts_finder.db import PartsDatabase
from parts_finder.models import BulbResult, VehicleRecord, VehicleSpecs
from parts_finder.lookup.bulbs import BulbLookup


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


# ── BulbResult construction ──────────────────────────────────────────


class TestBulbResultConstruction(unittest.TestCase):
    """BulbResult dataclass and factory method."""

    def test_from_vehicle_specs(self) -> None:
        specs = _make_specs(
            low_beam_bulb="H7",
            high_beam_bulb="H1",
            front_turn_bulb="PY21W",
            rear_turn_bulb="PY21W",
            tail_brake_bulb="P21/5W",
            reverse_bulb="W16W",
            fog_bulb="H11",
            license_plate_bulb="W5W",
        )
        result = BulbResult.from_vehicle_specs(specs, source="exact")

        self.assertEqual(result.low_beam, "H7")
        self.assertEqual(result.high_beam, "H1")
        self.assertEqual(result.front_turn, "PY21W")
        self.assertEqual(result.rear_turn, "PY21W")
        self.assertEqual(result.tail_brake, "P21/5W")
        self.assertEqual(result.reverse, "W16W")
        self.assertEqual(result.fog, "H11")
        self.assertEqual(result.license_plate, "W5W")
        self.assertEqual(result.source, "exact")

    def test_from_vehicle_specs_model_year_source(self) -> None:
        specs = _make_specs(low_beam_bulb="H7")
        result = BulbResult.from_vehicle_specs(specs, source="model_year")
        self.assertEqual(result.source, "model_year")

    def test_empty_fields_default(self) -> None:
        specs = _make_specs(low_beam_bulb="H7")
        result = BulbResult.from_vehicle_specs(specs, source="exact")
        self.assertEqual(result.high_beam, "")
        self.assertEqual(result.fog, "")

    def test_frozen(self) -> None:
        result = BulbResult(low_beam="H7", source="exact")
        with self.assertRaises(AttributeError):
            result.low_beam = "H4"  # type: ignore[misc]


# ── replaceable_positions ────────────────────────────────────────────


class TestReplaceablePositions(unittest.TestCase):
    """The replaceable_positions property filters LED and empty."""

    def test_filters_empty_entries(self) -> None:
        result = BulbResult(low_beam="H7", high_beam="", source="exact")
        rp = result.replaceable_positions
        self.assertIn("low_beam", rp)
        self.assertNotIn("high_beam", rp)

    def test_filters_led_entries(self) -> None:
        result = BulbResult(
            low_beam="LED", high_beam="H1", tail_brake="LED",
            source="exact",
        )
        rp = result.replaceable_positions
        self.assertNotIn("low_beam", rp)
        self.assertNotIn("tail_brake", rp)
        self.assertIn("high_beam", rp)
        self.assertEqual(rp["high_beam"], "H1")

    def test_all_led_returns_empty_dict(self) -> None:
        result = BulbResult(
            low_beam="LED", high_beam="LED", source="exact",
        )
        self.assertEqual(result.replaceable_positions, {})

    def test_all_empty_returns_empty_dict(self) -> None:
        result = BulbResult(source="exact")
        self.assertEqual(result.replaceable_positions, {})

    def test_mixed_positions(self) -> None:
        result = BulbResult(
            low_beam="H7",
            high_beam="LED",
            front_turn="PY21W",
            rear_turn="",
            fog="H11",
            license_plate="W5W",
            source="exact",
        )
        rp = result.replaceable_positions
        self.assertEqual(len(rp), 4)
        self.assertEqual(rp["low_beam"], "H7")
        self.assertEqual(rp["front_turn"], "PY21W")
        self.assertEqual(rp["fog"], "H11")
        self.assertEqual(rp["license_plate"], "W5W")


# ── Tier 1: exact match ─────────────────────────────────────────────


class TestBulbLookupExactMatch(unittest.TestCase):
    """Tier 1 — exact match on make + model + year + engine_code."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            low_beam_bulb="H7",
            high_beam_bulb="H1",
            fog_bulb="H11",
        ))
        self.bulbs = BulbLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_returns_result(self) -> None:
        vehicle = _make_vehicle()
        result = self.bulbs.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.low_beam, "H7")
        self.assertEqual(result.high_beam, "H1")
        self.assertEqual(result.fog, "H11")

    def test_exact_match_source(self) -> None:
        vehicle = _make_vehicle()
        result = self.bulbs.lookup(vehicle)
        self.assertEqual(result.source, "exact")

    def test_no_match_returns_none(self) -> None:
        vehicle = _make_vehicle(
            make_english="Honda", model_english="Civic",
            make_hebrew="הונדה", model_hebrew="סיוויק",
            engine_code="K20C2",
        )
        result = self.bulbs.lookup(vehicle)
        self.assertIsNone(result)


# ── Tier 2: model-year fallback ──────────────────────────────────────


class TestBulbLookupModelYearFallback(unittest.TestCase):
    """Tier 2 — model-year match (different engine, same chassis)."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        # DB has Corolla with 2ZR-FE — lookup for 1ZR-FE should fallback
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE",
            low_beam_bulb="H7",
            high_beam_bulb="H1",
        ))
        self.bulbs = BulbLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_different_engine_same_model_matches(self) -> None:
        """Corolla 1ZR-FE not in DB, but 2ZR-FE Corolla has bulb data."""
        vehicle = _make_vehicle(engine_code="1ZR-FE")
        result = self.bulbs.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.low_beam, "H7")

    def test_model_year_source(self) -> None:
        vehicle = _make_vehicle(engine_code="1ZR-FE")
        result = self.bulbs.lookup(vehicle)
        self.assertEqual(result.source, "model_year")

    def test_different_model_does_not_match(self) -> None:
        """Camry has no records — model-year fallback returns None."""
        vehicle = _make_vehicle(
            model_english="Camry", model_hebrew="קאמרי",
            engine_code="2AR-FE",
        )
        result = self.bulbs.lookup(vehicle)
        self.assertIsNone(result)


# ── Cascade integration ──────────────────────────────────────────────


class TestBulbLookupCascade(unittest.TestCase):
    """Integration: cascade short-circuits at the right tier."""

    def setUp(self) -> None:
        self.db = PartsDatabase(":memory:")
        self.db.insert_specs(_make_specs(
            engine_code="2ZR-FE",
            low_beam_bulb="H7",
            high_beam_bulb="H1",
        ))
        self.bulbs = BulbLookup(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_exact_match_short_circuits(self) -> None:
        """When exact match hits, cascade stops at tier 1."""
        vehicle = _make_vehicle()
        result = self.bulbs.lookup(vehicle)
        self.assertEqual(result.source, "exact")

    def test_falls_through_to_model_year(self) -> None:
        """Different engine on same model -> tier 2."""
        vehicle = _make_vehicle(engine_code="1ZR-FE")
        result = self.bulbs.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "model_year")

    def test_no_make_returns_none(self) -> None:
        """Vehicle with no usable make name returns None."""
        vehicle = _make_vehicle(
            make_english=None, make_hebrew="",
        )
        result = self.bulbs.lookup(vehicle)
        self.assertIsNone(result)

    def test_no_model_returns_none(self) -> None:
        """Vehicle with no usable model name returns None."""
        vehicle = _make_vehicle(
            model_english=None, model_hebrew="",
        )
        result = self.bulbs.lookup(vehicle)
        self.assertIsNone(result)

    def test_specs_with_no_bulb_data_returns_none(self) -> None:
        """VehicleSpecs with all empty bulb fields is treated as no match."""
        self.db.insert_specs(_make_specs(
            model="Yaris", engine_code="1NZ-FE",
            year_from=2019, year_to=2023,
            # No bulb fields set — all default to ""
        ))
        vehicle = _make_vehicle(
            model_english="Yaris", model_hebrew="יאריס",
            engine_code="1NZ-FE",
        )
        result = self.bulbs.lookup(vehicle)
        self.assertIsNone(result)

    def test_empty_engine_skips_tier1_tries_tier2(self) -> None:
        """Empty engine_code skips tier 1, hits tier 2."""
        vehicle = _make_vehicle(engine_code="")
        result = self.bulbs.lookup(vehicle)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "model_year")


if __name__ == "__main__":
    unittest.main()
